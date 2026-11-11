"""Discrete-event simulation engine: event queue, nodes with local clocks,
an unreliable network, and computation-time charging."""
from __future__ import annotations

import heapq
import itertools
import math
import time
from collections import defaultdict

import numpy as np


class Msg:
    __slots__ = ("kind", "src", "dst", "data", "size", "t_sent")

    def __init__(self, kind, data=None, size=64):
        self.kind = kind
        self.data = data
        self.size = size
        self.src = None
        self.dst = None
        self.t_sent = 0.0


# message kinds grouped by tier for overhead accounting
TIER = {
    "DET": "member-center", "HELLO": "member-center", "SYNC": "member-center",
    "SYNC_R": "member-center", "WHO": "member-center", "LEADER_IS": "member-center",
    "RV": "raft", "RV_R": "raft", "AE": "raft", "AE_R": "raft", "SNAP": "raft",
    "SNAP_R": "raft", "NOT_MEMBER": "raft",
    "OFFER": "center-center", "ACCEPT": "center-center", "RELEASE": "center-center",
    "RELINQ": "center-center",
    "REPORT": "center-agg", "MERGE": "center-agg", "MERGE_ACK": "center-agg",
    "SNAPREQ": "center-agg", "SNAPSHOT": "center-agg", "CSYNC": "center-agg",
    "CSYNC_R": "center-agg",
    "CHELLO": "directory", "DIRUPD": "directory", "JOIN": "directory", "JOIN_R": "directory",
    "LEAVE": "directory", "MFAIL": "directory", "MALIVE": "directory", "APPOINT": "directory",
}


class Sim:
    def __init__(self, cfg, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.now = 0.0
        self._q = []
        self._seq = itertools.count()
        self.nodes = {}
        self.stats = Stats()
        self.events = []          # (t, kind, info dict) for traces/analysis
        self.listeners = []       # callables(t, kind, info)

    # ---------------- events ----------------
    def at(self, t, fn, *args):
        heapq.heappush(self._q, (t, next(self._seq), fn, args))

    def after(self, d, fn, *args):
        self.at(self.now + d, fn, *args)

    def run_until(self, t_end):
        q = self._q
        while q and q[0][0] <= t_end:
            t, _, fn, args = heapq.heappop(q)
            self.now = t
            fn(*args)
        self.now = t_end

    def log(self, kind, **info):
        self.events.append((self.now, kind, info))
        for f in self.listeners:
            f(self.now, kind, info)

    # ---------------- network ----------------
    def delay(self):
        c = self.cfg
        return c.delta0 + c.delay_median * math.exp(c.delay_sigma * self.rng.standard_normal())

    def send(self, src, dst_id, msg: Msg, t_send=None):
        t_send = self.now if t_send is None else t_send
        msg.src = src.id
        msg.dst = dst_id
        msg.t_sent = t_send
        st = self.stats
        tier = TIER.get(msg.kind, "other")
        st.msgs[tier] += 1
        st.bytes[tier] += msg.size
        st.node_out[src.id] += msg.size
        if dst_id == src.id:
            self.at(t_send, self._deliver, msg)
            return
        if self.rng.random() < self.cfg.p_loss:
            st.lost[tier] += 1
            return
        self.at(t_send + self.delay(), self._deliver, msg)

    def _deliver(self, msg):
        node = self.nodes.get(msg.dst)
        if node is None:
            return
        node._deliver(msg)


class Stats:
    def __init__(self):
        self.msgs = defaultdict(int)
        self.bytes = defaultdict(int)
        self.lost = defaultdict(int)
        self.node_in = defaultdict(int)
        self.node_out = defaultdict(int)
        self.busy = defaultdict(float)     # charged compute seconds per node
        self.counters = defaultdict(int)


class Node:
    """An actor with a local clock. Handlers are methods named on_<KIND>.

    Charged handlers (see `charged`) measure their real execution time, scaled
    by kappa, and keep the node busy for that long; messages they send leave
    the node only when the computation finishes."""

    def __init__(self, sim: Sim, nid):
        self.sim = sim
        self.id = nid
        self.alive = True
        self.incarnation = 0
        self.busy_until = 0.0
        self._outbox = None
        c = sim.cfg
        self.theta = float(sim.rng.uniform(-c.theta_max, c.theta_max))
        self.rho = float(sim.rng.uniform(-c.rho_max, c.rho_max))
        self.theta_hat = 0.0          # estimated offset of local clock vs reference
        self.best_rtt = math.inf
        self.sync_time = -math.inf
        sim.nodes[nid] = self

    # ---------------- clocks ----------------
    def local(self, t=None):
        t = self.sim.now if t is None else t
        return t * (1.0 + self.rho) + self.theta

    def ref_now(self, t=None):
        """Node's estimate of reference (aggregator) time."""
        return self.local(t) - self.theta_hat

    def true_of_ref(self, x):
        return (x + self.theta_hat - self.theta) / (1.0 + self.rho)

    # ---------------- timers ----------------
    def timer(self, delay, fn, *args):
        inc = self.incarnation
        self.sim.after(delay, self._fire, inc, fn, args)

    def timer_at_ref(self, x, fn, *args):
        inc = self.incarnation
        self.sim.at(max(self.sim.now, self.true_of_ref(x)), self._fire, inc, fn, args)

    def _fire(self, inc, fn, args):
        if not self.alive or inc != self.incarnation:
            return
        if self.busy_until > self.sim.now:
            self.sim.at(self.busy_until, self._fire, inc, fn, args)
            return
        fn(*args)

    # ---------------- messaging ----------------
    def send(self, dst_id, msg):
        if self._outbox is not None:
            self._outbox.append((dst_id, msg))
        else:
            self.sim.send(self, dst_id, msg, max(self.sim.now, self.busy_until))

    def _deliver(self, msg):
        if not self.alive:
            return
        if self.busy_until > self.sim.now:
            self.sim.at(self.busy_until, self._deliver, msg)
            return
        self.sim.stats.node_in[self.id] += msg.size
        h = getattr(self, "on_" + msg.kind, None)
        if h is not None:
            h(msg)

    def charged(self, fn, *args):
        """Run fn, charging its measured compute time to this node."""
        sim = self.sim
        self._outbox = []
        t0 = time.perf_counter()
        try:
            res = fn(*args)
        finally:
            el = time.perf_counter() - t0
            out, self._outbox = self._outbox, None
        cost = el * sim.cfg.kappa if sim.cfg.cost_model == "measured" else 0.0
        self.busy_until = max(sim.now, self.busy_until) + cost
        sim.stats.busy[self.id] += cost
        for dst, m in out:
            sim.send(self, dst, m, self.busy_until)
        return res

    # ---------------- faults ----------------
    def crash(self):
        if not self.alive:
            return
        self.alive = False
        self.incarnation += 1
        self.busy_until = 0.0
        self.on_crash()

    def recover(self):
        if self.alive:
            return
        self.alive = True
        self.incarnation += 1
        self.on_recover()

    def on_crash(self):
        pass

    def on_recover(self):
        pass
