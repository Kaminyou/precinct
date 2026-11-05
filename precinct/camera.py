"""Camera node: every camera is a *member* (sends detections to its region's
center); R of them are Raft *voters*; the Raft leader acts as the *center*."""
from __future__ import annotations

import math

import numpy as np

from .raft import Entry, RaftPeer
from .sim import Msg, Node
from .tracker import RegionTracker, StepContext, TrackTable, Z_BYTES

AGG = "agg"
CTRL = ("OFFER", "ACCEPT", "RELEASE", "RELINQ", "MERGE")


class CameraNode(Node):
    def __init__(self, sim, system, cid, pos, region, voters):
        super().__init__(sim, cid)
        self.sys = system
        self.pos = pos
        self.phase = float(sim.rng.uniform(0, sim.cfg.dt))
        self.region = region
        self.dir = system.directory_snapshot
        self.center = None
        self.center_term = -1
        self.synced = False
        self.last_hello = sim.now
        self.sync_round = None
        self.raft = None
        self.committed = None
        self.role = None
        self.left = False
        self.joined = True
        if cid in voters:
            self._make_raft(voters)

    # ============================================================ lifecycle
    def _make_raft(self, voters, term=0):
        self.raft = RaftPeer(self, voters, term)
        self.committed = TrackTable(self.region)

    def start(self):
        if self.raft is not None:
            # the voter nearest the centroid campaigns first to speed up bootstrap
            first = self.raft.voters[0] == self.id
            self.raft.start(initial_delay=0.01 if first else None)
        self.timer(self.sim.cfg.T_hello, self._check_center)
        self.timer(self.sim.cfg.T_sync, self._periodic_sync)

    def on_crash(self):
        self.synced = False
        self.center = None
        if self.role is not None:
            self.role = None
        self.sys.on_node_crash(self)

    def on_recover(self):
        # durable Raft state survives; volatile state is rebuilt
        if self.raft is not None:
            r = self.raft
            r.role, r.leader, r.commit, r.applied = r.FOLLOWER, None, r.snap_index, r.snap_index
            self.committed = TrackTable(self.region)
            if r.snap_state is not None:
                self.sm_restore(r.snap_state)
            r.active = False                     # wait for the directory's answer
        self.center = None
        self.synced = False
        self.joined = False
        self.send(AGG, Msg("JOIN", {"cam": self.id, "pos": self.pos, "rejoin": True}, 48))
        self.timer(1.0, self._join_retry)
        self.timer(self.sim.cfg.T_hello, self._check_center)
        self.timer(self.sim.cfg.T_sync, self._periodic_sync)

    def _join_retry(self):
        if not self.joined:
            self.send(AGG, Msg("JOIN", {"cam": self.id, "pos": self.pos, "rejoin": True}, 48))
            self.timer(1.0, self._join_retry)

    def on_JOIN_R(self, m):
        d = m.data
        self.joined = True
        self.region = d["region"]
        self.dir = d["dir"]
        if self.raft is not None:
            if self.id in d["voters"]:
                if not self.raft.active:
                    self.raft.active = True
                    self.raft.start()
            else:
                self.raft = None
                self.committed = None
        if d["center"] is not None:
            self._set_center(d["center"], -1)

    # ============================================================ member role
    def can_emit(self):
        return self.alive and self.synced and self.center is not None and not self.left

    def _set_center(self, leader, term):
        if leader != self.center:
            self.center = leader
            self.synced = False
            self._sync_start()
        self.center_term = max(self.center_term, term)
        self.last_hello = self.sim.now

    def on_HELLO(self, m):
        d = m.data
        if d["term"] >= self.center_term or self.center is None:
            if d["region"] != self.region:
                self.region = d["region"]
            self._set_center(m.src, d["term"])
        if self.raft is None and self.id in d["voters"]:
            pass   # the leader will contact us with AppendEntries / snapshot

    def on_LEADER_IS(self, m):
        d = m.data
        if d["leader"] is not None and d["term"] >= self.center_term:
            self._set_center(d["leader"], d["term"])

    def on_WHO(self, m):
        if self.raft is not None and self.raft.leader is not None:
            self.send(m.src, Msg("LEADER_IS", {"leader": self.raft.leader, "term": self.raft.term}, 24))

    def _check_center(self):
        c = self.sim.cfg
        if self.role is None and self.sim.now - self.last_hello > c.T_center_timeout:
            for v in self.dir.voters.get(self.region, []):
                if v != self.id:
                    self.send(v, Msg("WHO", {}, 16))
            self.send(AGG, Msg("WHO", {"region": self.region}, 16))
            self.last_hello = self.sim.now - c.T_center_timeout + c.T_hello  # back off
        self.timer(c.T_hello, self._check_center)

    # ---- clock synchronisation (Cristian, min-RTT of a 3-probe burst)
    def _sync_start(self):
        target = AGG if self.role is not None else self.center
        if target is None:
            return
        self.sync_round = (self.incarnation, self.sim.now)
        self._best = (math.inf, None)
        kind = "CSYNC" if target == AGG else "SYNC"
        for i in range(3):
            self.send(target, Msg(kind, {"t0": self.local(), "round": self.sync_round}, 24))

    def _periodic_sync(self):
        self._sync_start()
        self.timer(self.sim.cfg.T_sync, self._periodic_sync)

    def on_SYNC(self, m):
        # center answers members with its estimate of reference time
        self.send(m.src, Msg("SYNC_R", {"t0": m.data["t0"], "ts": self.ref_now(),
                                        "round": m.data["round"]}, 24))

    def on_SYNC_R(self, m):
        d = m.data
        t1 = self.local()
        rtt = t1 - d["t0"]
        if d["round"] != self.sync_round or rtt >= self._best[0]:
            if d["round"] == self.sync_round and not self.synced:
                self.synced = True
            return
        self._best = (rtt, d["ts"])
        self.theta_hat = (d["t0"] + t1) / 2 - d["ts"]
        self.synced = True

    on_CSYNC_R = on_SYNC_R

    # ============================================================ voter role
    def sm_apply(self, e):
        if e.kind == "step" and self.committed is not None:
            self.committed.apply(e.data)

    def sm_snapshot(self):
        return self.committed.copy(), self.committed.size()

    def sm_restore(self, state):
        self.committed = state.copy()

    def raft_removed(self):
        self.raft = None
        self.committed = None
        self.sim.log("voter_removed", region=self.region, node=self.id)

    def on_RV(self, m):
        if self.raft is not None:
            self.raft.on_RV(m)
        else:
            self.send(m.src, Msg("NOT_MEMBER", {"term": m.data["term"]}, 16))

    def on_RV_R(self, m):
        if self.raft is not None:
            self.raft.on_RV_R(m)

    def on_AE(self, m):
        if self.raft is None:
            # a leader added us as a voter: join with an empty log
            self._make_raft([m.data["leader"]] + [self.id], m.data["term"])
            self.raft.active = True
            self.raft.start()
        self.raft.on_AE(m)

    def on_AE_R(self, m):
        if self.raft is not None:
            self.raft.on_AE_R(m)

    def on_SNAP(self, m):
        if self.raft is None:
            self._make_raft(list(m.data["voters"]), m.data["term"])
            self.raft.start()
        self.raft.on_SNAP(m)

    def on_NOT_MEMBER(self, m):
        if self.raft is not None:
            self.raft.on_NOT_MEMBER(m)

    def on_APPOINT(self, m):
        """No-replication ablation: the directory appoints a fresh center."""
        d = m.data
        self.region = d["region"]
        self.raft = RaftPeer(self, [self.id], d["term"])
        self.committed = TrackTable(self.region)
        self.raft.term = d["term"]
        self.raft._start_election()

    # ============================================================ center role
    def raft_became_leader(self):
        self.role = CenterRole(self)
        self.role.start()

    def raft_stepped_down(self):
        if self.role is not None:
            self.sim.log("center_down", region=self.region, node=self.id)
        self.role = None

    def raft_committed(self, e):
        if self.role is not None:
            self.role.committed(e)

    def _forward_or_drop(self, m):
        if self.raft is not None and self.raft.leader not in (None, self.id):
            self.send(self.raft.leader, Msg(m.kind, m.data, m.size))

    def on_DET(self, m):
        if self.role is not None:
            self.role.on_det(m)
        elif self.raft is not None and self.raft.leader not in (None, self.id):
            self.send(m.src, Msg("LEADER_IS", {"leader": self.raft.leader, "term": self.raft.term}, 24))

    def _ctrl(self, m):
        if self.role is not None:
            self.role.on_ctrl(m)
        else:
            self._forward_or_drop(m)

    on_OFFER = on_ACCEPT = on_RELEASE = on_RELINQ = on_MERGE = _ctrl

    def on_LEAVE(self, m):
        """Graceful departure: treat the camera as failed right away so that a
        departing voter is replaced while it is still up."""
        if self.role is not None:
            self.role.member_left(m.data["cam"])

    def on_SNAPREQ(self, m):
        if self.role is not None:
            self.role.on_snapreq(m)

    def on_DIRUPD(self, m):
        d = m.data["dir"]
        if d.version > self.dir.version:
            self.dir = d
        if self.role is not None:
            self.role.on_dir()


class CenterRole:
    """State and behaviour of a camera while it is its region's Raft leader."""

    def __init__(self, node: CameraNode):
        self.n = node
        self.sim = node.sim
        self.cfg = node.sim.cfg
        self.tracker = RegionTracker(self.cfg)
        self.buf = {}             # step -> {cam: batch}
        self.arr = {}             # step -> [n_det, sum capture, sum arrival]
        self.ctrl = []
        self.last_heard = {}
        self.failed = set()
        self.region_cache = {}    # region -> center node id
        self.pending_reports = {}
        self.next_k = None

    # ------------------------------------------------------------ setup
    def start(self):
        n = self.n
        raft = n.raft
        # working state = committed state + uncommitted suffix of our log
        self.work = n.committed.copy()
        for i in range(raft.applied + 1, raft.last_index + 1):
            e = raft.entry(i)
            if e.kind == "step":
                self.work.apply(e.data)
        self.term = raft.term
        now = self.sim.now
        for c in n.dir.members(n.region):
            self.last_heard[c] = now
        cfg = self.cfg
        ref = n.ref_now()
        k = max(self.work.last_step + 1, int(math.floor((ref - cfg.W) / cfg.dt)))
        while (k + 1) * cfg.dt + cfg.W <= ref:
            k += 1
        self.next_k = k
        n.timer_at_ref((k + 1) * cfg.dt + cfg.W, self._deadline, k)
        self._hello()
        n.center = n.id
        n.synced = True
        n._sync_start()
        self.sim.log("center_up", region=n.region, node=n.id, term=raft.term)

    def _hello(self):
        n = self.n
        if n.role is not self:
            return
        voters = list(n.raft.voters)
        for c in n.dir.members(n.region):
            if c != n.id:
                n.send(c, Msg("HELLO", {"region": n.region, "term": n.raft.term, "voters": voters}, 32))
        n.send(AGG, Msg("CHELLO", {"region": n.region, "term": n.raft.term, "voters": voters,
                                   "version": n.dir.version, "failed": sorted(self.failed)}, 48))
        n.timer(self.cfg.T_hello, self._hello)

    def on_dir(self):
        now = self.sim.now
        for c in self.n.dir.members(self.n.region):
            self.last_heard.setdefault(c, now)

    # ------------------------------------------------------------ inputs
    def on_det(self, m):
        d = m.data
        k = d["k"]
        self.last_heard[m.src] = self.sim.now
        if m.src in self.failed:
            self.failed.discard(m.src)
        if k < self.next_k:
            self.sim.stats.counters["late_det"] += 1
            self.sim.stats.counters["late_det_n"] += len(d["P"])
            return
        self.buf.setdefault(k, {})[m.src] = (m.src, d["tau"], d["P"], d["Z"], d["C"])
        a = self.arr.setdefault(k, [0, 0.0, 0.0, 0])
        a[0] += len(d["P"])
        a[1] += d["t_cap"] * len(d["P"])
        a[2] += self.sim.now * len(d["P"])
        a[3] += 1
        if k == self.next_k:
            expected = [c for c in self.n.dir.members(self.n.region) if c not in self.failed]
            if len(self.buf[k]) >= len(expected):
                self._close(k)

    def on_ctrl(self, m):
        src_region = m.data.get("src")
        if src_region is not None:
            self.region_cache[src_region] = m.src
        self.ctrl.append(m)

    def on_snapreq(self, m):
        c = self.n.committed
        tracks = [r.to_report(True) for r in c.tracks.values() if r.owner and r.status == "C"]
        lost = [r.to_report(True) + (r.lost_t,) for r in c.lost.values() if r.owner]
        self.n.send(AGG, Msg("SNAPSHOT", {"region": self.n.region, "step": c.last_step,
                                          "tracks": tracks, "lost": lost, "aliases": dict(c.aliases),
                                          "term": self.n.raft.term},
                             32 + (len(tracks) + len(lost)) * (40 + Z_BYTES) + 24 * len(c.aliases)))

    # ------------------------------------------------------------ step
    def _deadline(self, k):
        if self.n.role is self and k == self.next_k:
            self._close(k)

    def _close(self, k):
        n = self.n
        if n.role is not self:
            return
        self.next_k = k + 1
        n.timer_at_ref((k + 2) * self.cfg.dt + self.cfg.W, self._deadline, k + 1)
        t_close = self.sim.now
        n.charged(self._process, k)
        self.sys_record(k, t_close)

    def _process(self, k):
        n, cfg = self.n, self.cfg
        batches = list(self.buf.pop(k, {}).values())
        for kk in [x for x in self.buf if x < k]:
            del self.buf[kk]
        ctrl, self.ctrl = self.ctrl, []
        ctx = StepContext(cfg, n.region, self.term, n.dir, n.ref_now(), n.id)
        delta, size, out, rep, info = self.tracker.step(self.work, ctx, k, batches, ctrl)
        nz = sum(1 for t in rep if t[-1] is not None)
        report = Msg("REPORT", {"region": n.region, "k": k, "term": self.term, "tracks": rep,
                                "aliases": info["aliases"], "node": n.id},
                     32 + 40 * len(rep) + Z_BYTES * nz + 24 * len(info["aliases"]))
        out.append(("agg", None, report))
        self._log_info(info)
        self._failure_detection()
        self.pending_reports[k] = None
        n.raft.propose(Entry(self.term, "step", delta, size, out))
        self._last_info = info

    def _log_info(self, info):
        reg = self.n.region
        for g, prev, ep, why in info["accepts"]:
            self.sim.log("accept", region=reg, gid=g, prev=prev, epoch=ep, why=why)
        for g, to, ep in info["releases"]:
            self.sim.log("release", region=reg, gid=g, to=to, epoch=ep)
        for lo, su in info["merges"]:
            self.sim.log("merge_applied", region=reg, loser=lo, survivor=su)
        st = self.sim.stats.counters
        st["handoff_accepts"] += len(info["accepts"])
        st["early_merges"] += info["early_merge"]
        st["adopted"] += info["adopted"]
        st["revived"] += info["revived"]
        st["new_tracks"] += info["new"]

    def sys_record(self, k, t_close):
        a = self.arr.pop(k, None)
        for kk in [x for x in self.arr if x < k]:
            del self.arr[kk]
        self.n.sys.rec.center_step(self.n.region, k, t_close, self.n.busy_until, a, self.n.id)

    def committed(self, e):
        n = self.n
        if e.kind != "step" or not e.out:
            return
        for kind, tgt, msg in e.out:
            if kind == "agg":
                if msg.kind == "REPORT":
                    n.sys.rec.center_commit(n.region, msg.data["k"], max(self.sim.now, n.busy_until))
                n.send(AGG, msg)
            else:
                dst = self.region_cache.get(tgt) or n.dir.centers.get(tgt)
                if dst is None:
                    vs = n.dir.voters.get(tgt)
                    if not vs:
                        continue
                    dst = vs[int(self.sim.rng.integers(len(vs)))]
                n.send(dst, msg)

    # ------------------------------------------------------------ membership
    def member_left(self, cam):
        if cam not in self.failed:
            self.failed.add(cam)
            self.n.send(AGG, Msg("MFAIL", {"region": self.n.region, "cams": [cam]}, 20))

    def _failure_detection(self):
        n, cfg = self.n, self.cfg
        now = self.sim.now
        lim = cfg.n_hb * cfg.dt
        newly = []
        for c in n.dir.members(n.region):
            if c == n.id or c in self.failed:
                continue
            if now - self.last_heard.get(c, now) > lim:
                self.failed.add(c)
                newly.append(c)
        if newly:
            self.sim.log("member_failed", region=n.region, cams=newly)
            n.send(AGG, Msg("MFAIL", {"region": n.region, "cams": newly}, 16 + 4 * len(newly)))
        raft = n.raft
        if raft.config_pending() or len(raft.voters) == 0:
            return
        dead = [v for v in raft.voters if v in self.failed]
        if dead:
            nv = [v for v in raft.voters if v != dead[0]]
            if raft.change_voters(nv):
                self.sim.log("voter_change", region=n.region, removed=dead[0], voters=nv)
            return
        if len(raft.voters) < cfg.R:
            cand = [c for c in n.dir.members(n.region)
                    if c not in raft.voters and c not in self.failed
                    and now - self.last_heard.get(c, -1e9) < lim]
            if cand:
                cen = n.dir.centroids[n.region]
                best = min(cand, key=lambda c: np.linalg.norm(n.dir.cam_pos[c] - cen))
                nv = list(raft.voters) + [best]
                if raft.change_voters(nv):
                    self.sim.log("voter_change", region=n.region, added=best, voters=nv)
