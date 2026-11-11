"""Builds and drives a full simulated deployment."""
from __future__ import annotations

import json
import math
import time

import numpy as np

from .aggregator import AggregatorNode, Directory
from .camera import CameraNode
from .config import Config
from .metrics import Recorder, evaluate, failover_continuity, latency, overhead
from .regions import balanced_kmeans, choose_voters
from .sim import Msg, Sim
from .tracker import gid_str
from .world import World, place_cameras


class System:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        base = np.random.SeedSequence(cfg.seed)
        s_world, s_cam, s_net, s_fault = base.spawn(4)
        self.sim = Sim(cfg, np.random.default_rng(s_net))
        self.world = World(cfg, np.random.default_rng(s_world))
        self.fault_rng = np.random.default_rng(s_fault)
        cam_rng = np.random.default_rng(s_cam)
        P = place_cameras(cfg, cam_rng)
        lab, C = balanced_kmeans(P, cfg.K, seed=0)
        voters = choose_voters(P, lab, C, cfg.R)
        self.directory = Directory(P, lab, voters, C)
        self.chello_t = {}
        self.directory_snapshot = self.directory.snap
        self.rec = Recorder(cfg)
        self.agg = AggregatorNode(self.sim, self, self.directory)
        self.chello_t = self.directory.__dict__.setdefault("chello_t", {})
        self.cams = []
        for i in range(len(P)):
            self.cams.append(CameraNode(self.sim, self, i, P[i], int(lab[i]), voters[int(lab[i])]))
        self.cam_pos = P.copy()
        self.n_det_sent = 0
        self.trace = None
        self.frame_listeners = []
        self.k = 0
        for c in self.cams:
            c.start()
        self.agg.start()
        self.sim.at(0.0, self._tick, 0)
        for f in cfg.faults:
            self.sim.at(f[0], self.inject, *f[1:])
        if cfg.member_crash_rate > 0:
            self._schedule_random_crash()
        if cfg.join_rate > 0:
            self.sim.at(self.fault_rng.exponential(1 / cfg.join_rate), self._random_join)
        if cfg.leave_rate > 0:
            self.sim.at(self.fault_rng.exponential(1 / cfg.leave_rate), self._random_leave)
        if cfg.trace:
            self.trace = open(cfg.trace, "w")
            self.trace.write(json.dumps(self.header()) + "\n")

    # ------------------------------------------------------------ driving
    def run(self, until=None, progress=False):
        until = self.cfg.duration if until is None else until
        t0 = time.time()
        step = 5.0
        t = self.sim.now
        while t < until:
            t = min(until, t + step)
            self.sim.run_until(t)
            if progress:
                print(f"  t={t:6.1f}s  wall={time.time() - t0:7.1f}s", flush=True)
        if self.trace:
            self.trace.close()
            self.trace = None
        return time.time() - t0

    def _tick(self, k):
        cfg = self.cfg
        if k > 0:
            self.world.step(cfg.dt)
            ids, pos = self.world.snapshot()
            obs = self.directory_live_snapshot().covered(pos, cfg.r)
            self.rec.gt[k - 1] = (ids, pos, obs)
            self._emit_frame(k - 1)
        self.k = k
        self._detect(k)
        self.sim.at((k + 1) * cfg.dt, self._tick, k + 1)

    def directory_live_snapshot(self):
        # ground-truth coverage uses the actual live cameras (oracle, evaluation only)
        if getattr(self, "_live_cache_k", None) != self.k:
            self._live_cache_k = self.k
            self._live_cache = self.directory.snap
            alive = np.array([c.alive and not c.left for c in self.cams])
            if alive.sum() != len(self.directory.snap.live_ids) or \
                    not np.array_equal(np.flatnonzero(alive), self.directory.snap.live_ids):
                from .regions import DirSnapshot
                d = self.directory
                self._live_cache = DirSnapshot(-1, self.cam_pos, d.cam_region, alive, {}, {},
                                               d.centroids)
        return self._live_cache

    def _detect(self, k):
        cfg = self.cfg
        rng = self.world.rng
        emit = np.array([c.can_emit() for c in self.cams])
        emit &= rng.random(len(emit)) >= cfg.p_timeout
        phase = np.array([c.phase for c in self.cams])
        res = self.world.detect(self.cam_pos, phase, emit)
        t_k = k * cfg.dt
        groups = {}
        if res is not None:
            ci, gt, ph, p, z, conf, _ = res
            bounds = np.flatnonzero(np.diff(ci)) + 1
            starts = np.concatenate([[0], bounds])
            ends = np.concatenate([bounds, [len(ci)]])
            for s, e in zip(starts, ends):
                groups[int(ci[s])] = (s, e)
        for i in np.flatnonzero(emit):
            cam = self.cams[i]
            t_cap = t_k + cam.phase
            tau = cam.ref_now(t_cap)
            kc = int(math.floor(tau / cfg.dt))
            t_send = max(t_cap, cam.true_of_ref((kc + 1) * cfg.dt))
            if i in groups:
                s, e = groups[i]
                P, Z, C = res[3][s:e], res[4][s:e], res[5][s:e]
            else:
                P, Z, C = np.zeros((0, 2)), np.zeros((0, cfg.D), np.float32), np.zeros(0)
            n = len(P)
            self.n_det_sent += n
            msg = Msg("DET", {"k": kc, "tau": tau, "P": P, "Z": Z, "C": C, "t_cap": t_cap},
                      24 + 536 * n)
            self.sim.send(cam, cam.center, msg, t_send)

    # ------------------------------------------------------------ faults
    def center_of(self, region):
        for c in self.cams:
            if c.alive and c.role is not None and c.region == region:
                return c
        return None

    def inject(self, kind, *args):
        sim = self.sim
        if kind == "crash":
            node = sim.nodes[args[0]]
            node.crash()
            sim.log("crash", node=args[0])
            if len(args) > 1 and args[1]:
                sim.after(args[1], self.inject, "recover", args[0])
        elif kind == "recover":
            node = sim.nodes[args[0]]
            if node.alive or getattr(node, "left", False):
                return                      # nothing to recover; a retired camera stays retired
            node.recover()
            sim.log("recover", node=args[0])
        elif kind == "crash_center":
            c = self.center_of(args[0])
            if c is not None:
                sim.log("crash_center", region=args[0], node=c.id)
                c.crash()
                if len(args) > 1 and args[1]:
                    sim.after(args[1], self.inject, "recover", c.id)
        elif kind == "crash_agg":
            self.agg.crash()
            if len(args) > 0 and args[0]:
                sim.after(args[0], self.inject, "recover_agg")
        elif kind == "recover_agg":
            self.agg.recover()
        elif kind == "loss":
            self.cfg.p_loss = float(args[0])
            sim.log("loss", p=float(args[0]))
        elif kind == "join":
            self.join_camera(np.array(args[0], float))
        elif kind == "leave":
            self.leave_camera(args[0])

    def _schedule_random_crash(self):
        cfg = self.cfg
        rate = cfg.member_crash_rate / 3600.0 * len(self.cams)
        self.sim.after(self.fault_rng.exponential(1 / rate), self._random_crash)

    def _random_crash(self):
        cfg = self.cfg
        alive = [c for c in self.cams if c.alive and not c.left]
        if alive:
            c = alive[int(self.fault_rng.integers(len(alive)))]
            c.crash()
            self.sim.log("crash", node=c.id, center=c.role is not None)
            if cfg.member_mttr > 0:
                self.sim.after(self.fault_rng.exponential(cfg.member_mttr), self.inject, "recover", c.id)
        self._schedule_random_crash()

    def join_camera(self, pos):
        tmp = _Joiner(self.sim, self, pos)
        tmp.go()

    def spawn_camera(self, cid, pos, region, src):
        voters = []
        cam = CameraNode(self.sim, self, cid, pos, region, voters)
        cam.dir = self.directory.snap
        self.cams.append(cam)
        self.cam_pos = np.vstack([self.cam_pos, pos])
        cam.start()
        self.sim.log("join", node=cid, region=region)

    def leave_camera(self, cid):
        cam = self.sim.nodes[cid]
        if not cam.alive or cam.left:
            return
        cam.send("agg", Msg("LEAVE", {"cam": cid}, 16))
        if cam.center is not None and cam.center != cid:
            cam.send(cam.center, Msg("LEAVE", {"cam": cid}, 16))
        cam.left = True
        self.sim.log("leave", node=cid)
        # a departing voter/center stays up briefly so that it can be replaced
        self.sim.after(1.0, cam.crash)

    def _random_join(self):
        self.join_camera(self.fault_rng.uniform(0, self.cfg.L, 2))
        self.sim.after(self.fault_rng.exponential(1 / self.cfg.join_rate), self._random_join)

    def _random_leave(self):
        alive = [c for c in self.cams if c.alive and not c.left]
        if alive:
            self.leave_camera(alive[int(self.fault_rng.integers(len(alive)))].id)
        self.sim.after(self.fault_rng.exponential(1 / self.cfg.leave_rate), self._random_leave)

    def on_node_crash(self, node):
        pass

    def on_directory(self, snap):
        self.directory_snapshot = snap

    def cam_alive_hint(self, c, reg):
        return True

    # ------------------------------------------------------------ results
    def results(self):
        cfg = self.cfg
        acc = evaluate(self.rec, cfg, self.agg.aliases)
        lat = latency(self.rec, cfg)
        ovh = overhead(self.sim, cfg, self)
        return {"accuracy": acc, "latency": lat, "overhead": ovh,
                "rebuilds": self.rec.rebuilds,
                "events": _event_counts(self.sim.events)}

    def continuity(self, t_crash, region, after=3.0):
        return failover_continuity(self.rec, self.cfg, self.agg.aliases, t_crash, region, after=after)

    # ------------------------------------------------------------ trace / frames
    def header(self):
        d = self.directory
        return {"type": "header", "L": self.cfg.L, "r": self.cfg.r, "dt": self.cfg.dt,
                "M": len(self.cam_pos), "K": self.cfg.K,
                "cams": [[round(float(x), 1), round(float(y), 1), int(reg)]
                         for (x, y), reg in zip(d.cam_pos, d.cam_region)],
                "centroids": d.centroids.round(1).tolist()}

    def _emit_frame(self, k):
        if not self.trace and not self.frame_listeners:
            return
        fr = self.frame(k)
        if self.trace:
            self.trace.write(json.dumps(fr) + "\n")
        for f in self.frame_listeners:
            f(fr)

    def frame(self, k):
        cfg = self.cfg
        ids, pos, _ = self.rec.gt.get(k, (np.zeros(0), np.zeros((0, 2)), None))
        v = self.rec.views[max(self.rec.views)] if self.rec.views else None
        view = []
        if v is not None:
            for g, p, reg, st in zip(v["gids"], v["pos"], v["region"], v["stale"]):
                view.append([gid_str(g), round(float(p[0]), 1), round(float(p[1]), 1), int(reg), bool(st)])
        centers = {}
        regions = {}
        for c in self.cams:
            if c.alive and c.role is not None:
                centers[c.region] = c.id
                if cfg.trace_centers:
                    regions[c.region] = [[gid_str(r.gid), round(float(r.x[0]), 1), round(float(r.x[1]), 1),
                                          int(r.owner), r.status] for r in c.role.work.tracks.values()]
        dead = [c.id for c in self.cams if not c.alive or c.left]
        left = [c.id for c in self.cams if c.left]
        voters = {}
        for c in self.cams:
            if c.alive and c.role is not None:
                voters[c.region] = list(c.raft.voters)
        evs = []
        while self._ev_ptr < len(self.sim.events):
            t, kind, info = self.sim.events[self._ev_ptr]
            self._ev_ptr += 1
            if kind in ("accept", "release", "merge_applied", "alias") and len(evs) > 60:
                continue
            evs.append([round(t, 2), kind, _jsonable(info)])
        return {"type": "frame", "k": k, "t": round((k + 1) * cfg.dt, 2),
                "gt": [[int(i), round(float(p[0]), 1), round(float(p[1]), 1)] for i, p in zip(ids, pos)],
                "view": view, "view_k": None if v is None else int(max(self.rec.views)),
                "centers": centers, "voters": voters, "regions": regions, "dead": dead, "left": left,
                "agg_alive": self.agg.alive, "cams_new": self._new_cams(),
                "events": evs}

    _ev_ptr = 0
    _cams_sent = None

    def _new_cams(self):
        n0 = self._cams_sent if self._cams_sent is not None else self.cfg.M
        out = []
        d = self.directory
        for i in range(n0, len(d.cam_pos)):
            out.append([i, round(float(d.cam_pos[i][0]), 1), round(float(d.cam_pos[i][1]), 1),
                        int(d.cam_region[i])])
        self._cams_sent = len(d.cam_pos)
        return out


class _Joiner:
    """A not-yet-registered camera asking the directory to join."""

    def __init__(self, sim, system, pos):
        self.sim, self.sys, self.pos = sim, system, pos

    def go(self):
        from .sim import Node

        class _Tmp(Node):
            def on_JOIN_R(s, m):
                pass
        t = _Tmp(self.sim, f"joiner{self.sim.now:.3f}")
        t.send("agg", Msg("JOIN", {"cam": None, "pos": self.pos, "rejoin": False}, 48))


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, tuple) and len(x) == 3 and all(isinstance(i, (int, np.integer)) for i in x):
        return gid_str(x)          # gids are tuples; plain lists (e.g. voter sets) stay lists
    if isinstance(x, (list, tuple)):
        return [_jsonable(i) for i in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), 3)
    return x


def _event_counts(events):
    out = {}
    for _, k, _ in events:
        out[k] = out.get(k, 0) + 1
    return out
