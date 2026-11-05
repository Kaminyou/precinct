"""Global aggregator (soft state) and the co-located directory (durable)."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .regions import DirSnapshot
from .sim import Msg, Node


class Directory:
    """Durable camera directory (survives aggregator crashes)."""

    def __init__(self, cam_pos, cam_region, voters, centroids):
        self.cam_pos = np.array(cam_pos, float)
        self.cam_region = np.array(cam_region, int)
        self.live = np.ones(len(cam_pos), bool)
        self.voters = {k: list(v) for k, v in voters.items()}
        self.centroids = np.array(centroids)
        self.centers = {}
        self.region_term = {k: -1 for k in voters}
        self.version = 0
        self.snap = self._build()

    def _build(self):
        return DirSnapshot(self.version, self.cam_pos, self.cam_region, self.live.copy(),
                           self.centers, self.voters, self.centroids)

    def bump(self):
        self.version += 1
        self.snap = self._build()
        return self.snap

    def add_camera(self, pos):
        cid = len(self.cam_pos)
        self.cam_pos = np.vstack([self.cam_pos, pos])
        reg = int(np.argmin(np.linalg.norm(self.centroids - pos, axis=1)))
        self.cam_region = np.append(self.cam_region, reg)
        self.live = np.append(self.live, True)
        return cid, reg


class AggregatorNode(Node):
    def __init__(self, sim, system, directory: Directory):
        super().__init__(sim, "agg")
        self.theta, self.rho = 0.0, 0.0          # the aggregator clock is the reference
        self.sys = system
        self.dir = directory
        self._reset_soft()
        self._flush_pending = False

    def _reset_soft(self):
        self.last_rep = {}        # region -> (k, {gid: tuple})
        self.buf = {}             # k -> {region: data}
        self.recv_t = {}          # (region,k) -> t
        self.maxk = {}            # region -> highest step reported
        self.z = {}               # gid -> z
        self.created = {}
        self.aliases = {}
        self.merges = {}          # loser -> [survivor, region, last_sent, first_sent]
        self.dup = {}             # (g1, g2) -> consecutive count
        self.next_k = None
        self.recovering = None
        self.view = {}
        self.vanished = {}        # gid -> (x, y, t, region, epoch, created)
        self.seen = set()

    # ------------------------------------------------------------ lifecycle
    def start(self):
        cfg = self.sim.cfg
        self.next_k = 0
        self.timer_at_ref(cfg.dt + cfg.W + cfg.W_a, self._deadline, 0)
        self.timer(1.0, self._dir_check)

    def on_crash(self):
        self._reset_soft()
        self.sim.log("agg_crash")

    def on_recover(self):
        cfg = self.sim.cfg
        self._reset_soft()
        now = self.sim.now
        k = int(math.floor((now - cfg.W - cfg.W_a) / cfg.dt))
        self.next_k = k
        self.timer_at_ref((k + 1) * cfg.dt + cfg.W + cfg.W_a, self._deadline, k)
        # hellos sent while we were down were lost: give every region a grace period
        for reg in self.dir.voters:
            self.sys.chello_t[reg] = now
        self.timer(1.0, self._dir_check)
        self.recovering = {"t0": now, "waiting": {r for r in self._active_regions()
                                                   if self.dir.centers.get(r) is not None}}
        self._request_snapshots()
        self.timer(2.0, self._finish_recovery, True)
        self.sim.log("agg_recover")

    def _request_snapshots(self):
        """(Re)send snapshot requests to regions that have not answered yet."""
        if self.recovering is None:
            return
        for reg in list(self.recovering["waiting"]):
            c = self.dir.centers.get(reg)
            if c is not None:
                self.send(c, Msg("SNAPREQ", {}, 16))
        self.timer(0.1, self._request_snapshots)

    def _finish_recovery(self, timeout=False):
        if self.recovering is None:
            return
        dt = self.sim.now - self.recovering["t0"]
        self.sim.log("agg_rebuilt", duration=dt, complete=not timeout)
        self.sys.rec.agg_rebuilt(dt, not timeout)
        self.recovering = None

    def _active_regions(self):
        now = self.sim.now
        cfg = self.sim.cfg
        snap = self.dir.snap
        regs = set(int(r) for r in np.unique(snap.live_region)) if len(snap.live_ids) else set()
        return {r for r in regs if now - self.sys.chello_t.get(r, 0.0) < cfg.T_empty}

    # ------------------------------------------------------------ reports
    def on_REPORT(self, m):
        d = m.data
        k, reg = d["k"], d["region"]
        self.sys.rec.agg_recv(reg, k, self.sim.now)
        if k < self.next_k:
            self.sim.stats.counters["late_report"] += 1
            return
        self.buf.setdefault(k, {})[reg] = d
        self.maxk[reg] = max(self.maxk.get(reg, -1), k)
        for t in d["tracks"]:
            if t[8] is not None:
                self.z[t[0]] = t[8]
            self.created.setdefault(t[0], t[7])
        for lo, su in d["aliases"].items():
            self.aliases[lo] = su
            self.sim.log("alias", loser=lo, survivor=su, src="early")
        self._try_close()

    def _try_close(self):
        """Close step next_k as soon as every active region has reported it or
        a later step (a report for next_k can then no longer arrive)."""
        act = self._active_regions()
        while act and all(self.maxk.get(r, -1) >= self.next_k for r in act):
            self._close(self.next_k)

    def on_SNAPSHOT(self, m):
        d = m.data
        reg = d["region"]
        self.last_rep[reg] = (d["step"], {t[0]: t for t in d["tracks"]})
        for t in d["tracks"]:
            if t[8] is not None:
                self.z[t[0]] = t[8]
            self.created.setdefault(t[0], t[7])
        self.aliases.update(d["aliases"])
        for t in d.get("lost", []):          # re-seed the vanished pool for global re-id
            self.z[t[0]] = t[8]
            self.created.setdefault(t[0], t[7])
            self.vanished[t[0]] = (t[1], t[2], t[9], reg, t[6], t[7], t[0])
        if self.recovering is not None:
            self.recovering["waiting"].discard(reg)
            if not self.recovering["waiting"]:
                self._finish_recovery()

    def _deadline(self, k):
        if k == self.next_k:
            self._close(k)

    def _close(self, k):
        cfg = self.sim.cfg
        self.next_k = k + 1
        self.timer_at_ref((k + 2) * cfg.dt + cfg.W + cfg.W_a, self._deadline, k + 1)
        t_close = self.sim.now
        view = self.charged(self._build_view, k)
        self.sys.rec.agg_view(k, t_close, self.busy_until, view, self.included)

    def _resolve(self, g):
        seen = 0
        while g in self.aliases and seen < 16:
            g = self.aliases[g]
            seen += 1
        return g

    def _build_view(self, k):
        cfg = self.sim.cfg
        reps = self.buf.pop(k, {})
        for kk in [x for x in self.buf if x < k]:
            del self.buf[kk]
        for reg, d in reps.items():
            self.last_rep[reg] = (k, {t[0]: t for t in d["tracks"]})
        self.included = set(reps)
        view = {}
        t_view = (k + 1) * cfg.dt
        for reg, (kr, tracks) in self.last_rep.items():
            age = (k - kr) * cfg.dt
            if age > cfg.T_stale:
                continue
            stale = kr < k
            for t in tracks.values():
                g = self._resolve(t[0])
                x, y = t[1] + t[3] * age, t[2] + t[4] * age
                var = t[5] + (age * 3.0) ** 2
                cand = (g, x, y, reg, t[6], stale, var, t[0])
                cur = view.get(g)
                if cur is None or (t[6], not stale, -reg) > (cur[4], not cur[5], -cur[3]):
                    view[g] = cand
        for reg in [r for r, (kr, _) in self.last_rep.items() if (k - kr) * cfg.dt > cfg.T_stale]:
            del self.last_rep[reg]
        if cfg.dedup:
            self._dedup(view)
        if cfg.global_reid:
            self._global_reid(view, t_view)
        self._retry_merges()
        self.view = view
        return view

    # ------------------------------------------------------------ duplicates
    def _dedup(self, view):
        cfg = self.sim.cfg
        items = [v for v in view.values() if not v[5]]
        seen = set()
        if len(items) >= 2:
            P = np.array([(v[1], v[2]) for v in items])
            pairs = cKDTree(P).query_pairs(cfg.dup_radius, output_type="ndarray")
            for i, j in pairs:
                a, b = items[i], items[j]
                za, zb = self.z.get(a[7]), self.z.get(b[7])
                if za is None or zb is None:
                    continue
                d2 = ((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) / (a[6] + b[6] + 1e-6)
                cs = float(za @ zb)
                cost = cfg.alpha * d2 / cfg.gamma + (1 - cfg.alpha) * (1 - cs)
                key = (a[0], b[0]) if a[0] < b[0] else (b[0], a[0])
                if cost < cfg.C_dup and d2 < cfg.gamma:
                    seen.add(key)
                    self.dup[key] = self.dup.get(key, 0) + 1
                    if self.dup[key] >= cfg.n_dup:
                        self._decide_merge(view, key)
        for key in [k for k in self.dup if k not in seen]:
            del self.dup[key]

    def _global_reid(self, view, t):
        """Link gids that appear for the first time to gids that vanished
        recently anywhere (objects crossing uncovered gaps between regions)."""
        cfg = self.sim.cfg
        for g, v in self.view.items():
            if g not in view and g not in self.merges and g not in self.aliases:
                self.vanished[g] = (v[1], v[2], t, v[3], v[4], self.created.get(v[7], t), v[7])
        for g in list(self.vanished):
            if g in view or t - self.vanished[g][2] > cfg.T_gl:
                del self.vanished[g]
        for g, v in view.items():
            if g in self.seen:
                continue
            self.seen.add(g)
            if v[5] or not self.vanished:
                continue
            z = self.z.get(v[7])
            cg = self.created.get(v[7], t)
            if z is None:
                continue
            best, bc = None, -1.0
            for h, (x, y, tv, reg, ep, cr, raw) in self.vanished.items():
                if cr >= cg:
                    continue
                reach = cfg.v_max * (t - tv) + 10.0
                if (x - v[1]) ** 2 + (y - v[2]) ** 2 > reach ** 2:
                    continue
                zh = self.z.get(raw)
                if zh is None:
                    continue
                cs = float(zh @ z)
                if cs >= cfg.s_reid and cs > bc:
                    best, bc = h, cs
            if best is not None and g not in self.merges:
                x, y, tv, reg, ep, cr, raw = self.vanished.pop(best)
                self.merges[g] = [best, v[3], -1e9, self.sim.now, reg, ep, cr, True]
                self.sim.log("reid_decided", loser=g, survivor=best)
                self.sim.stats.counters["global_reid"] += 1
        if len(self.seen) > 200000:
            self.seen = set(view)

    def _decide_merge(self, view, key):
        g1, g2 = key
        c1 = self.created.get(view[g1][7], math.inf)
        c2 = self.created.get(view[g2][7], math.inf)
        surv, loser = (g1, g2) if (c1, g1) <= (c2, g2) else (g2, g1)
        if loser in self.merges:
            return
        sv = view[surv]
        self.merges[loser] = [surv, view[loser][3], -1e9, self.sim.now, sv[3], sv[4], min(c1, c2), False]
        self.sim.log("merge_decided", loser=loser, survivor=surv)
        self.sim.stats.counters["merges_decided"] += 1
        self._retry_merges()

    def _retry_merges(self):
        now = self.sim.now
        for lo, st in list(self.merges.items()):
            if now - st[3] > 5.0:
                del self.merges[lo]
                continue
            if now - st[2] >= 0.5:
                st[2] = now
                c = self.dir.centers.get(st[1])
                if c is not None:
                    self.send(c, Msg("MERGE", {"loser": lo, "survivor": st[0], "s_region": st[4],
                                               "s_epoch": st[5], "s_created": st[6], "take": st[7]}, 56))

    def on_MERGE_ACK(self, m):
        d = m.data
        lo, su = d["loser"], d["survivor"]
        st = self.merges.pop(lo, None)
        if st is not None or lo not in self.aliases:
            self.aliases[lo] = su
            self.sim.log("merge", loser=lo, survivor=su,
                         why="reid" if (st is not None and st[7]) else "duplicate")
            self.sim.stats.counters["merges"] += 1

    # ------------------------------------------------------------ clock
    def on_CSYNC(self, m):
        self.send(m.src, Msg("CSYNC_R", {"t0": m.data["t0"], "ts": self.ref_now(),
                                         "round": m.data["round"]}, 24))

    # ------------------------------------------------------------ directory
    def _publish(self):
        if self._flush_pending:
            return
        self._flush_pending = True
        self.timer(0.005, self._flush)

    def _flush(self):
        self._flush_pending = False
        snap = self.dir.bump()
        for reg, c in self.dir.centers.items():
            self.send(c, Msg("DIRUPD", {"dir": snap}, 96))
        self.sys.on_directory(snap)

    def on_CHELLO(self, m):
        d = m.data
        reg = d["region"]
        D = self.dir
        self.sys.chello_t[reg] = self.sim.now
        changed = False
        if d["term"] >= D.region_term.get(reg, -1):
            if D.centers.get(reg) != m.src or D.voters.get(reg) != d["voters"]:
                changed = True
            D.centers[reg] = m.src
            D.voters[reg] = list(d["voters"])
            D.region_term[reg] = d["term"]
        for c in d["failed"]:
            if D.live[c]:
                D.live[c] = False
                changed = True
        if changed:
            self._publish()
        elif d["version"] < D.version:
            self.send(m.src, Msg("DIRUPD", {"dir": D.snap}, 96))

    def on_MFAIL(self, m):
        for c in m.data["cams"]:
            self.dir.live[c] = False
        self._publish()

    def on_JOIN(self, m):
        d = m.data
        D = self.dir
        cid = d["cam"]
        if cid is None or cid >= len(D.cam_pos):
            cid, reg = D.add_camera(d["pos"])
            self.sys.spawn_camera(cid, d["pos"], reg, m.src)
        else:
            reg = int(D.cam_region[cid])
        D.live[cid] = True
        snap = D.bump()
        for r, c in D.centers.items():
            self.send(c, Msg("DIRUPD", {"dir": snap}, 96))
        self.sys.on_directory(snap)
        self.send(cid, Msg("JOIN_R", {"region": reg, "voters": D.voters.get(reg, []),
                                      "center": D.centers.get(reg), "dir": snap}, 128))

    def on_LEAVE(self, m):
        self.dir.live[m.data["cam"]] = False
        self._publish()

    def on_WHO(self, m):
        reg = m.data.get("region")
        c = self.dir.centers.get(reg)
        if c is not None:
            self.send(m.src, Msg("LEADER_IS", {"leader": c, "term": self.dir.region_term.get(reg, -1)}, 24))

    def _dir_check(self):
        cfg = self.sim.cfg
        now = self.sim.now
        D = self.dir
        changed = False
        for reg in list(D.voters):
            last = self.sys.chello_t.get(reg, 0.0)
            members = np.flatnonzero(D.live & (D.cam_region == reg))
            if len(members) == 0:
                continue
            if not cfg.replication and now - last > cfg.T_center_timeout and now > 2.0:
                # no-replication ablation: appoint a fresh center
                old = D.centers.get(reg)
                cand = [int(c) for c in members if c != old and self.sys.cam_alive_hint(c, reg)]
                if cand:
                    cen = D.centroids[reg]
                    c = min(cand, key=lambda i: np.linalg.norm(D.cam_pos[i] - cen))
                    D.region_term[reg] = D.region_term.get(reg, 0) + 1
                    D.voters[reg] = [c]
                    if old is not None:
                        D.live[old] = False
                    D.centers[reg] = c
                    self.sys.chello_t[reg] = now
                    self.send(c, Msg("APPOINT", {"region": reg, "term": D.region_term[reg]}, 24))
                    self.sim.log("appoint", region=reg, node=c)
                    changed = True
            elif now - last > cfg.T_empty and now > cfg.T_empty:
                D.live[members] = False             # the region is empty
                self.sim.log("region_empty", region=reg)
                changed = True
        if changed:
            self._publish()
        self.timer(1.0, self._dir_check)
