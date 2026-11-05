"""Regional tracking: Kalman filtering, per-camera association, track
management, ownership/handoff state, and the replicated track table.

The replicated state machine is `TrackTable`. A center processes step k on a
*working* table and emits a `delta` that is appended to the Raft log; every
replica (and the center's own committed copy) applies the same delta.
Track records are treated as immutable values (x, P and z are replaced, never
mutated in place), so a delta can hold shallow copies."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .sim import Msg

BIG = 1e6
TRACK_BYTES = 120          # gid, state, covariance (float32), flags
Z_BYTES = 512              # 256-d float16 appearance vector


def gid_str(g):
    return f"{g[0]}.{g[1]}.{g[2]}"


def gid_key(g):
    return g


# ---------------------------------------------------------------- Kalman
def _FQ(dt, q):
    n = len(dt)
    F = np.tile(np.eye(4), (n, 1, 1))
    F[:, 0, 2] = dt
    F[:, 1, 3] = dt
    dt2, dt3, dt4 = dt ** 2, dt ** 3 / 2, dt ** 4 / 4
    Q = np.zeros((n, 4, 4))
    Q[:, 0, 0] = Q[:, 1, 1] = dt4
    Q[:, 2, 2] = Q[:, 3, 3] = dt2
    Q[:, 0, 2] = Q[:, 2, 0] = Q[:, 1, 3] = Q[:, 3, 1] = dt3
    return F, Q * q


def kf_predict(X, P, dt, q):
    """Batch predict X [n,4], P [n,4,4] forward by dt [n] (negative dt -> 0)."""
    dt = np.maximum(dt, 0.0)
    F, Q = _FQ(dt, q)
    Xn = np.einsum("nij,nj->ni", F, X)
    Pn = F @ P @ F.transpose(0, 2, 1) + Q
    return Xn, Pn


def kf_update(X, P, z, rvar):
    """Batch update with position measurements z [n,2], variances rvar [n]."""
    S = P[:, :2, :2] + rvar[:, None, None] * np.eye(2)
    Sinv = np.linalg.inv(S)
    K = P[:, :, :2] @ Sinv
    y = z - X[:, :2]
    Xn = X + np.einsum("nij,nj->ni", K, y)
    Pn = P - K @ P[:, :2, :]
    return Xn, 0.5 * (Pn + Pn.transpose(0, 2, 1))


def maha2(X, P, Z, rvar):
    """Mahalanobis distances between n tracks and m detections -> [n,m]."""
    S = P[:, None, :2, :2] + rvar[None, :, None, None] * np.eye(2)
    d = Z[None, :, :] - X[:, None, :2]
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * c - b * b
    return (c * d[..., 0] ** 2 - 2 * b * d[..., 0] * d[..., 1] + a * d[..., 1] ** 2) / det


def cosine(A, B):
    A = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-9)
    return A @ B.T


def unit(z):
    return (z / (np.linalg.norm(z) + 1e-9)).astype(np.float32)


# ---------------------------------------------------------------- records
class Track:
    __slots__ = ("gid", "x", "P", "t", "z", "z_rep", "status", "hits", "hist", "age",
                 "misses", "last_hit", "owner", "owner_region", "epoch", "created",
                 "offered", "accept_to", "accept_sent", "lost_t", "from_offer", "z_dirty",
                 "relinq")

    def copy(self):
        t = Track.__new__(Track)
        for s in Track.__slots__:
            setattr(t, s, getattr(self, s))
        t.offered = dict(self.offered)
        return t

    def to_report(self, with_z):
        return (self.gid, float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3]),
                float(self.P[0, 0] + self.P[1, 1]) / 2, self.epoch, self.created,
                self.z if with_z else None)


class Offer:
    __slots__ = ("gid", "src", "x", "P", "t", "z", "epoch", "created", "recv", "orphan")

    def copy(self):
        o = Offer.__new__(Offer)
        for s in Offer.__slots__:
            setattr(o, s, getattr(self, s))
        return o


class TrackTable:
    """Replicated regional state."""

    def __init__(self, region):
        self.region = region
        self.tracks = {}       # gid -> Track
        self.lost = {}         # gid -> Track (lost pool)
        self.offers = {}       # gid -> Offer (pending pool)
        self.aliases = {}      # loser gid -> survivor gid
        self.seq = 0
        self.last_step = -1

    def copy(self):
        t = TrackTable(self.region)
        t.tracks = {g: r.copy() for g, r in self.tracks.items()}
        t.lost = {g: r.copy() for g, r in self.lost.items()}
        t.offers = {g: o.copy() for g, o in self.offers.items()}
        t.aliases = dict(self.aliases)
        t.seq = self.seq
        t.last_step = self.last_step
        return t

    def apply(self, delta):
        for g, r in delta["up"].items():
            self.tracks[g] = r
        for g in delta["del"]:
            self.tracks.pop(g, None)
        for g, r in delta["lost_up"].items():
            self.lost[g] = r
        for g in delta["lost_del"]:
            self.lost.pop(g, None)
        for g, o in delta["off_up"].items():
            self.offers[g] = o
        for g in delta["off_del"]:
            self.offers.pop(g, None)
        self.aliases.update(delta["alias"])
        self.seq = delta["seq"]
        self.last_step = max(self.last_step, delta["step"])

    def size(self):
        return 64 + (TRACK_BYTES + Z_BYTES) * (len(self.tracks) + len(self.lost)) \
            + 700 * len(self.offers) + 24 * len(self.aliases)


class _Diff:
    """Collects changes made while processing one step."""

    def __init__(self):
        self.up, self.dele, self.lup, self.ldel = {}, set(), {}, set()
        self.oup, self.odel, self.alias = {}, set(), {}

    def delta(self, tab, k):
        # records are copied so replicas never observe later speculative edits
        d = {"up": {g: r.copy() for g, r in self.up.items()}, "del": list(self.dele),
             "lost_up": {g: r.copy() for g, r in self.lup.items()},
             "lost_del": list(self.ldel), "off_up": {g: o.copy() for g, o in self.oup.items()}, "off_del": list(self.odel),
             "alias": self.alias, "seq": tab.seq, "step": k}
        size = 48 + sum(TRACK_BYTES + (Z_BYTES if r.z_dirty else 0) for r in self.up.values()) \
            + 8 * (len(self.dele) + len(self.ldel) + len(self.odel)) \
            + (TRACK_BYTES + Z_BYTES) * len(self.lup) + 700 * len(self.oup) + 24 * len(self.alias)
        return d, size


# ---------------------------------------------------------------- processing
class StepContext:
    """Everything the center needs to process one step."""

    def __init__(self, cfg, region, term, directory, now_ref, node_id):
        self.cfg = cfg
        self.region = region
        self.term = term
        self.dir = directory
        self.now = now_ref
        self.node_id = node_id


class RegionTracker:
    """Stateless processor: mutates a working TrackTable for one step and
    returns (delta, size, outgoing messages, report, info)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.q = cfg.sigma_a ** 2

    # ------------------------------------------------------------ helpers
    def _new_gid(self, tab, ctx):
        tab.seq += 1
        return (ctx.region, ctx.term, tab.seq)

    def _touch(self, diff, r):
        diff.up[r.gid] = r
        diff.dele.discard(r.gid)

    def _remove(self, tab, diff, g):
        tab.tracks.pop(g, None)
        diff.up.pop(g, None)
        diff.dele.add(g)

    def _z_update(self, r, z, c):
        lam = self.cfg.lam * c
        r.z = unit((1 - lam) * r.z + lam * unit(z))
        if r.z_rep is None or float(r.z @ r.z_rep) < 1 - self.cfg.z_resend:
            r.z_dirty = True

    def _send_region(self, out, ctx, region, msg):
        out.append(("region", region, msg))

    # ------------------------------------------------------------ control
    def apply_control(self, tab, ctx, diff, out, msgs, info):
        cfg = self.cfg
        for m in msgs:
            d = m.data
            k = m.kind
            if k == "OFFER":
                g = d["gid"]
                if g in tab.aliases:
                    continue
                r = tab.tracks.get(g)
                if r is not None:
                    if d["epoch"] > r.epoch or (not r.owner and d["epoch"] == r.epoch):
                        # a newer owner exists: (re)become its shadow
                        r.owner, r.epoch, r.owner_region, r.accept_to = False, d["epoch"], d["src"], None
                        self._touch(diff, r)
                    continue
                o = Offer()
                o.gid, o.src, o.x, o.P, o.t = g, d["src"], d["x"], d["P"], d["t"]
                o.z, o.epoch, o.created, o.recv, o.orphan = d["z"], d["epoch"], d["created"], ctx.now, False
                prev = tab.offers.get(g)
                if prev is not None and prev.orphan:
                    o.orphan = True
                tab.offers[g] = o
                diff.oup[g] = o
                diff.odel.discard(g)
            elif k == "ACCEPT":
                g, ep, src = d["gid"], d["epoch"], d["src"]
                r = tab.tracks.get(g) or tab.lost.get(g)
                granted, owner = True, src
                if r is not None:
                    if r.epoch < ep:
                        was_owner = r.owner
                        r.owner, r.owner_region, r.epoch, r.accept_to = False, src, ep, None
                        if was_owner:
                            info["releases"].append((g, src, ep))
                        if g in tab.tracks:
                            self._touch(diff, r)
                        else:                      # a released lost track is forgotten
                            tab.lost.pop(g)
                            diff.ldel.add(g)
                            diff.lup.pop(g, None)
                    elif not (r.epoch == ep and r.owner_region == src):
                        granted = False
                        owner = tab.region if r.owner else r.owner_region
                self._send_region(out, ctx, src, Msg("RELEASE", {
                    "gid": g, "epoch": ep, "granted": granted, "src": tab.region,
                    "owner": owner}, 48))
            elif k == "RELEASE":
                g, ep = d["gid"], d["epoch"]
                r = tab.tracks.get(g) or tab.lost.get(g)
                if r is None or r.epoch != ep:
                    continue
                r.accept_to = None
                if not d["granted"]:
                    r.owner, r.owner_region = False, d["owner"]
                self._touch(diff, r) if g in tab.tracks else diff.lup.__setitem__(g, r)
            elif k == "RELINQ":
                g, ep, src = d["gid"], d["epoch"], d["src"]
                r = tab.tracks.get(g)
                if r is not None and not r.owner and r.status == "C" and r.epoch <= ep:
                    self._take_ownership(r, ctx, out, info, reason="relinquish")
                    self._touch(diff, r)
                elif g in tab.offers:
                    tab.offers[g].orphan = True
                    diff.oup[g] = tab.offers[g]
            elif k == "MERGE":
                loser, surv = d["loser"], d["survivor"]
                if d.get("take"):
                    # the survivor vanished elsewhere: we become its owner
                    r = self._relabel(tab, diff, loser, surv, tab.region, d["s_epoch"] + 1, d["s_created"])
                    if r is not None and d["s_region"] not in (None, tab.region):
                        r.accept_to, r.accept_sent = d["s_region"], ctx.now
                        self._send_region(out, ctx, d["s_region"], Msg("ACCEPT", {
                            "gid": surv, "epoch": r.epoch, "src": tab.region}, 40))
                else:
                    self._relabel(tab, diff, loser, surv, d["s_region"], d["s_epoch"], d["s_created"])
                info["merges"].append((loser, surv))
                out.append(("agg", None, Msg("MERGE_ACK", {"loser": loser, "survivor": surv,
                                                           "region": ctx.region}, 40)))

    def _relabel(self, tab, diff, loser, surv, s_region, s_epoch, s_created):
        tab.aliases[loser] = surv
        diff.alias[loser] = surv
        r = tab.tracks.pop(loser, None)
        if r is None:
            r = tab.lost.pop(loser, None)
            if r is not None:
                diff.ldel.add(loser)
                diff.lup.pop(loser, None)
            return None
        diff.dele.add(loser)
        diff.up.pop(loser, None)
        if surv in tab.tracks:
            return None                  # keep the existing track for the survivor
        if surv in tab.lost:
            tab.lost.pop(surv)
            diff.ldel.add(surv)
            diff.lup.pop(surv, None)
        r = r.copy()
        r.gid = surv
        r.created = min(r.created, s_created)
        if s_region == tab.region:
            r.owner = True
            r.epoch = max(r.epoch, s_epoch)
        else:
            r.owner, r.owner_region, r.epoch = False, s_region, max(r.epoch, s_epoch)
        r.accept_to = None
        r.offered = {}
        r.z_dirty = True
        r.z_rep = None
        tab.tracks[surv] = r
        self._touch(diff, r)
        return r

    def _take_ownership(self, r, ctx, out, info, reason="accept"):
        prev = r.owner_region
        r.epoch += 1
        r.owner = True
        r.owner_region = ctx.region
        r.offered = {}
        if prev is not None and prev != ctx.region:
            r.accept_to = prev
            r.accept_sent = ctx.now
            self._send_region(out, ctx, prev, Msg("ACCEPT", {"gid": r.gid, "epoch": r.epoch,
                                                            "src": ctx.region}, 40))
        info["accepts"].append((r.gid, prev, r.epoch, reason))

    # ------------------------------------------------------------ main step
    def step(self, tab: TrackTable, ctx: StepContext, k, batches, ctrl):
        """batches: list of (cam_id, tau, P[n,2], Z[n,D], C[n]) received for step k."""
        cfg = self.cfg
        diff = _Diff()
        out = []
        info = {"accepts": [], "releases": [], "merges": [], "new": 0, "revived": 0,
                "adopted": 0, "early_merge": 0, "n_det": 0, "matched": 0}
        t_end = (k + 1) * cfg.dt
        self.apply_control(tab, ctx, diff, out, ctrl, info)

        tracks = list(tab.tracks.values())
        matched = set()
        unmatched = []        # (cam, tau, p, z, c)
        if tracks:
            tpos = np.array([r.x[:2] for r in tracks])
            ttree = cKDTree(tpos)
        cam_pos = ctx.dir.cam_pos
        reporting = []
        for cam, tau, Pd, Zd, Cd in sorted(batches, key=lambda b: b[1]):
            reporting.append(cam)
            n = len(Pd)
            info["n_det"] += n
            if n == 0:
                continue
            cand = []
            if tracks:
                cand = ttree.query_ball_point(cam_pos[cam], cfg.r + 15.0 + cfg.v_max * cfg.dt)
            if not cand:
                unmatched.extend((cam, tau, Pd[i], Zd[i], Cd[i]) for i in range(n))
                continue
            rs = [tracks[j] for j in cand]
            X = np.array([r.x for r in rs])
            Pm = np.array([r.P for r in rs])
            dts = np.array([tau - r.t for r in rs])
            X, Pm = kf_predict(X, Pm, dts, self.q)
            rvar = cfg.sigma_p(Cd) ** 2
            d2 = maha2(X, Pm, Pd, rvar)
            cs = cosine(np.array([r.z for r in rs]), Zd)
            cost = cfg.alpha * d2 / cfg.gamma + (1 - cfg.alpha) * (1 - cs)
            gate = (d2 > cfg.gamma) | (cs < cfg.s_min)
            cost = np.where(gate, BIG, cost)
            ri, ci = linear_sum_assignment(cost)
            ok = cost[ri, ci] < BIG
            ri, ci = ri[ok], ci[ok]
            used = set(ci.tolist())
            if len(ri):
                Xu, Pu = kf_update(X[ri], Pm[ri], Pd[ci], rvar[ci])
                for a, (ti, di) in enumerate(zip(ri, ci)):
                    r = rs[ti]
                    r.x, r.P, r.t = Xu[a], Pu[a], max(r.t, tau)
                    self._z_update(r, Zd[di], float(Cd[di]))
                    matched.add(r.gid)
            # second pass: relaxed gate for detections left over by the first pass
            # (absorbs abrupt manoeuvres such as turns at waypoints)
            rest = [i for i in range(n) if i not in used]
            free = [j for j in range(len(rs)) if rs[j].gid not in matched]
            if rest and free:
                sub = d2[np.ix_(free, rest)]
                csub = cs[np.ix_(free, rest)]
                c2 = np.where((sub > 4 * cfg.gamma) | (csub < cfg.s_reid), BIG,
                              cfg.alpha * sub / (4 * cfg.gamma) + (1 - cfg.alpha) * (1 - csub))
                r2, c2i = linear_sum_assignment(c2)
                for ti, di in zip(r2, c2i):
                    if c2[ti, di] >= BIG:
                        continue
                    r = rs[free[ti]]
                    i = rest[di]
                    Xu, Pu = kf_update(X[free[ti]][None], Pm[free[ti]][None], Pd[i][None], rvar[i:i + 1])
                    r.x, r.P, r.t = Xu[0], Pu[0], max(r.t, tau)
                    self._z_update(r, Zd[i], float(Cd[i]))
                    matched.add(r.gid)
                    used.add(i)
            for i in range(n):
                if i not in used:
                    unmatched.append((cam, tau, Pd[i], Zd[i], Cd[i]))
        info["matched"] = len(matched)

        # ---- cluster unmatched detections across cameras
        for cl in self._cluster(unmatched):
            self._birth(tab, ctx, diff, out, info, cl)

        # ---- track management
        rep_cams = np.array([cam_pos[c] for c in reporting]) if reporting else np.zeros((0, 2))
        rtree = cKDTree(rep_cams) if len(rep_cams) else None
        for r in list(tab.tracks.values()):
            if r.t < t_end:
                X, Pm = kf_predict(r.x[None], r.P[None], np.array([t_end - r.t]), self.q)
                r.x, r.P, r.t = X[0], Pm[0], t_end
            hit = r.gid in matched
            r.age += 1
            r.hist = ((r.hist << 1) | int(hit)) & ((1 << cfg.n_w) - 1)
            if hit:
                r.hits += 1
                r.misses = 0
                r.last_hit = t_end
            else:
                expected = rtree is not None and np.isfinite(
                    rtree.query(r.x[:2], distance_upper_bound=cfg.r)[0])
                if expected:
                    r.misses += 1
            if r.status == "T":
                if bin(r.hist).count("1") >= cfg.n_c:
                    r.status = "C"
                elif r.age >= cfg.n_w:
                    self._remove(tab, diff, r.gid)
                    continue
            elif r.misses >= cfg.n_d or t_end - r.last_hit > cfg.T_coast_max:
                self._lose(tab, ctx, diff, out, r)
                continue
            self._touch(diff, r)

        # ---- lost pool and offer pool expiry
        for g, r in list(tab.lost.items()):
            if t_end - r.lost_t > cfg.T_lost:
                del tab.lost[g]
                diff.ldel.add(g)
                diff.lup.pop(g, None)
        for g, o in list(tab.offers.items()):
            if ctx.now - o.recv > cfg.T_h or g in tab.tracks:
                del tab.offers[g]
                diff.odel.add(g)
                diff.oup.pop(g, None)

        # ---- early merge of young tracks with older offered gids
        if cfg.handoff and tab.offers:
            self._early_merge(tab, ctx, diff, info)

        # ---- ownership and handoff
        self._ownership(tab, ctx, diff, out, info)

        # ---- report (owned, confirmed)
        rep = []
        for r in tab.tracks.values():
            if r.owner and r.status == "C":
                with_z = r.z_rep is None or float(r.z @ r.z_rep) < 1 - cfg.z_resend
                if with_z:
                    r.z_rep = r.z
                rep.append(r.to_report(with_z))
        info["aliases"] = dict(diff.alias)
        delta, size = diff.delta(tab, k)
        for r in diff.up.values():
            r.z_dirty = False
        tab.last_step = k
        return delta, size, out, rep, info

    # ------------------------------------------------------------ pieces
    def _cluster(self, dets):
        cfg = self.cfg
        n = len(dets)
        if n == 0:
            return []
        if n == 1:
            return [dets]
        P = np.array([d[2] for d in dets])
        tree = cKDTree(P)
        pairs = tree.query_pairs(3 * (cfg.sigma_p0 + cfg.sigma_p1) * 1.5 + cfg.v_max * cfg.dt,
                                 output_type="ndarray")
        parent = list(range(n))
        cams = [{dets[i][0]} for i in range(n)]

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        if len(pairs):
            dd = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
            for (i, j), dist in sorted(zip(pairs.tolist(), dd.tolist()), key=lambda x: x[1]):
                di, dj = dets[i], dets[j]
                if di[0] == dj[0]:
                    continue
                s2 = cfg.sigma_p(di[4]) ** 2 + cfg.sigma_p(dj[4]) ** 2
                lim = 3 * math.sqrt(s2) + cfg.v_max * abs(di[1] - dj[1])
                if dist > lim or float(unit(di[3]) @ unit(dj[3])) < cfg.s_min:
                    continue
                a, b = find(i), find(j)
                if a == b or cams[a] & cams[b]:
                    continue
                parent[b] = a
                cams[a] |= cams[b]
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(dets[i])
        return list(groups.values())

    def _fuse(self, cl):
        cfg = self.cfg
        w = np.array([1.0 / cfg.sigma_p(d[4]) ** 2 for d in cl])
        p = (np.array([d[2] for d in cl]) * w[:, None]).sum(0) / w.sum()
        tau = max(d[1] for d in cl)
        c = max(d[4] for d in cl)
        z = unit(sum(unit(d[3]) * max(d[4], 0.05) for d in cl))
        return p, tau, z, c, 1.0 / w.sum()

    def _birth(self, tab, ctx, diff, out, info, cl):
        cfg = self.cfg
        p, tau, z, c, var = self._fuse(cl)
        # (a) pending handoff offers
        best, bc = None, BIG
        if cfg.handoff:
            for o in tab.offers.values():
                if o.gid in tab.tracks:
                    continue
                X, Pm = kf_predict(o.x[None], o.P[None], np.array([tau - o.t]), self.q)
                d2 = maha2(X, Pm, p[None], np.array([var]))[0, 0]
                cs = float(o.z @ z)
                if d2 > 2 * cfg.gamma or cs < cfg.s_reid:
                    continue
                cost = cfg.alpha * d2 / cfg.gamma + (1 - cfg.alpha) * (1 - cs)
                if cost < bc:
                    best, bc = (o, X[0], Pm[0]), cost
        if best is not None:
            o, X, Pm = best
            Xu, Pu = kf_update(X[None], Pm[None], p[None], np.array([var]))
            r = self._make(o.gid, Xu[0], Pu[0], tau, z, o.created, status="C")
            r.owner, r.owner_region, r.epoch, r.from_offer = False, o.src, o.epoch, True
            tab.tracks[r.gid] = r
            if o.orphan:
                self._take_ownership(r, ctx, out, info, reason="orphan")
            del tab.offers[o.gid]
            diff.odel.add(o.gid)
            diff.oup.pop(o.gid, None)
            self._touch(diff, r)
            info["adopted"] += 1
            return
        # (b) lost pool (reachability + appearance)
        best, bc = None, -1
        for r in tab.lost.values():
            dt = tau - r.last_hit
            reach = cfg.v_max * max(dt, 0) + 3 * math.sqrt(var) + 3
            if np.linalg.norm(p - r.x[:2]) > reach:
                continue
            cs = float(r.z @ z)
            if cs >= cfg.s_reid and cs > bc:
                best, bc = r, cs
        if best is not None:
            r = best.copy()
            del tab.lost[r.gid]
            diff.ldel.add(r.gid)
            diff.lup.pop(r.gid, None)
            r.x = np.array([p[0], p[1], 0.0, 0.0])
            r.P = np.diag([var, var, 50.0, 50.0])
            r.t, r.status, r.misses, r.last_hit, r.hist = tau, "C", 0, tau, 0
            r.age = 0
            tab.tracks[r.gid] = r
            self._touch(diff, r)
            info["revived"] += 1
            return
        # (c) new tentative track
        g = self._new_gid(tab, ctx)
        r = self._make(g, np.array([p[0], p[1], 0.0, 0.0]), np.diag([var, var, 100.0, 100.0]),
                       tau, z, tau, status="T")
        r.owner, r.owner_region, r.epoch = True, ctx.region, 0
        tab.tracks[g] = r
        self._touch(diff, r)
        info["new"] += 1

    def _make(self, gid, x, P, t, z, created, status):
        r = Track.__new__(Track)
        r.gid, r.x, r.P, r.t, r.z = gid, x, P, t, z
        r.z_rep, r.status, r.hits, r.hist, r.age, r.misses = None, status, 1, 0, 0, 0
        r.last_hit = t
        r.owner, r.owner_region, r.epoch, r.created = True, None, 0, created
        r.offered, r.accept_to, r.accept_sent, r.lost_t = {}, None, 0.0, None
        r.from_offer, r.z_dirty, r.relinq = False, True, False
        return r

    def _lose(self, tab, ctx, diff, out, r):
        self._remove(tab, diff, r.gid)
        if not r.owner:
            return               # shadows are dropped
        r = r.copy()
        r.lost_t = r.t
        r.status = "L"
        # a young track that is this object re-acquired under a new gid?
        cfg = self.cfg
        best, bc = None, -1.0
        for y in tab.tracks.values():
            if y.from_offer or y.epoch != 0 or not y.owner or ctx.now - y.created > cfg.T_young:
                continue
            reach = cfg.v_max * max(y.t - r.last_hit, 0) + 3 * math.sqrt(y.P[0, 0]) + 3
            cs = float(y.z @ r.z)
            if np.linalg.norm(y.x[:2] - r.x[:2]) <= reach and cs >= cfg.s_reid and cs > bc:
                best, bc = y, cs
        if best is not None:
            self._relabel(tab, diff, best.gid, r.gid, tab.region, r.epoch, r.created)
            nr = tab.tracks.get(r.gid)
            if nr is not None:
                nr.offered = dict(r.offered)
            return
        tab.lost[r.gid] = r
        diff.lup[r.gid] = r
        diff.ldel.discard(r.gid)
        if self.cfg.handoff:
            for reg in r.offered:
                self._send_region(out, ctx, reg, Msg("RELINQ", {"gid": r.gid, "epoch": r.epoch,
                                                               "src": ctx.region}, 40))

    def _early_merge(self, tab, ctx, diff, info):
        cfg = self.cfg
        for r in list(tab.tracks.values()):
            if r.from_offer or r.epoch != 0 or not r.owner or ctx.now - r.created > cfg.T_young:
                continue
            for o in tab.offers.values():
                if o.gid in tab.tracks or (o.created, o.gid) >= (r.created, r.gid):
                    continue
                X, Pm = kf_predict(o.x[None], o.P[None], np.array([r.t - o.t]), self.q)
                d2 = maha2(X, Pm + r.P[None], r.x[None, :2], np.array([0.0]))[0, 0]
                if d2 <= cfg.gamma and float(o.z @ r.z) >= cfg.s_reid:
                    self._relabel(tab, diff, r.gid, o.gid, o.src, o.epoch, o.created)
                    del tab.offers[o.gid]
                    diff.odel.add(o.gid)
                    diff.oup.pop(o.gid, None)
                    info["early_merge"] += 1
                    break

    def _ownership(self, tab, ctx, diff, out, info):
        cfg = self.cfg
        d = ctx.dir
        for r in tab.tracks.values():
            if r.status != "C":
                continue
            p = r.x[:2]
            if not cfg.handoff:
                own = d.region_at(p) == ctx.region
                if own != r.owner:
                    r.owner = own
                    self._touch(diff, r)
                continue
            if r.owner:
                for b in d.border_regions(p, ctx.region, cfg.r + cfg.m):
                    if ctx.now - r.offered.get(b, -1e9) >= cfg.T_reoffer:
                        r.offered[b] = ctx.now
                        self._send_region(out, ctx, b, Msg("OFFER", {
                            "gid": r.gid, "src": ctx.region, "x": r.x, "P": r.P, "t": r.t,
                            "z": r.z, "epoch": r.epoch, "created": r.created}, 32 + 16 + 64 + Z_BYTES))
                        self._touch(diff, r)
                if r.accept_to is not None and ctx.now - r.accept_sent >= cfg.T_retry:
                    if ctx.now - r.accept_sent > cfg.T_h:
                        r.accept_to = None
                    else:
                        r.accept_sent = ctx.now
                        self._send_region(out, ctx, r.accept_to, Msg("ACCEPT", {
                            "gid": r.gid, "epoch": r.epoch, "src": ctx.region}, 40))
                    self._touch(diff, r)
            else:
                if d.margin(p, ctx.region) > cfg.h:
                    self._take_ownership(r, ctx, out, info)
                    self._touch(diff, r)
