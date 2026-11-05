"""Recording and evaluation: MOTA, IDF1, ID switches, RMSE, duplicates,
latency breakdown, late-drop rates, overhead, and failover continuity."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

GATE = 5.0  # metres, matching gate between view tracks and ground truth


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gt = {}            # k -> (ids, pos, observable)
        self.views = {}         # k -> dict
        self.cstep = {}         # (region,k) -> [t_close, t_done, n, sum_cap, sum_arr, node]
        self.ccommit = {}
        self.arecv = {}
        self.rebuilds = []
        self.n_det_total = 0

    def center_step(self, region, k, t_close, t_done, a, node):
        if a is None:
            a = [0, 0.0, 0.0, 0]
        self.cstep[(region, k)] = (t_close, t_done, a[0], a[1], a[2], node)

    def center_commit(self, region, k, t):
        self.ccommit.setdefault((region, k), t)

    def agg_recv(self, region, k, t):
        self.arecv.setdefault((region, k), t)

    def agg_view(self, k, t_close, t_pub, view, included):
        vals = list(view.values())
        self.views[k] = {
            "t_close": t_close, "t_pub": t_pub,
            "gids": [v[0] for v in vals],
            "pos": np.array([(v[1], v[2]) for v in vals]).reshape(-1, 2),
            "region": np.array([v[3] for v in vals], int),
            "stale": np.array([v[5] for v in vals], bool),
            "included": set(included),
        }

    def agg_rebuilt(self, dt, complete):
        self.rebuilds.append((dt, complete))


def _resolve(aliases):
    cache = {}

    def f(g):
        if g in cache:
            return cache[g]
        h, n = g, 0
        while h in aliases and n < 32:
            h = aliases[h]
            n += 1
        cache[g] = h
        return h
    return f


def match_step(gt_pos, view_pos):
    """Greedy nearest matching within GATE (exact for well-separated objects)."""
    if len(gt_pos) == 0 or len(view_pos) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    sp = cKDTree(view_pos).sparse_distance_matrix(cKDTree(gt_pos), GATE, output_type="coo_matrix")
    order = np.argsort(sp.data, kind="stable")
    used_v, used_g = set(), set()
    vi, gi, dd = [], [], []
    for o in order:
        v, g = int(sp.row[o]), int(sp.col[o])
        if v in used_v or g in used_g:
            continue
        used_v.add(v)
        used_g.add(g)
        vi.append(v)
        gi.append(g)
        dd.append(sp.data[o])
    return np.array(vi, int), np.array(gi, int), np.array(dd)


def idf1_tp(pairs):
    """Max one-to-one sum of co-occurrence counts, solved per connected component."""
    if not pairs:
        return 0
    gts = {g: i for i, g in enumerate({p[0] for p in pairs})}
    ids = {g: i for i, g in enumerate({p[1] for p in pairs})}
    rows = np.array([gts[p[0]] for p in pairs])
    cols = np.array([ids[p[1]] for p in pairs])
    w = np.array(list(pairs.values()), float)
    n1, n2 = len(gts), len(ids)
    adj = coo_matrix((np.ones(len(w)), (rows, cols + n1)), shape=(n1 + n2, n1 + n2))
    ncomp, lab = connected_components(adj, directed=False)
    by = defaultdict(list)
    for e in range(len(w)):
        by[lab[rows[e]]].append(e)
    tp = 0
    for es in by.values():
        r = np.unique(rows[es])
        c = np.unique(cols[es])
        if len(es) == 1:
            tp += w[es[0]]
            continue
        rm = {x: i for i, x in enumerate(r)}
        cm = {x: i for i, x in enumerate(c)}
        M = np.zeros((len(r), len(c)))
        for e in es:
            M[rm[rows[e]], cm[cols[e]]] = w[e]
        a, b = linear_sum_assignment(-M)
        tp += M[a, b].sum()
    return tp


def evaluate(rec: Recorder, cfg, aliases, k_from=None, k_to=None, track_matches=False):
    res = _resolve(aliases)
    k0 = int(cfg.warmup / cfg.dt) if k_from is None else k_from
    ks = sorted(k for k in rec.views if k >= k0 and (k_to is None or k <= k_to) and k in rec.gt)
    n_gt = fn = fp = idsw = idsw_raw = dup = n_match = 0
    se = 0.0
    last = {}
    last_raw = {}
    co = defaultdict(int)
    n_view = 0
    matches = {} if track_matches else None
    for k in ks:
        ids, gpos, obs = rec.gt[k]
        v = rec.views[k]
        vpos = v["pos"]
        vi, gi, dd = match_step(gpos, vpos)
        matched_g = np.zeros(len(ids), bool)
        matched_g[gi] = True
        counted = obs | matched_g
        n_gt += int(counted.sum())
        fn += int((obs & ~matched_g).sum())
        n_match += len(vi)
        se += float((dd ** 2).sum())
        n_view += len(vpos)
        unv = np.ones(len(vpos), bool)
        unv[vi] = False
        fp_k = int(unv.sum())
        if fp_k and len(gpos):
            dist, _ = cKDTree(gpos).query(vpos[unv], distance_upper_bound=GATE)
            d_k = int(np.isfinite(dist).sum())
            dup += d_k
        fp += fp_k
        gids = v["gids"]
        step_m = {} if track_matches else None
        for a, b in zip(vi, gi):
            g = int(ids[b])
            raw = gids[a]
            gid = res(raw)
            co[(g, gid)] += 1
            if g in last and last[g] != gid:
                idsw += 1
            if g in last_raw and last_raw[g] != raw:
                idsw_raw += 1
            last[g] = gid
            last_raw[g] = raw
            if step_m is not None:
                step_m[g] = (gid, int(v["region"][a]))
        if matches is not None:
            matches[k] = step_m
    tp = idf1_tp(co)
    out = {
        "steps": len(ks), "gt": n_gt, "fn": fn, "fp": fp, "idsw": idsw, "idsw_online": idsw_raw,
        "mota": 1 - (fn + fp + idsw) / max(n_gt, 1),
        "idf1": 2 * tp / max(n_gt + n_view, 1),
        "rmse": float(np.sqrt(se / max(n_match, 1))),
        "dup_rate": dup / max(n_gt, 1),
        "recall": n_match / max(n_gt, 1),
        "precision": n_match / max(n_view, 1),
        "idsw_per_obj_min": idsw / max(n_gt * cfg.dt / 60.0, 1e-9),
    }
    if matches is not None:
        out["matches"] = matches
    return out


def latency(rec: Recorder, cfg, k_from=None):
    k0 = int(cfg.warmup / cfg.dt) if k_from is None else k_from
    comps = defaultdict(list)
    tot, w = [], []
    for k, v in rec.views.items():
        if k < k0:
            continue
        for reg in v["included"]:
            cs = rec.cstep.get((reg, k))
            if cs is None or cs[2] == 0:
                continue
            t_close, t_done, n, s_cap, s_arr, _ = cs
            t_com = rec.ccommit.get((reg, k))
            t_rcv = rec.arecv.get((reg, k))
            if t_com is None or t_rcv is None:
                continue
            cap, arr = s_cap / n, s_arr / n
            parts = {
                "batching+member->center": arr - cap,
                "wait deadline": t_close - arr,
                "fusion": t_done - t_close,
                "raft commit": t_com - t_done,
                "center->agg": t_rcv - t_com,
                "wait agg deadline": v["t_close"] - t_rcv,
                "aggregation": v["t_pub"] - v["t_close"],
            }
            for kk, val in parts.items():
                comps[kk].append(val)
            tot.append(v["t_pub"] - cap)
            w.append(n)
    if not tot:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "mean": float("nan"),
                "parts": {}}
    tot = np.array(tot)
    w = np.array(w, float)
    order = np.argsort(tot)
    cw = np.cumsum(w[order]) / w.sum()

    def q(p):
        return float(tot[order][np.searchsorted(cw, p)])
    return {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "mean": float((tot * w).sum() / w.sum()),
            "parts": {k: float(np.mean(v)) for k, v in comps.items()}}


def overhead(sim, cfg, system):
    T = max(sim.now - 0.0, 1e-9)
    st = sim.stats
    out = {"msgs_per_s": {k: v / T for k, v in st.msgs.items()},
           "bytes_per_s": {k: v / T for k, v in st.bytes.items()}}
    busy = st.busy
    centers = [n for n in busy if n != "agg"]
    out["center_cpu_max"] = max((busy[n] / T for n in centers), default=0.0)
    out["center_cpu_mean"] = float(np.mean([busy[n] / T for n in centers])) if centers else 0.0
    out["agg_cpu"] = busy.get("agg", 0.0) / T
    ins = {n: v / T for n, v in st.node_in.items()}
    out["center_in_max_Bps"] = max((v for n, v in ins.items() if n != "agg"), default=0.0)
    out["agg_in_Bps"] = ins.get("agg", 0.0)
    out["late_det_rate"] = st.counters["late_det_n"] / max(system.n_det_sent, 1)
    out["late_reports"] = st.counters["late_report"]
    out["counters"] = dict(st.counters)
    return out


def failover_continuity(rec, cfg, aliases, t_crash, region, before=0.3, after=3.0):
    """Among objects owned by `region` just before t_crash, the fraction that
    are reported under the same (alias-resolved) gid `after` seconds later."""
    res = _resolve(aliases)
    kb = int(round((t_crash - before) / cfg.dt))
    ka = int(round((t_crash + after) / cfg.dt))

    def m(k):
        if k not in rec.views or k not in rec.gt:
            return {}
        ids, gpos, _ = rec.gt[k]
        v = rec.views[k]
        vi, gi, _ = match_step(gpos, v["pos"])
        return {int(ids[b]): (res(v["gids"][a]), int(v["region"][a])) for a, b in zip(vi, gi)}
    mb, ma = m(kb), m(ka)
    objs = [g for g, (gid, reg) in mb.items() if reg == region]
    both = [g for g in objs if g in ma]
    kept = sum(1 for g in both if ma[g][0] == mb[g][0])
    # recovery: first commit of the region after the crash
    t_rec = min((t for (r, k), t in rec.ccommit.items() if r == region and t > t_crash + 1e-6),
                default=float("nan"))
    return {"objects": len(objs), "still_tracked": len(both), "kept": kept,
            "kept_frac": kept / max(len(both), 1), "recovery_s": t_rec - t_crash}


# ------------------------------------------------------------------ live viewer support
def window_payload(rec: Recorder, aliases, k_from, k_to):
    """Copy only the recorder entries of steps [k_from, k_to] (cheap to pickle)."""
    w = Recorder(rec.cfg)
    w.gt = {k: v for k, v in rec.gt.items() if k_from <= k <= k_to}
    w.views = {k: v for k, v in rec.views.items() if k_from <= k <= k_to}
    w.cstep = {key: v for key, v in rec.cstep.items() if k_from <= key[1] <= k_to}
    w.ccommit = {key: v for key, v in rec.ccommit.items() if k_from <= key[1] <= k_to}
    w.arecv = {key: v for key, v in rec.arecv.items() if k_from <= key[1] <= k_to}
    return w, dict(aliases), k_from


def window_metrics(payload):
    """Runs in a worker process: accuracy and latency over a recent window."""
    w, aliases, k_from = payload
    acc = evaluate(w, w.cfg, aliases, k_from=k_from)
    lat = latency(w, w.cfg, k_from=k_from)
    return {"mota": acc["mota"], "idf1": acc["idf1"], "idsw": acc["idsw"],
            "lat_p50": lat["p50"], "lat_p99": lat["p99"]}


def prune(rec: Recorder, k_keep_from):
    """Drop recorder history older than k_keep_from (live mode keeps memory bounded)."""
    for d in (rec.gt, rec.views):
        for k in [k for k in d if k < k_keep_from]:
            del d[k]
    for d in (rec.cstep, rec.ccommit, rec.arecv):
        for key in [key for key in d if key[1] < k_keep_from]:
            del d[key]
