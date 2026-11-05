"""Experiment driver. Each run is one simulation; results are written as
JSON to results/<experiment>/<run>.json so sweeps can be resumed.

    python experiments/run.py --exp scale --workers 16
    python experiments/run.py --exp all --workers 24
    python experiments/run.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from precinct.config import Config  # noqa: E402
from precinct.system import System  # noqa: E402

OUT = os.path.join(ROOT, "results")
SEEDS = [1, 2, 3]


def runs_scale():
    out = []
    for M in [100, 300, 1000, 3000, 10000]:
        dur = 30.0 if M <= 3000 else 20.0
        seeds = SEEDS if M <= 1000 else SEEDS[:1]
        for s in seeds:
            out.append((f"dist_M{M}_s{s}", dict(M=M, seed=s, duration=dur)))
            if M <= 3000:
                out.append((f"cent_M{M}_s{s}", dict(M=M, seed=s, duration=dur, centralized=True)))
    return out


def runs_region():
    return [(f"S{S}_s{s}", dict(M=1000, S=S, seed=s, duration=30.0))
            for S in [5, 10, 20, 50, 100] for s in SEEDS]


def runs_deadline():
    out = [(f"W{int(W * 1000)}_s{s}", dict(M=300, W=W, delay_sigma=1.0, seed=s, duration=30.0))
           for W in [0.005, 0.01, 0.025, 0.05, 0.1, 0.15] for s in SEEDS]
    out += [(f"Wa{int(W * 1000)}_s{s}", dict(M=1000, W_a=W, seed=s, duration=30.0))
            for W in [0.02, 0.05, 0.1, 0.2] for s in SEEDS]
    return out


def runs_network():
    out = [(f"loss{int(p * 100)}_s{s}", dict(M=300, p_loss=p, seed=s, duration=30.0))
           for p in [0.0, 0.01, 0.05, 0.1, 0.2] for s in SEEDS]
    out += [(f"skew{int(th * 1000)}_s{s}", dict(M=300, theta_max=th, seed=s, duration=30.0))
            for th in [0.0, 0.01, 0.05, 0.2] for s in SEEDS]
    return out


def runs_noise():
    out = [(f"la{int(r * 100)}_s{s}", dict(M=300, rho_la=r, seed=s, duration=30.0))
           for r in [0.0, 0.1, 0.3, 0.5] for s in SEEDS]
    out += [(f"pn{x}_s{s}", dict(M=300, sigma_p0=0.5 * x, sigma_p1=2.0 * x, seed=s, duration=30.0))
            for x in [0.5, 1, 2, 3] for s in SEEDS]
    return out


def runs_crash():
    out = []
    for s in [1, 2, 3, 4, 5]:
        for rep in [True, False]:
            tag = "rep" if rep else "norep"
            out.append((f"center_{tag}_s{s}", dict(M=300, seed=s, duration=30.0, replication=rep,
                                                  faults=[(15.0, "crash_center", (s * 3) % 15)],
                                                  _crash=(15.0, (s * 3) % 15))))
    for rate in [0, 30, 120, 360]:
        for s in SEEDS:
            out.append((f"member{rate}_s{s}", dict(M=300, seed=s, duration=30.0,
                                                  member_crash_rate=rate, member_mttr=10.0)))
    return out


def runs_agg():
    out = []
    for down in [1.0, 5.0]:
        for s in SEEDS:
            out.append((f"agg{int(down)}_s{s}", dict(M=300, seed=s, duration=30.0,
                                                    faults=[(15.0, "crash_agg", down)], _agg=(15.0, down))))
    out += [(f"agg_none_s{s}", dict(M=300, seed=s, duration=30.0)) for s in SEEDS]
    return out


def runs_churn():
    return [(f"churn{int(r * 10)}_s{s}", dict(M=300, seed=s, duration=30.0, join_rate=r, leave_rate=r))
            for r in [0.0, 0.2, 1.0, 2.0] for s in SEEDS]


def runs_ablation():
    out = []
    for M in [300, 1000]:
        for s in SEEDS:
            out.append((f"full_M{M}_s{s}", dict(M=M, seed=s, duration=30.0)))
            out.append((f"nohandoff_M{M}_s{s}", dict(M=M, seed=s, duration=30.0, handoff=False)))
            out.append((f"nohandoff_noagg_M{M}_s{s}", dict(M=M, seed=s, duration=30.0, handoff=False,
                                                           dedup=False, global_reid=False)))
            out.append((f"noglobalreid_M{M}_s{s}", dict(M=M, seed=s, duration=30.0, global_reid=False)))
    return out


EXPS = {"scale": runs_scale, "region": runs_region, "deadline": runs_deadline,
        "network": runs_network, "noise": runs_noise, "crash": runs_crash, "agg": runs_agg,
        "churn": runs_churn, "ablation": runs_ablation}


def one(exp, name, kw):
    path = os.path.join(OUT, exp, name + ".json")
    if os.path.exists(path):
        return name, "cached"
    kw = dict(kw)
    crash = kw.pop("_crash", None)
    agg = kw.pop("_agg", None)
    cfg = Config(**kw)
    t0 = time.time()
    s = System(cfg)
    wall = s.run()
    res = s.results()
    res["config"] = {k: v for k, v in cfg.to_dict().items() if k != "faults"}
    res["faults"] = [list(f) for f in cfg.faults]
    res["wall"] = wall
    res["n_regions"] = cfg.K
    if crash is not None:
        res["continuity"] = s.continuity(crash[0], crash[1], after=3.0)
        res["continuity_10s"] = s.continuity(crash[0], crash[1], after=10.0)
        # ID switches in the 5 s after the crash vs the 5 s before
        from precinct.metrics import evaluate
        kc = int(crash[0] / cfg.dt)
        res["acc_before"] = evaluate(s.rec, cfg, s.agg.aliases, k_from=kc - 50, k_to=kc)
        res["acc_after"] = evaluate(s.rec, cfg, s.agg.aliases, k_from=kc, k_to=kc + 50)
    if agg is not None:
        from precinct.metrics import evaluate
        t_c, down = agg
        kc = int(t_c / cfg.dt)
        kr = int((t_c + down) / cfg.dt)
        res["acc_before"] = evaluate(s.rec, cfg, s.agg.aliases, k_from=kc - 50, k_to=kc)
        res["acc_after"] = evaluate(s.rec, cfg, s.agg.aliases, k_from=kr + 10, k_to=kr + 60)
        # identity preservation: gids in the view just before the crash that are
        # still tracked (same object) after the rebuild
        kb = max(k for k in s.rec.views if k < kc)       # last view published before the crash
        res["id_preserved"] = _id_preserved(s, kb, kr + 20)
    res["events_n"] = len(s.sim.events)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(res, f, default=_ser)
    return name, f"{time.time() - t0:.0f}s"


def _id_preserved(s, kb, ka):
    from precinct.metrics import _resolve, match_step
    res = _resolve(s.agg.aliases)

    def m(k):
        if k not in s.rec.views:
            return {}
        ids, gpos, _ = s.rec.gt[k]
        v = s.rec.views[k]
        vi, gi, _ = match_step(gpos, v["pos"])
        return {int(ids[b]): res(v["gids"][a]) for a, b in zip(vi, gi)}
    a, b = m(kb), m(ka)
    both = [g for g in a if g in b]
    return {"objects": len(both), "kept": sum(1 for g in both if a[g] == b[g]),
            "frac": sum(1 for g in both if a[g] == b[g]) / max(len(both), 1)}


def _ser(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on run names")
    a = ap.parse_args()
    exps = list(EXPS) if a.exp == "all" else a.exp.split(",")
    jobs = [(e, n, kw) for e in exps for n, kw in EXPS[e]()]
    if a.only:
        jobs = [j for j in jobs if a.only in j[1]]
    # longest first
    jobs.sort(key=lambda j: -j[2].get("M", 100) * (5 if j[2].get("centralized") else 1))
    if a.list:
        for j in jobs:
            print(j[0], j[1])
        print(len(jobs), "runs")
        return
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for f in as_completed(futs):
            e, n, _ = futs[f]
            try:
                print(e, *f.result(), flush=True)
            except Exception:
                print("FAILED", e, n, traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
