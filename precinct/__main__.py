"""Run one simulation and print its metrics.

    python -m precinct --M 300 --duration 30
    python -m precinct --M 100 --duration 90 --scenario demo --trace demo.jsonl
    python -m precinct --M 1000 --centralized
    python -m precinct --M 300 --set p_loss=0.1 --set S=10
"""
import argparse
import json

from .config import Config
from .system import System
from .viz.server import DEMO_FAULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objects", type=float, default=None, help="mean number of objects (default: = M)")
    ap.add_argument("--centralized", action="store_true")
    ap.add_argument("--scenario", choices=["demo"], default=None)
    ap.add_argument("--trace", default=None, help="write a replayable .jsonl trace")
    ap.add_argument("--set", action="append", default=[], help="key=value config override")
    a = ap.parse_args()
    kw = {}
    for s in a.set:
        k, v = s.split("=", 1)
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
    cfg = Config(M=a.M, N_bar=a.objects, duration=a.duration, seed=a.seed, centralized=a.centralized, trace=a.trace,
                 faults=DEMO_FAULTS if a.scenario == "demo" else [], **kw)
    s = System(cfg)
    wall = s.run(progress=True)
    r = s.results()
    acc, lat, ovh = r["accuracy"], r["latency"], r["overhead"]
    print(f"\nM={cfg.M} K={cfg.K} sim={cfg.duration}s wall={wall:.1f}s")
    print(f"MOTA {acc['mota']:.3f}  IDF1 {acc['idf1']:.3f}  IDSW {acc['idsw']} (online {acc['idsw_online']})"
          f"  RMSE {acc['rmse']:.2f} m  dup {acc['dup_rate']:.4f}")
    print(f"latency p50 {lat['p50'] * 1e3:.0f} ms  p99 {lat['p99'] * 1e3:.0f} ms")
    for k, v in lat["parts"].items():
        print(f"   {k:28s} {v * 1e3:7.1f} ms")
    print(f"busiest center CPU {ovh['center_cpu_max'] * 100:.1f}%  aggregator CPU {ovh['agg_cpu'] * 100:.1f}%"
          f"  late detections {ovh['late_det_rate'] * 100:.2f}%")
    print("bytes/s per tier:", {k: round(v) for k, v in ovh["bytes_per_s"].items()})
    print("events:", r["events"])


if __name__ == "__main__":
    main()
