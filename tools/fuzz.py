"""Fuzz the simulator: run many short simulations with random mixes of faults,
churn, loss and ablations, and report any that crash.

    python tools/fuzz.py --runs 200            # seeds 0..199
    python tools/fuzz.py --start 1000 --runs 50

Each failing seed prints its exception, location and full config, so it can be
replayed with System(Config(**kw)).run()."""
import argparse
import json
import os
import random
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct.config import Config  # noqa: E402
from precinct.system import System  # noqa: E402


def random_config(i):
    R = random.Random(i)
    dur = R.choice([12.0, 18.0])
    faults = []
    for _ in range(R.randint(0, 3)):
        t = R.uniform(3, dur - 2)
        f = R.choice(["crash_center", "crash_agg", "loss"])
        if f == "crash_center":
            faults.append((t, "crash_center", R.randrange(5), R.choice([1.0, 3.0, None])))
        elif f == "crash_agg":
            faults.append((t, "crash_agg", R.choice([0.5, 2.0])))
        else:
            faults.append((t, "loss", R.choice([0.3, 0.0])))
    return dict(M=R.choice([60, 100, 150]), seed=i, duration=dur,
                cost_model=R.choice(["none", "none", "measured"]),
                p_loss=R.choice([0.0, 0.01, 0.05, 0.2]),
                join_rate=R.choice([0, 0, 0.5, 2.0]), leave_rate=R.choice([0, 0, 0.5, 2.0]),
                member_crash_rate=R.choice([0, 0, 60, 600]), member_mttr=R.choice([3.0, 10.0]),
                S=R.choice([5, 10, 20, 50]), R=R.choice([1, 3, 5]),
                replication=R.random() > 0.25, handoff=R.random() > 0.2,
                dedup=R.random() > 0.2, global_reid=R.random() > 0.2,
                T_snap=R.choice([5, 50]), centralized=R.random() < 0.15, faults=faults)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    failed = 0
    for i in range(args.start, args.start + args.runs):
        kw = random_config(i)
        try:
            System(Config(**kw)).run()
        except Exception as e:
            failed += 1
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(json.dumps({"seed": i, "error": f"{type(e).__name__}: {e}",
                              "at": f"{os.path.basename(tb.filename)}:{tb.lineno} {tb.name}",
                              "config": {k: repr(v) for k, v in kw.items()}}), flush=True)
    print(f"{args.runs - failed}/{args.runs} runs completed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
