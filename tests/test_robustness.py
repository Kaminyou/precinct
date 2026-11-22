"""Robustness tests: a center that is told it is no longer a voter, and a fuzz
smoke test over random fault mixes. Run: python -m pytest -q tests"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct.config import Config  # noqa: E402
from precinct.sim import Msg  # noqa: E402
from precinct.system import System  # noqa: E402


def _running_center(**kw):
    s = System(Config(M=60, S=20, duration=30, seed=3, cost_model="none", **kw))
    s.run(until=4.0)
    c = next(c for c in s.cams if c.alive and c.role is not None)
    return s, c


def _join_r(s, c, voters, term):
    D = s.agg.dir
    m = Msg("JOIN_R", {"region": c.region, "voters": voters, "center": voters[0],
                       "dir": D.snap, "cam": c.id, "term": term}, 128)
    m.src = "agg"
    return m


def test_center_steps_down_when_directory_has_newer_center():
    # no-replication ablation: the directory appointed someone else in a newer term
    s, c = _running_center(replication=False)
    other = next(x.id for x in s.cams if x.region == c.region and x.id != c.id)
    c.on_JOIN_R(_join_r(s, c, [other], c.raft.term + 1))
    assert c.role is None and c.raft is None
    s.run(until=8.0)                       # used to raise AttributeError on n.raft


def test_center_ignores_stale_join_reply():
    s, c = _running_center(replication=False)
    other = next(x.id for x in s.cams if x.region == c.region and x.id != c.id)
    term = c.raft.term
    c.on_JOIN_R(_join_r(s, c, [other], term - 1))   # answer to an older, retried JOIN
    assert c.role is not None and c.raft is not None and c.raft.term == term
    s.run(until=8.0)


def test_no_replication_regression_case():
    # the fuzz case that first exposed the bug
    s = System(Config(M=100, seed=1011, duration=18.0, cost_model="none", p_loss=0.2,
                      leave_rate=2.0, member_crash_rate=600, member_mttr=10.0,
                      faults=[(5.319327701206319, "crash_agg", 2.0),
                              (13.621554705130126, "loss", 0.0)],
                      S=5, R=3, replication=False, T_snap=5))
    s.run()


def _random_config(i):
    R = random.Random(i)
    faults = []
    for _ in range(R.randint(0, 3)):
        t = R.uniform(3, 8)
        f = R.choice(["crash_center", "crash_agg", "loss"])
        if f == "crash_center":
            faults.append((t, "crash_center", R.randrange(5), R.choice([1.0, 3.0, None])))
        elif f == "crash_agg":
            faults.append((t, "crash_agg", R.choice([0.5, 2.0])))
        else:
            faults.append((t, "loss", R.choice([0.3, 0.0])))
    return Config(M=R.choice([60, 100]), seed=i, duration=10.0, cost_model="none",
                  p_loss=R.choice([0.0, 0.05, 0.2]), join_rate=R.choice([0, 1.0]),
                  leave_rate=R.choice([0, 1.0]), member_crash_rate=R.choice([0, 600]),
                  member_mttr=3.0, S=R.choice([5, 10, 20]), R=R.choice([1, 3]),
                  replication=R.random() > 0.3, T_snap=R.choice([5, 50]),
                  centralized=R.random() < 0.15, faults=faults)


def test_fuzz_smoke():
    """A handful of random fault mixes must run to completion and produce
    finite accuracy. For a longer search run tools/fuzz.py."""
    for i in range(8):
        s = System(_random_config(i))
        s.run()
        a = s.results()["accuracy"]
        assert a["mota"] == a["mota"] and a["idf1"] == a["idf1"], f"NaN accuracy for seed {i}"
