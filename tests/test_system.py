"""End-to-end and unit tests. Run: python -m pytest -q tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct.config import Config  # noqa: E402
from precinct.regions import balanced_kmeans, choose_voters  # noqa: E402
from precinct.system import System  # noqa: E402
from precinct.tracker import kf_predict, kf_update, maha2  # noqa: E402


def test_balanced_kmeans_deterministic_and_balanced():
    P = np.random.default_rng(0).uniform(0, 1000, (200, 2))
    a, ca = balanced_kmeans(P, 10, seed=7)
    b, cb = balanced_kmeans(P, 10, seed=7)
    assert (a == b).all() and np.allclose(ca, cb)
    assert np.bincount(a).max() <= 20
    v = choose_voters(P, a, ca, 3)
    assert all(len(x) == 3 for x in v.values())


def test_kalman_tracks_constant_velocity():
    X = np.array([[0.0, 0.0, 0.0, 0.0]])
    P = np.diag([1.0, 1.0, 100.0, 100.0])[None]
    for t in range(1, 30):
        X, P = kf_predict(X, P, np.array([0.1]), 1.0)
        X, P = kf_update(X, P, np.array([[1.0 * t, 0.5 * t]]) * 0.1 * 10, np.array([0.25]))
    assert abs(X[0, 2] - 10.0) < 0.5 and abs(X[0, 3] - 5.0) < 0.5
    d = maha2(X, P, X[:, :2] + 0.0, np.array([0.25]))
    assert d[0, 0] < 1e-9


@pytest.fixture(scope="module")
def base_run():
    s = System(Config(M=100, duration=20, seed=3))
    s.run()
    return s


def test_one_center_per_region(base_run):
    s = base_run
    centers = {}
    for c in s.cams:
        if c.alive and c.role is not None:
            assert c.region not in centers
            centers[c.region] = c.id
    assert len(centers) == s.cfg.K


def test_tracking_quality(base_run):
    r = base_run.results()["accuracy"]
    assert r["mota"] > 0.85
    assert r["idf1"] > 0.85
    assert r["rmse"] < 2.5


def test_gids_unique_and_replicas_agree(base_run):
    s = base_run
    for reg in range(s.cfg.K):
        leader = s.center_of(reg)
        committed = leader.committed
        for v in leader.raft.voters:
            node = s.sim.nodes[v]
            if node is leader or not node.alive or node.raft is None:
                continue
            # a follower's committed table is a prefix of the leader's
            if node.committed.last_step == committed.last_step:
                assert set(node.committed.tracks) == set(committed.tracks)
    all_gids = [g for c in s.cams if c.role for g in c.role.work.tracks if c.role.work.tracks[g].owner]
    assert len(all_gids) == len(set(all_gids))


def test_handoffs_happen(base_run):
    ev = base_run.results()["events"]
    assert ev.get("accept", 0) > 0 and ev.get("release", 0) > 0


def test_center_failover_preserves_ids():
    s = System(Config(M=100, duration=25, seed=2, faults=[(12.0, "crash_center", 1)]))
    s.run()
    c = s.continuity(12.0, 1, after=3.0)
    assert c["recovery_s"] < 1.5
    assert c["still_tracked"] > 0 and c["kept_frac"] > 0.8
    assert s.center_of(1) is not None


def test_aggregator_restart_rebuilds():
    s = System(Config(M=100, duration=25, seed=2, faults=[(12.0, "crash_agg", 2.0)]))
    s.run()
    assert s.rec.rebuilds and s.rec.rebuilds[0][1]
    assert s.rec.rebuilds[0][0] < 0.5
    k = max(s.rec.views)
    assert len(s.rec.views[k]["gids"]) > 50


def test_centralized_baseline_runs():
    s = System(Config(M=100, duration=10, seed=1, centralized=True))
    s.run()
    assert s.cfg.K == 1
    assert s.results()["accuracy"]["mota"] > 0.8
