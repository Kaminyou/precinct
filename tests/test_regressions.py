"""Regression tests for the latency breakdown, Raft commit monotonicity, and
bounded aggregator caches. Run: python -m pytest -q tests"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct.config import Config  # noqa: E402
from precinct.raft import Entry, RaftPeer  # noqa: E402
from precinct.sim import Msg  # noqa: E402
from precinct.system import System  # noqa: E402


def test_latency_parts_non_negative_centralized():
    # With a single voter the entry commits inside the charged step handler;
    # the breakdown used to report a negative "raft commit" equal to -fusion.
    s = System(Config(M=100, duration=10, seed=1, centralized=True))
    s.run()
    lat = s.results()["latency"]
    parts = lat["parts"]
    assert parts, "no latency samples"
    for name, v in parts.items():
        assert v >= -1e-9, f"{name} is negative: {v}"
    # the parts still add up to (roughly) the mean end-to-end latency
    assert abs(sum(parts.values()) - lat["mean"]) < 0.02


class _FakeHost:
    """Minimal host for driving a RaftPeer by hand."""

    def __init__(self, nid="f"):
        cfg = SimpleNamespace(election_timeout=0.3, raft_hb=0.05, T_snap=10_000)
        self.sim = SimpleNamespace(cfg=cfg, now=0.0, rng=np.random.default_rng(0),
                                   log=lambda *a, **k: None)
        self.id = nid
        self.region = 0
        self.sent = []
        self.applied = []

    def send(self, dst, msg):
        self.sent.append((dst, msg))

    def timer(self, *a, **k):
        pass

    def sm_apply(self, e):
        self.applied.append(e)

    def sm_snapshot(self):
        return None, 0

    def sm_restore(self, st):
        pass

    def raft_became_leader(self):
        pass

    def raft_stepped_down(self):
        pass

    def raft_committed(self, e):
        pass


def _ae(prev, pterm, ents, commit, term=1, src="L"):
    m = Msg("AE", {"term": term, "leader": src, "prev": prev, "pterm": pterm,
                   "ents": ents, "commit": commit}, 40)
    m.src = src
    return m


def test_follower_commit_never_moves_backwards():
    h = _FakeHost()
    r = RaftPeer(h, ["L", "f", "g"])
    ents = [Entry(1, "noop") for _ in range(10)]
    # leader replicates entries 1..10 with commit 10
    r.on_AE(_ae(0, 0, ents, commit=10))
    assert r.commit == 10 and len(h.applied) == 10
    # a delayed / reordered AppendEntries covering only up to index 5 arrives,
    # carrying a newer leader commit of 12
    r.on_AE(_ae(3, 1, ents[3:5], commit=12))
    assert r.commit == 10, "commit index regressed"
    assert len(h.applied) == 10           # nothing re-applied
    # config_pending() slices the log from the commit index; it must stay valid
    assert r.config_pending() is False


def test_aggregator_caches_are_pruned():
    s = System(Config(M=100, duration=12, seed=1, cost_model="none"))
    s.run()
    a = s.agg
    live = {t[0] for _, tracks in a.last_rep.values() for t in tracks.values()}
    live_z = live & set(a.z)
    assert live_z, "expected cached appearance vectors for live tracks"
    k = max(s.rec.views)
    # a gid nobody has referenced for a long time
    ghost = ("ghost", 0, 0)
    a.z[ghost] = np.zeros(4)
    a.created[ghost] = 0.0
    a.last_ref[ghost] = k - 10_000
    a._prune_soft(k)
    assert ghost not in a.z and ghost not in a.created
    # gids still reported by some region keep their appearance vectors
    assert live_z <= set(a.z)


def test_aggregator_memory_bounded_over_long_run():
    s = System(Config(M=100, duration=150, seed=1, cost_model="none"))
    s.run()
    a = s.agg
    # without pruning every gid ever created stays cached (~450 after 150 s);
    # with pruning the cache stays within a small multiple of the live view
    assert len(a.z) < 4 * max(len(a.view), 1)
