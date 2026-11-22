"""Tests for the faster aggregator: incremental step closing and vectorized
global re-identification. Run: python -m pytest -q tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct import aggregator as agg_mod  # noqa: E402
from precinct.config import Config  # noqa: E402
from precinct.system import System  # noqa: E402


def test_active_regions_computed_once_per_step(monkeypatch):
    # closing a step used to re-derive the active region set on every report:
    # O(regions) recomputations per step
    calls = []
    orig = agg_mod.AggregatorNode._active_regions

    def counted(self):
        calls.append(self.sim.now)
        return orig(self)
    monkeypatch.setattr(agg_mod.AggregatorNode, "_active_regions", counted)
    s = System(Config(M=300, duration=6, seed=1, cost_model="none"))
    s.run()
    steps = len(s.rec.views)
    regions = len(s.agg.dir.voters)
    assert regions >= 10
    assert len(calls) <= 3 * steps, (len(calls), steps)


def test_steps_still_close_early_and_completely():
    s = System(Config(M=300, duration=6, seed=1, cost_model="none"))
    s.run()
    lat = s.results()["latency"]["parts"]
    # every region reports well before the aggregation deadline, so a step closes
    # as soon as the last report arrives instead of waiting the full W_a
    assert 0.0 <= lat["wait agg deadline"] < 0.8 * s.cfg.W_a
    assert len(s.rec.views) >= int(6 / s.cfg.dt) - 2         # no step was skipped


def test_directory_change_recomputes_pending_regions():
    s = System(Config(M=300, duration=4, seed=1, cost_model="none"))
    s.run()
    a = s.agg
    a._try_close()
    key = a._pend_key
    a.dir.bump()                    # e.g. a camera joined or failed
    a._try_close()
    assert a._pend_key != key and a._pend_key[1] == a.dir.version


def _reference_reid(agg, view, new, t):
    """The original per-gid Python loop, kept here as the specification."""
    cfg = agg.sim.cfg
    picks = {}
    vanished = dict(agg.vanished)
    for g in new:
        v = view[g]
        if v[5]:
            continue
        z = agg.z.get(v[7])
        cg = agg.created.get(v[7], t)
        if z is None:
            continue
        best, bc = None, -1.0
        for h, (x, y, tv, reg, ep, cr, raw) in vanished.items():
            if cr >= cg:
                continue
            reach = cfg.v_max * (t - tv) + 10.0
            if (x - v[1]) ** 2 + (y - v[2]) ** 2 > reach ** 2:
                continue
            zh = agg.z.get(raw)
            if zh is None:
                continue
            cs = float(zh @ z)
            if cs >= cfg.s_reid and cs > bc:
                best, bc = h, cs
        if best is not None:
            vanished.pop(best)
            picks[g] = best
    return picks


def test_vectorized_reid_matches_reference():
    s = System(Config(M=60, duration=2, seed=1, cost_model="none"))
    s.run()
    a = s.agg
    rng = np.random.default_rng(7)
    for trial in range(20):
        a.vanished, a.merges, a.z, a.created = {}, {}, {}, {}
        t = 50.0
        base = rng.normal(size=(6, 16))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        for i in range(300):                         # vanished pool
            raw = ("v", trial, i)
            z = base[i % 6] + 0.3 * rng.normal(size=16)
            a.z[raw] = z / np.linalg.norm(z)
            if rng.random() < 0.1:
                del a.z[raw]                          # some have no appearance
            a.vanished[raw] = (rng.uniform(0, 500), rng.uniform(0, 500), t - rng.uniform(0, 30),
                               0, 0, rng.uniform(0, 40), raw)
        view, new = {}, []
        for i in range(60):                          # gids appearing for the first time
            g = ("n", trial, i)
            z = base[i % 6] + 0.3 * rng.normal(size=16)
            a.z[g] = z / np.linalg.norm(z)
            a.created[g] = rng.uniform(0, 60)
            view[g] = (g, rng.uniform(0, 500), rng.uniform(0, 500), 1, 0, rng.random() < 0.1, 1.0, g)
            new.append(g)
        want = _reference_reid(a, view, new, t)
        a._reid_new(view, new, t)
        got = {g: m[0] for g, m in a.merges.items()}
        assert got == want, trial
        assert len(want) > 0
