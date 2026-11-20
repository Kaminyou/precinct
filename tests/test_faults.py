"""Regression tests for crash-stop message semantics and camera joins.
Run: python -m pytest -q tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precinct import system as system_mod  # noqa: E402
from precinct.config import Config  # noqa: E402
from precinct.sim import Msg, Node, Sim  # noqa: E402
from precinct.system import System  # noqa: E402


class _Sink(Node):
    def __init__(self, sim, nid):
        super().__init__(sim, nid)
        self.got = []

    def on_PING(self, m):
        self.got.append(m)


def _sim(p_loss=0.0):
    cfg = Config(M=10, p_loss=p_loss, cost_model="none")
    return Sim(cfg, np.random.default_rng(0))


def test_crashed_node_does_not_send_scheduled_messages():
    sim = _sim()
    a, b = _Sink(sim, "a"), _Sink(sim, "b")
    # `a` schedules a message to leave at t=0.10 (like a DET sent at the end of
    # a capture step), then crashes at t=0.05 before it departs
    sim.send(a, "b", Msg("PING"), t_send=0.10)
    sim.at(0.05, a.crash)
    sim.run_until(1.0)
    assert b.got == []
    assert sim.stats.counters["dropped_sender_crashed"] == 1


def test_message_already_in_flight_survives_sender_crash():
    sim = _sim()
    a, b = _Sink(sim, "a"), _Sink(sim, "b")
    sim.send(a, "b", Msg("PING"), t_send=0.0)     # departs now
    sim.at(1e-4, a.crash)                         # crashes while it is on the wire
    sim.run_until(1.0)
    assert len(b.got) == 1


def test_message_sent_after_recovery_is_delivered():
    sim = _sim()
    a, b = _Sink(sim, "a"), _Sink(sim, "b")
    a.crash()
    a.recover()
    sim.send(a, "b", Msg("PING"), t_send=0.1)
    sim.run_until(1.0)
    assert len(b.got) == 1


def test_member_crashes_drop_pending_detections():
    s = System(Config(M=100, duration=15, seed=1, member_crash_rate=360, member_mttr=5.0,
                      cost_model="none"))
    s.run()
    assert s.sim.stats.counters["dropped_sender_crashed"] > 0


def test_joins_do_not_crash_center():
    # Used to raise IndexError in tracker.step: a new camera learned its center
    # from JOIN_R and sent detections before that center's directory listed it.
    s = System(Config(M=100, duration=20, seed=1, join_rate=1.0, cost_model="none"))
    s.run()
    assert len(s.cams) > 100


def test_joins_are_retried_and_idempotent(monkeypatch):
    attempts = []
    orig = system_mod._Joiner.go

    def go(self):
        attempts.append(self)
        orig(self)
    monkeypatch.setattr(system_mod._Joiner, "go", go)
    # heavy loss plus an aggregator outage: without retries some joins vanish
    s = System(Config(M=100, duration=20, seed=2, join_rate=1.0, p_loss=0.2,
                      faults=[(8.0, "crash_agg", 2.0)], cost_model="none"))
    s.run(until=17.0)
    # every join requested more than a few retry periods before the end has registered
    done = [j for j in attempts if j.done]
    assert attempts and len(done) >= len(attempts) - 3
    # retries never register the same camera twice
    D = s.agg.dir
    assert len(D.join_tokens) == len(set(D.join_tokens.values()))
    assert len(s.cams) == len(D.cam_pos) == 100 + len(D.join_tokens)
