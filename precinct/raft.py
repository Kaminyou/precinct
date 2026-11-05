"""Raft consensus among a region's voters: leader election, log replication,
snapshots (InstallSnapshot), single-server membership changes, leader
stickiness and check-quorum.

The Raft peer is embedded in a CameraNode (`host`), which provides send/timer
and the state-machine callbacks:
    host.sm_apply(entry)          apply a committed entry to the committed state
    host.sm_snapshot()            -> (state, size) of the committed state
    host.sm_restore(state)        replace the committed state
    host.raft_became_leader()
    host.raft_stepped_down()
    host.raft_committed(entry)    leader only, after sm_apply
"""
from __future__ import annotations

from .sim import Msg


class Entry:
    __slots__ = ("term", "kind", "data", "size", "out")

    def __init__(self, term, kind, data=None, size=16, out=None):
        self.term, self.kind, self.data, self.size, self.out = term, kind, data, size, out


class RaftPeer:
    LEADER, FOLLOWER, CANDIDATE = "leader", "follower", "candidate"

    def __init__(self, host, voters, term=0):
        self.h = host
        self.cfg = host.sim.cfg
        # durable
        self.term = term
        self.voted_for = None
        self.log = []                  # entries snap_index+1 ...
        self.snap_index = 0
        self.snap_term = 0
        self.snap_state = None
        self.snap_size = 0
        self.snap_voters = list(voters)
        # volatile
        self.voters = list(voters)
        self.commit = 0
        self.applied = 0
        self.role = self.FOLLOWER
        self.leader = None
        self.last_heard = host.sim.now
        self.next = {}
        self.match = {}
        self.last_ack = {}
        self.last_sent = {}
        self.votes = set()
        self.timeout = self._rand_timeout()
        self.active = True

    # ------------------------------------------------------------ helpers
    def _rand_timeout(self):
        T = self.cfg.election_timeout
        return T + self.h.sim.rng.random() * T

    @property
    def last_index(self):
        return self.snap_index + len(self.log)

    def term_at(self, i):
        if i == self.snap_index:
            return self.snap_term
        if i < self.snap_index or i > self.last_index:
            return -1
        return self.log[i - self.snap_index - 1].term

    def entry(self, i):
        return self.log[i - self.snap_index - 1]

    def majority(self):
        return len(self.voters) // 2 + 1

    def _recompute_voters(self):
        v = self.snap_voters
        for e in self.log:
            if e.kind == "config":
                v = e.data
        self.voters = list(v)

    def start(self, initial_delay=None):
        self.last_heard = self.h.sim.now
        d = self.timeout if initial_delay is None else initial_delay
        self.h.timer(d, self._election_check)

    # ------------------------------------------------------------ elections
    def _election_check(self):
        if not self.active:
            return
        if self.role == self.LEADER:
            return
        now = self.h.sim.now
        if now - self.last_heard >= self.timeout:
            self._start_election()
            self.h.timer(self.timeout, self._election_check)
        else:
            self.h.timer(max(1e-3, self.last_heard + self.timeout - now), self._election_check)

    def _start_election(self):
        if self.h.id not in self.voters:
            return
        self.term += 1
        self.role = self.CANDIDATE
        self.voted_for = self.h.id
        self.votes = {self.h.id}
        self.leader = None
        self.last_heard = self.h.sim.now
        self.timeout = self._rand_timeout()
        self.h.sim.log("election", region=self.h.region, node=self.h.id, term=self.term)
        if len(self.votes) >= self.majority():
            self._become_leader()
            return
        for v in self.voters:
            if v != self.h.id:
                self.h.send(v, Msg("RV", {"term": self.term, "cand": self.h.id,
                                          "lli": self.last_index, "llt": self.term_at(self.last_index),
                                          "force": False}, 32))

    def on_RV(self, m):
        d = m.data
        if m.src not in self.voters:
            self.h.send(m.src, Msg("NOT_MEMBER", {"term": self.term}, 16))
            return
        now = self.h.sim.now
        # leader stickiness: ignore disruptive candidates while a leader is alive
        if self.role != self.CANDIDATE and self.leader is not None and \
                now - self.last_heard < self.cfg.election_timeout and not d["force"]:
            self.h.send(m.src, Msg("RV_R", {"term": self.term, "granted": False}, 16))
            return
        if d["term"] > self.term:
            self._step_down(d["term"])
        up_to_date = (d["llt"], d["lli"]) >= (self.term_at(self.last_index), self.last_index)
        granted = d["term"] == self.term and up_to_date and self.voted_for in (None, d["cand"])
        if granted:
            self.voted_for = d["cand"]
            self.last_heard = now
        self.h.send(m.src, Msg("RV_R", {"term": self.term, "granted": granted}, 16))

    def on_RV_R(self, m):
        d = m.data
        if d["term"] > self.term:
            self._step_down(d["term"])
            return
        if self.role != self.CANDIDATE or d["term"] != self.term or not d["granted"]:
            return
        self.votes.add(m.src)
        if len(self.votes & set(self.voters)) >= self.majority():
            self._become_leader()

    def _become_leader(self):
        self.role = self.LEADER
        self.leader = self.h.id
        for v in self.voters:
            self.next[v] = self.last_index + 1
            self.match[v] = 0
            self.last_ack[v] = self.h.sim.now
            self.last_sent[v] = -1.0
        self.match[self.h.id] = self.last_index
        self.h.sim.log("leader", region=self.h.region, node=self.h.id, term=self.term)
        self.h.raft_became_leader()
        self.propose(Entry(self.term, "noop"))
        self.h.timer(self.cfg.raft_hb, self._heartbeat)

    def _step_down(self, term):
        was_leader = self.role == self.LEADER
        if term > self.term:
            self.term = term
            self.voted_for = None
        self.role = self.FOLLOWER
        if was_leader:
            self.leader = None
            self.h.raft_stepped_down()
            self.last_heard = self.h.sim.now
            self.h.timer(self.timeout, self._election_check)

    def on_NOT_MEMBER(self, m):
        if self.role != self.LEADER and m.data["term"] >= self.term:
            self.active = False
            self.h.raft_removed()

    # ------------------------------------------------------------ replication
    def propose(self, entry):
        assert self.role == self.LEADER
        entry.term = self.term
        self.log.append(entry)
        if entry.kind == "config":
            self._recompute_voters()
            for v in self.voters:
                if v not in self.next:
                    self.next[v] = self.last_index
                    self.match[v] = 0
                    self.last_ack[v] = self.h.sim.now
                    self.last_sent[v] = -1.0
        self.match[self.h.id] = self.last_index
        for v in self.voters:
            if v != self.h.id:
                self._send_ae(v)
        self._advance_commit()
        return self.last_index

    def _send_ae(self, v):
        nxt = self.next.get(v, self.last_index + 1)
        if nxt <= self.snap_index:
            self.h.send(v, Msg("SNAP", {"term": self.term, "leader": self.h.id,
                                        "index": self.snap_index, "sterm": self.snap_term,
                                        "state": self.snap_state, "voters": self.snap_voters},
                               48 + self.snap_size))
            self.next[v] = self.snap_index + 1
            self.last_sent[v] = self.h.sim.now
            return
        prev = nxt - 1
        ents = self.log[prev - self.snap_index:]
        if len(ents) > 64:
            ents = ents[:64]
        size = 40 + sum(e.size for e in ents)
        self.h.send(v, Msg("AE", {"term": self.term, "leader": self.h.id, "prev": prev,
                                  "pterm": self.term_at(prev), "ents": ents,
                                  "commit": self.commit}, size))
        self.next[v] = prev + len(ents) + 1        # optimistic pipelining
        self.last_sent[v] = self.h.sim.now

    def _heartbeat(self):
        if self.role != self.LEADER or not self.active:
            return
        now = self.h.sim.now
        # check quorum: step down if a majority has not answered recently
        alive = sum(1 for v in self.voters
                    if v == self.h.id or now - self.last_ack.get(v, -1e9) < 2 * self.cfg.election_timeout)
        if alive < self.majority():
            self.h.sim.log("stepdown", region=self.h.region, node=self.h.id, term=self.term)
            self._step_down(self.term)
            return
        for v in self.voters:
            if v != self.h.id and now - self.last_sent.get(v, -1) >= self.cfg.raft_hb * 0.99:
                if self.match.get(v, 0) < self.last_index:
                    self.next[v] = self.match.get(v, 0) + 1   # resend what is unacknowledged
                self._send_ae(v)
        self.h.timer(self.cfg.raft_hb, self._heartbeat)

    def on_AE(self, m):
        d = m.data
        if d["term"] < self.term:
            self.h.send(m.src, Msg("AE_R", {"term": self.term, "ok": False, "hint": self.last_index}, 24))
            return
        if d["term"] > self.term or self.role != self.FOLLOWER:
            self._step_down(d["term"])
        self.leader = d["leader"]
        self.last_heard = self.h.sim.now
        self.active = True
        prev = d["prev"]
        if prev > self.last_index or (prev >= self.snap_index and self.term_at(prev) != d["pterm"]):
            hint = min(self.last_index, prev - 1) if prev <= self.last_index else self.last_index
            self.h.send(m.src, Msg("AE_R", {"term": self.term, "ok": False,
                                            "hint": max(hint, self.snap_index)}, 24))
            return
        idx = prev
        conf = False
        for e in d["ents"]:
            idx += 1
            if idx <= self.snap_index:
                continue
            if idx <= self.last_index:
                if self.term_at(idx) == e.term:
                    continue
                del self.log[idx - self.snap_index - 1:]
                conf = True
            self.log.append(e)
            if e.kind == "config":
                conf = True
        if conf:
            self._recompute_voters()
        match = prev + len(d["ents"])
        if d["commit"] > self.commit:
            self.commit = min(d["commit"], match)
            self._apply()
        self.h.send(m.src, Msg("AE_R", {"term": self.term, "ok": True, "match": match}, 24))

    def on_AE_R(self, m):
        d = m.data
        if d["term"] > self.term:
            self._step_down(d["term"])
            return
        if self.role != self.LEADER:
            return
        v = m.src
        self.last_ack[v] = self.h.sim.now
        if d["ok"]:
            if d["match"] > self.match.get(v, 0):
                self.match[v] = d["match"]
            self._advance_commit()
        else:
            self.next[v] = max(1, min(self.next.get(v, 1), d["hint"] + 1))
            self._send_ae(v)

    def on_SNAP(self, m):
        d = m.data
        if d["term"] < self.term:
            return
        if d["term"] > self.term or self.role != self.FOLLOWER:
            self._step_down(d["term"])
        self.leader = d["leader"]
        self.last_heard = self.h.sim.now
        self.active = True
        if d["index"] > self.snap_index:
            if d["index"] <= self.last_index and self.term_at(d["index"]) == d["sterm"]:
                self.log = self.log[d["index"] - self.snap_index:]
            else:
                self.log = []
            self.snap_index, self.snap_term = d["index"], d["sterm"]
            self.snap_state, self.snap_voters = d["state"], list(d["voters"])
            self.snap_size = m.size
            self._recompute_voters()
            if self.applied < self.snap_index:
                self.h.sm_restore(d["state"])
                self.applied = self.commit = self.snap_index
        self.h.send(m.src, Msg("AE_R", {"term": self.term, "ok": True, "match": self.snap_index}, 24))

    def _advance_commit(self):
        if self.role != self.LEADER:
            return
        ms = sorted((self.match.get(v, 0) for v in self.voters), reverse=True)
        n = ms[self.majority() - 1] if len(ms) >= self.majority() else 0
        if n > self.commit and self.term_at(n) == self.term:
            self.commit = n
            self._apply()

    def _apply(self):
        while self.applied < self.commit:
            self.applied += 1
            e = self.entry(self.applied)
            self.h.sm_apply(e)
            if self.role == self.LEADER:
                self.h.raft_committed(e)
        # snapshot / log compaction
        if self.applied - self.snap_index >= self.cfg.T_snap:
            st, size = self.h.sm_snapshot()
            self.snap_term = self.term_at(self.applied)
            v = self.snap_voters
            for e in self.log[: self.applied - self.snap_index]:
                if e.kind == "config":
                    v = e.data
            self.snap_voters = list(v)
            self.log = self.log[self.applied - self.snap_index:]
            self.snap_index = self.applied
            self.snap_state, self.snap_size = st, size

    # ------------------------------------------------------------ membership
    def config_pending(self):
        return any(e.kind == "config" for e in self.log[self.commit - self.snap_index:])

    def change_voters(self, new_voters):
        """Single-server change (caller ensures |diff| == 1)."""
        if self.role != self.LEADER or self.config_pending():
            return False
        self.propose(Entry(self.term, "config", list(new_voters), 16 + 8 * len(new_voters)))
        if self.h.id not in self.voters:
            pass
        return True
