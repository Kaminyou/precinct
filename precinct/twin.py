"""Centralized twin for the live demo.

Runs the *same* city (same cameras, objects, seed, loss rate) with the
centralized baseline (one node fuses every detection) in a separate process,
advancing in step with the live simulation, and reports its measured load.
Injected faults are not mirrored: the twin only shows what one node would
have to do."""
from __future__ import annotations

from .config import Config
from .metrics import prune, window_metrics, window_payload
from .system import System


def load_window(sim, prev, node_filter):
    """CPU utilization and ingress over the simulated time since `prev`."""
    st = sim.stats
    now = sim.now
    busy = {n: v for n, v in st.busy.items() if node_filter(n)}
    nin = {n: v for n, v in st.node_in.items() if node_filter(n)}
    out = {}
    if prev is not None:
        dt = max(now - prev["t"], 1e-9)
        out["cpu"] = {n: (busy.get(n, 0.0) - prev["busy"].get(n, 0.0)) / dt for n in busy}
        out["in"] = {n: (nin.get(n, 0) - prev["in"].get(n, 0)) / dt for n in nin}
    return out, {"t": now, "busy": busy, "in": nin}


def _snapshot(s):
    """The twin's own ground truth and its published (centralized) view, for drawing."""
    from .tracker import gid_str
    snap = {"now": round(s.sim.now, 2), "gt": [], "view": [], "view_t": None}
    k_gt = max(s.rec.gt) if s.rec.gt else None
    if k_gt is not None:
        ids, pos, _ = s.rec.gt[k_gt]
        snap["gt"] = [[int(i), round(float(p[0]), 1), round(float(p[1]), 1)] for i, p in zip(ids, pos)]
        snap["gt_t"] = round((k_gt + 1) * s.cfg.dt, 2)
    if s.rec.views:
        k = max(s.rec.views)
        v = s.rec.views[k]
        snap["view"] = [[gid_str(g), round(float(p[0]), 1), round(float(p[1]), 1), 0, bool(st)]
                        for g, p, st in zip(v["gids"], v["pos"], v["stale"])]
        snap["view_t"] = round((k + 1) * s.cfg.dt, 2)
    center = next((c.id for c in s.cams if c.alive and c.role is not None), None)
    snap["center"] = center
    return snap


def twin_main(conn, cfg_dict):
    """Advance one step at a time towards the newest target time sent by the
    live session, replying after every step (snapshot) and every simulated
    second (load and accuracy statistics)."""
    cfg = Config(**cfg_dict)
    s = System(cfg)
    conn.send({"ready": True})
    prev = None
    next_stats = 0.0
    target, want_view = 0.0, False
    while True:
        if s.sim.now + cfg.dt > target + 1e-9:
            msg = conn.recv()                  # caught up: wait for the live session
            if msg is None:
                return
            target, s.cfg.p_loss, want_view = msg
        while conn.poll():                     # always act on the newest request
            msg = conn.recv()
            if msg is None:
                return
            target, s.cfg.p_loss, want_view = msg
        if s.sim.now + cfg.dt > target + 1e-9:
            continue
        s.sim.run_until(s.sim.now + cfg.dt)
        out = {"snap": _snapshot(s)} if want_view else {}
        if s.sim.now >= next_stats:
            next_stats = s.sim.now + 1.0
            load, prev = load_window(s.sim, prev, lambda n: n != "agg")
            res = {"t": round(s.sim.now, 1)}
            if load:
                res["cpu"] = max(load["cpu"].values(), default=0.0)
                res["in"] = max(load["in"].values(), default=0.0)
            if s.rec.views:
                k_now = max(s.rec.views)
                res["lag"] = s.sim.now - (k_now + 1) * cfg.dt    # how far the published view trails
                k_from = max(int(cfg.warmup / cfg.dt), k_now - 100)
                if k_now - k_from >= 5:
                    res.update(window_metrics(window_payload(s.rec, s.agg.aliases, k_from, k_now)))
                prune(s.rec, k_now - 1200)
            else:
                res["saturated"] = s.sim.now > cfg.warmup     # nothing published yet
            out["stats"] = res
        if out:
            conn.send(out)
