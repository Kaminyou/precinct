"""Live / replay web viewer.

    python -m precinct.viz.server                      # live simulation (M=100)
    python -m precinct.viz.server --M 200 --speed 2
    python -m precinct.viz.server --L 1500 --set r=60   # bigger city / wider camera views
    python -m precinct.viz.server --replay trace.jsonl # replay a recorded trace
    python -m precinct.viz.server --scenario demo      # scripted demo faults

Open http://localhost:8765 . In live mode the simulation is paced to wall
clock; the browser can inject faults (kill a center, a camera, the
aggregator, raise message loss, add cameras) while it runs."""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

from aiohttp import WSMsgType, web

from ..config import Config
from ..metrics import prune, window_metrics, window_payload
from ..system import System
from ..twin import load_window, twin_main

STATIC = os.path.join(os.path.dirname(__file__), "static")

DEMO_FAULTS = [
    (20.0, "crash_center", 1, 15.0),     # kill region 1's center, recover the camera 15 s later
    (45.0, "crash_agg", 5.0),            # aggregator down for 5 s
    (60.0, "loss", 0.10),                # 10 % message loss ...
    (75.0, "loss", 0.01),                # ... back to normal
]


class Client:
    """One browser. Frames are coalesced: if the browser is slower than the
    simulation, it gets the newest frame (with the events of the frames it
    skipped) instead of an ever-growing backlog."""

    def __init__(self, ws):
        self.ws = ws
        self.msgs = []           # non-frame messages, in order
        self.frame = None        # newest unsent frame
        self.events = []         # events of skipped frames
        self.cams_new = []
        self.wake = asyncio.Event()

    def push_frame(self, fr):
        if self.frame is not None:
            self.events.extend(self.frame["events"])
            self.cams_new.extend(self.frame.get("cams_new", []))
        self.frame = fr
        self.wake.set()

    def push_text(self, text):
        self.msgs.append(text)
        self.wake.set()

    async def run(self):
        while not self.ws.closed:
            await self.wake.wait()
            self.wake.clear()
            msgs, self.msgs = self.msgs, []
            fr, self.frame = self.frame, None
            for t in msgs:
                await self.ws.send_str(t)
            if fr is not None:
                if self.events or self.cams_new:
                    fr = dict(fr, events=self.events[-200:] + fr["events"],
                              cams_new=self.cams_new + fr.get("cams_new", []))
                    self.events, self.cams_new = [], []
                await self.ws.send_str(json.dumps(fr))


class LiveSession:
    def __init__(self, cfg: Config, speed: float):
        self.cfg = cfg
        self.speed = speed
        self.paused = False
        self.clients = set()
        self.frames = []
        self.metrics = None
        # live accuracy is computed in a separate process so it never stalls the simulation
        self.pool = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
        self._metrics_busy = False
        self.behind = 0
        self.want_twin_view = False
        self.want_regions = False
        self._new_system()

    def _new_system(self):
        self._start_twin()
        self._prev_load = None
        self.sys = System(self.cfg)
        self.sys.cfg.trace_centers = getattr(self, "want_regions", False)
        self.sys.frame_listeners.append(self._on_frame)
        self.header = self.sys.header()
        self.frames = []

    def _on_frame(self, fr):
        if self.want_twin_view and self.twin_snap is not None:
            fr["twin"] = self.twin_snap
        self.frames.append(fr)
        if len(self.frames) > 1200:
            del self.frames[:-600]
        for cl in self._live_clients():
            cl.push_frame(fr)

    def _live_clients(self):
        for cl in list(self.clients):
            if cl.ws.closed:
                self.clients.discard(cl)
            else:
                yield cl

    def _broadcast(self, text):
        for cl in self._live_clients():
            cl.push_text(text)

    def command(self, c):
        s = self.sys
        kind = c.get("cmd")
        if kind == "pause":
            self.paused = True
        elif kind == "resume":
            self.paused = False
        elif kind == "speed":
            self.speed = max(0.1, min(20.0, float(c["v"])))
        elif kind == "twin_view":
            self.want_twin_view = bool(c.get("on"))
        elif kind == "regions":
            # per-region track lists are large: only send them while a viewer shows them
            self.want_regions = bool(c.get("on"))
            self.sys.cfg.trace_centers = self.want_regions
        elif kind == "reset":
            self._new_system()
            self._broadcast(json.dumps(self.header))
        elif kind == "crash_center":
            s.inject("crash_center", int(c["region"]))
        elif kind == "crash":
            s.inject("crash", int(c["node"]))
        elif kind == "recover":
            s.inject("recover", int(c["node"]))
        elif kind == "crash_agg":
            s.inject("crash_agg")
        elif kind == "recover_agg":
            s.inject("recover_agg")
        elif kind == "loss":
            s.inject("loss", float(c["p"]))
        elif kind == "join":
            s.inject("join", (float(c["x"]), float(c["y"])))
        elif kind == "leave":
            s.inject("leave", int(c["node"]))
        self._broadcast(json.dumps({"type": "status", "paused": self.paused, "speed": self.speed,
                                    "loss": self.cfg.p_loss}))

    def _start_twin(self):
        """(Re)start the centralized twin of this city in its own process."""
        old = getattr(self, "twin", None)
        if old is not None:
            try:
                self.twin_conn.send(None)
            except Exception:
                pass
            old.join(timeout=1.0)
            if old.is_alive():
                old.terminate()
        self.baseline = None
        self.twin_snap = None
        self.twin_ready = False
        self.twin_wait_start = time.perf_counter()
        d = self.cfg.to_dict()
        d.update(centralized=True, faults=[], trace=None)
        ctx = mp.get_context("spawn")
        self.twin_conn, child = ctx.Pipe()
        self.twin = ctx.Process(target=twin_main, args=(child, d), daemon=True)
        self.twin.start()

    def _poll_twin(self):
        """Collect the twin's replies and ask it to advance to our current time."""
        try:
            while self.twin_conn.poll():
                r = self.twin_conn.recv()
                if r.get("ready"):
                    self.twin_ready = True
                if "stats" in r:
                    self.baseline = r["stats"]
                if "snap" in r:
                    self.twin_snap = r["snap"]
            self.twin_conn.send((self.sys.sim.now, self.cfg.p_loss, self.want_twin_view))
        except (EOFError, BrokenPipeError, OSError):
            pass

    def _load(self):
        """Measured CPU and ingress of every center and the aggregator since the last call."""
        s = self.sys
        load, self._prev_load = load_window(s.sim, self._prev_load, lambda n: True)
        if not load:
            return None
        centers = {c.id: c.region for c in s.cams if c.alive and c.role is not None}
        reg_cpu = {str(r): round(load["cpu"].get(n, 0.0), 4) for n, r in centers.items()}
        cen_in = [load["in"].get(n, 0.0) for n in centers]
        return {"centers": reg_cpu,
                "busiest": max(reg_cpu.values(), default=0.0),
                "agg": load["cpu"].get("agg", 0.0),
                "center_in_max": max(cen_in, default=0.0),
                "agg_in": load["in"].get("agg", 0.0)}

    KEEP_STEPS = 1200          # live mode keeps the last 2 minutes of history

    def _submit_metrics(self):
        """Score the last 10 s in the worker process; broadcast when done."""
        s = self.sys
        if self._metrics_busy or not s.rec.views:
            return
        k_now = max(s.rec.views)
        k_from = max(int(self.cfg.warmup / self.cfg.dt), k_now - 100)
        if k_now - k_from < 5:
            return
        payload = window_payload(s.rec, s.agg.aliases, k_from, k_now)
        n_obj = len(s.rec.gt.get(k_now, ([], None, None))[0])
        n_trk = len(s.rec.views[k_now]["gids"])
        t_sim = round(s.sim.now, 1)
        sys_id = id(s)
        load = self._load()
        self._metrics_busy = True
        try:
            fut = asyncio.get_running_loop().run_in_executor(self.pool, window_metrics, payload)
        except Exception:              # worker died: start a fresh one next time
            self.pool = ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"))
            self._metrics_busy = False
            return

        def done(f):
            self._metrics_busy = False
            if f.exception() is not None or id(self.sys) != sys_id:
                return
            m = dict(f.result(), type="metrics", t=t_sim, objects=n_obj, tracks=n_trk,
                     load=load, baseline=self.baseline)
            self.metrics = m
            self._broadcast(json.dumps(m))
        fut.add_done_callback(done)
        prune(s.rec, k_now - self.KEEP_STEPS)

    async def loop(self):
        dt = self.cfg.dt
        next_metrics = 0.0
        while True:
            if self.paused:
                await asyncio.sleep(0.05)
                continue
            if not self.twin_ready and time.perf_counter() - self.twin_wait_start < 60:
                self._poll_twin()               # start together with the centralized twin
                await asyncio.sleep(0.1)
                continue
            t0 = time.perf_counter()
            self.sys.sim.run_until(self.sys.sim.now + dt)
            self._poll_twin()
            if time.perf_counter() > next_metrics:
                next_metrics = time.perf_counter() + 1.0
                self._submit_metrics()
            el = time.perf_counter() - t0
            if el > dt / self.speed:
                self.behind += 1     # the simulation cannot keep up with this speed
            await asyncio.sleep(max(0.0, dt / self.speed - el))


def make_app(session=None, replay=None):
    app = web.Application()

    async def index(_):
        return web.FileResponse(os.path.join(STATIC, "index.html"))

    async def trace(_):
        if replay is None:
            raise web.HTTPNotFound()
        return web.FileResponse(replay)

    async def mode(_):
        return web.json_response({"mode": "replay" if replay else "live"})

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=0)
        await ws.prepare(request)
        if session is None:
            await ws.close()
            return ws
        await ws.send_str(json.dumps(session.header))
        # catch the new client up with the recent past (for scrubbing), without heavy fields
        for fr in session.frames[-150:]:
            await ws.send_str(json.dumps({k: v for k, v in fr.items() if k not in ("regions", "twin")}))
        await ws.send_str(json.dumps({"type": "status", "paused": session.paused,
                                      "speed": session.speed, "loss": session.cfg.p_loss}))
        cl = Client(ws)
        session.clients.add(cl)
        sender = asyncio.create_task(cl.run())
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    session.command(json.loads(msg.data))
                except Exception as e:  # keep the demo alive on bad input
                    await ws.send_str(json.dumps({"type": "error", "msg": str(e)}))
        session.clients.discard(cl)
        sender.cancel()
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/mode", mode)
    app.router.add_get("/trace", trace)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", STATIC)
    if session is not None:
        async def start_bg(app):
            app["loop_task"] = asyncio.create_task(session.loop())
        app.on_startup.append(start_bg)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=100)
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objects", type=float, default=None, help="mean number of objects (default: = M)")
    ap.add_argument("--L", type=float, default=None, help="city side length in m (default: 1000*sqrt(M/100))")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--scenario", default=None, choices=[None, "demo"])
    ap.add_argument("--set", action="append", default=[], help="key=value config override, e.g. r=80")
    a = ap.parse_args()
    if a.replay:
        app = make_app(None, a.replay)
    else:
        faults = DEMO_FAULTS if a.scenario == "demo" else []
        kw = {}
        for kv in a.set:
            k, v = kv.split("=", 1)
            kw[k] = json.loads(v)
        cfg = Config(M=a.M, S=a.S, seed=a.seed, duration=1e9, faults=faults, N_bar=a.objects, L=a.L, **kw)
        print(f"M={cfg.M} cameras, L={cfg.L:.0f} m, ~{cfg.N_bar:.0f} objects, camera radius r={cfg.r:.0f} m")
        app = make_app(LiveSession(cfg, a.speed))
    print(f"viewer on http://localhost:{a.port}")
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
