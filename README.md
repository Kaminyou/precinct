# Precinct — Scalable Distributed Real-Time City-Wide Multi-Camera Tracking

Implementation of the Precinct: cameras are grouped into regions;
each region's Raft leader (the *center*) fuses its cameras' detections, assigns
global IDs, replicates its track table with Raft, and hands tracks off to
neighbouring regions with an epoch protocol; a global aggregator merges regional
reports, removes duplicates, re-identifies objects across regions, and rebuilds
itself from the centers after a crash. Everything runs in a deterministic
discrete-event simulator (Python), with a live / replay web viewer.

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`, `aiohttp` (viewer), `pytest` (tests).
The Anaconda base environment already has all of them.

## Quick start

```bash
# one simulation (prints accuracy, latency breakdown, overhead, events)
python -m precinct --M 300 --duration 30
python -m precinct --M 1000 --centralized              # baseline
python -m precinct --M 300 --set p_loss=0.1 --set S=10   # any Config field

# live demo: open http://localhost:8765
python -m precinct.viz.server                    # M=100, real time
python -m precinct.viz.server --scenario demo    # scripted: center crash @20s, aggregator crash @45s, 10% loss @60-75s
python -m precinct.viz.server --replay results/demo_trace.jsonl
python -m precinct.viz.server --objects 300        # denser traffic (mean number of objects)
python -m precinct.viz.server --L 1500              # bigger city (side length in m; default 1000*sqrt(M/100))
python -m precinct.viz.server --set r=60            # any other Config field

# record a replayable trace
python -m precinct --M 100 --duration 90 --scenario demo --trace demo.jsonl

# tests
python -m pytest -q tests
```

### Demo in the browser

* **Click a ★ (a region's center)** to crash it. Members time out, the remaining
  Raft voters elect a new center within ~1 s, and the tracks keep their IDs
  (watch the labels, or click a track to see its ID history).
* **Click a track** to follow its ID across handoffs, failovers, and gaps.
* **Scroll** to zoom around the cursor, **drag** to pan, **double-click** (or *Fit city*) to see the whole city again.
* **Click a camera** to crash, recover, or retire it. **Shift-click empty ground** to add one.
* **Kill aggregator** freezes the global view; regions keep tracking, and on recovery the
  view is rebuilt from center snapshots in ~20 ms.
* The **Compute load** panel compares, every second, the measured CPU and ingress of the
  busiest distributed center with a *centralized twin*: the same city (cameras, objects, seed,
  loss) fused by one node, running in a second process. Per-region center CPU and the
  aggregator are shown as bars. To see the centralized node saturate live:
  `python -m precinct.viz.server --M 300 --objects 900` (centralized ≈ 96% CPU and latency
  growing past 1.5 s, vs ≈ 11% / 130 ms distributed; the demo then plays ~1.5× slower than real
  time, but all numbers are in simulated time).
* The **Distributed / Centralized** switch in the header chooses which system's published view
  is drawn. In *Centralized* mode the map shows the twin's own vehicles (gray) and the single
  node's published tracks (colored); when that node saturates, the tracks visibly trail the
  vehicles and the banner shows by how many seconds. Faults are only injected into the
  distributed system.
* The **loss** slider changes message loss; the timeline at the bottom marks faults
  and lets you scrub back.

## Experiments

```bash
python experiments/run.py --list                 # 184 runs
python experiments/run.py --exp all --workers 20 # writes results/<exp>/<run>.json (resumable)
python experiments/plot.py                       # report/figures/*.pdf + tables.tex
```

Experiments: `scale` (M = 100…10,000, distributed vs centralized), `region` (S),
`deadline` (W, W_a), `network` (loss, clock skew), `noise` (look-alikes, position
noise), `crash` (center crash with/without replication, member crash rates),
`agg` (aggregator restart), `churn` (joins/leaves), `ablation` (handoff, dedup,
global re-id).

Note: computation time is *measured* and charged to the simulated node
(`cost_model="measured"`, scale `kappa`), so latency/CPU numbers depend on the
host and on how many runs execute in parallel. Use `--set cost_model='"none"'` for
fully deterministic runs.

## Code map

| File | Contents (report section) |
|---|---|
| `precinct/config.py` | all parameters (Table 2) |
| `precinct/sim.py` | event engine, clocks, lossy network, compute charging (§4) |
| `precinct/world.py` | random-waypoint objects, cameras, detection & appearance model (§3.1) |
| `precinct/regions.py` | balanced k-means, voters, territory / border queries (§3.3) |
| `precinct/tracker.py` | Kalman, association, track management, handoff & merge state machine, replicated track table (§3.5–3.6) |
| `precinct/raft.py` | Raft: election, replication, snapshots, membership change, stickiness, check-quorum (§3.5, §3.8) |
| `precinct/camera.py` | camera node: member, voter, and center roles; clock sync (§3.3–3.8) |
| `precinct/aggregator.py` | aggregator view, dedup, global re-id, merges, recovery; directory (§3.7–3.8) |
| `precinct/metrics.py` | MOTA / IDF1 / ID switches / latency breakdown / overhead / failover continuity (§5) |
| `precinct/system.py` | builds a deployment, drives frames, injects faults & churn, traces |
| `precinct/viz/` | aiohttp server + vanilla-JS canvas viewer (§6) |
| `experiments/` | sweep driver and plotting |
| `tests/` | unit and end-to-end tests |
