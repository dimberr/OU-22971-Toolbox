# Ray Capstone: Per-Zone Demand Recommendations

Tick-based replay of NYC TLC Green Taxi demand. One Ray **actor per zone** owns
that zone's state; stateless scoring **tasks** compare current demand to a
baseline and emit a `NEED` / `OK` recommendation. The project contrasts a
**blocking** controller (wait for all zones) with an **asynchronous** controller
(tasks report straight to actors; the driver closes ticks under a
partial-readiness policy), and shows how the async path tolerates stragglers.

Built and pinned for **Python 3.14 + Ray 2.55.1** (see `requirements.txt`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare the assets

Reference month (Nov) builds the baseline; replay month (Dec) is replayed:

```bash
python main.py prepare \
  --reference-parquet ../TLC_Data/green_tripdata_2025-11.parquet \
  --replay-parquet ../TLC_Data/green_tripdata_2025-12.parquet \
  --output-dir prepared \
  --n-zones 15
```

This writes `prepared/{baseline.parquet,replay.parquet,metadata.json}`.

## Run locally

```bash
# Blocking baseline (skew hurts: tick latency tracks the slowest zone)
python main.py run --prepared-dir prepared --output-dir runs/blocking \
  --mode blocking --max-ticks 20 --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0

# Async controller (driver closes ticks once completion-fraction reports)
python main.py run --prepared-dir prepared --output-dir runs/async \
  --mode async --max-ticks 20 --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.0 \
  --completion-fraction 0.75 --tick-timeout-s 2.0

# Stress (escalated skew; async progresses via fallbacks)
python main.py run --prepared-dir prepared --output-dir runs/stress \
  --mode stress --max-ticks 20
```

## Tests (fault-tolerance invariants)

```bash
python -m pytest test_invariants.py -v
```

Covers idempotent writes and rejection of duplicate, late, and inactive-tick
reports, plus the `previous_else_ok` fallback.

## Docker cluster + interactive console

A virtual cluster of **1 head (`--num-cpus 0`, dashboard/driver only) + 2 workers**
capped at **1 CPU** each. The CPU cap bounds the stress run and makes skew
visible; keeping the head at 0 CPU prevents actors from landing on it. The image
is a thin layer over
the official `rayproject/ray:2.55.1-py313-cpu` (no official py314 image exists;
the cluster runs Python 3.13, behaviorally equivalent here). The compose stack
also starts a small web console for launching runs and viewing the generated
metrics.

**Memory on a laptop.** All containers share one Docker Desktop VM (~7.75GB).
Measured with `docker stats`, the dominant cost is the **head: ~3.55GB** for the
`ray[default]` dashboard (required by the job-submission API) - that is fixed and
nearly half the VM. After the head, SonarQube/MCP containers (~0.5GB), the UI
(~0.15GB), and VM overhead (~0.8GB), only **~2.75GB is left for the workers**, or
roughly **1.4GB per worker** with 2 workers. Mitigations baked in:

- **Head runs no workload (`--num-cpus 0`)** so ZoneActors/ScoreHelpers never get
  scheduled onto the head on top of its dashboard. Without this, the driver on
  the head gets OOM-killed (exit 137) under load.
- **Per-container `memory` caps** (head 6g, workers 2g, ui 0.5g). These are hard
  cgroup limits; Ray reads each node's own limit instead of the whole VM. NOTE:
  caps are limits, not reservations - if the sum of *actual* peaks exceeds the VM
  the kernel still OOM-kills a victim (often the driver). So the real constraint
  is keeping actual peak under ~7GB, not the cap values.
- **`--object-store-memory 256000000`** (256MB/node) - we use almost none.
- **`RAY_memory_monitor_refresh_ms=0`** disables the threshold killer so a
  transient spike doesn't kill our driver/actors mid-run.

**Workload must fit the worker envelope.** The full stress demo (`--mode stress`
forces ~8 slow zones, each promoting a ScoreHelper plus many sleeping tasks that
spawn Python workers) needs more than ~1.4GB/worker and OOMs on this VM. A run
that completes cleanly (verified end-to-end, ~6.9GB peak VM, no OOM) keeps the
slow-zone count and concurrency bounded:

```bash
RUN_LABEL=fit1 docker/submit.sh async \
  --slow-zone-fraction 0.2 --slow-zone-sleep-s 3.0 --max-inflight-zones 3 \
  --use-subactors --subactor-trigger 1 --n-helpers 1 --max-ticks 96
```

To run the *full* stress demo with subactors without OOM, give the Docker Desktop
VM more RAM (Settings -> Resources -> Memory -> ~12GB); that frees the worker
envelope and lets you raise `--slow-zone-fraction`/`--n-helpers` or add workers.

```bash
# Build the image and start the cluster (from this folder)
docker compose -f docker/docker-compose.yml up -d --build

# Capstone console: http://localhost:8080
# Ray dashboard:     http://localhost:8265

# Tear down
docker compose -f docker/docker-compose.yml down
```

The console submits Ray jobs through the head's job API and writes artifacts to
`./runs/<label>` through the shared `/app` bind mount. It includes buttons for
the required use cases, a custom parameter form, job status, run summaries,
latency charts, per-zone heatmaps, and an optional animated TLC zone map.

For map animation, download these files from the TLC "Taxi Zone Maps and Lookup
Tables" section into the sibling `../TLC_Data/` folder:

- `Taxi Zone Lookup Table (CSV)` for borough and zone names.
- `Taxi Zone Shapefile (PARQUET)` for polygons. GeoJSON also works if you
  already converted it.

`submit.sh` is still available for terminal-only workflows, but it is no longer
needed for the interactive demo.

## Decision rule

For each `(zone, tick)`: `current_pickups / baseline_pickups >= need-threshold`
(default `1.1`, i.e. demand at least 10% above the normal profile for that
zone's `(hour_of_day, day_of_week)` slot) emits `NEED`, otherwise `OK`. A
non-positive baseline is guarded and yields `OK`.

## Partial-readiness policy (async)

Per tick the driver submits scoring tasks with bounded concurrency
(`--max-inflight-zones`), then polls actor readiness. It closes the tick as soon
as `--completion-fraction` of zones have reported, or when `--tick-timeout-s`
elapses, whichever comes first. Zones that have not reported are finalized with
the fallback policy `previous_else_ok`: reuse the zone's previous accepted
decision, or `OK` on first use. Late/duplicate/inactive reports after close are
rejected and counted (`late_reports`, `duplicate_reports`).

## Demo runbook (every required case)

Run from this folder. Steps 0-1 are one-time/local; steps 2+ can be launched
from `http://localhost:8080`.

```bash
# 0. One-time setup + prepare assets (local)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python main.py prepare \
  --reference-parquet ../TLC_Data/green_tripdata_2025-11.parquet \
  --replay-parquet ../TLC_Data/green_tripdata_2025-12.parquet \
  --output-dir prepared --n-zones 15

# 1. Fault-tolerance invariants (idempotency, late/duplicate/inactive, fallback,
#    delayed arrivals, subactor promotion) - 10 tests
python -m pytest test_invariants.py -v

# 2. Bring up the Docker cluster + web console
docker compose -f docker/docker-compose.yml up -d --build
#    Capstone console: http://localhost:8080
#    Ray dashboard:     http://localhost:8265
```

The console has preset buttons for blocking, async, stress, blocking harsh skew,
delayed arrivals, and subactors. The custom form exposes the same run parameters
as `main.py`, plus a replay window picker (`start day`, `start hour`, and
`window hours`) that maps to `--start-tick` and `--max-ticks`.

Each run writes `runs/<label>/{run_config,metrics,latency_log,tick_summary}.json`.
Compare `metrics.csv` (tick latency, max/mean ratio, fallbacks) across runs.

## Stretch A: delayed arrivals

Models delayed *information* (not delayed task completion). Each tick a zone
withholds `floor(true_pickups * --withhold-fraction)` of its demand; that hidden
amount resurfaces, **added on top of**, the snapshot `--arrival-delay-ticks`
later. Committed ticks are never rewritten (idempotency), so the late demand can
only color the release tick onward.

- `--withhold-fraction` - fraction hidden each tick (`0` disables the feature)
- `--arrival-delay-ticks` - base delay `D` before withheld demand resurfaces
- `--delay-spread` - `0` = same delay for all zones (system-wide); `>0` = seeded
  per-zone jitter in `[D, D+spread]`, reproducible for a given `--seed`

```bash
python main.py run --prepared-dir prepared --output-dir runs/delay \
  --mode blocking --max-ticks 96 --withhold-fraction 0.5 --arrival-delay-ticks 3
```

Effect: a genuinely busy tick can be mislabeled `OK` (its demand is partly
hidden), while a later quiet tick is mislabeled `NEED` (stale demand resurfaces)
- the scoring logic cannot distinguish resurfaced-old from real-new demand.
Totals (`withheld_total`, `released_total`, `unreleased_total`) are written to
`tick_summary.json`; demand withheld within the last `D` ticks never resurfaces
and is counted as `unreleased`. See `test_invariants.py` for a decision-flip
demonstration.

## Stretch B: adaptive load balancing via zone subactors

A zone that is a **repeat straggler** (misses `--subactor-trigger` ticks in a row
via fallback) promotes itself: the `ZoneActor` spawns `--n-helpers` helper
subactors (`ScoreHelper`) once and exposes their handles. From then on, that
zone's scoring task fans the tick's demand out as integer shards to the helpers
in parallel (each sleeps `sleep_s / n_helpers`), sums the partials, and reports.

Key properties:
- **Decision semantics preserved**: `sum(shards) == current_pickups`, so the
  sharded result equals a single scoring of the total (`split_int` guarantees it).
- **Actors stay responsive**: the parallel fan-out runs in the *task*, never
  inside an actor method, so `has_report()` polling is never blocked.
- **Detection reuses fallbacks**: a real report resets the straggler streak.

```bash
python main.py run --prepared-dir prepared --output-dir runs/subactors \
  --mode async --max-ticks 12 \
  --slow-zone-fraction 0.25 --slow-zone-sleep-s 1.5 \
  --tick-timeout-s 1.0 --completion-fraction 1.0 \
  --use-subactors --subactor-trigger 3 --n-helpers 3
```

Observed: per-tick fallbacks `[4,4,4,3,3,0,0,...]` and tick latency falling from
~1.5s to ~0.67s as slow zones promote and recover. `tick_summary.json` reports
`promoted_zones` and `subactor_ticks`. Note the quorum matters: if
`--completion-fraction` is low enough to close on the fast zones alone, promoted
zones never get a chance to report - set it high enough to require them.

## Artifacts (per run, in `--output-dir`)

- `run_config.json` - exact config + selected slow zones
- `metrics.csv` - per-tick latency, mean/max zone latency, max/mean ratio, completed vs fallback, NEED/OK
- `latency_log.json` - per-`(tick, zone)` task latency (`null` for fallbacks)
- `tick_summary.json` - per-tick rows + run totals (fallbacks, late, duplicate,
  withheld/released/unreleased, promoted_zones, subactor_ticks)
