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

## Docker cluster + `ray job submit`

A virtual cluster of **1 head + 3 workers**, each capped at **1 CPU** (4 logical
CPUs total). The CPU cap bounds the stress run and makes skew visible. The image
is a thin layer over the official `rayproject/ray:2.55.1-py313-cpu` (no official
py314 image exists; the cluster runs Python 3.13, behaviorally equivalent here).

```bash
# Build the image and start the cluster (from this folder)
docker compose -f docker/docker-compose.yml up -d --build

# Dashboard / job UI: http://localhost:8265

# Submit the three demo runs to the cluster (artifacts land in ./runs/<mode>)
docker/submit.sh blocking --max-ticks 20 --slow-zone-sleep-s 1.0
docker/submit.sh async    --max-ticks 20 --slow-zone-sleep-s 1.0
docker/submit.sh stress   --max-ticks 20

# Tear down
docker compose -f docker/docker-compose.yml down
```

`submit.sh` runs `ray job submit --working-dir /app` inside the head container;
`--prepared-dir` / `--output-dir` use absolute `/app/...` paths (the bind mount)
so artifacts persist to the host under `./runs/`.

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
