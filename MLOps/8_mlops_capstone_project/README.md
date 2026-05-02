# MLOps Capstone — NYC Green Taxi tip prediction

A small but complete monitoring + retraining + promotion loop for a tip-prediction
model on NYC Green Taxi trip records. The full pipeline runs from a Streamlit UI
(or from the CLI), with all decisions and artifacts logged to MLflow.

Built with **Metaflow** (workflow orchestration), **MLflow** (tracking + registry),
**NannyML** (drift / data integrity), **scikit-learn** (model), and **Streamlit** (UI).

---

## What it does

Each time a new month of taxi data lands in `TLC_Data/`, you can:

- **Bootstrap** the very first champion model from the reference month (one click).
- **Run flow** on a new batch — runs the full end-to-end pipeline:
  1. Hard integrity gate (schema, ranges, datetimes, zone validity)
  2. Soft integrity gate (NannyML: missingness + unseen-value drift)
  3. Feature engineering (filter to credit-card trips; calendar features; clip + log1p heavy-tail columns)
  4. Load `@champion` from the MLflow registry
  5. **Model gate** — evaluate champion on the new batch's eval slice with NannyML's PerformanceCalculator → `retrain_needed=true/false`
  6. **Conditional retrain** — train a candidate on the rolling N-month window (eval slice held out)
  7. **Promotion gate** — four criteria (P1 valid, P2 beats champion, P3 no reference regression, P4 integrity sanity)
  8. Move `@champion` alias if all pass; tag previous champion as `previous_champion`
- **Score** runs offline batch inference with the active `@champion` and logs `predictions.parquet`.

Every step writes a structured `decision.json` artifact to MLflow so reviewers can
see exactly which rule fired and what the values were.

---

## Quick start

### Prerequisites

- Docker Desktop (or any Docker engine) with Docker Compose v2.
- About 2 GB free disk for one image + a handful of monthly parquet files.

### 1. Clone and enter the project

```bash
git clone <this repo>
cd MLOps/8_mlops_capstone_project
```

### 2. Download the data

NYC TLC publishes monthly Green Taxi trip records as parquet files. Drop them in
`TLC_Data/` next to the auxiliary zone file. Filenames must match
`green_tripdata_YYYY-MM.parquet`.

```bash
mkdir -p TLC_Data
cd TLC_Data

# A reference month (used to fit the FeatureSpec) — required.
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-01.parquet

# A few extra months to play with (pick any).
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-04.parquet
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-08.parquet
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-04.parquet
curl -O https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-08.parquet

# Optional: NYC taxi zone lookup (used by the hard integrity gate to validate
# PULocationID / DOLocationID values). Without it that check is skipped.
curl -L -o NYC_Taxi_Zones.geojson "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"

cd ..
```

Source pages:

- [NYC TLC trip record data index](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- [NYC Taxi Zones (GeoJSON)](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc)

### 3. Start the stack

```bash
docker compose up -d --build
```

This brings up two containers, both bound to `127.0.0.1` only (no LAN exposure):

| Service | URL | Purpose |
|---|---|---|
| `mlflow` | http://localhost:5000 | MLflow tracking + model registry UI |
| `ui` | http://localhost:8501 | Streamlit control panel for the flow |

State that survives container restarts is mounted from the host:

| Host path | Container path | Contents |
|---|---|---|
| `./TLC_Data` | `/data` | Input parquet files + zone lookup (read-only in practice) |
| `./mlflow_tracking` | `/mlflow` | SQLite backing DB + artifact store |
| `./ui_state` | `/app/ui_state` | Run history, logs, predictions, result sidecars for the UI |
| `./metaflow_state` | `/app/.metaflow` | Metaflow's per-run checkpoint store |

### 4. First-time use

1. Open the Streamlit UI at http://localhost:8501.
2. The "Champion model" section will show **No champion model exists yet**. Click
   **Bootstrap champion from reference**.
3. After ~10 s the UI flips to **Active champion: green_taxi_tip_model v1**.
4. Pick any non-reference parquet in the **Available batches** table and click
   **Run flow**. The live log streams in below; on completion the row's "retrain?"
   and "outcome" columns reflect the gate decision.
5. Open the MLflow UI at http://localhost:5000 to see the run, its metrics, tags,
   and `decision.json` artifacts. Experiment name: `green_taxi_tip_model`.

---

## The three demo scenarios

The capstone rubric requires three flow runs in the demo video. The UI plus the
parquet files above is enough to reproduce all three from a clean state.

### Scenario 1 — baseline (no action)

**Story:** champion is healthy on a recent batch; the gate correctly says
"don't retrain".

```text
Bootstrap champion from /data/green_tripdata_2020-01.parquet
Run flow on   /data/green_tripdata_2020-04.parquet
```

Expected outcome in MLflow + UI:

- `retrain_recommended = false`
- `model_gate/decision.json`: `alert_chunks=0/11`, `retrain_needed=false`
- UI history row: `retrain? = no`, `outcome = no_retrain`

### Scenario 2 — retrain + promotion (automatic within the run)

**Story:** champion was trained on early-2020 data and is being shown 2025
traffic. The gate detects degradation, retrain runs, candidate beats champion,
and the alias flips.

```text
(start from the baseline state above, with v1 champion bootstrapped)
In the UI sidebar, expand "Advanced parameters" and set:
    max_ref_regression_pct = 0.30
Then Run flow on /data/green_tripdata_2025-08.parquet
```

Why the slider matters: with the strict default `max_ref_regression_pct = 0.01`,
the candidate trained on the rolling 12 months ending in 2025-08 would be
correctly rejected because it regresses on the 2020-01 reference by ~21%
(catastrophic forgetting). This is honest behaviour but doesn't show a
promotion. Setting the budget to 0.30 expresses the policy "we accept the
2020 reference is stale" and lets the candidate through.

Expected outcome in MLflow + UI:

- New model version v2 registered
- Tags on v2: `role=champion`, `promotion_reason=...`, `decision_reason=...`
- Tags on v1: `role=previous_champion`, `decision_reason=demoted_for_v2`
- `@champion` alias now on v2
- `promote/decision.json`: all four P-criteria pass
- UI history row: `retrain? = yes`, `outcome = promoted`

### Scenario 3 — failure + resume (workflow robustness)

**Story:** an exception is intentionally raised in the `retrain` step. Metaflow
records the failure, you fix the bug, then `resume` picks up from the failed
step without re-running the earlier ones.

Easiest way to inject the failure: temporarily add `raise RuntimeError("demo")`
at the top of the `retrain` step in `flow_starter.py`, run the flow on a batch
that triggers retrain (e.g. 2025-08 with the wider reference budget). The flow
will fail in `retrain`. Then remove the line and:

```bash
docker compose exec ui python flow_starter.py resume
```

Metaflow restarts from the failed step; earlier steps (`load_data`,
`hard_integrity_gate`, `soft_integrity_gate`, `feature_engineering`,
`load_champion`, `model_gate`) are not re-executed because their `self.*`
artifacts are already persisted.

The CLI works inside the running `ui` container; if you prefer running it
directly on the host, `pip install -r requirements.txt` into a venv and point
`MLFLOW_TRACKING_URI=http://localhost:5000` first.

---

## Where to look in MLflow for grading evidence

Open http://localhost:5000.

| Rubric item | Where it lives |
|---|---|
| Champion eval metrics (`rmse_champion`, `rmse_increase_pct`) | Run → "Metrics" tab |
| `decision.json` for each gate | Run → "Artifacts" tab → `hard_integrity/`, `soft_integrity/`, `model_gate/`, `promote/` |
| NannyML soft-gate report (HTML) | Run → "Artifacts" → `soft_integrity/` |
| Tags reflecting decisions (`retrain_recommended`, `promoted`, etc.) | Run → "Tags" panel |
| New model version (Scenario 2) | "Models" → `green_taxi_tip_model` → Versions list |
| `@champion` alias movement | "Models" → `green_taxi_tip_model` → "Aliases" column |
| `previous_champion` tag on demoted v1 | "Models" → `green_taxi_tip_model` → click v1 → Tags |
| Inference predictions artifact | Run with `pipeline=capstone_inference` tag → Artifacts → `predictions.parquet` |

All flow runs use experiment name **`green_taxi_tip_model`**.

---

## Using the CLI directly (no UI)

The UI is a thin wrapper over three Python entry points. Anything you can do in
the UI you can also do from a shell inside (or instead of) the container.

```bash
# Inside the ui container:
docker compose exec ui bash

# 1. Bootstrap the very first champion (idempotent — no-op if one exists).
python bootstrap.py --reference-path /data/green_tripdata_2020-01.parquet

# 2. Run the full flow on a batch.
python flow_starter.py run \
    --reference-path /data/green_tripdata_2020-01.parquet \
    --batch-path     /data/green_tripdata_2025-08.parquet \
    --historical-dir /data \
    --model-name     green_taxi_tip_model \
    --rolling-window-months 12 \
    --batch-eval-pct        0.2 \
    --min-improvement-pct   0.01 \
    --max-ref-regression-pct 0.30 \
    --taxi-zone-lookup-path /data/NYC_Taxi_Zones.geojson

# 3. Score a batch with @champion and log predictions.parquet.
python score_batch.py \
    --batch-path /data/green_tripdata_2025-08.parquet \
    --model-name green_taxi_tip_model

# 4. Resume a failed flow from the failed step.
python flow_starter.py resume
```

To run the same commands on the host instead of in the container:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://localhost:5000
# ...then any of the python ... commands above, with paths
# adjusted from /data/... to ./TLC_Data/...
```

---

## Project layout

```
8_mlops_capstone_project/
├── flow_starter.py            # Metaflow flow: integrity → gate → retrain → promote
├── bootstrap.py               # One-shot: train + register the very first @champion
├── score_batch.py             # Offline batch inference → predictions.parquet
├── lib/
│   ├── features.py            # FeatureSpec + engineer_features (clip+log1p, calendar feats)
│   ├── green_taxi_schema.py   # Column names, dtypes, valid ranges
│   ├── helper.py              # init_mlflow, flow_run context manager, parquet loaders
│   ├── integrity/
│   │   ├── hard.py            # Schema / range / datetime / zone-validity rules
│   │   └── soft.py            # NannyML missingness + unseen-value drift
│   ├── model_registry.py      # Champion load/bootstrap/promote, @champion alias, FeatureSpec fetch
│   ├── model_gate.py          # NannyML PerformanceCalculator → retrain decision
│   ├── promotion.py           # P1..P4 promotion criteria
│   ├── retrain.py             # Rolling-window training set, candidate train + eval
│   └── mlflow_log.py          # All MLflow logging helpers (metrics, tags, decision.json)
├── ui/
│   ├── app.py                 # Streamlit entry point
│   ├── runner.py              # Subprocess wrapper for flow / bootstrap / score
│   └── state.py               # On-disk run history + current-run lock
├── Dockerfile                 # python:3.12-slim + requirements.txt
├── docker-compose.yml         # mlflow + ui services, localhost-only ports
├── requirements.txt           # Pinned dependencies
└── TLC_Data/                  # Mounted into containers as /data (you populate this)
```

---

## Design notes worth knowing

### Champion vs candidate

`@champion` is just an MLflow alias — the model currently bound to the
production endpoint. It is a *deployment* status, not a *quality* grade.

- The very first model becomes champion automatically via `bootstrap.py` (no
  competition; it's the only model in the registry).
- Every subsequent training run produces a **candidate** that must pass all four
  P-criteria in `lib/promotion.py` before the alias moves.
- A demoted version is tagged `role=previous_champion` for auditability.

### `rmse_baseline` vs `rmse_champion`

Both are RMSEs of the *same model*, on *different datasets*:

- **`rmse_baseline`** — RMSE the champion got on the reference data at
  *training time*, stamped on the model version.
- **`rmse_champion`** — RMSE the champion gets on *this* run's eval slice,
  recomputed every flow run. Read together with `rmse_increase_pct`, it tells
  you how much performance has drifted since training.

### Why retrain is decided per-chunk, not per-average

The model gate uses NannyML's PerformanceCalculator with a 3-sigma threshold,
which alerts only when an *individual chunk* of the batch crosses the
upper bound derived from the reference RMSE distribution. A small steady
increase in average RMSE will not trip the gate by design — this is
NannyML's "robust to noise" philosophy. If you want the gate to also catch
slow gradual drift, add an aggregate-RMSE rule on top in `lib/model_gate.py`.

### Self-contained inference

`score_batch.py` does not need the original reference parquet. It loads the
`FeatureSpec` (clip bounds + feature column order) directly from the same
MLflow run that trained the champion, via `feature_spec.json`. This means
the inference pipeline can never silently use a different feature spec than
the model was trained on, even if the reference data on disk drifts.

### Concurrency

Only one job runs at a time. The lock lives on disk
(`ui_state/current.json`); both the UI buttons and the Cancel control read /
write it. The flow subprocess writes its real exit code to a `.exit` sidecar
file as its final action, which is the runner's authoritative "done" signal
(PID-based liveness checks are unreliable inside containers when PID 1
doesn't reap zombies).

### Stretch goals from the design doc

- Stretch A (automation / event triggering) — partially done: the UI watches
  `TLC_Data/` and exposes new files as a one-click "Run flow" trigger
  (manual button rather than `cron`-based polling, by request).
- Stretch B (Giskard) — not implemented.
- Stretch C (web deployment) — not implemented (the project ships with a
  containerised local stack instead).
