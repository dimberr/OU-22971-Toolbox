"""Streamlit control panel for the MLOps capstone flow.

Layout:
- Sidebar: data dir info, reference selector, model name, hyperparams, MLflow link.
- Main:
  * Available batches table with a "Run flow" button per row.
  * Live log panel (visible while a flow is running, auto-refreshes).
  * Recent runs history.

Concurrency: only one flow run at a time. The "Run" buttons are disabled
while a run is in progress; the lock lives on disk (ui/state.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ui import runner, state


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "TLC_Data"
DEFAULT_STATE_DIR = PROJECT_ROOT / "ui_state"
DEFAULT_MODEL_NAME = "green_taxi_tip_model"
DEFAULT_REFERENCE = "green_tripdata_2020-01.parquet"
DEFAULT_ZONE_FILE = "NYC_Taxi_Zones.geojson"
MLFLOW_UI_URL_DEFAULT = "http://localhost:5000"

STATUS_ICON = {"running": "...", "success": "OK", "failed": "FAIL"}


def _data_dir() -> Path:
    return Path(os.environ.get("DATA_DIR", str(DEFAULT_DATA_DIR)))


def _state_dir() -> Path:
    return Path(os.environ.get("UI_STATE_DIR", str(DEFAULT_STATE_DIR)))


def _mlflow_ui_url() -> str:
    # The browser hits MLflow on the host port-mapping, not the in-network name,
    # so default to localhost:5000 even when MLFLOW_TRACKING_URI is mlflow:5000.
    return os.environ.get("MLFLOW_UI_URL", MLFLOW_UI_URL_DEFAULT)


def _list_parquet_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("green_tripdata_*.parquet"))


def _last_status_for(file_name: str, history: list[state.RunRecord]) -> tuple[str, str]:
    for record in history:
        if record.batch_file == file_name:
            return record.status, record.started_at
    return "—", ""


def _render_sidebar(data_dir: Path, parquet_files: list[Path]) -> dict:
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Data dir: `{data_dir}`")
        st.markdown(f"[Open MLflow UI]({_mlflow_ui_url()})")

        file_names = [p.name for p in parquet_files]
        default_idx = file_names.index(DEFAULT_REFERENCE) if DEFAULT_REFERENCE in file_names else 0
        reference_file = st.selectbox(
            "Reference file",
            options=file_names if file_names else ["(no parquet files found)"],
            index=default_idx if file_names else 0,
        )
        model_name = st.text_input("Model name", value=DEFAULT_MODEL_NAME)

        with st.expander("Advanced parameters"):
            rolling_window_months = st.number_input(
                "rolling_window_months", min_value=1, max_value=120, value=12,
            )
            batch_eval_pct = st.slider(
                "batch_eval_pct", min_value=0.05, max_value=0.5, value=0.2, step=0.05,
            )
            min_improvement_pct = st.slider(
                "min_improvement_pct", min_value=0.0, max_value=0.10, value=0.01, step=0.005,
            )
            max_ref_regression_pct = st.slider(
                "max_ref_regression_pct", min_value=0.0, max_value=0.10, value=0.01, step=0.005,
            )

        return {
            "reference_file": reference_file,
            "model_name": model_name,
            "rolling_window_months": int(rolling_window_months),
            "batch_eval_pct": float(batch_eval_pct),
            "min_improvement_pct": float(min_improvement_pct),
            "max_ref_regression_pct": float(max_ref_regression_pct),
        }


def _build_run_params(
    *,
    data_dir: Path,
    settings: dict,
    batch_file: str,
) -> runner.RunParams:
    zone_path = data_dir / DEFAULT_ZONE_FILE
    return runner.RunParams(
        reference_path=str(data_dir / settings["reference_file"]),
        batch_path=str(data_dir / batch_file),
        historical_dir=str(data_dir),
        model_name=settings["model_name"],
        rolling_window_months=settings["rolling_window_months"],
        batch_eval_pct=settings["batch_eval_pct"],
        min_improvement_pct=settings["min_improvement_pct"],
        max_ref_regression_pct=settings["max_ref_regression_pct"],
        taxi_zone_lookup_path=str(zone_path) if zone_path.exists() else None,
    )


def _render_batches_table(
    *,
    parquet_files: list[Path],
    settings: dict,
    history: list[state.RunRecord],
    is_running: bool,
    data_dir: Path,
    state_dir: Path,
) -> None:
    st.subheader("Available batches")

    reference_file = settings["reference_file"]
    if not parquet_files:
        st.info(f"No `green_tripdata_*.parquet` files found in `{data_dir}`.")
        return

    header = st.columns([3, 1, 2, 2, 2])
    for col, label in zip(header, ["File", "Size (MB)", "Modified", "Last status", "Action"]):
        col.markdown(f"**{label}**")

    for path in parquet_files:
        is_reference = path.name == reference_file
        size_mb = path.stat().st_size / (1024 * 1024)
        modified = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        last_status, last_at = _last_status_for(path.name, history)
        status_text = f"{STATUS_ICON.get(last_status, '—')} {last_status}"
        if last_at:
            status_text += f"  ({last_at[:16]})"

        cols = st.columns([3, 1, 2, 2, 2])
        cols[0].write(f"`{path.name}`" + (" *(reference)*" if is_reference else ""))
        cols[1].write(f"{size_mb:.1f}")
        cols[2].write(modified)
        cols[3].write(status_text)

        button_disabled = is_running or is_reference
        button_help = (
            "A flow is already running" if is_running
            else "This file is the reference; pick a different batch" if is_reference
            else "Run the full pipeline on this batch"
        )
        if cols[4].button("Run flow", key=f"run-{path.name}", disabled=button_disabled, help=button_help):
            params = _build_run_params(data_dir=data_dir, settings=settings, batch_file=path.name)
            runner.start_run(project_root=PROJECT_ROOT, state_dir=state_dir, params=params)
            st.rerun()


@st.fragment(run_every=2.0)
def _render_live_log(state_dir: Path) -> None:
    finalized = runner.finalize_if_done(state_dir)
    if finalized is not None:
        st.rerun()
        return

    current = state.load_current(state_dir)
    if current is None:
        return

    st.subheader(f"Running: `{current.batch_file}`  (run id: `{current.id}`)")
    st.caption(f"Started: {current.started_at}  •  PID: {current.pid}")

    log_text = runner.read_log_tail(Path(current.log_path))
    st.code(log_text or "(waiting for output...)", language="bash", line_numbers=False)

    if st.button("Cancel run", key="cancel-run"):
        runner.cancel_current_run(state_dir)
        st.warning("Sent SIGTERM to the running flow. It will be marked as failed.")


def _render_history(history: list[state.RunRecord], state_dir: Path) -> None:
    st.subheader("Recent runs")
    if not history:
        st.caption("No runs yet.")
        return

    rows = []
    for r in history:
        rows.append({
            "id": r.id,
            "batch": r.batch_file,
            "reference": r.reference_file,
            "started": r.started_at,
            "finished": r.finished_at or "",
            "status": f"{STATUS_ICON.get(r.status, '?')} {r.status}",
            "exit": r.exit_code if r.exit_code is not None else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("View log of a past run"):
        run_ids = [r.id for r in history]
        chosen = st.selectbox("Run id", options=run_ids, key="hist-log-select")
        if chosen:
            chosen_record = next(r for r in history if r.id == chosen)
            log_text = runner.read_log_tail(Path(chosen_record.log_path), max_chars=200_000)
            st.code(log_text or "(no log available)", language="bash")


def main() -> None:
    st.set_page_config(
        page_title="Green Taxi - MLOps Capstone",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data_dir = _data_dir()
    state_dir = _state_dir()
    state.ensure_state_dir(state_dir)

    parquet_files = _list_parquet_files(data_dir)
    settings = _render_sidebar(data_dir, parquet_files)

    st.title("Green Taxi tip prediction - MLOps capstone")
    st.markdown(
        "Drop a new `green_tripdata_YYYY-MM.parquet` into the data folder; "
        "it will appear below. Click **Run flow** to execute integrity gate, "
        "feature engineering, performance gate, retrain (conditional), and "
        f"promotion. Evidence lands in [MLflow]({_mlflow_ui_url()})."
    )

    runner.finalize_if_done(state_dir)
    is_running = state.load_current(state_dir) is not None
    history = state.load_history(state_dir)

    _render_batches_table(
        parquet_files=parquet_files,
        settings=settings,
        history=history,
        is_running=is_running,
        data_dir=data_dir,
        state_dir=state_dir,
    )

    if is_running:
        st.divider()
        _render_live_log(state_dir)

    st.divider()
    _render_history(history, state_dir)


if __name__ == "__main__":
    main()
