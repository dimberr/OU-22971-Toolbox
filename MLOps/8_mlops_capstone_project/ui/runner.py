"""Subprocess flow runner.

Spawns the flow (or bootstrap) as a detached `sh -c` subprocess so the
parent shell can write the real exit code to a sidecar file once the
process finishes. We need this because Streamlit reruns the script on
every interaction, which throws away any in-memory `Popen` handle, so
we can't call `proc.wait()` to learn the exit code later.

Liveness check: probe the recorded PID with `os.kill(pid, 0)`.
Exit code: read the sidecar file (only present after the process exits).
"""

from __future__ import annotations

import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import state


@dataclass
class RunParams:
    reference_path: str
    batch_path: str
    historical_dir: str
    model_name: str
    rolling_window_months: int
    batch_eval_pct: float
    min_improvement_pct: float
    max_ref_regression_pct: float
    taxi_zone_lookup_path: str | None = None


@dataclass
class BootstrapParams:
    reference_path: str
    model_name: str


@dataclass
class ScoreParams:
    reference_path: str
    batch_path: str
    model_name: str


def start_run(*, project_root: Path, state_dir: Path, params: RunParams) -> state.RunRecord:
    return _start_subprocess(
        project_root=project_root,
        state_dir=state_dir,
        flow_cmd=_build_flow_cmd(params),
        batch_file=Path(params.batch_path).name,
        reference_file=Path(params.reference_path).name,
        params_summary=_params_summary(params),
    )


def start_bootstrap(
    *,
    project_root: Path,
    state_dir: Path,
    params: BootstrapParams,
) -> state.RunRecord:
    return _start_subprocess(
        project_root=project_root,
        state_dir=state_dir,
        flow_cmd=_build_bootstrap_cmd(params),
        batch_file="(bootstrap)",
        reference_file=Path(params.reference_path).name,
        params_summary={"action": "bootstrap", "model_name": params.model_name},
    )


def start_score(
    *,
    project_root: Path,
    state_dir: Path,
    params: ScoreParams,
) -> state.RunRecord:
    return _start_subprocess(
        project_root=project_root,
        state_dir=state_dir,
        flow_cmd=_build_score_cmd(params),
        batch_file=f"(score) {Path(params.batch_path).name}",
        reference_file=Path(params.reference_path).name,
        params_summary={
            "action": "score",
            "model_name": params.model_name,
            "batch_path": params.batch_path,
        },
    )


def _start_subprocess(
    *,
    project_root: Path,
    state_dir: Path,
    flow_cmd: list[str],
    batch_file: str,
    reference_file: str,
    params_summary: dict[str, str],
) -> state.RunRecord:
    state.ensure_state_dir(state_dir)

    run_id = uuid.uuid4().hex[:8]
    log_path = state.log_path_for(state_dir, run_id)
    exit_code_path = _exit_code_path(state_dir, run_id)

    log_handle = log_path.open("w", buffering=1)
    log_handle.write(f"$ {' '.join(flow_cmd)}\n\n")
    log_handle.flush()

    shell_cmd = (
        " ".join(shlex.quote(p) for p in flow_cmd)
        + f" ; echo $? > {shlex.quote(str(exit_code_path))}"
    )
    proc = subprocess.Popen(
        ["sh", "-c", shell_cmd],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=project_root,
    )

    record = state.RunRecord(
        id=run_id,
        batch_file=batch_file,
        reference_file=reference_file,
        started_at=state.utc_now_iso(),
        status="running",
        log_path=str(log_path),
        pid=proc.pid,
        params=params_summary,
    )
    state.save_current(state_dir, record)
    return record


def finalize_if_done(state_dir: Path) -> state.RunRecord | None:
    """If the current run has finished, move it to history.

    The shell wrapper writes the exit code to a sidecar file as its final
    action, so the existence of that file is our authoritative "done"
    signal. PID-based liveness is unreliable inside containers because
    zombie children persist when PID 1 doesn't reap.

    Returns the finalized record (or None if no current run / still running).
    """
    current = state.load_current(state_dir)
    if current is None:
        return None

    exit_code_path = _exit_code_path(state_dir, current.id)
    if exit_code_path.exists():
        exit_code = _read_exit_code(state_dir, current.id)
    elif current.pid is not None and state.is_pid_alive(current.pid):
        return None
    else:
        # PID gone but no exit-code file: process was killed externally.
        exit_code = 137

    current.finished_at = state.utc_now_iso()
    current.exit_code = exit_code
    current.status = "success" if exit_code == 0 else "failed"
    current.pid = None

    state.append_history(state_dir, current)
    state.clear_current(state_dir)
    return current


def cancel_current_run(state_dir: Path) -> None:
    current = state.load_current(state_dir)
    if current is None or current.pid is None:
        return
    state.kill_pid(current.pid)


def read_log_tail(log_path: Path, max_chars: int = 20_000) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace")
    return text if len(text) <= max_chars else text[-max_chars:]


def _build_flow_cmd(params: RunParams) -> list[str]:
    cmd = [
        "python", "flow_starter.py", "run",
        "--reference-path", params.reference_path,
        "--batch-path", params.batch_path,
        "--historical-dir", params.historical_dir,
        "--model-name", params.model_name,
        "--rolling-window-months", str(params.rolling_window_months),
        "--batch-eval-pct", str(params.batch_eval_pct),
        "--min-improvement-pct", str(params.min_improvement_pct),
        "--max-ref-regression-pct", str(params.max_ref_regression_pct),
    ]
    if params.taxi_zone_lookup_path:
        cmd += ["--taxi-zone-lookup-path", params.taxi_zone_lookup_path]
    return cmd


def _build_bootstrap_cmd(params: BootstrapParams) -> list[str]:
    return [
        "python", "bootstrap.py",
        "--reference-path", params.reference_path,
        "--model-name", params.model_name,
    ]


def _build_score_cmd(params: ScoreParams) -> list[str]:
    return [
        "python", "score_batch.py",
        "--reference-path", params.reference_path,
        "--batch-path", params.batch_path,
        "--model-name", params.model_name,
    ]


def _params_summary(params: RunParams) -> dict[str, str]:
    return {
        "model_name": params.model_name,
        "rolling_window_months": str(params.rolling_window_months),
        "batch_eval_pct": str(params.batch_eval_pct),
        "min_improvement_pct": str(params.min_improvement_pct),
        "max_ref_regression_pct": str(params.max_ref_regression_pct),
    }


def _exit_code_path(state_dir: Path, run_id: str) -> Path:
    return state.log_path_for(state_dir, run_id).with_suffix(".exit")


def _read_exit_code(state_dir: Path, run_id: str) -> int:
    path = _exit_code_path(state_dir, run_id)
    if not path.exists():
        return 1
    return int(path.read_text().strip() or "1")
