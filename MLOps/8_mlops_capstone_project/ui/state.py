"""Persistent UI state: run history + current-run lock.

All UI state lives on disk under UI_STATE_DIR so it survives Streamlit
script reruns (which spawn fresh interpreters) and container restarts.

Files:
- runs.json         : list of finished runs (most recent first), bounded.
- current.json      : metadata of the currently-running flow (deleted on finalize).
- logs/<id>.log     : stdout/stderr of each flow run.
- results/<id>.json : structured outcome the flow's end step writes,
                      consumed by the runner on finalize (retrain
                      decision, promotion, summary).
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


_HISTORY_FILE = "runs.json"
_CURRENT_FILE = "current.json"
_LOGS_DIR = "logs"
_RESULTS_DIR = "results"
_HISTORY_LIMIT = 50


@dataclass
class RunRecord:
    id: str
    batch_file: str
    reference_file: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    exit_code: int | None = None
    log_path: str = ""
    pid: int | None = None
    params: dict[str, str] = field(default_factory=dict)
    result: dict | None = None


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def ensure_state_dir(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / _LOGS_DIR).mkdir(parents=True, exist_ok=True)
    (state_dir / _RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def log_path_for(state_dir: Path, run_id: str) -> Path:
    return state_dir / _LOGS_DIR / f"{run_id}.log"


def result_path_for(state_dir: Path, run_id: str) -> Path:
    return state_dir / _RESULTS_DIR / f"{run_id}.json"


def read_result(state_dir: Path, run_id: str) -> dict | None:
    path = result_path_for(state_dir, run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_history(state_dir: Path) -> list[RunRecord]:
    path = state_dir / _HISTORY_FILE
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [RunRecord(**item) for item in raw]


def save_history(state_dir: Path, records: list[RunRecord]) -> None:
    path = state_dir / _HISTORY_FILE
    bounded = records[:_HISTORY_LIMIT]
    path.write_text(json.dumps([asdict(r) for r in bounded], indent=2))


def append_history(state_dir: Path, record: RunRecord) -> None:
    history = load_history(state_dir)
    history.insert(0, record)
    save_history(state_dir, history)


def load_current(state_dir: Path) -> RunRecord | None:
    path = state_dir / _CURRENT_FILE
    if not path.exists():
        return None
    return RunRecord(**json.loads(path.read_text()))


def save_current(state_dir: Path, record: RunRecord) -> None:
    path = state_dir / _CURRENT_FILE
    path.write_text(json.dumps(asdict(record), indent=2))


def clear_current(state_dir: Path) -> None:
    path = state_dir / _CURRENT_FILE
    if path.exists():
        path.unlink()


def is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
