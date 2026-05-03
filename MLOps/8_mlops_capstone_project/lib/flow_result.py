"""Build and persist the structured outcome record the UI reads after a run.

The Streamlit history table consumes the JSON written here (one file per
flow run, at `--ui-result-path`) to show the retrain decision and outcome
label without having to query MLflow on every page render.

Pure data formatting; no Metaflow or MLflow imports so it can be unit-
tested in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_flow_result(
    *,
    batch_rejected: bool,
    retrain_needed: bool | None = None,
    rmse_champion: float | None = None,
    integrity_warn: bool = False,
    promoted: bool | None = None,
    candidate_version: str | None = None,
    candidate_rmse: float | None = None,
    candidate_rmse_delta_pct: float | None = None,
) -> dict:
    """Compose the outcome dict for one of the four terminal branches:
    batch_rejected, no_retrain, promoted, candidate_rejected.

    All metric fields are optional because the values that exist depend on
    which branch the flow took (e.g. retrain-related fields are only set
    when retrain actually ran).
    """
    if batch_rejected:
        return {
            "outcome": "batch_rejected",
            "retrain_needed": None,
            "promoted": None,
            "summary": (
                "Batch rejected by hard integrity checks. "
                "No evaluation, retrain, or promotion performed."
            ),
        }

    if not retrain_needed:
        return {
            "outcome": "no_retrain",
            "retrain_needed": False,
            "promoted": False,
            "rmse_champion": float(rmse_champion) if rmse_champion is not None else None,
            "integrity_warn": bool(integrity_warn),
            "summary": (
                "Champion still healthy; no retrain triggered. "
                f"rmse_champion={rmse_champion:.4f} on batch_eval "
                f"(integrity_warn={integrity_warn})."
            ),
        }

    if promoted:
        return {
            "outcome": "promoted",
            "retrain_needed": True,
            "promoted": True,
            "candidate_version": str(candidate_version),
            "rmse_champion": float(rmse_champion) if rmse_champion is not None else None,
            "rmse_candidate": float(candidate_rmse) if candidate_rmse is not None else None,
            "rmse_delta_pct": float(candidate_rmse_delta_pct) if candidate_rmse_delta_pct is not None else None,
            "summary": (
                f"Candidate v{candidate_version} PROMOTED to @champion. "
                f"rmse_candidate={candidate_rmse:.4f} vs "
                f"rmse_champion={rmse_champion:.4f} on batch_eval "
                f"(delta={candidate_rmse_delta_pct:+.2f}%)."
            ),
        }

    return {
        "outcome": "candidate_rejected",
        "retrain_needed": True,
        "promoted": False,
        "candidate_version": str(candidate_version),
        "rmse_champion": float(rmse_champion) if rmse_champion is not None else None,
        "rmse_candidate": float(candidate_rmse) if candidate_rmse is not None else None,
        "rmse_delta_pct": float(candidate_rmse_delta_pct) if candidate_rmse_delta_pct is not None else None,
        "summary": (
            f"Candidate v{candidate_version} registered but REJECTED for promotion. "
            f"rmse_candidate={candidate_rmse:.4f} vs "
            f"rmse_champion={rmse_champion:.4f} on batch_eval "
            f"(delta={candidate_rmse_delta_pct:+.2f}%). "
            "Champion alias unchanged."
        ),
    }


def write_flow_result(path: str | None, result: dict) -> None:
    """Write the result dict as JSON to `path`. No-op if path is falsy.

    Parent directories are created if needed (Metaflow steps run in a
    fresh subprocess on each invocation, so the dir may not exist yet).
    """
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
