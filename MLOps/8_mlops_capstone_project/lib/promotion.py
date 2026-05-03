"""Step G - candidate acceptance criteria.

Evaluates the four promotion criteria from the design doc and produces a
single `PromotionResult` the caller can act on (flip alias or reject).

P1 - eval is valid: `rmse_candidate` exists and is finite.
P2 - beats champion: `rmse_candidate < rmse_champion * (1 - min_improvement_pct)`.
     Both RMSE values come from the SAME held-out batch slice (Step E + Step F).
P3 - stability (no reference regression): the candidate must not regress on the
     reference set by more than `max_ref_regression_pct`. Catches one-batch overfit.
P4 - integrity sanity: hard failures already gate the flow at Step B; this step
     records the soft `integrity_warn` flag for auditability but does not block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


@dataclass
class CriterionOutcome:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionResult:
    rmse_candidate_eval: float
    rmse_champion_eval: float
    rmse_candidate_ref: float
    rmse_champion_ref: float
    rmse_ref_delta_pct: float
    min_improvement_pct: float
    max_ref_regression_pct: float
    integrity_warn: bool
    criteria: list[CriterionOutcome]
    promoted: bool
    decision_reason: str


def evaluate_promotion(
    *,
    champion_model: Pipeline,
    candidate_model: Pipeline,
    X_ref: pd.DataFrame,
    y_ref: np.ndarray,
    rmse_champion_eval: float,
    rmse_candidate_eval: float,
    integrity_warn: bool,
    min_improvement_pct: float,
    max_ref_regression_pct: float,
) -> PromotionResult:
    rmse_champion_ref = _rmse(champion_model, X_ref, y_ref)
    rmse_candidate_ref = _rmse(candidate_model, X_ref, y_ref)
    rmse_ref_delta_pct = (rmse_candidate_ref - rmse_champion_ref) / rmse_champion_ref * 100.0

    criteria = [
        _evaluate_p1(rmse_candidate_eval),
        _evaluate_p2(rmse_candidate_eval, rmse_champion_eval, min_improvement_pct),
        _evaluate_p3(rmse_candidate_ref, rmse_champion_ref, max_ref_regression_pct),
        _evaluate_p4(integrity_warn),
    ]
    promoted = all(c.passed for c in criteria)
    decision_reason = _format_decision_reason(criteria, promoted)

    return PromotionResult(
        rmse_candidate_eval=rmse_candidate_eval,
        rmse_champion_eval=rmse_champion_eval,
        rmse_candidate_ref=rmse_candidate_ref,
        rmse_champion_ref=rmse_champion_ref,
        rmse_ref_delta_pct=rmse_ref_delta_pct,
        min_improvement_pct=min_improvement_pct,
        max_ref_regression_pct=max_ref_regression_pct,
        integrity_warn=integrity_warn,
        criteria=criteria,
        promoted=promoted,
        decision_reason=decision_reason,
    )


def _rmse(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def _evaluate_p1(rmse_candidate_eval: float) -> CriterionOutcome:
    valid = math.isfinite(rmse_candidate_eval)
    detail = (
        f"rmse_candidate={rmse_candidate_eval:.4f} is finite"
        if valid
        else f"rmse_candidate={rmse_candidate_eval} is not finite"
    )
    return CriterionOutcome(name="P1_eval_valid", passed=valid, detail=detail)


def _evaluate_p2(
    rmse_candidate: float,
    rmse_champion: float,
    min_improvement_pct: float,
) -> CriterionOutcome:
    threshold = rmse_champion * (1.0 - min_improvement_pct)
    passed = rmse_candidate < threshold
    detail = (
        f"rmse_candidate={rmse_candidate:.4f} vs threshold={threshold:.4f} "
        f"(rmse_champion={rmse_champion:.4f}, min_improvement={min_improvement_pct * 100:.2f}%)"
    )
    return CriterionOutcome(name="P2_beats_champion", passed=passed, detail=detail)


def _evaluate_p3(
    rmse_candidate_ref: float,
    rmse_champion_ref: float,
    max_ref_regression_pct: float,
) -> CriterionOutcome:
    threshold = rmse_champion_ref * (1.0 + max_ref_regression_pct)
    passed = rmse_candidate_ref <= threshold
    delta_pct = (rmse_candidate_ref - rmse_champion_ref) / rmse_champion_ref * 100.0
    detail = (
        f"rmse_candidate_ref={rmse_candidate_ref:.4f} vs rmse_champion_ref={rmse_champion_ref:.4f} "
        f"(delta={delta_pct:+.2f}%, max allowed={max_ref_regression_pct * 100:.2f}%)"
    )
    return CriterionOutcome(name="P3_no_ref_regression", passed=passed, detail=detail)


def _evaluate_p4(integrity_warn: bool) -> CriterionOutcome:
    detail = (
        "Soft integrity warning present (informational only; does not block promotion)"
        if integrity_warn
        else "No integrity warnings"
    )
    return CriterionOutcome(name="P4_integrity_sanity", passed=True, detail=detail)


def _format_decision_reason(criteria: list[CriterionOutcome], promoted: bool) -> str:
    if promoted:
        return "All criteria passed; candidate approved for promotion to @champion"
    failed = [c for c in criteria if not c.passed]
    failed_names = ", ".join(c.name for c in failed)
    failed_details = "; ".join(c.detail for c in failed)
    return f"Rejected by {failed_names}: {failed_details}"
