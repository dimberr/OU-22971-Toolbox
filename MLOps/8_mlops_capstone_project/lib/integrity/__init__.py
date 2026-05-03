"""Integrity checks for the green taxi raw batch.

Two gates:
- hard: deterministic rules (schema, ranges, datetime sanity, etc.).
  Failures stop the flow.
- soft: NannyML-based data quality (missingness spikes, unseen categoricals).
  Failures emit warnings only; flow proceeds.
"""

from ._base import CheckResult
from .hard import hard_failure_reasons, hard_is_ok, run_hard_integrity_checks
from .soft import run_soft_integrity_checks


__all__ = [
    "CheckResult",
    "hard_failure_reasons",
    "hard_is_ok",
    "run_hard_integrity_checks",
    "run_soft_integrity_checks",
]
