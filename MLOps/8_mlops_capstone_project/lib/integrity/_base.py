"""Shared types for integrity gates.

CheckResult is intentionally pure data so both hard and soft gates can
populate the same structure. Gate-specific decision logic (failure reasons,
warning derivation) lives in `hard.py` / `soft.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    metrics: dict[str, float]
    tables: dict[str, pd.DataFrame]
    warnings: list[str] = field(default_factory=list)
