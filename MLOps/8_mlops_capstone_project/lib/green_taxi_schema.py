"""Green taxi raw-batch schema: presence, dtype family, range, and domain policy.

Slice identity (which month a batch represents) lives in run/flow metadata,
not as a column. Do NOT add a "month" entry here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


PICKUP_COL = "lpep_pickup_datetime"
DROPOFF_COL = "lpep_dropoff_datetime"
TARGET_COL = "tip_amount"
TRIP_DISTANCE_COL = "trip_distance"


class DType(Enum):
    NUMERIC = "numeric"
    STRING = "string"
    DATETIME = "datetime"
    BOOL = "bool"
    CATEGORY = "category"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: DType
    required: bool = True
    value_range: tuple[float, float] | None = None
    allowed_values: tuple[int | str, ...] | None = None


GREEN_TAXI_SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec(name="VendorID", dtype=DType.NUMERIC),
    ColumnSpec(name="lpep_pickup_datetime", dtype=DType.DATETIME),
    ColumnSpec(name="lpep_dropoff_datetime", dtype=DType.DATETIME),
    ColumnSpec(name="store_and_fwd_flag", dtype=DType.STRING, allowed_values=("Y", "N")),
    ColumnSpec(name="RatecodeID", dtype=DType.NUMERIC, allowed_values=(1, 2, 3, 4, 5, 6)),
    ColumnSpec(name="PULocationID", dtype=DType.NUMERIC),
    ColumnSpec(name="DOLocationID", dtype=DType.NUMERIC),
    ColumnSpec(name="passenger_count", dtype=DType.NUMERIC, value_range=(0.0, 10.0)),
    ColumnSpec(name="trip_distance", dtype=DType.NUMERIC, value_range=(0.0, 200.0)),
    ColumnSpec(name="fare_amount", dtype=DType.NUMERIC, value_range=(0.0, 500.0)),
    ColumnSpec(name="extra", dtype=DType.NUMERIC),
    ColumnSpec(name="mta_tax", dtype=DType.NUMERIC),
    ColumnSpec(name="tip_amount", dtype=DType.NUMERIC, value_range=(0.0, 200.0)),
    ColumnSpec(name="tolls_amount", dtype=DType.NUMERIC, value_range=(0.0, 200.0)),
    # ehail_fee is historically a junk column with mixed/empty values; permitted as STRING.
    ColumnSpec(name="ehail_fee", dtype=DType.STRING),
    ColumnSpec(name="improvement_surcharge", dtype=DType.NUMERIC),
    ColumnSpec(name="total_amount", dtype=DType.NUMERIC, value_range=(0.0, 1000.0)),
    ColumnSpec(name="payment_type", dtype=DType.NUMERIC, allowed_values=(1, 2, 3, 4, 5, 6)),
    ColumnSpec(name="trip_type", dtype=DType.NUMERIC, allowed_values=(1, 2)),
    ColumnSpec(name="congestion_surcharge", dtype=DType.NUMERIC),
    # Derived feature; only checked when present in the frame.
    ColumnSpec(name="duration_min", dtype=DType.NUMERIC, required=False, value_range=(0.0, 360.0)),
)


def family_ok(*, actual_dtype: object, family: DType) -> bool:
    t = pd.api.types
    match family:
        case DType.DATETIME:
            return bool(t.is_datetime64_any_dtype(actual_dtype))
        case DType.NUMERIC:
            return bool(t.is_numeric_dtype(actual_dtype))
        case DType.STRING:
            # TLC flags arrive as object; some engines yield pandas string dtype or category.
            return bool(
                t.is_object_dtype(actual_dtype)
                or t.is_string_dtype(actual_dtype)
                or isinstance(actual_dtype, pd.CategoricalDtype)
            )
        case DType.BOOL:
            return bool(t.is_bool_dtype(actual_dtype))
        case DType.CATEGORY:
            return isinstance(actual_dtype, pd.CategoricalDtype)
