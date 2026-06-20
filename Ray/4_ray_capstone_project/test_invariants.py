"""Fault-tolerance invariant checks for ZoneActor.

These exercise the failure model from the design doc against a single actor:
idempotent writes, and rejection of duplicate, late, and inactive-tick reports.
They use no skew sleeps, so they run quickly and will not stall a laptop.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
import ray
from ray.actor import ActorHandle

from main import ZoneActor


@pytest.fixture(scope="module")
def ray_session():
    ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def counters_of(actor: ActorHandle) -> dict[str, int]:
    return cast("dict[str, int]", ray.get(actor.counters.remote()))


def make_actor() -> ActorHandle:
    return cast(
        "ActorHandle",
        ZoneActor.remote(
            zone_id=1,
            pickups_by_tick={},
            baseline_by_slot={},
            month_start=pd.Timestamp("2025-12-01"),
        ),
    )


def test_idempotent_blocking_write(ray_session):
    actor = make_actor()
    assert ray.get(actor.write_decision.remote(0, "NEED")) is True
    assert ray.get(actor.write_decision.remote(0, "NEED")) is False
    assert ray.get(actor.accepted.remote()) == {
        0: {"decision": "NEED", "used_fallback": False, "task_latency_s": 0.0}
    }


def test_duplicate_report_rejected(ray_session):
    actor = make_actor()
    ray.get(actor.mark_tick_active.remote(0))
    assert ray.get(actor.report_decision.remote(0, "NEED")) == "accepted"
    assert ray.get(actor.report_decision.remote(0, "NEED")) == "duplicate"
    assert counters_of(actor)["duplicate_reports"] == 1


def test_late_report_does_not_overwrite(ray_session):
    actor = make_actor()
    ray.get(actor.mark_tick_active.remote(0))
    ray.get(actor.report_decision.remote(0, "NEED"))
    ray.get(actor.finalize_tick.remote(0))

    assert ray.get(actor.report_decision.remote(0, "OK")) == "late"
    assert ray.get(actor.accepted.remote())[0]["decision"] == "NEED"
    assert counters_of(actor)["late_reports"] == 1


def test_inactive_tick_report_rejected(ray_session):
    actor = make_actor()
    ray.get(actor.mark_tick_active.remote(0))
    assert ray.get(actor.report_decision.remote(5, "NEED")) == "late"
    assert 5 not in ray.get(actor.accepted.remote())


def test_fallback_first_use_ok_then_previous(ray_session):
    actor = make_actor()
    ray.get(actor.mark_tick_active.remote(0))
    assert ray.get(actor.finalize_tick.remote(0)) == {
        "decision": "OK",
        "used_fallback": True,
        "task_latency_s": None,
    }

    ray.get(actor.mark_tick_active.remote(1))
    ray.get(actor.report_decision.remote(1, "NEED"))
    ray.get(actor.finalize_tick.remote(1))

    ray.get(actor.mark_tick_active.remote(2))
    assert ray.get(actor.finalize_tick.remote(2)) == {
        "decision": "NEED",
        "used_fallback": True,
        "task_latency_s": None,
    }
    assert counters_of(actor)["fallbacks"] == 2
