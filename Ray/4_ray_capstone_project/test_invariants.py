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

from main import ScoreHelper, ZoneActor, decide, score_with_helpers, split_int


@pytest.fixture(scope="module")
def ray_session():
    ray.init(num_cpus=2, log_to_driver=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


def counters_of(actor: ActorHandle) -> dict[str, int]:
    return cast("dict[str, int]", ray.get(actor.counters.remote()))


def snapshot_at(actor: ActorHandle, tick_id: int) -> dict:
    return cast("dict", ray.get(actor.next_snapshot.remote(tick_id)))


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


def test_delayed_arrival_shifts_demand_and_flips_decision(ray_session):
    # Monday 2025-12-01 00:00 -> ticks 0,1,2 share the (hour=0, dow=0) baseline.
    actor = cast(
        "ActorHandle",
        ZoneActor.remote(
            zone_id=1,
            pickups_by_tick={0: 20, 2: 8},
            baseline_by_slot={(0, 0): 12.0},
            month_start=pd.Timestamp("2025-12-01"),
            withhold_fraction=0.5,
            arrival_delay_ticks=2,
        ),
    )

    s0 = snapshot_at(actor, 0)
    s1 = snapshot_at(actor, 1)
    s2 = snapshot_at(actor, 2)

    # Tick 0: true demand 20 is busy, but half is withheld -> visible 10.
    assert s0["current_pickups"] == 10
    assert decide(20, 12.0, 1.1) == "NEED"  # what the truth deserved
    assert decide(s0["current_pickups"], 12.0, 1.1) == "OK"  # mislabeled at T

    assert s1["current_pickups"] == 0

    # Tick 2: own true demand 8 (would be OK), but 10 resurfaces from tick 0,
    # plus 4 of tick 2's own demand withheld -> visible 4 + 10 = 14 -> false NEED.
    assert s2["current_pickups"] == 14
    assert decide(8, 12.0, 1.1) == "OK"  # what tick 2 alone deserved
    assert decide(s2["current_pickups"], 12.0, 1.1) == "NEED"  # false spike

    c = counters_of(actor)
    assert c["withheld_total"] == 14  # 10 from tick 0 + 4 from tick 2
    assert c["released_total"] == 10  # only tick 0's release surfaced by tick 2
    assert c["unreleased_total"] == 4  # tick 2's withheld demand is still pending


def make_subactor_zone(trigger: int = 3, n_helpers: int = 3) -> ActorHandle:
    return cast(
        "ActorHandle",
        ZoneActor.remote(
            zone_id=1,
            pickups_by_tick={},
            baseline_by_slot={},
            month_start=pd.Timestamp("2025-12-01"),
            subactor_trigger=trigger,
            n_helpers=n_helpers,
        ),
    )


def test_split_int_sums_back():
    assert split_int(20, 3) == [7, 7, 6]
    assert sum(split_int(20, 3)) == 20
    assert split_int(0, 3) == [0, 0, 0]


def test_promotion_after_consecutive_fallbacks(ray_session):
    actor = make_subactor_zone(trigger=3, n_helpers=3)
    for t in range(3):  # three ticks with no report -> three fallbacks in a row
        ray.get(actor.mark_tick_active.remote(t))
        ray.get(actor.finalize_tick.remote(t))

    assert len(ray.get(actor.helper_handles.remote())) == 3
    assert counters_of(actor)["promoted"] == 1


def test_no_promotion_when_streak_breaks(ray_session):
    actor = make_subactor_zone(trigger=3)
    for t in (0, 1):  # two misses
        ray.get(actor.mark_tick_active.remote(t))
        ray.get(actor.finalize_tick.remote(t))
    ray.get(actor.mark_tick_active.remote(2))  # a real report resets the streak
    ray.get(actor.report_decision.remote(2, "OK"))
    ray.get(actor.finalize_tick.remote(2))
    ray.get(actor.mark_tick_active.remote(3))  # one more miss -> only 1 in a row
    ray.get(actor.finalize_tick.remote(3))

    assert ray.get(actor.helper_handles.remote()) == []
    assert counters_of(actor)["promoted"] == 0


def test_helpers_preserve_decision(ray_session):
    actor = make_subactor_zone()
    ray.get(actor.mark_tick_active.remote(0))
    helpers = [ScoreHelper.remote() for _ in range(3)]
    snapshot = {"zone_id": 1, "tick_id": 0, "current_pickups": 20, "baseline_pickups": 12.0}

    result = cast("dict", ray.get(score_with_helpers.remote(snapshot, actor, helpers, 1.1, 0.0)))

    # Sharded fan-out yields the same decision as a single scoring of the total.
    assert result["decision"] == decide(20, 12.0, 1.1) == "NEED"
    ray.get(actor.finalize_tick.remote(0))
    assert counters_of(actor)["subactor_ticks"] == 1
