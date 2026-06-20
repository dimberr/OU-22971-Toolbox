# Ray capstone scaffold starter: TLC-backed per-zone recommendations under skew.
# Runs: prepare TLC replay assets -> initialize per-zone actors -> compare blocking and async execution.

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import ray
from ray.actor import ActorHandle

PICKUP_COL = "lpep_pickup_datetime"
ZONE_COL = "PULocationID"
TICK_MINUTES = 15
NEED_THRESHOLD = 1.1


def load_and_clean_month(parquet_path: Path) -> tuple[pd.DataFrame, pd.Period]:
    """Load a Green Taxi file, keep only the pickup/zone columns, and drop rows
    whose pickup time does not fall in the file's dominant calendar month.

    Returns the cleaned frame plus the nominal month as a monthly Period.
    """
    df = pd.read_parquet(parquet_path, columns=[PICKUP_COL, ZONE_COL])
    df[PICKUP_COL] = pd.to_datetime(df[PICKUP_COL])

    months = df[PICKUP_COL].dt.to_period("M")
    nominal_month = months.mode().iloc[0]

    cleaned = df.loc[months == nominal_month].reset_index(drop=True)
    return cleaned, nominal_month


def validate_adjacent_months(ref_month: pd.Period, replay_month: pd.Period) -> None:
    """Enforce the design rule: the two files must be adjacent months from the
    same year (so a Dec -> Jan pair is rejected)."""
    if ref_month + 1 != replay_month:
        raise ValueError(
            f"Reference and replay months must be adjacent: got {ref_month} -> {replay_month}"
        )
    if ref_month.year != replay_month.year:
        raise ValueError(
            f"Reference and replay months must share a year: got {ref_month} and {replay_month}"
        )


@ray.remote
class ZoneActor:
    def __init__(
        self,
        zone_id: int,
        pickups_by_tick: dict[int, int],
        baseline_by_slot: dict[tuple[int, int], float],
        month_start: pd.Timestamp,
        tick_minutes: int = TICK_MINUTES,
        withhold_fraction: float = 0.0,
        arrival_delay_ticks: int = 0,
        subactor_trigger: int = 0,
        n_helpers: int = 3,
    ):
        self.zone_id = zone_id
        self.pickups_by_tick = pickups_by_tick
        self.baseline_by_slot = baseline_by_slot
        self.month_start = pd.Timestamp(month_start)
        self.tick_minutes = tick_minutes

        # Stretch A: delayed arrivals. A fraction of each tick's demand is hidden
        # and resurfaces (added) at tick T + arrival_delay_ticks.
        self.withhold_fraction = withhold_fraction
        self.arrival_delay_ticks = arrival_delay_ticks
        self.pending_releases: dict[int, int] = {}
        self.withheld_total = 0
        self.released_total = 0

        # Stretch B: adaptive load balancing. A repeat straggler (consecutive
        # fallbacks) promotes itself by spawning n_helpers helper subactors.
        self.subactor_trigger = subactor_trigger
        self.n_helpers = n_helpers
        self.consecutive_fallbacks = 0
        self.promoted = False
        self.helpers: list = []
        self.subactor_ticks = 0

        self.active_tick_id: int | None = None
        self.accepted_decisions: dict[int, dict] = {}

        self.reported_tick: int | None = None
        self.reported_decision: str | None = None
        self.reported_latency: float | None = None
        self.last_decision: str | None = None
        self.duplicate_reports = 0
        self.late_reports = 0
        self.fallbacks = 0

    def _slot_for_tick(self, tick_id: int) -> tuple[int, int]:
        """Derive the recurring (hour_of_day, day_of_week) baseline key for any
        tick_id, including ticks with zero observed demand (no replay row)."""
        tick_start = self.month_start + pd.Timedelta(minutes=tick_id * self.tick_minutes)
        return int(tick_start.hour), int(tick_start.dayofweek)

    def _is_delayed(self) -> bool:
        return self.withhold_fraction > 0 and self.arrival_delay_ticks > 0

    def _visible_pickups(self, tick_id: int) -> int:
        """Demand visible at this tick: this tick's true demand minus what we
        withhold, plus any earlier-withheld demand scheduled to resurface now.
        The resurfaced amount is additive, so the release tick is inflated."""
        true = self.pickups_by_tick.get(tick_id, 0)
        released = self.pending_releases.pop(tick_id, 0)
        self.released_total += released
        if not self._is_delayed() or true <= 0:
            return true + released
        withheld = int(true * self.withhold_fraction)
        self.withheld_total += withheld
        release_at = tick_id + self.arrival_delay_ticks
        self.pending_releases[release_at] = self.pending_releases.get(release_at, 0) + withheld
        return (true - withheld) + released

    def next_snapshot(self, tick_id: int) -> dict:
        """Mark this tick active and return the minimal snapshot the scoring task
        needs. Missing demand defaults to 0; a missing baseline defaults to 0.0."""
        self.active_tick_id = tick_id
        current_pickups = self._visible_pickups(tick_id)
        baseline_pickups = self.baseline_by_slot.get(self._slot_for_tick(tick_id), 0.0)
        return {
            "zone_id": self.zone_id,
            "tick_id": tick_id,
            "current_pickups": current_pickups,
            "baseline_pickups": baseline_pickups,
        }

    def write_decision(
        self, tick_id: int, decision: str, task_latency_s: float = 0.0, used_fallback: bool = False
    ) -> bool:
        """Idempotent accepted-decision write keyed by tick_id. A duplicate write
        for an already-recorded tick is a safe no-op. Returns True if this call
        actually recorded the decision, False if it was a duplicate."""
        if tick_id in self.accepted_decisions:
            return False
        self.accepted_decisions[tick_id] = {
            "decision": decision,
            "used_fallback": used_fallback,
            "task_latency_s": task_latency_s,
        }
        self.last_decision = decision
        return True

    def accepted(self) -> dict[int, dict]:
        """Expose the actor-owned accepted decisions so artifacts derive from
        actor state rather than raw task completions."""
        return self.accepted_decisions

    def counters(self) -> dict[str, int]:
        return {
            "duplicate_reports": self.duplicate_reports,
            "late_reports": self.late_reports,
            "fallbacks": self.fallbacks,
            "withheld_total": self.withheld_total,
            "released_total": self.released_total,
            # Demand withheld so late it never resurfaced before the run ended.
            "unreleased_total": sum(self.pending_releases.values()),
            "promoted": int(self.promoted),
            "subactor_ticks": self.subactor_ticks,
        }

    def helper_handles(self) -> list:
        """Expose this zone's helper subactors (empty until promoted). The driver
        routes a promoted zone's scoring through these handles."""
        return self.helpers

    def mark_tick_active(self, tick_id: int) -> None:
        """Open a tick for reporting. The actor only accepts reports for this id."""
        self.active_tick_id = tick_id

    def report_decision(
        self, tick_id: int, decision: str, task_latency_s: float = 0.0, used_subactors: bool = False
    ) -> str:
        """Async report path from the scoring task. Rejects (and counts) reports
        that are late (tick already finalized), inactive (not the open tick), or
        duplicate (already reported this tick)."""
        if tick_id in self.accepted_decisions or tick_id != self.active_tick_id:
            self.late_reports += 1
            return "late"
        if self.reported_tick == tick_id:
            self.duplicate_reports += 1
            return "duplicate"
        self.reported_tick = tick_id
        self.reported_decision = decision
        self.reported_latency = task_latency_s
        if used_subactors:
            self.subactor_ticks += 1
        return "accepted"

    def has_report(self, tick_id: int) -> bool:
        """Cheap readiness read the driver polls to decide when to close a tick."""
        return self.reported_tick == tick_id

    def finalize_tick(self, tick_id: int) -> dict:
        """Commit the tick using the reported decision, or the fallback if none
        arrived in time. Idempotent: re-finalizing a closed tick is a no-op."""
        if tick_id in self.accepted_decisions:
            return self.accepted_decisions[tick_id]

        if self.reported_tick == tick_id and self.reported_decision is not None:
            entry = {
                "decision": self.reported_decision,
                "used_fallback": False,
                "task_latency_s": self.reported_latency,
            }
            self.consecutive_fallbacks = 0
        else:
            entry = {"decision": self._fallback_decision(), "used_fallback": True, "task_latency_s": None}
            self.fallbacks += 1
            self.consecutive_fallbacks += 1
            self._maybe_promote()

        self.accepted_decisions[tick_id] = entry
        self.last_decision = entry["decision"]
        self.reported_tick = None
        self.reported_decision = None
        self.reported_latency = None
        return entry

    def _maybe_promote(self) -> None:
        """A repeat straggler that has missed subactor_trigger ticks in a row
        spawns its helper subactors once and becomes a dispatcher zone."""
        if self.subactor_trigger <= 0 or self.promoted:
            return
        if self.consecutive_fallbacks >= self.subactor_trigger:
            self.helpers = [ScoreHelper.remote() for _ in range(self.n_helpers)]
            self.promoted = True

    def _fallback_decision(self) -> str:
        """previous_else_ok: reuse the last accepted decision, defaulting to OK on
        the first tick when no previous decision exists."""
        if self.last_decision is not None:
            return self.last_decision
        return "OK"


def decide(current_pickups: float, baseline_pickups: float, need_threshold: float) -> str:
    """Pure decision rule: NEED when current demand is at least need_threshold
    times the recent norm. With no usable baseline, fall back to OK."""
    if baseline_pickups <= 0:
        return "OK"
    return "NEED" if current_pickups / baseline_pickups >= need_threshold else "OK"


def run_score(snapshot: dict, need_threshold: float, sleep_s: float) -> dict:
    """Shared scoring body. The sleep simulates a slow zone and only inflates
    latency; it never changes the decision (deterministic from the snapshot
    input, so retries are safe)."""
    start = time.perf_counter()
    if sleep_s > 0:
        time.sleep(sleep_s)
    decision = decide(snapshot["current_pickups"], snapshot["baseline_pickups"], need_threshold)
    return {
        "zone_id": snapshot["zone_id"],
        "tick_id": snapshot["tick_id"],
        "decision": decision,
        "task_latency_s": time.perf_counter() - start,
    }


@ray.remote
def score_zone(snapshot: dict, need_threshold: float, sleep_s: float) -> dict:
    """Blocking-mode task: compute and return the decision to the driver."""
    return run_score(snapshot, need_threshold, sleep_s)


@ray.remote
def score_and_report(
    snapshot: dict, actor: ActorHandle, need_threshold: float, sleep_s: float
) -> dict:
    """Async-mode task: compute the decision and report it directly to the owning
    actor. A late report (tick already closed) is rejected by the actor."""
    result = run_score(snapshot, need_threshold, sleep_s)
    ray.get(
        actor.report_decision.remote(
            result["tick_id"], result["decision"], result["task_latency_s"]
        )
    )
    return result


def split_int(total: int, k: int) -> list[int]:
    """Split a non-negative integer into k parts that sum back to total."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


@ray.remote
class ScoreHelper:
    """Subactor owned by a promoted ZoneActor. Processes one shard of a tick's
    demand, simulating heavy per-shard work via sleep, and returns its partial."""

    def score_shard(self, shard_value: int, sleep_s: float) -> int:
        if sleep_s > 0:
            time.sleep(sleep_s)
        return shard_value


@ray.remote
def score_with_helpers(
    snapshot: dict, actor: ActorHandle, helpers: list, need_threshold: float, sleep_s: float
) -> dict:
    """Promoted-zone task: fan the tick's demand out as shards to the zone's
    helper subactors (each sleeps sleep_s/len(helpers), in parallel), sum the
    partials -> identical current_pickups -> identical decision, then report.
    The fan-out runs in this task (not the actor) so actor methods stay poll-fast."""
    start = time.perf_counter()
    shards = split_int(int(snapshot["current_pickups"]), len(helpers))
    per_helper_sleep = sleep_s / len(helpers) if sleep_s > 0 else 0.0
    partial_refs = [
        helper.score_shard.remote(shard, per_helper_sleep)
        for helper, shard in zip(helpers, shards)
    ]
    current = sum(ray.get(partial_refs))
    decision = decide(current, snapshot["baseline_pickups"], need_threshold)
    latency = time.perf_counter() - start
    ray.get(actor.report_decision.remote(snapshot["tick_id"], decision, latency, True))
    return {
        "zone_id": snapshot["zone_id"],
        "tick_id": snapshot["tick_id"],
        "decision": decision,
        "task_latency_s": latency,
    }


def select_active_zones(reference_df: pd.DataFrame, n_zones: int) -> list[int]:
    """Pick the busiest pickup zones in the reference month.

    Ranking is total pickups descending, with ties broken by lowest zone_id, so
    the selection is fully reproducible for the same inputs.
    """
    counts = reference_df[ZONE_COL].value_counts().reset_index(name="pickups")
    ranked = counts.sort_values(by=["pickups", ZONE_COL], ascending=[False, True])
    top_zones = ranked.head(n_zones)[ZONE_COL]
    return sorted(int(zone_id) for zone_id in top_zones)


def build_baseline_table(reference_df: pd.DataFrame, active_zones: list[int]) -> pd.DataFrame:
    """Collapse the reference month into a per-(zone, hour_of_day, day_of_week)
    'normal demand' table, in units of typical pickups per 15-minute window.
    """
    active = reference_df[reference_df[ZONE_COL].isin(active_zones)].copy()
    active["tick_start"] = active[PICKUP_COL].dt.floor("15min")

    per_tick = active.groupby([ZONE_COL, "tick_start"]).size().reset_index(name="pickups")
    per_tick["hour_of_day"] = per_tick["tick_start"].dt.hour
    per_tick["day_of_week"] = per_tick["tick_start"].dt.dayofweek

    baseline = (
        per_tick.groupby([ZONE_COL, "hour_of_day", "day_of_week"])["pickups"]
        .mean()
        .reset_index(name="baseline_pickups")
    )
    return baseline


def build_replay_table(
    replay_df: pd.DataFrame, active_zones: list[int], replay_month: pd.Period
) -> pd.DataFrame:
    """Build the time-ordered replay stream: one row per observed (zone_id, tick).

    tick_id is a global index anchored to the month's first 15-minute window, so
    the same tick_id means the same wall-clock window for every zone. Windows
    with no pickups are simply absent (treated as zero demand at runtime).
    """
    active = replay_df[replay_df[ZONE_COL].isin(active_zones)].copy()
    active["tick_start"] = active[PICKUP_COL].dt.floor("15min")

    replay = active.groupby([ZONE_COL, "tick_start"]).size().reset_index(name="pickups")

    month_start = replay_month.start_time
    steps = (replay["tick_start"] - month_start) // pd.Timedelta("15min")
    replay["tick_id"] = steps.astype(int)
    replay["hour_of_day"] = replay["tick_start"].dt.hour
    replay["day_of_week"] = replay["tick_start"].dt.dayofweek

    columns = [ZONE_COL, "tick_id", "tick_start", "hour_of_day", "day_of_week", "pickups"]
    return replay[columns].sort_values(["tick_id", ZONE_COL]).reset_index(drop=True)


def crosscheck_replay(
    replay_df: pd.DataFrame,
    replay_table: pd.DataFrame,
    replay_month: pd.Period,
    sample_size: int,
    seed: int,
) -> None:
    """Independently re-count a few sampled (zone, tick) windows directly from the
    cleaned rows and assert they match the prepared replay counts exactly.
    """
    month_start = replay_month.start_time
    window = pd.Timedelta("15min")
    sample = replay_table.sample(n=min(sample_size, len(replay_table)), random_state=seed)

    for record in sample.to_dict("records"):
        start = month_start + record["tick_id"] * window
        in_window = (
            (replay_df[ZONE_COL] == record[ZONE_COL])
            & (replay_df[PICKUP_COL] >= start)
            & (replay_df[PICKUP_COL] < start + window)
        )
        direct_count = int(in_window.sum())
        if direct_count != record["pickups"]:
            raise AssertionError(
                f"cross-check failed for zone {record[ZONE_COL]} tick {record['tick_id']}: "
                f"prepared={record['pickups']} direct={direct_count}"
            )

    print(f"cross-check passed on {len(sample)} sampled (zone, tick) windows")


def write_assets(
    output_dir: Path,
    baseline: pd.DataFrame,
    replay: pd.DataFrame,
    active_zones: list[int],
    n_ticks: int,
    reference_month: pd.Period,
    replay_month: pd.Period,
    seed: int,
) -> None:
    """Persist the prepared assets the runtime consumes: two parquet tables plus
    a JSON metadata file with the facts the runtime cannot recompute."""
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(output_dir / "baseline.parquet", index=False)
    replay.to_parquet(output_dir / "replay.parquet", index=False)

    metadata = {
        "active_zones": active_zones,
        "n_zones": len(active_zones),
        "n_ticks": n_ticks,
        "tick_minutes": TICK_MINUTES,
        "reference_month": str(reference_month),
        "replay_month": str(replay_month),
        "seed": seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"wrote assets to {output_dir}: baseline.parquet, replay.parquet, metadata.json")


def prepare_assets(
    reference_parquet: Path,
    replay_parquet: Path,
    output_dir: Path,
    n_zones: int,
    seed: int,
) -> None:
    reference_df, reference_month = load_and_clean_month(reference_parquet)
    replay_df, replay_month = load_and_clean_month(replay_parquet)
    validate_adjacent_months(reference_month, replay_month)

    print(f"reference month {reference_month}: {len(reference_df)} cleaned rows, "
          f"{reference_df[ZONE_COL].nunique()} zones")
    print(f"replay month    {replay_month}: {len(replay_df)} cleaned rows, "
          f"{replay_df[ZONE_COL].nunique()} zones")

    active_zones = select_active_zones(reference_df, n_zones)
    print(f"selected {len(active_zones)} active zones: {active_zones}")

    baseline = build_baseline_table(reference_df, active_zones)
    print(f"baseline table: {len(baseline)} (zone, hour, dow) cells")
    print(baseline.head(6).to_string(index=False))

    replay = build_replay_table(replay_df, active_zones, replay_month)
    n_ticks = int(replay["tick_id"].max()) + 1
    print(f"replay table: {len(replay)} (zone, tick) rows across {n_ticks} global ticks")
    print(replay.head(6).to_string(index=False))

    crosscheck_replay(replay_df, replay, replay_month, sample_size=3, seed=seed)

    write_assets(
        output_dir, baseline, replay, active_zones, n_ticks, reference_month, replay_month, seed
    )


def load_prepared(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load the prepared assets the runtime consumes instead of raw parquet."""
    baseline = pd.read_parquet(prepared_dir / "baseline.parquet")
    replay = pd.read_parquet(prepared_dir / "replay.parquet")
    metadata = json.loads((prepared_dir / "metadata.json").read_text())
    return baseline, replay, metadata


def build_zone_lookups(
    replay: pd.DataFrame, baseline: pd.DataFrame, zone_id: int
) -> tuple[dict[int, int], dict[tuple[int, int], float]]:
    """Slice the two prepared tables down to one zone and turn them into the
    keyed lookups the actor owns."""
    zone_replay = replay[replay[ZONE_COL] == zone_id]
    pickups_by_tick = {
        int(tick_id): int(pickups)
        for tick_id, pickups in zip(zone_replay["tick_id"], zone_replay["pickups"])
    }
    zone_baseline = baseline[baseline[ZONE_COL] == zone_id]
    baseline_by_slot = {
        (int(hour), int(dow)): float(value)
        for hour, dow, value in zip(
            zone_baseline["hour_of_day"],
            zone_baseline["day_of_week"],
            zone_baseline["baseline_pickups"],
        )
    }
    return pickups_by_tick, baseline_by_slot


def select_zone_delays(zone_ids: list[int], base_delay: int, spread: int, seed: int) -> dict[int, int]:
    """Per-zone arrival delay. spread=0 gives every zone the same base delay (a
    system-wide delay); spread>0 adds a seeded per-zone jitter, reproducible for
    a given seed."""
    rng = random.Random(seed)
    return {
        z: base_delay + (rng.randint(0, spread) if spread > 0 else 0)
        for z in sorted(zone_ids)
    }


def create_actors(
    baseline: pd.DataFrame, replay: pd.DataFrame, metadata: dict, args: argparse.Namespace
) -> dict[int, ActorHandle]:
    """Create one ZoneActor per active zone, each owning its own replay partition
    and baseline slice, plus its (reproducible) delayed-arrival config."""
    month_start = pd.Period(metadata["replay_month"], "M").start_time
    tick_minutes = metadata["tick_minutes"]
    zone_delays = select_zone_delays(
        metadata["active_zones"], args.arrival_delay_ticks, args.delay_spread, args.seed
    )
    actors = {}
    for zone_id in metadata["active_zones"]:
        pickups_by_tick, baseline_by_slot = build_zone_lookups(replay, baseline, zone_id)
        actors[zone_id] = ZoneActor.remote(
            zone_id,
            pickups_by_tick,
            baseline_by_slot,
            month_start,
            tick_minutes,
            withhold_fraction=args.withhold_fraction,
            arrival_delay_ticks=zone_delays[zone_id],
            subactor_trigger=args.subactor_trigger if args.use_subactors else 0,
            n_helpers=args.n_helpers,
        )
    return actors


def resolve_tick_range(metadata: dict, start_tick: int, max_ticks: int | None) -> range:
    """Resolve the replay window so demos can target a specific time slice."""
    n_ticks = metadata["n_ticks"]
    if start_tick < 0 or start_tick >= n_ticks:
        raise ValueError(f"start_tick must be in [0, {n_ticks - 1}], got {start_tick}")
    end_tick = min(n_ticks, start_tick + max_ticks) if max_ticks else n_ticks
    return range(start_tick, end_tick)


def select_slow_zones(zone_ids: list[int], slow_zone_fraction: float, seed: int) -> set[int]:
    """Deterministically pick which zones are stragglers. Same seed + same zones
    always yields the same slow set."""
    n_slow = round(len(zone_ids) * slow_zone_fraction)
    rng = random.Random(seed)
    return set(rng.sample(sorted(zone_ids), n_slow))


def build_tick_records(
    accepted_by_zone: dict[int, dict], tick_ids: list[int], tick_latencies: list[float]
) -> list[dict]:
    """Per-tick metrics derived from actor-owned accepted state plus the driver's
    measured tick latencies. Fallback zones carry no task latency."""
    records = []
    for tick_id, tick_latency in zip(tick_ids, tick_latencies):
        entries = [decisions[tick_id] for decisions in accepted_by_zone.values() if tick_id in decisions]
        latencies = [
            e["task_latency_s"] for e in entries
            if not e["used_fallback"] and e["task_latency_s"] is not None
        ]
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
        max_lat = max(latencies) if latencies else 0.0
        records.append(
            {
                "tick_id": tick_id,
                "tick_latency_s": tick_latency,
                "mean_zone_latency_s": mean_lat,
                "max_zone_latency_s": max_lat,
                "max_mean_ratio": (max_lat / mean_lat) if mean_lat > 0 else 0.0,
                "zones_completed": sum(1 for e in entries if not e["used_fallback"]),
                "zones_fallback": sum(1 for e in entries if e["used_fallback"]),
                "need": sum(1 for e in entries if e["decision"] == "NEED"),
                "ok": sum(1 for e in entries if e["decision"] == "OK"),
            }
        )
    return records


def build_latency_log(accepted_by_zone: dict[int, dict]) -> dict:
    """Per-(tick, zone) task latency for completed reports."""
    log: dict[str, dict] = {}
    for zone_id, decisions in accepted_by_zone.items():
        for tick_id, entry in decisions.items():
            log.setdefault(str(tick_id), {})[str(zone_id)] = entry["task_latency_s"]
    return log


def write_run_artifacts(output_dir, args, mode, slow_zones, tick_records, latency_log, counters, total_s) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "mode": mode,
        "seed": args.seed,
        "need_threshold": args.need_threshold,
        "start_tick": args.start_tick,
        "max_ticks": args.max_ticks,
        "max_inflight_zones": args.max_inflight_zones,
        "tick_timeout_s": args.tick_timeout_s,
        "completion_fraction": args.completion_fraction,
        "slow_zone_fraction": args.slow_zone_fraction,
        "slow_zone_sleep_s": args.slow_zone_sleep_s,
        "fallback_policy": args.fallback_policy,
        "withhold_fraction": args.withhold_fraction,
        "arrival_delay_ticks": args.arrival_delay_ticks,
        "delay_spread": args.delay_spread,
        "use_subactors": args.use_subactors,
        "subactor_trigger": args.subactor_trigger,
        "n_helpers": args.n_helpers,
        "slow_zones": sorted(slow_zones),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    pd.DataFrame(tick_records).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "latency_log.json").write_text(json.dumps(latency_log, indent=2))

    summary = {
        "mode": mode,
        "n_ticks": len(tick_records),
        "total_s": total_s,
        "late_reports": sum(c["late_reports"] for c in counters),
        "duplicate_reports": sum(c["duplicate_reports"] for c in counters),
        "fallbacks": sum(c["fallbacks"] for c in counters),
        "withheld_total": sum(c["withheld_total"] for c in counters),
        "released_total": sum(c["released_total"] for c in counters),
        "unreleased_total": sum(c["unreleased_total"] for c in counters),
        "promoted_zones": sum(c["promoted"] for c in counters),
        "subactor_ticks": sum(c["subactor_ticks"] for c in counters),
        "ticks": tick_records,
    }
    (output_dir / "tick_summary.json").write_text(json.dumps(summary, indent=2))


def finalize_run(output_dir, args, mode, zone_ids, actors, slow_zones, tick_ids, tick_latencies, total_s) -> None:
    """Collect actor state, build metrics, write the four artifacts, and print a
    one-line summary. Shared by blocking and async so artifacts are identical."""
    accepted_states = ray.get([actors[z].accepted.remote() for z in zone_ids])
    counters = ray.get([actors[z].counters.remote() for z in zone_ids])
    accepted_by_zone = dict(zip(zone_ids, accepted_states))

    tick_records = build_tick_records(accepted_by_zone, tick_ids, tick_latencies)
    latency_log = build_latency_log(accepted_by_zone)
    write_run_artifacts(output_dir, args, mode, slow_zones, tick_records, latency_log, counters, total_s)

    need = sum(r["need"] for r in tick_records)
    decisions = sum(r["need"] + r["ok"] for r in tick_records)
    totals = (sum(c["fallbacks"] for c in counters), sum(c["late_reports"] for c in counters),
              sum(c["duplicate_reports"] for c in counters))
    print(
        f"{mode}: {len(tick_records)} ticks, {decisions} decisions ({need} NEED), "
        f"total {total_s:.2f}s, mean tick {sum(tick_latencies) / len(tick_latencies) * 1000:.1f}ms, "
        f"fallbacks={totals[0]} late={totals[1]} dup={totals[2]} -> {output_dir}"
    )


def run_blocking(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    baseline, replay, metadata = load_prepared(prepared_dir)
    actors = create_actors(baseline, replay, metadata, args)
    zone_ids = metadata["active_zones"]
    tick_range = resolve_tick_range(metadata, args.start_tick, args.max_ticks)
    slow_zones = select_slow_zones(zone_ids, args.slow_zone_fraction, args.seed)
    print(f"blocking: {len(slow_zones)} slow zones {sorted(slow_zones)} sleep {args.slow_zone_sleep_s}s")

    tick_ids = []
    tick_latencies = []
    run_start = time.perf_counter()
    for tick_id in tick_range:
        tick_start = time.perf_counter()

        snapshots = ray.get([actors[z].next_snapshot.remote(tick_id) for z in zone_ids])
        result_refs = [
            score_zone.remote(
                snap,
                args.need_threshold,
                args.slow_zone_sleep_s if zone_id in slow_zones else 0.0,
            )
            for zone_id, snap in zip(zone_ids, snapshots)
        ]
        results = ray.get(result_refs)
        ray.get(
            [
                actors[r["zone_id"]].write_decision.remote(
                    r["tick_id"], r["decision"], r["task_latency_s"]
                )
                for r in results
            ]
        )

        tick_ids.append(tick_id)
        tick_latencies.append(time.perf_counter() - tick_start)

    total_s = time.perf_counter() - run_start
    finalize_run(output_dir, args, "blocking", zone_ids, actors, slow_zones, tick_ids, tick_latencies, total_s)


def _scoring_ref(actors, zone_id, snapshot, sleep_s, args, helpers_by_zone):
    """A promoted zone fans out to its helper subactors; everyone else uses the
    normal single scoring task."""
    helpers = helpers_by_zone.get(zone_id)
    if helpers:
        return score_with_helpers.remote(
            snapshot, actors[zone_id], helpers, args.need_threshold, sleep_s
        )
    return score_and_report.remote(snapshot, actors[zone_id], args.need_threshold, sleep_s)


def submit_scoring_bounded(actors, zone_ids, snapshots, slow_zones, args, helpers_by_zone) -> None:
    """Submit async scoring tasks while keeping at most max_inflight_zones in
    flight (ray.wait throttles submission). Tasks report straight to actors."""
    pending = []
    for zone_id, snapshot in zip(zone_ids, snapshots):
        if len(pending) >= args.max_inflight_zones:
            _, pending = ray.wait(pending, num_returns=1)
        sleep_s = args.slow_zone_sleep_s if zone_id in slow_zones else 0.0
        pending.append(_scoring_ref(actors, zone_id, snapshot, sleep_s, args, helpers_by_zone))


def refresh_promotions(actors, slow_zones, helpers_by_zone) -> None:
    """After a tick, pick up helper handles for any slow zone that just promoted
    itself, so subsequent ticks route through its subactors."""
    pending_zones = [z for z in slow_zones if z not in helpers_by_zone]
    if not pending_zones:
        return
    handles = ray.get([actors[z].helper_handles.remote() for z in pending_zones])
    for zone_id, helpers in zip(pending_zones, handles):
        if helpers:
            helpers_by_zone[zone_id] = helpers


def poll_until_ready(actors, zone_ids, tick_id, args) -> None:
    """Poll actor readiness; return once completion_fraction of zones have
    reported or tick_timeout_s elapses."""
    needed = math.ceil(args.completion_fraction * len(zone_ids))
    deadline = time.perf_counter() + args.tick_timeout_s
    while True:
        ready = sum(ray.get([actors[z].has_report.remote(tick_id) for z in zone_ids]))
        if ready >= needed or time.perf_counter() >= deadline:
            return
        time.sleep(0.01)


def run_async(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    baseline, replay, metadata = load_prepared(prepared_dir)
    actors = create_actors(baseline, replay, metadata, args)
    zone_ids = metadata["active_zones"]
    tick_range = resolve_tick_range(metadata, args.start_tick, args.max_ticks)
    slow_zones = select_slow_zones(zone_ids, args.slow_zone_fraction, args.seed)
    print(f"async: {len(slow_zones)} slow zones {sorted(slow_zones)} sleep {args.slow_zone_sleep_s}s")

    helpers_by_zone: dict[int, list] = {}
    tick_ids = []
    tick_latencies = []
    run_start = time.perf_counter()
    for tick_id in tick_range:
        tick_start = time.perf_counter()

        ray.get([actors[z].mark_tick_active.remote(tick_id) for z in zone_ids])
        snapshots = ray.get([actors[z].next_snapshot.remote(tick_id) for z in zone_ids])
        submit_scoring_bounded(actors, zone_ids, snapshots, slow_zones, args, helpers_by_zone)
        poll_until_ready(actors, zone_ids, tick_id, args)
        ray.get([actors[z].finalize_tick.remote(tick_id) for z in zone_ids])
        if args.use_subactors:
            refresh_promotions(actors, slow_zones, helpers_by_zone)

        tick_ids.append(tick_id)
        tick_latencies.append(time.perf_counter() - tick_start)

    total_s = time.perf_counter() - run_start
    finalize_run(output_dir, args, "async", zone_ids, actors, slow_zones, tick_ids, tick_latencies, total_s)


def run_stress(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Reuse the async controller under aggressively harsher skew: more slow
    zones, longer stalls. The async path should still progress via fallbacks
    while a blocking run on the same settings would degrade sharply."""
    args.slow_zone_fraction = max(args.slow_zone_fraction, 0.5)
    args.slow_zone_sleep_s = max(args.slow_zone_sleep_s, 3.0)
    print(
        f"stress: escalated skew -> fraction={args.slow_zone_fraction} "
        f"sleep={args.slow_zone_sleep_s}s"
    )
    run_async(prepared_dir, output_dir, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capstone starter for TLC-backed per-zone recommendations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-parquet", type=Path, required=True)
    prepare.add_argument("--replay-parquet", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--n-zones", type=int, default=15)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.set_defaults(handler=handle_prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--prepared-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--mode", choices=("blocking", "async", "stress"), required=True)
    run.add_argument("--max-inflight-zones", type=int, default=4)
    run.add_argument("--tick-timeout-s", type=float, default=2.0)
    run.add_argument("--completion-fraction", type=float, default=0.75)
    run.add_argument("--slow-zone-fraction", type=float, default=0.25)
    run.add_argument("--slow-zone-sleep-s", type=float, default=1.0)
    run.add_argument("--fallback-policy", default="previous_else_ok")
    run.add_argument("--need-threshold", type=float, default=NEED_THRESHOLD)
    run.add_argument("--withhold-fraction", type=float, default=0.0)
    run.add_argument("--arrival-delay-ticks", type=int, default=0)
    run.add_argument("--delay-spread", type=int, default=0)
    run.add_argument("--use-subactors", action="store_true")
    run.add_argument("--subactor-trigger", type=int, default=3)
    run.add_argument("--n-helpers", type=int, default=3)
    run.add_argument("--start-tick", type=int, default=0)
    run.add_argument("--max-ticks", type=int, default=96)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--ray-address", default=None)
    run.set_defaults(handler=handle_run)

    return parser


def handle_prepare(args: argparse.Namespace) -> None:
    prepare_assets(
        args.reference_parquet, args.replay_parquet, args.output_dir, args.n_zones, args.seed
    )


def handle_run(args: argparse.Namespace) -> None:
    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    if args.mode == "blocking":
        run_blocking(args.prepared_dir, args.output_dir, args)
    elif args.mode == "async":
        run_async(args.prepared_dir, args.output_dir, args)
    else:
        run_stress(args.prepared_dir, args.output_dir, args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
