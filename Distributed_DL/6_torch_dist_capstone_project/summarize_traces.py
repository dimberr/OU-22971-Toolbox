"""Summarize named capstone profiler spans from Chrome trace files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SPAN_NAMES = {
    "prepare_views",
    "stage0_forward",
    "send_boundary",
    "recv_boundary",
    "stage1_forward",
    "gather_embeddings",
    "loss_calculation",
    "send_boundary_grad",
    "recv_boundary_grad",
    "stage0_backward",
    "grad_sync_stage0",
    "grad_sync_stage1",
    "optimizer_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    return parser.parse_args()


def summarize_trace(path: Path) -> list[dict[str, float | int | str]]:
    events = json.loads(path.read_text())["traceEvents"]
    totals: dict[str, list[float]] = defaultdict(list)
    for event in events:
        name = event.get("name")
        duration = event.get("dur")
        if name in SPAN_NAMES and isinstance(duration, (int, float)):
            totals[name].append(duration / 1_000)
    rank = int(path.stem.removeprefix("trace_rank"))
    return [
        {
            "rank": rank,
            "span": name,
            "count": len(durations),
            "total_ms": sum(durations),
            "mean_ms": sum(durations) / len(durations),
        }
        for name, durations in sorted(totals.items())
    ]


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.run_directory.glob("trace_rank*.json")):
        rows.extend(summarize_trace(path))
    if not rows:
        raise SystemExit(f"No trace files found in {args.run_directory}")
    output_path = args.run_directory / "trace_summary.csv"
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
