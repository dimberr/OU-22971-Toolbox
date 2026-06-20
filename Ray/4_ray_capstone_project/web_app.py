from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import struct
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
from ray.job_submission import JobSubmissionClient

APP_ROOT = Path("/app")
PREPARED_DIR = APP_ROOT / "prepared"
RUNS_DIR = APP_ROOT / "runs"
TLC_DIR = APP_ROOT / "tlc"
TLC_DATA_DIR = Path("/tlc-data")
TLC_DIRS = [TLC_DATA_DIR, TLC_DIR]
ZONE_COL = "PULocationID"


PRESETS: list[dict[str, Any]] = [
    {
        "id": "blocking",
        "title": "Blocking Baseline",
        "description": "Waits for every zone. Tick latency follows the slowest selected zones.",
        "label": "blocking",
        "mode": "blocking",
        "params": {"max_ticks": 20, "slow_zone_fraction": 0.25, "slow_zone_sleep_s": 1.0},
    },
    {
        "id": "async",
        "title": "Async Partial Readiness",
        "description": "Closes ticks once enough actors report, then applies deterministic fallbacks.",
        "label": "async",
        "mode": "async",
        "params": {
            "max_ticks": 20,
            "slow_zone_fraction": 0.25,
            "slow_zone_sleep_s": 1.0,
            "completion_fraction": 0.75,
            "tick_timeout_s": 2.0,
        },
    },
    {
        "id": "stress",
        "title": "Async Stress",
        "description": "Escalates skew internally and demonstrates forward progress via fallbacks.",
        "label": "stress",
        "mode": "stress",
        "params": {"max_ticks": 20},
    },
    {
        "id": "blocking_harsh",
        "title": "Blocking Harsh Skew",
        "description": "Same harsh skew as stress, but blocking waits for the slowest zones.",
        "label": "blocking_harsh",
        "mode": "blocking",
        "params": {"max_ticks": 20, "slow_zone_fraction": 0.5, "slow_zone_sleep_s": 3.0},
    },
    {
        "id": "delay",
        "title": "Delayed Arrivals",
        "description": "Withholds demand and releases it later to show decision mis-timing.",
        "label": "delay",
        "mode": "blocking",
        "params": {"max_ticks": 96, "withhold_fraction": 0.5, "arrival_delay_ticks": 3},
    },
    {
        "id": "subactors",
        "title": "Adaptive Subactors",
        "description": "Repeat stragglers promote helper subactors and recover latency.",
        "label": "subactors",
        "mode": "async",
        "params": {
            "max_ticks": 12,
            "slow_zone_fraction": 0.25,
            "slow_zone_sleep_s": 1.5,
            "tick_timeout_s": 1.0,
            "completion_fraction": 1.0,
            "use_subactors": True,
            "subactor_trigger": 3,
            "n_helpers": 3,
        },
    },
]


PARAM_SPECS: dict[str, tuple[str, type]] = {
    "max_inflight_zones": ("--max-inflight-zones", int),
    "tick_timeout_s": ("--tick-timeout-s", float),
    "completion_fraction": ("--completion-fraction", float),
    "slow_zone_fraction": ("--slow-zone-fraction", float),
    "slow_zone_sleep_s": ("--slow-zone-sleep-s", float),
    "fallback_policy": ("--fallback-policy", str),
    "need_threshold": ("--need-threshold", float),
    "withhold_fraction": ("--withhold-fraction", float),
    "arrival_delay_ticks": ("--arrival-delay-ticks", int),
    "delay_spread": ("--delay-spread", int),
    "subactor_trigger": ("--subactor-trigger", int),
    "n_helpers": ("--n-helpers", int),
    "start_tick": ("--start-tick", int),
    "max_ticks": ("--max-ticks", int),
    "seed": ("--seed", int),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(handler: BaseHTTPRequestHandler, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_html(handler: BaseHTTPRequestHandler, body: str) -> None:
    payload = body.encode()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def load_request(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length))


def valid_label(label: str) -> str:
    cleaned = label.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", cleaned):
        raise ValueError("label must contain only letters, numbers, dot, underscore, or dash")
    return cleaned


def coerce_params(raw: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, (_, caster) in PARAM_SPECS.items():
        value = raw.get(key)
        if value in (None, ""):
            continue
        params[key] = str(value) if caster is str else caster(value)
    if raw.get("use_subactors"):
        params["use_subactors"] = True
    return params


def build_entrypoint(label: str, mode: str, params: dict[str, Any]) -> str:
    args = [
        "python",
        "main.py",
        "run",
        "--prepared-dir",
        str(PREPARED_DIR),
        "--output-dir",
        str(RUNS_DIR / label),
        "--mode",
        mode,
    ]
    for key, value in params.items():
        if key == "use_subactors":
            args.append("--use-subactors")
            continue
        flag, _ = PARAM_SPECS[key]
        args.extend([flag, str(value)])
    return shlex.join(args)


def submit_run(client: JobSubmissionClient, request: dict[str, Any]) -> dict[str, Any]:
    mode = request.get("mode")
    if mode not in {"blocking", "async", "stress"}:
        raise ValueError("mode must be blocking, async, or stress")
    label = valid_label(str(request.get("label") or mode))
    params = coerce_params(request.get("params", {}))
    entrypoint = build_entrypoint(label, mode, params)
    submission_id = f"{label}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env={"working_dir": str(APP_ROOT), "excludes": ["runs/", "tlc/", "__pycache__/", ".pytest_cache/"]},
        metadata={"label": label, "mode": mode},
        submission_id=submission_id,
    )
    return {"job_id": job_id, "submission_id": submission_id, "label": label, "entrypoint": entrypoint}


def list_jobs(client: JobSubmissionClient) -> list[dict[str, Any]]:
    jobs = client.list_jobs()
    return [job_to_dict(job) for job in jobs]


def job_to_dict(job: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(job) and not isinstance(job, type):
        return serializable(dataclasses.asdict(job))
    if isinstance(job, dict):
        return serializable(job)
    return {key: serializable(getattr(job, key)) for key in dir(job) if not key.startswith("_")}


def serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return str(value)


def load_runs() -> list[dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    runs = [load_run(path) for path in sorted(RUNS_DIR.iterdir()) if path.is_dir()]
    return [run for run in runs if run is not None]


def load_run(path: Path) -> dict[str, Any] | None:
    required = ["run_config.json", "metrics.csv", "latency_log.json", "tick_summary.json"]
    if not all((path / name).exists() for name in required):
        return None
    metrics = pd.read_csv(path / "metrics.csv").to_dict("records")
    return {
        "label": path.name,
        "config": read_json(path / "run_config.json"),
        "summary": compact_summary(read_json(path / "tick_summary.json")),
        "metrics": metrics,
        "latencyLog": read_json(path / "latency_log.json"),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k != "ticks"}


def load_prepared(max_ticks: int | None = None) -> dict[str, Any] | None:
    if not (PREPARED_DIR / "metadata.json").exists():
        return None
    metadata = read_json(PREPARED_DIR / "metadata.json")
    replay = pd.read_parquet(PREPARED_DIR / "replay.parquet")
    baseline = pd.read_parquet(PREPARED_DIR / "baseline.parquet")
    lookup = read_lookup(find_lookup_csv())
    return {
        "metadata": metadata,
        "zones": [zone_record(zone_id, lookup.get(zone_id, {})) for zone_id in metadata["active_zones"]],
        "frames": build_zone_frames(metadata, replay, baseline, max_ticks),
    }


def find_lookup_csv() -> Path | None:
    for folder in TLC_DIRS:
        candidates = [
            folder / "taxi_zone_lookup.csv",
            folder / "taxi+_zone_lookup.csv",
            folder / "Taxi Zone Lookup Table.csv",
        ]
        match = first_existing(candidates) or first_match(folder, "*.csv")
        if match:
            return match
    return None


def find_geometry_file() -> Path | None:
    for folder in TLC_DIRS:
        candidates = [
            folder / "taxi_zones.geojson",
            folder / "taxi_zones.json",
            folder / "taxi_zones.parquet",
            folder / "taxi_zone_shapes.parquet",
            folder / "taxi_zone_shapefile.parquet",
            folder / "taxi_zones" / "taxi_zones.shp",
        ]
        match = (
            first_existing(candidates)
            or first_match(folder, "*zone*.geojson")
            or first_match(folder, "*zone*.json")
            or first_match(folder, "*zone*.parquet")
            or first_match(folder, "*zone*.shp")
            or first_recursive_match(folder, "*zone*.shp")
        )
        if match:
            return match
    return None


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_match(folder: Path, pattern: str) -> Path | None:
    if not folder.exists():
        return None
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def first_recursive_match(folder: Path, pattern: str) -> Path | None:
    if not folder.exists():
        return None
    matches = sorted(folder.rglob(pattern))
    return matches[0] if matches else None


def read_lookup(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    lookup = pd.read_csv(path)
    id_col = first_column(lookup, ["LocationID", "location_id", "zone_id"])
    return {int(row[id_col]): {str(k): clean_cell(v) for k, v in row.items()} for row in lookup.to_dict("records")}


def first_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"missing expected column: {candidates}")


def clean_cell(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def zone_record(zone_id: int, details: dict[str, str]) -> dict[str, Any]:
    zone = details.get("Zone") or details.get("zone") or f"Zone {zone_id}"
    borough = details.get("Borough") or details.get("borough") or ""
    return {"zone_id": zone_id, "zone": zone, "borough": borough, "label": f"{borough} - {zone}" if borough else zone}


def build_zone_frames(metadata: dict[str, Any], replay: pd.DataFrame, baseline: pd.DataFrame, max_ticks: int | None) -> list[dict[str, Any]]:
    n_ticks = min(int(metadata["n_ticks"]), max_ticks) if max_ticks else int(metadata["n_ticks"])
    pickup_lookup = {(int(row[ZONE_COL]), int(row["tick_id"])): int(row["pickups"]) for row in replay.to_dict("records")}
    baseline_lookup = make_baseline_lookup(baseline)
    return [build_frame(tick_id, metadata, pickup_lookup, baseline_lookup) for tick_id in range(n_ticks)]


def make_baseline_lookup(baseline: pd.DataFrame) -> dict[tuple[int, int, int], float]:
    return {
        (int(row[ZONE_COL]), int(row["hour_of_day"]), int(row["day_of_week"])): float(row["baseline_pickups"])
        for row in baseline.to_dict("records")
    }


def build_frame(
    tick_id: int,
    metadata: dict[str, Any],
    pickup_lookup: dict[tuple[int, int], int],
    baseline_lookup: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    tick_start = tick_timestamp(metadata, tick_id)
    return {
        "tick_id": tick_id,
        "time": tick_start.isoformat(),
        "zones": {
            str(zone_id): zone_value(zone_id, tick_id, tick_start, pickup_lookup, baseline_lookup)
            for zone_id in metadata["active_zones"]
        },
    }


def tick_timestamp(metadata: dict[str, Any], tick_id: int) -> pd.Timestamp:
    start = pd.Period(metadata["replay_month"], "M").start_time
    return start + pd.Timedelta(minutes=int(metadata["tick_minutes"]) * tick_id)


def zone_value(
    zone_id: int,
    tick_id: int,
    tick_start: pd.Timestamp,
    pickup_lookup: dict[tuple[int, int], int],
    baseline_lookup: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    pickups = pickup_lookup.get((zone_id, tick_id), 0)
    baseline = baseline_lookup.get((zone_id, int(tick_start.hour), int(tick_start.dayofweek)), 0.0)
    return {"pickups": pickups, "baseline": round(baseline, 3), "ratio": pickups / baseline if baseline > 0 else None}


def load_geometry(prepared: dict[str, Any] | None) -> dict[str, Any] | None:
    geometry_path = find_geometry_file()
    if geometry_path is None or prepared is None:
        return None
    active = {int(zone["zone_id"]) for zone in prepared["zones"]}
    geometry = read_geometry(geometry_path)
    features = geometry.get("features", [])
    return {"path": str(geometry_path), "features": features, "activeZoneIds": sorted(active)}


def read_geometry(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".json", ".geojson"}:
        return read_json(path)
    if path.suffix.lower() == ".parquet":
        return read_geo_parquet(path)
    if path.suffix.lower() == ".shp":
        return read_shapefile(path)
    raise ValueError(f"unsupported geometry file: {path}")


def read_geo_parquet(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    id_col = first_column(frame, ["LocationID", "location_id", "zone_id", "OBJECTID", "objectid"])
    geometry_col = first_column(frame, ["geometry", "geom"])
    features = []
    for row in frame.to_dict("records"):
        geometry = parse_wkb(row[geometry_col])
        if geometry is not None:
            features.append(
                {
                    "type": "Feature",
                    "properties": {str(k): serializable(v) for k, v in row.items() if k != geometry_col},
                    "geometry": geometry,
                }
            )
    return {"type": "FeatureCollection", "features": features, "id_column": id_col}


def read_shapefile(path: Path) -> dict[str, Any]:
    geometries = read_shp_geometries(path)
    properties = read_dbf_records(path.with_suffix(".dbf"))
    features = []
    for index, geometry in enumerate(geometries):
        props = properties[index] if index < len(properties) else {}
        features.append({"type": "Feature", "properties": props, "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


def read_shp_geometries(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    offset = 100
    geometries = []
    while offset + 8 <= len(data):
        content_words = struct.unpack_from(">i", data, offset + 4)[0]
        content_start = offset + 8
        content_end = content_start + (content_words * 2)
        geometry = parse_shp_record(data[content_start:content_end])
        if geometry:
            geometries.append(geometry)
        offset = content_end
    return geometries


def parse_shp_record(record: bytes) -> dict[str, Any] | None:
    if len(record) < 44:
        return None
    shape_type = struct.unpack_from("<i", record, 0)[0]
    if shape_type == 0:
        return None
    if shape_type not in {5, 15, 25}:
        raise ValueError(f"unsupported shapefile geometry type: {shape_type}")
    part_count = struct.unpack_from("<i", record, 36)[0]
    point_count = struct.unpack_from("<i", record, 40)[0]
    parts_offset = 44
    points_offset = parts_offset + (part_count * 4)
    part_starts = list(struct.unpack_from(f"<{part_count}i", record, parts_offset))
    points = read_shp_points(record, points_offset, point_count)
    rings = build_shp_rings(points, part_starts)
    return {"type": "Polygon", "coordinates": rings}


def read_shp_points(record: bytes, offset: int, point_count: int) -> list[list[float]]:
    points = []
    for index in range(point_count):
        x, y = struct.unpack_from("<2d", record, offset + (index * 16))
        points.append([float(x), float(y)])
    return points


def build_shp_rings(points: list[list[float]], part_starts: list[int]) -> list[list[list[float]]]:
    ends = part_starts[1:] + [len(points)]
    return [points[start:end] for start, end in zip(part_starts, ends)]


def read_dbf_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    record_count = struct.unpack_from("<I", data, 4)[0]
    header_len = struct.unpack_from("<H", data, 8)[0]
    record_len = struct.unpack_from("<H", data, 10)[0]
    fields = read_dbf_fields(data)
    return [
        parse_dbf_record(data[header_len + (index * record_len): header_len + ((index + 1) * record_len)], fields)
        for index in range(record_count)
    ]


def read_dbf_fields(data: bytes) -> list[dict[str, Any]]:
    fields = []
    offset = 32
    while offset + 32 <= len(data) and data[offset] != 0x0D:
        descriptor = data[offset: offset + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        fields.append({"name": name, "type": chr(descriptor[11]), "length": descriptor[16]})
        offset += 32
    return fields


def parse_dbf_record(record: bytes, fields: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    offset = 1
    for field in fields:
        raw = record[offset: offset + field["length"]]
        values[field["name"]] = parse_dbf_value(raw, field["type"])
        offset += field["length"]
    return values


def parse_dbf_value(raw: bytes, field_type: str) -> Any:
    text = raw.decode("latin1", errors="ignore").strip()
    if text == "":
        return None
    if field_type in {"N", "F"}:
        return float(text) if "." in text else int(text)
    if field_type == "L":
        return text.upper() in {"Y", "T"}
    return text


def parse_wkb(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    data = bytes(value)
    geometry, _ = parse_wkb_at(data, 0)
    return geometry


def parse_wkb_at(data: bytes, offset: int) -> tuple[dict[str, Any] | None, int]:
    endian, offset = read_endian(data, offset)
    raw_type, offset = read_uint32(data, offset, endian)
    base_type, dimensions, offset = normalize_wkb_type(raw_type, data, offset, endian)
    if base_type == 3:
        return parse_polygon(data, offset, endian, dimensions)
    if base_type == 6:
        return parse_multipolygon(data, offset, endian)
    return None, len(data)


def read_endian(data: bytes, offset: int) -> tuple[str, int]:
    byte_order = data[offset]
    return ("<" if byte_order == 1 else ">"), offset + 1


def read_uint32(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    return struct.unpack_from(f"{endian}I", data, offset)[0], offset + 4


def read_point(data: bytes, offset: int, endian: str, dimensions: int) -> tuple[list[float], int]:
    values = struct.unpack_from(f"{endian}{dimensions}d", data, offset)
    return [float(values[0]), float(values[1])], offset + (8 * dimensions)


def normalize_wkb_type(raw_type: int, data: bytes, offset: int, endian: str) -> tuple[int, int, int]:
    has_srid = bool(raw_type & 0x20000000)
    has_z = bool(raw_type & 0x80000000)
    clean_type = raw_type & 0xFFFF
    dimensions = 3 if has_z or clean_type in range(1000, 4000) else 2
    base_type = clean_type % 1000
    if has_srid:
        _, offset = read_uint32(data, offset, endian)
    return base_type, dimensions, offset


def parse_polygon(data: bytes, offset: int, endian: str, dimensions: int) -> tuple[dict[str, Any], int]:
    ring_count, offset = read_uint32(data, offset, endian)
    rings = []
    for _ in range(ring_count):
        ring, offset = parse_ring(data, offset, endian, dimensions)
        rings.append(ring)
    return {"type": "Polygon", "coordinates": rings}, offset


def parse_multipolygon(data: bytes, offset: int, endian: str) -> tuple[dict[str, Any], int]:
    polygon_count, offset = read_uint32(data, offset, endian)
    polygons = []
    for _ in range(polygon_count):
        polygon, offset = parse_wkb_at(data, offset)
        if polygon and polygon["type"] == "Polygon":
            polygons.append(polygon["coordinates"])
    return {"type": "MultiPolygon", "coordinates": polygons}, offset


def parse_ring(data: bytes, offset: int, endian: str, dimensions: int) -> tuple[list[list[float]], int]:
    point_count, offset = read_uint32(data, offset, endian)
    points = []
    for _ in range(point_count):
        point, offset = read_point(data, offset, endian, dimensions)
        points.append(point)
    return points, offset


def feature_zone_id(feature: dict[str, Any]) -> int | None:
    props = feature.get("properties", {})
    for key in ("LocationID", "location_id", "zone_id", "OBJECTID", "objectid"):
        if key in props and props[key] is not None:
            return int(props[key])
    return None


def app_state(ray_address: str) -> dict[str, Any]:
    client, ray = connect_to_ray(ray_address)
    prepared = load_prepared()
    return {
        "ray": ray,
        "presets": PRESETS,
        "jobs": safe_jobs(client),
        "runs": load_runs(),
        "prepared": prepared,
        "geometry": safe_geometry(prepared),
        "tlc": tlc_state(),
    }


def connect_to_ray(ray_address: str) -> tuple[JobSubmissionClient | None, dict[str, Any]]:
    try:
        client = JobSubmissionClient(ray_address)
        return client, {"ready": True, "address": client.get_address(), "version": serializable(client.get_version())}
    except Exception as exc:
        return None, {"ready": False, "address": ray_address, "error": str(exc)}


def safe_jobs(client: JobSubmissionClient | None) -> list[dict[str, Any]]:
    if client is None:
        return []
    try:
        return list_jobs(client)
    except Exception:
        return []


def safe_geometry(prepared: dict[str, Any] | None) -> dict[str, Any] | None:
    try:
        return load_geometry(prepared)
    except Exception as exc:
        return {"error": str(exc), "features": []}


def tlc_state() -> dict[str, Any]:
    lookup = find_lookup_csv()
    geometry = find_geometry_file()
    return {
        "folders": [str(folder) for folder in TLC_DIRS],
        "lookup": str(lookup) if lookup else None,
        "geometry": str(geometry) if geometry else None,
    }


def get_logs(client: JobSubmissionClient, job_id: str) -> str:
    return client.get_job_logs(job_id)


class CapstoneHandler(BaseHTTPRequestHandler):
    ray_address: str

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            write_html(self, INDEX_HTML)
        elif parsed.path == "/api/state":
            write_json(self, app_state(self.ray_address))
        elif parsed.path == "/api/logs":
            self.handle_logs(parsed.query)
        else:
            write_json(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/submit":
            write_json(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self.handle_submit()

    def handle_submit(self) -> None:
        try:
            result = submit_run(JobSubmissionClient(self.ray_address), load_request(self))
            write_json(self, result, HTTPStatus.ACCEPTED)
        except Exception as exc:
            write_json(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_logs(self, query: str) -> None:
        job_id = parse_qs(query).get("job_id", [""])[0]
        if not job_id:
            write_json(self, {"error": "job_id is required"}, HTTPStatus.BAD_REQUEST)
            return
        write_json(self, {"job_id": job_id, "logs": get_logs(JobSubmissionClient(self.ray_address), job_id)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def build_handler(ray_address: str) -> type[CapstoneHandler]:
    class Handler(CapstoneHandler):
        pass

    Handler.ray_address = ray_address
    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ray capstone web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ray-address", default="http://ray-head:8265")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(args.ray_address))
    print(f"serving Ray capstone UI on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ray Capstone Console</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --ink: #1f2933;
      --muted: #667085;
      --line: #d8d0c4;
      --panel: #fffaf2;
      --panel-2: #ebe4d8;
      --accent: #b45309;
      --accent-2: #0f766e;
      --danger: #b91c1c;
      --ok: #15803d;
      --shadow: 0 18px 45px rgba(52, 43, 32, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 10% 0%, rgba(180, 83, 9, 0.16), transparent 26rem),
        linear-gradient(135deg, #f7f1e7, #ece7dd 52%, #f8f4ed);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 24px;
      align-items: end;
      padding: 34px 36px 26px;
      border-bottom: 1px solid rgba(31, 41, 51, 0.12);
    }
    h1 { margin: 0; font-size: clamp(30px, 5vw, 54px); letter-spacing: -0.055em; line-height: 0.94; }
    h2 { margin: 0 0 14px; font-size: 20px; letter-spacing: -0.02em; }
    h3 { margin: 0 0 8px; font-size: 15px; }
    p { margin: 0; color: var(--muted); line-height: 1.45; }
    button, input, select {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fffdf8;
      color: var(--ink);
      padding: 9px 11px;
    }
    button {
      cursor: pointer;
      background: var(--ink);
      color: #fffaf2;
      border-color: var(--ink);
      font-weight: 700;
      transition: transform 150ms ease, opacity 150ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button.secondary { background: transparent; color: var(--ink); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    input, select { width: 100%; min-width: 0; }
    main { padding: 26px 36px 42px; display: grid; gap: 24px; }
    section {
      background: rgba(255, 250, 242, 0.82);
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 24px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      animation: rise 420ms ease both;
    }
    @keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
    .status { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 8px 11px;
      border-radius: 999px;
      background: rgba(255, 250, 242, 0.78);
      border: 1px solid rgba(31, 41, 51, 0.12);
      color: var(--muted);
      font-size: 13px;
    }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--danger); }
    .dot.ready { background: var(--ok); }
    .layout { display: grid; grid-template-columns: 380px minmax(0, 1fr); gap: 24px; align-items: start; }
    .stack { display: grid; gap: 16px; }
    .presets { display: grid; gap: 10px; }
    .preset {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 14px;
      background: rgba(235, 228, 216, 0.6);
      border: 1px solid rgba(31, 41, 51, 0.10);
      border-radius: 18px;
    }
    .preset small, .field label, .mini { color: var(--muted); font-size: 12px; }
    .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .field { display: grid; gap: 5px; }
    .wide { grid-column: 1 / -1; }
    .checkbox { display: flex; align-items: center; gap: 8px; }
    .checkbox input { width: auto; }
    .actions { margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .actions button { width: auto; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
    .card {
      padding: 16px;
      background: rgba(255, 253, 248, 0.72);
      border: 1px solid rgba(31, 41, 51, 0.10);
      border-radius: 18px;
    }
    .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
    .value { margin-top: 6px; font-size: 26px; font-weight: 800; letter-spacing: -0.04em; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
    .tab { background: transparent; color: var(--ink); border-color: var(--line); }
    .tab.active { background: var(--accent); color: white; border-color: var(--accent); }
    .chart, .heatmap, .map { min-height: 250px; overflow: auto; }
    svg { width: 100%; height: auto; display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 7px; border-bottom: 1px solid rgba(31, 41, 51, 0.10); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .split { display: grid; grid-template-columns: minmax(0, 1fr) 270px; gap: 16px; }
    .tooltip { min-height: 130px; padding: 14px; border-radius: 16px; background: #1f2933; color: #fffaf2; white-space: pre-line; font-size: 13px; }
    .empty { padding: 18px; border: 1px dashed var(--line); border-radius: 18px; color: var(--muted); background: rgba(255, 253, 248, 0.52); }
    @media (max-width: 980px) { header, .layout, .split { grid-template-columns: 1fr; } .status { justify-content: flex-start; } }
  </style>
</head>
<body>
  <header>
    <div>
      <p class="label">Ray Capstone</p>
      <h1>Per-zone demand operations console</h1>
      <p>Launch replay use cases, tune parameters, compare latency behavior, and animate TLC zones from one Docker-served page.</p>
    </div>
    <div id="status" class="status"></div>
  </header>
  <main class="layout">
    <aside class="stack">
      <section>
        <h2>Use Cases</h2>
        <div id="presets" class="presets"></div>
      </section>
      <section>
        <h2>Custom Run</h2>
        <div id="customForm" class="fields"></div>
        <div class="actions">
          <button onclick="submitCustom()">Submit Run</button>
          <button class="secondary" onclick="loadState()">Refresh</button>
        </div>
      </section>
    </aside>
    <div class="stack">
      <section>
        <h2>Run Summary</h2>
        <div id="summary" class="cards"></div>
      </section>
      <section>
        <h2>Jobs</h2>
        <div id="jobs"></div>
      </section>
      <section>
        <h2>Metric Timelines</h2>
        <div id="tabs" class="tabs"></div>
        <div id="chart" class="chart"></div>
      </section>
      <section>
        <h2>Per-Zone Latency Heatmap</h2>
        <div id="heatControls" class="tabs"></div>
        <div id="heatmap" class="heatmap"></div>
      </section>
      <section>
        <h2>Animated TLC Zone Map</h2>
        <div id="mapControls" class="tabs"></div>
        <div id="mapContent"></div>
      </section>
    </div>
  </main>
<script>
let STATE = null;
let chartMetric = "tick_latency_s";
let heatRun = null;
let mapRun = null;
let mapMetric = "pickups";
let mapTickIndex = 0;
let playTimer = null;
let customFormRendered = false;
let hoveredZoneId = null;

const METRICS = [
  ["tick_latency_s", "Tick latency"],
  ["mean_zone_latency_s", "Mean zone latency"],
  ["max_zone_latency_s", "Max zone latency"],
  ["max_mean_ratio", "Latency skew"],
  ["zones_fallback", "Fallbacks"],
  ["need", "NEED count"],
];

const CUSTOM_FIELDS = [
  ["label", "Label", "text", "custom"],
  ["mode", "Mode", "select", "async"],
  ["start_day", "Start day in December", "number", "1"],
  ["start_hour", "Start hour", "number", "0"],
  ["window_hours", "Window hours", "number", "5"],
  ["slow_zone_fraction", "Slow zone fraction", "number", "0.25"],
  ["slow_zone_sleep_s", "Slow sleep seconds", "number", "1.0"],
  ["completion_fraction", "Completion fraction", "number", "0.75"],
  ["tick_timeout_s", "Tick timeout seconds", "number", "2.0"],
  ["max_inflight_zones", "Max inflight zones", "number", "4"],
  ["need_threshold", "Need threshold", "number", "1.1"],
  ["withhold_fraction", "Withhold fraction", "number", "0"],
  ["arrival_delay_ticks", "Arrival delay ticks", "number", "0"],
  ["delay_spread", "Delay spread", "number", "0"],
  ["subactor_trigger", "Subactor trigger", "number", "3"],
  ["n_helpers", "Helpers", "number", "3"],
  ["seed", "Seed", "number", "0"],
];

async function loadState() {
  const response = await fetch("/api/state");
  STATE = await response.json();
  heatRun = heatRun || firstRunLabel();
  mapRun = mapRun || firstRunLabel();
  renderAll();
}

function renderAll() {
  renderStatus();
  renderPresets();
  if (!customFormRendered) renderCustomForm();
  renderSummary();
  renderJobs();
  renderChart();
  renderHeatmap();
  renderMap();
}

function renderStatus() {
  const ray = STATE.ray || {};
  const tlc = STATE.tlc || {};
  document.getElementById("status").innerHTML = [
    pill(ray.ready ? "ready" : "", ray.ready ? "Ray Jobs ready" : "Ray Jobs offline"),
    pill("", `${STATE.runs.length} completed runs`),
    pill("", tlc.geometry ? "TLC geometry found" : "TLC geometry missing"),
  ].join("");
}

function pill(dotClass, text) {
  return `<span class="pill"><span class="dot ${dotClass}"></span>${escapeHtml(text)}</span>`;
}

function renderPresets() {
  document.getElementById("presets").innerHTML = STATE.presets.map(preset => `
    <article class="preset">
      <div>
        <h3>${escapeHtml(preset.title)}</h3>
        <small>${escapeHtml(preset.description)}</small>
      </div>
      <button ${STATE.ray.ready ? "" : "disabled"} onclick='submitPreset(${JSON.stringify(preset.id)})'>Run</button>
    </article>
  `).join("");
}

async function submitPreset(id) {
  const preset = STATE.presets.find(item => item.id === id);
  await submitRun({ label: preset.label, mode: preset.mode, params: preset.params });
}

function renderCustomForm() {
  const fields = CUSTOM_FIELDS.map(([name, label, type, value]) => fieldHtml(name, label, type, value));
  fields.push(`<label class="checkbox wide"><input id="custom-use_subactors" type="checkbox"> Use subactors</label>`);
  document.getElementById("customForm").innerHTML = fields.join("");
  customFormRendered = true;
}

function fieldHtml(name, label, type, value) {
  if (type === "select") {
    return `<div class="field"><label>${label}</label><select id="custom-${name}"><option>blocking</option><option selected>async</option><option>stress</option></select></div>`;
  }
  return `<div class="field"><label>${label}</label><input id="custom-${name}" type="${type}" value="${value}" step="any"></div>`;
}

async function submitCustom() {
  const mode = valueOf("mode");
  const label = valueOf("label") || mode;
  const params = windowParams();
  CUSTOM_FIELDS.filter(([name]) => !["label", "mode", "start_day", "start_hour", "window_hours"].includes(name)).forEach(([name]) => {
    const value = valueOf(name);
    if (value !== "") params[name] = value;
  });
  params.use_subactors = document.getElementById("custom-use_subactors").checked;
  await submitRun({ label, mode, params });
}

function windowParams() {
  const startDay = Math.max(1, Number(valueOf("start_day") || 1));
  const startHour = Math.max(0, Math.min(23, Number(valueOf("start_hour") || 0)));
  const windowHours = Math.max(0.25, Number(valueOf("window_hours") || 5));
  return {
    start_tick: Math.floor((startDay - 1) * 96 + startHour * 4),
    max_ticks: Math.ceil(windowHours * 4),
  };
}

function valueOf(name) {
  return document.getElementById(`custom-${name}`).value;
}

async function submitRun(payload) {
  const response = await fetch("/api/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) alert(result.error || "Submit failed");
  await loadState();
}

function renderSummary() {
  if (STATE.runs.length === 0) {
    document.getElementById("summary").innerHTML = `<div class="empty">No completed run artifacts yet. Launch a preset to populate this section.</div>`;
    return;
  }
  document.getElementById("summary").innerHTML = STATE.runs.map(run => {
    const s = run.summary;
    return `<article class="card">
      <div class="label">${escapeHtml(run.label)} | ${escapeHtml(run.config.mode)}</div>
      <div class="value">${fmt(s.total_s)}s</div>
      <p>start tick ${run.config.start_tick || 0} | ${s.n_ticks} ticks | fallbacks ${s.fallbacks} | late ${s.late_reports}</p>
      <p>withheld ${s.withheld_total} | released ${s.released_total} | subactor ticks ${s.subactor_ticks}</p>
    </article>`;
  }).join("");
}

function renderJobs() {
  if (!STATE.jobs.length) {
    document.getElementById("jobs").innerHTML = `<div class="empty">No Ray jobs reported yet.</div>`;
    return;
  }
  const rows = STATE.jobs.slice().reverse().map(job => {
    const metadata = job.metadata || {};
    const id = job.submission_id || job.job_id || job.jobId || "";
    return `<tr>
      <td><code>${escapeHtml(id)}</code></td>
      <td>${escapeHtml(metadata.label || "")}</td>
      <td>${escapeHtml(job.status || "")}</td>
      <td><code>${escapeHtml((job.entrypoint || "").slice(0, 110))}</code></td>
    </tr>`;
  }).join("");
  document.getElementById("jobs").innerHTML = `<table><thead><tr><th>Job</th><th>Label</th><th>Status</th><th>Entrypoint</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderChart() {
  renderMetricTabs();
  if (STATE.runs.length === 0) {
    document.getElementById("chart").innerHTML = `<div class="empty">Charts appear after a run finishes.</div>`;
    return;
  }
  lineChart("chart", chartMetric);
}

function renderMetricTabs() {
  document.getElementById("tabs").innerHTML = METRICS.map(([key, label]) =>
    `<button class="tab ${chartMetric === key ? "active" : ""}" onclick="chartMetric='${key}'; renderChart();">${label}</button>`
  ).join("");
}

function lineChart(id, field) {
  const width = 920, height = 270, pad = 36;
  const xmax = Math.max(1, ...STATE.runs.flatMap(run => run.metrics.map(row => Number(row.tick_id))));
  const ymax = Math.max(1, ...STATE.runs.flatMap(run => run.metrics.map(row => Number(row[field] || 0))));
  const x = value => pad + (Number(value) / xmax) * (width - pad * 2);
  const y = value => height - pad - (Number(value || 0) / ymax) * (height - pad * 2);
  const colors = ["#b45309", "#0f766e", "#1d4ed8", "#b91c1c", "#6d28d9", "#475569"];
  const lines = STATE.runs.map((run, i) => {
    const points = run.metrics.map(row => `${x(row.tick_id)},${y(row[field])}`).join(" ");
    return `<polyline points="${points}" fill="none" stroke="${colors[i % colors.length]}" stroke-width="3"/>`;
  }).join("");
  const legend = STATE.runs.map((run, i) => `<span class="pill"><span class="dot" style="background:${colors[i % colors.length]}"></span>${escapeHtml(run.label)}</span>`).join("");
  document.getElementById(id).innerHTML = `<svg viewBox="0 0 ${width} ${height}">
    <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#9a9186"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#9a9186"/>
    <text x="${pad}" y="22" fill="#667085" font-size="12">max ${fmt(ymax)}</text>
    ${lines}
  </svg><div class="status" style="justify-content:flex-start">${legend}</div>`;
}

function renderHeatmap() {
  if (STATE.runs.length === 0) {
    document.getElementById("heatControls").innerHTML = "";
    document.getElementById("heatmap").innerHTML = `<div class="empty">Heatmap appears after a run finishes.</div>`;
    return;
  }
  heatRun = heatRun || firstRunLabel();
  document.getElementById("heatControls").innerHTML = STATE.runs.map(run =>
    `<button class="tab ${heatRun === run.label ? "active" : ""}" onclick="heatRun='${run.label}'; renderHeatmap();">${escapeHtml(run.label)}</button>`
  ).join("");
  drawHeatmap(STATE.runs.find(run => run.label === heatRun) || STATE.runs[0]);
}

function drawHeatmap(run) {
  const zones = heatZones(run);
  const ticks = run.metrics.map(row => String(row.tick_id));
  const cell = 18, left = 58, top = 24;
  const width = left + ticks.length * cell + 20;
  const height = top + zones.length * cell + 28;
  const maxLatency = maxLatencyFor(run);
  const labels = zones.map((zone, i) => `<text x="5" y="${top + i * cell + 13}" font-size="11" fill="#667085">${zone}</text>`).join("");
  const cells = zones.flatMap((zone, zi) => ticks.map((tick, ti) => {
    const value = (run.latencyLog[tick] || {})[zone];
    return `<rect x="${left + ti * cell}" y="${top + zi * cell}" width="${cell - 1}" height="${cell - 1}" fill="${heatColor(value, maxLatency)}"><title>zone ${zone}, tick ${tick}, latency ${latencyLabel(value)}</title></rect>`;
  })).join("");
  document.getElementById("heatmap").innerHTML = `<svg viewBox="0 0 ${width} ${height}">${labels}${cells}</svg><p class="mini">Orange cells are fallback-finalized zones.</p>`;
}

function heatZones(run) {
  if (STATE.prepared) return STATE.prepared.zones.map(zone => String(zone.zone_id));
  return [...new Set(Object.values(run.latencyLog).flatMap(row => Object.keys(row)))].sort((a, b) => Number(a) - Number(b));
}

function renderMap() {
  if (!STATE.prepared) {
    document.getElementById("mapControls").innerHTML = "";
    document.getElementById("mapContent").innerHTML = `<div class="empty">Missing prepared assets at /app/prepared.</div>`;
    return;
  }
  if (!STATE.geometry || !STATE.geometry.features || STATE.geometry.features.length === 0) {
    const message = STATE.geometry && STATE.geometry.error ? STATE.geometry.error : "Download TLC zone lookup CSV and zone shapefile parquet or GeoJSON into Ray/TLC_Data.";
    document.getElementById("mapControls").innerHTML = "";
    document.getElementById("mapContent").innerHTML = `<div class="empty">${escapeHtml(message)} Search folders: <code>${escapeHtml((STATE.tlc.folders || []).join(", "))}</code></div>`;
    return;
  }
  mapRun = mapRun || firstRunLabel();
  const runOptions = STATE.runs.map(run => `<option value="${escapeHtml(run.label)}" ${mapRun === run.label ? "selected" : ""}>${escapeHtml(run.label)}</option>`).join("");
  const tickIds = mapTickIds();
  mapTickIndex = Math.min(mapTickIndex, Math.max(0, tickIds.length - 1));
  const currentTickId = tickIds[mapTickIndex] ?? 0;
  document.getElementById("mapControls").innerHTML = `
    <select onchange="mapRun=this.value; mapTickIndex=0; renderMap();">${runOptions}</select>
    <select onchange="mapMetric=this.value; renderMap();">
      ${option("pickups", "Observed pickups")}
      ${option("ratio", "Observed / baseline")}
      ${option("latency", "Task latency")}
      ${option("fallback", "Fallbacks")}
      ${option("slow", "Slow zones")}
    </select>
    <input id="mapTickSlider" type="range" min="${tickIds[0] ?? 0}" max="${tickIds[tickIds.length - 1] ?? 0}" value="${currentTickId}" step="1" oninput="mapTickIndex=tickIndexForValue(Number(this.value)); drawMap();">
    <button class="secondary" onclick="togglePlay()">${playTimer ? "Pause" : "Play"}</button>
    <span id="mapTickLabel" class="pill"></span>`;
  document.getElementById("mapContent").innerHTML = `<div class="split"><div id="map" class="map"></div><div id="tip" class="tooltip">${escapeHtml(currentTipText())}</div></div>`;
  drawMap();
}

function option(value, label) {
  return `<option value="${value}" ${mapMetric === value ? "selected" : ""}>${label}</option>`;
}

function drawMap() {
  if (!STATE || !STATE.prepared || !STATE.geometry || !document.getElementById("map")) return;
  const width = 880, height = 620;
  const run = currentMapRun();
  const frame = currentMapFrame(run);
  const bounds = geoBounds(STATE.geometry.features);
  const activeZones = activeZoneSet();
  const features = STATE.geometry.features
    .slice()
    .sort((a, b) => Number(activeZones.has(String(featureZoneId(a)))) - Number(activeZones.has(String(featureZoneId(b)))));
  const paths = features.map(feature => zonePath(feature, bounds, width, height, frame, run, activeZones)).join("");
  document.getElementById("map").innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-label="NYC TLC zone map">${paths}</svg>
    <p class="mini">${activeZones.size} active zones highlighted over ${STATE.geometry.features.length} TLC zones.</p>`;
  document.getElementById("mapTickLabel").textContent = `tick ${frame.tick_id} | ${frame.time}`;
  syncTickSlider(frame.tick_id);
  updateTooltip(frame, run);
}

function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    renderMap();
    return;
  }
  playTimer = setInterval(() => {
    mapTickIndex = (mapTickIndex + 1) % Math.max(1, mapTickIds().length);
    drawMap();
  }, 350);
  renderMap();
}

function mapTickIds(run = null) {
  const selected = run || STATE.runs.find(item => item.label === mapRun) || STATE.runs[0];
  if (selected && selected.metrics && selected.metrics.length > 0) {
    return selected.metrics.map(row => Number(row.tick_id));
  }
  return STATE.prepared.frames.map(frame => Number(frame.tick_id));
}

function tickIndexForValue(value) {
  const tickIds = mapTickIds();
  let bestIndex = 0;
  let bestDistance = Infinity;
  tickIds.forEach((tickId, index) => {
    const distance = Math.abs(tickId - value);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function syncTickSlider(tickId) {
  const slider = document.getElementById("mapTickSlider");
  if (slider) slider.value = String(tickId);
}

function currentMapRun() {
  return STATE.runs.find(item => item.label === mapRun) || STATE.runs[0];
}

function currentMapFrame(run = currentMapRun()) {
  const tickIds = mapTickIds(run);
  const tickId = tickIds[Math.min(mapTickIndex, Math.max(0, tickIds.length - 1))] ?? 0;
  return frameForTick(tickId) || STATE.prepared.frames[0];
}

function frameForTick(tickId) {
  return STATE.prepared.frames.find(frame => Number(frame.tick_id) === Number(tickId));
}

function activeZoneSet() {
  const ids = STATE.geometry.activeZoneIds || STATE.prepared.zones.map(zone => zone.zone_id);
  return new Set(ids.map(String));
}

function zonePath(feature, bounds, width, height, frame, run, activeZones) {
  const zoneId = String(featureZoneId(feature));
  const d = geometryPath(feature.geometry, bounds, width, height);
  const active = activeZones.has(zoneId);
  const fill = active ? mapColor(zoneId, frame, run) : "#ded6c8";
  const stroke = active ? "#1f2933" : "rgba(31, 41, 51, 0.14)";
  const widthValue = active ? 2.2 : 0.35;
  return `<path d="${d}" fill="${fill}" fill-rule="evenodd" stroke="${stroke}" stroke-width="${widthValue}" data-zone-id="${escapeHtml(zoneId)}" onmousemove="hoverZone(this.dataset.zoneId)"></path>`;
}

function mapColor(zoneId, frame, run) {
  const value = (frame.zones[zoneId] || {});
  if (mapMetric === "fallback") return latencyFor(run, frame.tick_id, zoneId) === null ? "#b45309" : "#c7e6da";
  if (mapMetric === "slow") return (run && run.config.slow_zones || []).map(String).includes(zoneId) ? "#6d28d9" : "#e8dfd1";
  if (mapMetric === "latency") {
    const latency = latencyFor(run, frame.tick_id, zoneId);
    return latency === undefined ? "#f2b84b" : heatColor(latency, maxLatencyFor(run));
  }
  if (mapMetric === "ratio") return ratioColor(value.ratio);
  return pickupColor(value.pickups, maxPickups());
}

function geometryPath(geometry, bounds, width, height) {
  const rings = geometry.type === "Polygon" ? geometry.coordinates : geometry.coordinates.flat();
  return rings.map(ring => "M" + ring.map(([lon, lat]) => project(lon, lat, bounds, width, height).join(",")).join("L") + "Z").join("");
}

function geoBounds(features) {
  const coords = features.flatMap(feature => allCoords(feature.geometry));
  const lons = coords.map(c => c[0]);
  const lats = coords.map(c => c[1]);
  return { minLon: Math.min(...lons), maxLon: Math.max(...lons), minLat: Math.min(...lats), maxLat: Math.max(...lats) };
}

function allCoords(geometry) {
  if (geometry.type === "Polygon") return geometry.coordinates.flat();
  if (geometry.type === "MultiPolygon") return geometry.coordinates.flat(2);
  return [];
}

function project(lon, lat, bounds, width, height) {
  const pad = 18;
  const dx = bounds.maxLon - bounds.minLon || 1;
  const dy = bounds.maxLat - bounds.minLat || 1;
  const scale = Math.min((width - pad * 2) / dx, (height - pad * 2) / dy);
  const left = (width - dx * scale) / 2;
  const top = (height - dy * scale) / 2;
  const x = left + (lon - bounds.minLon) * scale;
  const y = top + (bounds.maxLat - lat) * scale;
  return [x, y];
}

function zoneTip(zoneId, frame, run) {
  const zone = STATE.prepared.zones.find(item => String(item.zone_id) === zoneId) || { label: `Zone ${zoneId}` };
  const value = frame.zones[zoneId] || {};
  const latency = latencyFor(run, frame.tick_id, zoneId);
  return `${zone.label}\nzone_id: ${zoneId}\ntick: ${frame.tick_id}\ntime: ${frame.time}\npickups: ${value.pickups ?? 0}\nbaseline: ${fmt(value.baseline)}\nratio: ${fmt(value.ratio)}\nlatency: ${latencyLabel(latency)}`;
}

function inactiveZoneTip(feature, zoneId) {
  const props = feature.properties || {};
  const borough = props.borough || props.Borough || "";
  const zone = props.zone || props.Zone || `Zone ${zoneId}`;
  return `${borough ? borough + " - " : ""}${zone}\nzone_id: ${zoneId}\nNot in this run's active zone subset.`;
}

function hoverZone(zoneId) {
  hoveredZoneId = zoneId;
  updateTooltip();
}

function updateTooltip(frame = null, run = null) {
  const tip = document.getElementById("tip");
  if (!tip) return;
  tip.textContent = currentTipText(frame, run);
}

function currentTipText(frame = null, run = null) {
  if (!hoveredZoneId) return "Hover a zone for details.";
  const active = activeZoneSet().has(String(hoveredZoneId));
  if (active) {
    const currentFrame = frame || currentMapFrame(run || currentMapRun());
    return zoneTip(String(hoveredZoneId), currentFrame, run || currentMapRun());
  }
  const feature = featureForZone(hoveredZoneId);
  return feature ? inactiveZoneTip(feature, String(hoveredZoneId)) : "Hover a zone for details.";
}

function featureForZone(zoneId) {
  return STATE.geometry.features.find(feature => String(featureZoneId(feature)) === String(zoneId));
}

function featureZoneId(feature) {
  const props = feature.properties || {};
  return props.LocationID ?? props.location_id ?? props.zone_id ?? props.OBJECTID ?? props.objectid;
}

function latencyFor(run, tick, zoneId) {
  if (!run) return undefined;
  return (run.latencyLog[String(tick)] || {})[String(zoneId)];
}

function latencyLabel(value) {
  if (value === null) return "fallback";
  if (value === undefined) return "not emitted for selected run";
  return fmt(value);
}

function maxLatencyFor(run) {
  if (!run) return 1;
  return Math.max(0.001, ...Object.values(run.latencyLog).flatMap(row => Object.values(row).filter(v => v !== null).map(Number)));
}

function maxPickups() {
  return Math.max(1, ...STATE.prepared.frames.flatMap(frame => Object.values(frame.zones).map(z => z.pickups)));
}

function heatColor(value, maxLatency) {
  if (value === null) return "#b45309";
  if (value === undefined) return "#e8dfd1";
  const t = Math.min(1, Number(value) / maxLatency);
  return `rgb(${Math.round(225 - 170 * t)}, ${Math.round(236 - 150 * t)}, ${Math.round(229 - 70 * t)})`;
}

function pickupColor(value, maxValue) {
  const t = 0.18 + Math.min(0.82, Number(value || 0) / maxValue);
  return `rgb(${Math.round(232 - 150 * t)}, ${Math.round(223 - 115 * t)}, ${Math.round(209 - 170 * t)})`;
}

function ratioColor(value) {
  if (value === null || value === undefined) return "#f2b84b";
  if (value >= 1.1) return "#b91c1c";
  if (value >= 0.8) return "#b45309";
  return "#0f766e";
}

function firstRunLabel() {
  return STATE && STATE.runs[0] ? STATE.runs[0].label : "";
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits).replace(/\.00$/, "");
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}

loadState();
setInterval(loadState, 5000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
