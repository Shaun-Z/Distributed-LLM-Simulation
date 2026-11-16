"""Artifact-only readers. Existing experiments remain untouched."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ..spec import HardwareSpec
from ..spec.schema import parse_spec
from .drafts import Draft

FILES = {
    "hardware.resolved.yaml",
    "hardware.json",
    "workload.json",
    "placement.json",
    "config.json",
    "execution.json",
    "manifest.json",
    "summary.json",
    "resources.csv",
    "requests.csv",
    "timeline.csv",
    "timeline.json",
    "trace.jsonl",
    "environment_trace.jsonl",
    "critical_path.json",
    "bounds.json",
    "llm_metrics.json",
    "failure_snapshot.json",
    "hardware.graph.json",
    "workload.graph.json",
    "invalid.json",
}


def read_json(directory, name, default=None):
    path = directory / name
    return json.loads(path.read_text()) if path.is_file() else default


def read_csv(directory, name):
    path = directory / name
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = {}
        self.series_cache = {}

    def scan(self):
        paths = set()
        for name in ("summary.json", "invalid.json", "manifest.json"):
            paths.update(
                p.parent
                for p in self.root.rglob(name)
                if ".ui" not in p.relative_to(self.root).parts
            )
        resolved_paths = {}
        output = []
        for directory in paths:
            resolved = directory.resolve()
            if not resolved.is_relative_to(self.root):
                continue
            # Reject escaping artifact symlinks as well as escaping run directories.
            if any(
                p.exists() and not p.resolve().is_relative_to(self.root)
                for p in (directory / name for name in FILES)
            ):
                continue
            rid = hashlib.sha256(str(directory.relative_to(self.root)).encode()).hexdigest()[:24]
            resolved_paths[rid] = resolved
            try:
                summary = read_json(resolved, "summary.json", {})
                manifest = read_json(resolved, "manifest.json", {})
                config = read_json(resolved, "config.json", {})
                workload = read_json(resolved, "workload.json", {})
                invalid = read_json(resolved, "invalid.json", {})
                output.append(
                    {
                        "id": rid,
                        "directory": str(directory.relative_to(self.root)),
                        "label": str(directory.relative_to(self.root)),
                        "workload": workload.get("name", "Unknown workload"),
                        "status": summary.get("status", invalid.get("status", "INCOMPLETE")),
                        "makespan_s": summary.get("makespan_s"),
                        "metrics": summary.get("metrics", {}),
                        "model_level": manifest.get("model_level"),
                        "scheduler": config.get("scheduler"),
                        "network_mode": config.get("network_mode"),
                        "modified_at": directory.stat().st_mtime,
                    }
                )
            except (ValueError, OSError):
                output.append(
                    {
                        "id": rid,
                        "directory": str(directory.relative_to(self.root)),
                        "label": directory.name,
                        "workload": "Unreadable artifacts",
                        "status": "INCOMPLETE",
                        "metrics": {},
                        "modified_at": directory.stat().st_mtime,
                    }
                )
        self.paths = resolved_paths
        return sorted(output, key=lambda row: row["modified_at"], reverse=True)

    def directory(self, rid):
        self.scan()
        if rid not in self.paths:
            raise KeyError("Unknown run")
        directory = self.paths[rid]
        if not directory.is_relative_to(self.root):
            raise KeyError("Unknown run")
        return directory

    def artifact(self, rid, name):
        if name not in FILES:
            raise KeyError("Unknown artifact")
        path = self.directory(rid) / name
        if not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise KeyError("Artifact is not available")
        return path

    def draft(self, rid):
        d = self.directory(rid)
        hardware = d / "hardware.resolved.yaml"
        if not hardware.is_file():
            hardware = d / "hardware.json"
        return Draft(
            label=f"Copy of {d.name}",
            hardware=hardware.read_text(),
            **{key: (d / f"{key}.json").read_text() for key in ("workload", "placement", "config")},
        )

    def detail(self, rid):
        d = self.directory(rid)
        return {
            "id": rid,
            "directory": str(d.relative_to(self.root)),
            "summary": read_json(d, "summary.json", {}),
            "manifest": read_json(d, "manifest.json", {}),
            "critical_path": read_json(d, "critical_path.json"),
            "bounds": read_json(d, "bounds.json"),
            "llm_metrics": read_json(d, "llm_metrics.json"),
            "failure": read_json(d, "failure_snapshot.json"),
            "invalid": read_json(d, "invalid.json"),
            "resources": read_csv(d, "resources.csv"),
            "requests": read_csv(d, "requests.csv"),
            "artifacts": sorted(name for name in FILES if (d / name).is_file()),
        }

    def specs(self, rid):
        draft = self.draft(rid)
        from .drafts import parse_draft

        h, w, p, c = parse_draft(draft)
        execution = read_json(self.directory(rid), "execution.json")
        return {
            "hardware": h.model_dump(mode="json"),
            "workload": w.model_dump(mode="json"),
            "placement": p.model_dump(mode="json"),
            "config": c.model_dump(mode="json"),
            "execution": execution,
        }

    def timeline(self, rid, start=None, end=None, request=None, search=""):
        d = self.directory(rid)
        rows = read_json(d, "timeline.json")
        if rows is None:
            rows = read_csv(d, "timeline.csv")
            for row in rows:
                for key in ("start_s", "finish_s", "setup_s"):
                    row[key] = float(row[key]) if row.get(key) else None
        return [
            r
            for r in rows
            if (start is None or r.get("finish_s") is None or r["finish_s"] >= start)
            and (end is None or r.get("start_s") is None or r["start_s"] <= end)
            and (not request or r.get("request_id") == request)
            and search.lower() in json.dumps(r).lower()
        ]

    def trace(self, rid, offset=0, limit=200, event_type=None, activity=None, start=None, end=None):
        path = self.artifact(rid, "trace.jsonl")
        rows, total = [], 0
        with path.open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (
                    event_type
                    and row["event_type"] != event_type
                    or activity
                    and row.get("activity_id") != activity
                ):
                    continue
                if (
                    start is not None
                    and row["sim_time"] < start
                    or end is not None
                    and row["sim_time"] > end
                ):
                    continue
                if offset <= total < offset + limit:
                    rows.append(row)
                total += 1
        return {"rows": rows, "total": total, "offset": offset, "limit": limit}

    def series(self, rid, start=None, end=None, max_points=2000):
        d = self.directory(rid)
        names = (
            "trace.jsonl",
            "execution.json",
            "config.json",
            "hardware.resolved.yaml",
            "hardware.json",
        )
        stamp = tuple(
            (name, (d / name).stat().st_mtime_ns, (d / name).stat().st_size)
            for name in names
            if (d / name).is_file()
        )
        if rid not in self.series_cache or self.series_cache[rid][0] != stamp:
            if len(self.series_cache) >= 8:
                self.series_cache.pop(next(iter(self.series_cache)))
            self.series_cache[rid] = (stamp, reconstruct_series(d))
        data = self.series_cache[rid][1]
        output = {**data, "series": [], "downsampled": False}
        for item in data["series"]:
            points = window_points(item["points"], start, end)
            sampled = envelope(points, max_points)
            output["downsampled"] |= len(sampled) < len(points)
            output["series"].append({**item, "points": sampled})
        return output


def window_points(points, start, end):
    if start is None and end is None:
        return points
    selected = [
        p for p in points if (start is None or p[0] >= start) and (end is None or p[0] <= end)
    ]
    before = [p for p in points if start is not None and p[0] < start]
    if before:
        selected.insert(0, [start, before[-1][1]])
    return selected


def envelope(points, maximum):
    """Keep ordered endpoints and min/max per bucket, including instantaneous peaks."""
    if len(points) <= maximum:
        return points
    width = max(1, (len(points) + max(1, maximum // 4) - 1) // max(1, maximum // 4))
    result = []
    for i in range(0, len(points), width):
        chunk = points[i : i + width]
        keep = {
            0,
            len(chunk) - 1,
            min(range(len(chunk)), key=lambda j: chunk[j][1]),
            max(range(len(chunk)), key=lambda j: chunk[j][1]),
        }
        result.extend(chunk[j] for j in sorted(keep))
    return result


def reconstruct_series(directory):
    hardware_path = directory / "hardware.resolved.yaml"
    if not hardware_path.is_file():
        hardware_path = directory / "hardware.json"
    if not hardware_path.is_file():
        return {
            "series": [],
            "memory_available": False,
            "rates_available": False,
            "reason": "Hardware specification is missing.",
        }
    hardware = parse_spec(hardware_path.read_text(), HardwareSpec)
    resources = {r.id: r for r in hardware.resources}
    config = read_json(directory, "config.json", {})
    execution = read_json(directory, "execution.json", {})
    summary = read_json(directory, "summary.json", {})
    activities = {a["id"]: a for a in execution.get("activities", [])}
    pools = {
        hardware.pool(n.id): resources[n.capacity_resource_id].capacity
        for n in hardware.nodes
        if n.kind == "memory"
    }
    points = {("memory", pool): [[0, 0]] for pool in pools}
    allocations, used = {}, defaultdict(int)
    rates, demands = {}, defaultdict(dict)
    for aid, a in activities.items():
        work = dict(a.get("rate_demands", []))
        for rid, amount in a.get("service_work", []):
            if a["total_work"]:
                work[rid] = work.get(rid, 0) + amount / a["total_work"]
        demands[aid] = work
    rate_resources = {r.id: r.capacity for r in hardware.resources if r.kind == "rate"}
    rate_resources.update(
        {
            r["id"]: r["capacity"]
            for r in execution.get("extra_resources", [])
            if r["kind"] == "rate"
        }
    )
    for rid, capacity in rate_resources.items():
        points["service", rid] = [[0, 0]]
        points["capacity", rid] = [[0, capacity]]
    for process in config.get("capacity_processes", []):
        points["capacity", process["resource_id"]] = [list(p) for p in process["points"]]
    allocation_seen, rate_seen = False, False
    trace = directory / "trace.jsonl"
    if trace.is_file():
        with trace.open() as file:
            for line in file:
                if not line.strip():
                    continue
                event = json.loads(line)
                time, kind = event["sim_time"], event["event_type"]
                if kind in ("ALLOCATED", "FREED"):
                    allocation_seen = True
                    pool, aid = event["pool_id"], event["allocation_id"]
                    if kind == "ALLOCATED":
                        if aid not in allocations:
                            allocations[aid] = (pool, event["allocation_bytes"])
                            used[pool] += event["allocation_bytes"]
                    elif aid in allocations:
                        owner, size = allocations.pop(aid)
                        used[owner] -= size
                    points.setdefault(("memory", pool), [[0, 0]]).append([time, used[pool]])
                aid = event.get("activity_id")
                if kind == "RATE_CHANGED":
                    rate_seen = True
                    rates[aid] = event["rate"]
                elif kind == "ACTIVITY_FINISHED":
                    rates[aid] = 0
                else:
                    continue
                for rid in demands[aid]:
                    total = sum(
                        rates.get(other, 0) * dem.get(rid, 0) for other, dem in demands.items()
                    )
                    points.setdefault(("service", rid), [[0, 0]]).append([time, total])
    finish = summary.get("finish_time_s", 0)
    level = config.get("trace_level", "resource_changes")
    memory_available = level != "summary" and (allocation_seen or not pools)
    rates_available = level in ("resource_changes", "debug") and rate_seen and bool(activities)
    output = []
    for (kind, rid), values in points.items():
        if (
            kind == "memory"
            and not memory_available
            or kind in ("service", "capacity")
            and not rates_available
        ):
            continue
        if values[-1][0] < finish:
            values.append([finish, values[-1][1]])
        values = [p for p in values if p[0] <= finish]
        output.append(
            {
                "kind": kind,
                "id": rid,
                "unit": "byte"
                if kind == "memory"
                else resources[rid].unit
                if rid in resources
                else "byte/s",
                "capacity": pools.get(rid) if kind == "memory" else None,
                "points": values,
            }
        )
    return {
        "series": output,
        "memory_available": memory_available,
        "rates_available": rates_available,
        "reason": None
        if memory_available and rates_available
        else "The saved trace does not contain all events required for these curves.",
    }
