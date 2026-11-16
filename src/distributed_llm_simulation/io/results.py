from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import yaml

from .. import __version__


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, data):
    Path(path).write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                    for k, v in row.items()
                }
            )


def write_results(
    result, output_dir, *, hardware, workload, placement, config, execution, overwrite=False
):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"nonempty run directory: {output}; use --overwrite or a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    for name in ("bounds.json", "llm_metrics.json"):
        if overwrite:
            (output / name).unlink(missing_ok=True)
    (output / "hardware.resolved.yaml").write_text(
        yaml.safe_dump(hardware.model_dump(mode="json"), sort_keys=False)
    )
    for name, spec in (
        ("workload", workload),
        ("placement", placement),
        ("config", config),
        ("execution", execution),
    ):
        write_json(output / f"{name}.json", spec.model_dump(mode="json"))
    try:
        repository = Path(__file__).resolve().parents[3]
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        source_revision, dirty = None, None
    package_root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        source_hash.update(str(path.relative_to(package_root)).encode())
        source_hash.update(path.read_bytes())
    manifest = {
        "schema_version": 1,
        "simulator_version": __version__,
        "python_version": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("networkx", "pydantic", "pyyaml")},
        "hashes": {
            "hardware": hardware.digest(),
            "workload": workload.digest(),
            "placement": placement.digest(),
            "config": config.digest(),
            "execution": execution.digest(),
        },
        "source_revision": source_revision,
        "source_sha256": source_hash.hexdigest(),
        "source_dirty": dirty,
        "seed": config.seed,
        "model_level": result.metrics["model_level"],
        "network_mode": config.network_mode,
        "assumptions": result.assumptions,
        "synthetic_hardware": hardware.synthetic,
    }
    write_json(output / "manifest.json", manifest)
    summary = {
        "status": result.status,
        "makespan_s": result.makespan_s,
        "finish_time_s": result.finish_time_s,
        "failure_reason": result.failure_reason,
        "metrics": result.metrics,
        "assumptions": result.assumptions,
    }
    write_json(output / "summary.json", summary)
    _csv(output / "resources.csv", result.resources)
    _csv(output / "requests.csv", result.requests)
    _csv(output / "timeline.csv", result.timeline)
    write_json(output / "timeline.json", result.timeline)
    (output / "trace.jsonl").write_text(
        "".join(
            json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            for row in result.trace
        )
    )
    environment = [
        {"sim_time": t, "resource_id": p.resource_id, "capacity": c}
        for p in config.capacity_processes
        for t, c in p.points
    ]
    environment.sort(key=lambda r: (r["sim_time"], r["resource_id"]))
    (output / "environment_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in environment)
    )
    failure_path = output / "failure_snapshot.json"
    if result.failure_snapshot is not None:
        write_json(failure_path, result.failure_snapshot)
    elif failure_path.exists():
        failure_path.unlink()
    from ..analysis.critical_path import explain_critical_path

    write_json(output / "critical_path.json", explain_critical_path(execution, result))
    return output


def export_graphs(hardware, workload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "hardware.graph.json",
        {
            "nodes": [n.model_dump() for n in hardware.nodes],
            "channels": [c.model_dump() for c in hardware.channels],
            "resources": [r.model_dump() for r in hardware.resources],
        },
    )
    g = workload.graph()
    write_json(
        directory / "workload.graph.json",
        {
            "nodes": [{"kind": kind, "id": tid} for kind, tid in g],
            "edges": [[list(a), list(b)] for a, b in g.edges],
        },
    )
