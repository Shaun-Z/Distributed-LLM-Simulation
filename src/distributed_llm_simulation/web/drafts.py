from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..compiler import Lowerer
from ..runtime import SimulationConfig, Simulator
from ..spec import HardwareSpec, PlacementPlan, WorkloadSpec
from ..spec.schema import UniqueLoader, load_hardware, parse_spec, quantity
from ..workloads import (
    SyntheticWorkload,
    TransformerConfig,
    TransformerWorkloadBuilder,
    image_pipeline,
)

SECTIONS = {
    "hardware": HardwareSpec,
    "workload": WorkloadSpec,
    "placement": PlacementPlan,
    "config": SimulationConfig,
}
TEMPLATES = {
    "pipeline": {
        "title": "CPU → GPU pipeline",
        "description": "A hand-verifiable two-stage pipeline.",
        "profile": "pipeline_oracle",
        "parameters": {
            "requests": 2,
            "duration_a": 0.002,
            "duration_b": 0.003,
            "tensor_bytes": 8000000,
        },
    },
    "fork_join": {
        "title": "Fork & join",
        "description": "Parallel branches sharing one physical UMA pool.",
        "profile": "uma_workloads",
        "parameters": {"duration_s": 0.002, "tensor_bytes": 16},
    },
    "image": {
        "title": "Image pipeline",
        "description": "Normalize, extract features, and classify.",
        "profile": "discrete_workloads",
        "parameters": {"images": 2, "width": 64, "height": 64},
    },
    "matmul": {
        "title": "Tiled matrix multiplication",
        "description": "Expose tile dependencies and intermediate buffers.",
        "profile": "discrete_workloads",
        "parameters": {"tiles": 2, "tile_size": 16},
    },
    "transformer": {
        "title": "Transformer decoder",
        "description": "Prefill, autoregressive decode, TP / PP, and offload.",
        "profile": "discrete_workloads",
        "parameters": {
            **asdict(TransformerConfig()),
            "stages": [["gpu0"]],
            "pools": [["hbm0"]],
            "offload_pool": "",
            "arrival_s": 0,
        },
    },
}


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Draft(InputModel):
    label: str = Field(default="Untitled experiment", max_length=160)
    hardware: str = Field(max_length=8_000_000)
    workload: str = Field(max_length=16_000_000)
    placement: str = Field(max_length=8_000_000)
    config: str = Field(max_length=1_000_000)


class TemplateRequest(InputModel):
    template: str
    profile: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class Edit(InputModel):
    section: Literal["hardware", "workload", "placement", "config"]
    path: list[str | int]
    value: str
    kind: Literal["number", "integer", "string", "json", "boolean"] = "string"
    unit: str | None = None


class PatchRequest(InputModel):
    draft: Draft
    edits: list[Edit] = Field(max_length=100000)


def wire(value):
    """Only unsafe integers become decimal strings; canonical source text stays exact."""
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    if isinstance(value, dict):
        return {str(k): wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def dump(spec):
    return yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)


def parse_draft(draft):
    return tuple(parse_spec(getattr(draft, key), model) for key, model in SECTIONS.items())


def normalized(draft, specs):
    return Draft(
        label=draft.label, **{key: dump(spec) for key, spec in zip(SECTIONS, specs, strict=True)}
    )


def profile_directory(custom=None):
    if custom:
        return Path(custom).resolve()
    repository = Path(__file__).resolve().parents[3]
    source_profiles = repository / "profiles" / "hardware"
    if (repository / "pyproject.toml").is_file() and source_profiles.is_dir():
        return source_profiles
    bundled = Path(__file__).parent / "data" / "hardware"
    if bundled.is_dir():
        return bundled
    return Path(__file__).resolve().parents[3] / "profiles" / "hardware"


def catalog(directory):
    profiles = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            h = load_hardware(path)
            profiles.append(
                {
                    "id": path.stem,
                    "name": h.name,
                    "synthetic": h.synthetic,
                    "devices": [
                        {"id": n.id, "type": n.device_type, "pools": list(n.accessible_pool_ids)}
                        for n in h.nodes
                        if n.kind == "compute"
                    ],
                }
            )
        except ValueError:
            continue
    return {
        "profiles": profiles,
        "templates": [{"id": key, **value} for key, value in TEMPLATES.items()],
    }


def build_template(request, directory):
    if request.template not in TEMPLATES:
        raise ValueError("Unknown workload template")
    info = TEMPLATES[request.template]
    profile = request.profile or info["profile"]
    if profile not in {p["id"] for p in catalog(directory)["profiles"]}:
        raise ValueError("Unknown hardware profile")
    h = load_hardware(directory / f"{profile}.yaml")
    p = {**info["parameters"], **request.parameters}
    if request.template == "pipeline":
        w, placement = SyntheticWorkload.pipeline(**p)
    elif request.template == "fork_join":
        w, placement = SyntheticWorkload.fork_join(left="uc0", right="ug0", pool="uma0", **p)
    elif request.template == "image":
        w, placement = image_pipeline(**p)
    elif request.template == "matmul":
        w, placement = SyntheticWorkload.tiled_matmul(**p)
    else:
        settings = {k: p.pop(k) for k in ("stages", "pools", "offload_pool", "arrival_s")}
        settings["offload_pool"] = settings["offload_pool"] or None
        w, placement = TransformerWorkloadBuilder(TransformerConfig(**p)).build(**settings)
    draft = Draft(
        label=info["title"],
        hardware=dump(h),
        workload=dump(w),
        placement=dump(placement),
        config=dump(SimulationConfig()),
    )
    return draft


def patch(request):
    sources = {key: yaml.load(getattr(request.draft, key), Loader=UniqueLoader) for key in SECTIONS}
    for edit in request.edits:
        value: Any = edit.value
        if edit.kind in ("number", "integer"):
            if edit.unit and " " in value.strip():
                value = quantity(value, edit.unit)
            else:
                try:
                    number = Decimal(value)
                    if (
                        not number.is_finite()
                        or edit.kind == "integer"
                        and number != number.to_integral_value()
                    ):
                        raise ValueError("Expected a finite integer/number")
                    value = int(number) if number == number.to_integral_value() else float(number)
                except InvalidOperation as error:
                    raise ValueError(f"Invalid number: {value}") from error
        elif edit.kind == "json":
            value = json.loads(value)
        elif edit.kind == "boolean":
            if value not in ("true", "false"):
                raise ValueError("Expected true or false")
            value = value == "true"
        obj = sources[edit.section]
        if not edit.path:
            raise ValueError("An edit requires a field path")
        for part in edit.path[:-1]:
            obj = obj[part]
        obj[edit.path[-1]] = value
    return Draft(
        label=request.draft.label,
        **{key: yaml.safe_dump(value, sort_keys=False) for key, value in sources.items()},
    )


def preview(draft):
    specs = parse_draft(draft)
    h, w, p, c = specs
    e = Lowerer(h).compile(w, p)
    Simulator(h, config=c).validate(e)
    return wire(
        {
            "draft": normalized(draft, specs).model_dump(),
            "specs": {
                key: spec.model_dump(mode="json") for key, spec in zip(SECTIONS, specs, strict=True)
            },
            "counts": {
                "operators": len(w.operators),
                "tensors": len(w.tensors),
                "activities": len(e.activities),
                "replicas": len(e.replicas),
            },
            "hashes": {key: spec.digest() for key, spec in zip(SECTIONS, specs, strict=True)},
            "execution": e.model_dump(mode="json"),
        }
    )
