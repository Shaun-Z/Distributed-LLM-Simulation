"""Safe YAML/JSON loading with explicit unit conversion and error paths."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import yaml

from .base import SpecError
from .hardware import HardwareSpec
from .placement import PlacementPlan
from .workload import WorkloadSpec

UNITS = {
    "byte": (1, "byte"),
    "B": (1, "byte"),
    "GB": (10**9, "byte"),
    "GiB": (2**30, "byte"),
    "MB": (10**6, "byte"),
    "MiB": (2**20, "byte"),
    "s": (1, "s"),
    "ms": (0.001, "s"),
    "us": (0.000001, "s"),
    "byte/s": (1, "byte/s"),
    "B/s": (1, "byte/s"),
    "GB/s": (10**9, "byte/s"),
    "Gbit/s": (125000000, "byte/s"),
    "FLOP/s": (1, "FLOP/s"),
    "TFLOP/s": (10**12, "FLOP/s"),
    "op/s": (1, "op/s"),
    "count": (1, "count"),
}


def quantity(value, unit: str):
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s+([A-Za-z/]+)\s*", value)
    if not match or match[2] not in UNITS or UNITS[match[2]][1] != unit:
        raise SpecError("INVALID_UNIT", "quantity", f"expected {unit}: {value!r}")
    result = Decimal(match[1]) * Decimal(str(UNITS[match[2]][0]))
    if unit in ("byte", "count"):
        if result != result.to_integral_value():
            raise SpecError("INVALID_UNIT", "quantity", "fractional byte/count")
        return int(result)
    return float(result)


def _normalize(value, path=""):
    if isinstance(value, list):
        return [_normalize(v, f"{path}/{i}") for i, v in enumerate(value)]
    if not isinstance(value, dict):
        return value
    out = {}
    pairs = {
        "resource_demands",
        "slot_demands",
        "read_bytes",
        "write_bytes",
        "shape_params",
        "attrs",
        "op_compute",
        "op_kernel",
        "tensor_target_pool",
        "final_output_pools",
        "markers",
    }
    for key, v in value.items():
        if key in pairs and isinstance(v, dict):
            v = list(v.items())
        elif key == "capacity":
            v = quantity(v, value.get("unit", "byte"))
        elif key == "effective_ops_per_s":
            v = quantity(v, "FLOP/s" if isinstance(v, str) and "FLOP" in v else "op/s")
        elif key.endswith("_s"):
            v = quantity(v, "s")
        elif key in ("payload_Bps",):
            v = quantity(v, "byte/s")
        elif key.endswith("_bytes") and not isinstance(v, (list, dict)):
            v = quantity(v, "byte")
        out[key] = _normalize(v, f"{path}/{key}")
    return out


class UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, val_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise SpecError("DUPLICATE_KEY", "yaml", str(key))
        result[key] = loader.construct_object(val_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
UniqueLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][+-]?[0-9]+$"),
    list("-+0123456789."),
)


def parse_spec(text, model):
    """Parse the same safe YAML/JSON syntax from files or Web draft text."""
    data = yaml.load(text, Loader=UniqueLoader)
    return model.model_validate(_normalize(data))


def load(path, model):
    try:
        return parse_spec(Path(path).read_text(), model)
    except (ValueError, TypeError, yaml.YAMLError) as error:
        raise SpecError("INVALID_SPEC", str(path), str(error)) from error


def load_hardware(path):
    return load(path, HardwareSpec)


def load_workload(path):
    return load(path, WorkloadSpec)


def load_placement(path):
    return load(path, PlacementPlan)


def save_spec(spec, path):
    Path(path).write_text(json.dumps(spec.model_dump(mode="json"), indent=2) + "\n")
