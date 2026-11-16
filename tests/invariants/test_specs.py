import json

import pytest

from distributed_llm_simulation import load_hardware
from distributed_llm_simulation.spec import (
    HardwareSpec,
    OperatorSpec,
    PlacementPlan,
    SpecError,
    TensorSpec,
    WorkloadSpec,
)
from distributed_llm_simulation.spec.schema import load, quantity, save_spec
from distributed_llm_simulation.workloads import SyntheticWorkload


@pytest.mark.parametrize(
    "path",
    ["discrete", "uma", "pipeline_oracle", "discrete_workloads", "uma_workloads", "two_hosts"],
)
def test_profile_roundtrip(path, tmp_path):
    h = load_hardware(f"profiles/hardware/{path}.yaml")
    save_spec(h, tmp_path / "h.json")
    assert load_hardware(tmp_path / "h.json").digest() == h.digest()
    assert h.graph().number_of_nodes() == len(h.nodes)


def test_units_and_integer_precision():
    assert quantity("100 Gbit/s", "byte/s") == 12_500_000_000
    assert quantity("32 GiB", "byte") == 32 * 2**30
    assert quantity("9007199254740993 byte", "byte") == 9007199254740993
    for invalid in ("100G", "32 GiB/s", "0.5 byte"):
        with pytest.raises(SpecError):
            quantity(invalid, "byte")


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d["nodes"].append(d["nodes"][0]), "DUPLICATE_ID"),
        (lambda d: d["channels"][0].update(src="missing"), "UNKNOWN_ENDPOINT"),
        (lambda d: d["resources"][0].update(capacity=-1), "INVALID_CAPACITY"),
        (lambda d: d["resources"][0].update(capacity=1.5), "exact integer"),
        (lambda d: d["resources"][0].update(capacity=float("nan")), "finite"),
        (lambda d: d["resources"][0].update(unit="byte"), "INVALID_UNIT"),
        (
            lambda d: d["routes"][0].update(channel_ids=["pcie_hbm", "ram_pcie"]),
            "DISCONTINUOUS_ROUTE",
        ),
        (lambda d: d["routes"][0].update(operation="p2p_copy"), "ILLEGAL_OPERATION"),
        (lambda d: d["kernel_profiles"][0].update(slot_resource_id="missing"), "INVALID_SLOT"),
        (lambda d: d.update(unknown_field=1), "Extra inputs"),
    ],
)
def test_invalid_hardware(mutation, message):
    d = load_hardware("profiles/hardware/discrete.yaml").model_dump(mode="json")
    mutation(d)
    with pytest.raises(ValueError, match=message):
        HardwareSpec.model_validate(d)


def test_duplicate_yaml_key_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: one\nname: two\n")
    with pytest.raises(SpecError, match="DUPLICATE_KEY"):
        load_hardware(path)


def test_cycle_and_producer_mismatch():
    with pytest.raises(ValueError, match="DEPENDENCY_CYCLE"):
        WorkloadSpec(
            name="cycle",
            tensors=(TensorSpec(id="x", storage_bytes=4, producer_id="a", consumer_ids=("a",)),),
            operators=(OperatorSpec(id="a", op_type="gemm", inputs=("x",), outputs=("x",)),),
        )
    w, _ = SyntheticWorkload.pipeline()
    d = w.model_dump(mode="json")
    d["tensors"][0]["consumer_ids"] = []
    with pytest.raises(ValueError, match="INVALID_CONSUMERS"):
        WorkloadSpec.model_validate(d)


def test_mapping_yaml_and_hash(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text("op_compute: {a: cpu0}\n")
    assert load(path, PlacementPlan).op_compute == (("a", "cpu0"),)
    path.write_text(json.dumps({"op_compute": [["a", "cpu0"]]}))
    assert load(path, PlacementPlan).digest() == PlacementPlan(op_compute=(("a", "cpu0"),)).digest()


def test_shape_layout_and_algorithm_capability():
    from distributed_llm_simulation import Lowerer
    from distributed_llm_simulation.workloads import DAGBuilder

    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    data = h.model_dump()
    for k in data["kernel_profiles"]:
        if k["id"] == "cpu0_gemm_fp32":
            k.update(layouts=("dense",), algorithms=("tiled",), shape_constraints=(("m", 1, 32),))
    h = HardwareSpec.model_validate(data)
    b = DAGBuilder("unsupported")
    x = b.tensor("x", 4, pools=("ram0",))
    b.op("op", [x], [], "cpu0", algorithm="tiled", shape_params=(("m", 64),))
    w, p = b.build()
    with pytest.raises(SpecError, match="UNSUPPORTED_KERNEL"):
        Lowerer(h).compile(w, p)


def test_duplicate_activity_demands_rejected():
    from distributed_llm_simulation import ActivitySpec

    with pytest.raises(ValueError, match="DUPLICATE_DEMAND"):
        ActivitySpec(id="op", kind="compute", slot_demands=(("s", 1), ("s", 1)))


def test_bit_packed_storage_and_allocation(tmp_path):
    tensor = TensorSpec(
        id="packed", shape=(3,), dtype="int4", storage_bytes=2, initial_pool_ids=("ram0",)
    )
    assert tensor.storage_bytes == 2
    with pytest.raises(ValueError, match="INVALID_SIZE"):
        TensorSpec(id="bad", shape=(3,), dtype="int4", storage_bytes=1)


def test_friendly_throughput_units_and_numeric_boolean_rejection(tmp_path):
    data = load_hardware("profiles/hardware/discrete.yaml").model_dump(mode="json")
    data["kernel_profiles"][0]["effective_ops_per_s"] = "1 TFLOP/s"
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(data))
    assert load_hardware(path).kernel_profiles[0].effective_ops_per_s == 1e12
    data["resources"][0]["capacity"] = True
    with pytest.raises(ValueError, match="boolean"):
        HardwareSpec.model_validate(data)
