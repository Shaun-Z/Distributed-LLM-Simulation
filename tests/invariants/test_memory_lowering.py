import pytest

from distributed_llm_simulation import Lowerer, Simulator, load_hardware
from distributed_llm_simulation.models.memory import MemoryManager
from distributed_llm_simulation.spec import HardwareSpec, RequestSpec, SpecError
from distributed_llm_simulation.workloads import DAGBuilder, SyntheticWorkload


def modified(h, change):
    data = h.model_dump(mode="json")
    change(data)
    return HardwareSpec.model_validate(data)


def test_atomic_oom_rollback_and_alignment():
    h = load_hardware("profiles/hardware/discrete.yaml")
    m = MemoryManager(h)
    ok, _ = m.try_reserve_many([("a", "ram0", 1), ("b", "hbm0", 10**20)])
    assert not ok and not m.allocations and all(v == 0 for v in m.used.values())
    assert all(v == 0 for v in m.peak.values())
    assert m.try_reserve_many([("a", "ram0", 1)])[0]
    assert m.used["ram0"] == 64
    m.mark_valid("a")
    m.pin("a")
    with pytest.raises(AssertionError):
        m.release("a")
    m.unpin("a")
    m.release("a")
    m.assert_invariants()


def test_fanout_one_copy_and_source_pinned_to_completion():
    h = load_hardware("profiles/hardware/pipeline_oracle.yaml")
    b = DAGBuilder("fanout")
    x = b.tensor("input", 8_000_000, pools=("ram0",))
    y, z = b.tensor("left", 4), b.tensor("right", 4)
    b.op("left", [x], [y], "gpu0", duration_s=0.003, request_id="q")
    b.op("right", [x], [z], "gpu0", duration_s=0.003, request_id="q")
    w, p = b.build([RequestSpec(id="q", output_tensor_ids=(y, z))])
    e = Lowerer(h).compile(w, p)
    assert len([a for a in e.activities if a.kind == "transfer"]) == 1
    r = Simulator(h).run(e)
    assert r.status == "SUCCESS" and r.makespan_s == pytest.approx(0.0071)
    local = next(rep for rep in e.replicas if rep.tensor_id == x and rep.pool_id == "hbm0")
    free = next(
        row for row in r.trace if row["event_type"] == "FREED" and row["allocation_id"] == local.id
    )
    assert free["sim_time"] == r.finish_time_s
    source = next(rep for rep in e.replicas if rep.tensor_id == x and rep.pool_id == "ram0")
    source_free = next(
        row for row in r.trace if row["event_type"] == "FREED" and row["allocation_id"] == source.id
    )
    assert source_free["sim_time"] == pytest.approx(0.0011)


def test_workspace_and_output_atomic_capacity_block():
    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    h = modified(
        h,
        lambda d: next(r for r in d["resources"] if r["id"] == "ram_capacity").update(capacity=128),
    )
    b = DAGBuilder("oom")
    x = b.tensor("x", 64, pools=("ram0",))
    y = b.tensor("y", 64)
    b.op("op", [x], [y], "cpu0", duration_s=1, workspace_bytes=64)
    w, p = b.build()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert r.status == "NO_PROGRESS"
    assert r.timeline[0]["start_s"] is None
    assert r.metrics["peak_memory_bytes"]["ram0"] == 64
    assert len(r.failure_snapshot["allocations"]) == 1
    assert r.failure_snapshot["waiting"]


def test_output_return_includes_transfer():
    h = load_hardware("profiles/hardware/pipeline_oracle.yaml")
    w, p = SyntheticWorkload.pipeline()
    p = p.model_copy(update={"final_output_pools": (("r0/output", "ram0"),)})
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert r.makespan_s == pytest.approx(0.0061 + 0.0001 + 4 / 8e9)


def test_uma_one_owner_and_zero_copy():
    h = load_hardware("profiles/hardware/uma_workloads.yaml")
    w, p = SyntheticWorkload.fork_join(left="uc0", right="ug0", pool="uma0")
    e = Lowerer(h).compile(w, p)
    assert not any(a.kind == "transfer" for a in e.activities)
    assert len([x for x in e.replicas if x.tensor_id == "input"]) == 1
    # Fixed kernels reserve UMA bandwidth exclusively, even across CPU/GPU slots.
    r = Simulator(h).run(e)
    assert r.status == "SUCCESS" and r.makespan_s == pytest.approx(4.000003)


def test_uma_alias_shared_capacity_and_conflicting_owners_rejected():
    h = load_hardware("profiles/hardware/uma.yaml")
    data = h.model_dump(mode="json")
    memory = next(n for n in data["nodes"] if n["kind"] == "memory")
    data["nodes"] += [{**memory, "id": "uma_view"}]
    h = HardwareSpec.model_validate(data)
    m = MemoryManager(h)
    assert h.pool("uma_view") == h.pool("uma0") and len(m.capacity) == 1
    data["nodes"][-1]["physical_pool_id"] = "fake_second_pool"
    with pytest.raises(ValueError, match="multiple physical owners"):
        HardwareSpec.model_validate(data)


def test_staging_without_p2p_and_illegal_route():
    h = load_hardware("profiles/hardware/two_hosts.yaml")
    data = h.model_dump(mode="json")
    data["routes"] = [r for r in data["routes"] if not r["id"].startswith("hbm_")]
    h = HardwareSpec.model_validate(data)
    b = DAGBuilder("staging")
    x = b.tensor("x", 256, pools=("hbm0",))
    y = b.tensor("y", 4)
    b.op("op", [x], [y], "gpu1", duration_s=0.001)
    w, p = b.build()
    e = Lowerer(h).compile(w, p)
    moves = [a for a in e.activities if a.kind == "transfer"]
    assert len(moves) == 3
    assert {r.pool_id for r in e.replicas if r.tensor_id == "x"} == {"hbm0", "ram0", "ram1", "hbm1"}
    assert Simulator(h).run(e).status == "SUCCESS"
    data["routes"] = []
    h = HardwareSpec.model_validate(data)
    with pytest.raises(SpecError, match="NO_ROUTE"):
        Lowerer(h).compile(w, p)


def test_pool_budget_single_6_gib_allocation():
    h = load_hardware("profiles/hardware/uma_workloads.yaml")
    h = modified(
        h,
        lambda d: next(r for r in d["resources"] if r["id"] == "uma_capacity").update(
            capacity=8 * 2**30
        ),
    )
    b = DAGBuilder("large_shared")
    x = b.tensor("x", 6 * 2**30, pools=("uma0",))
    b.op("cpu", [x], [], "uc0", duration_s=1)
    b.op("gpu", [x], [], "ug0", duration_s=1)
    w, p = b.build()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert r.status == "SUCCESS" and r.metrics["peak_memory_bytes"]["uma0"] == 6 * 2**30


def test_workspace_released():
    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    b = DAGBuilder("workspace")
    x = b.tensor("x", 64, pools=("ram0",))
    y = b.tensor("y", 64)
    b.op("op", [x], [y], "cpu0", duration_s=1, workspace_bytes=128)
    w, p = b.build()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert any(
        e["event_type"] == "FREED" and e["allocation_id"].startswith("workspace:") for e in r.trace
    )
