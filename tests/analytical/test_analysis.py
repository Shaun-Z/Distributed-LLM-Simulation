import pytest

from distributed_llm_simulation import (
    Lowerer,
    Simulator,
    load_hardware,
)
from distributed_llm_simulation.analysis.bounds import (
    capacity_integral_bound,
    critical_path_bound,
    execution_bounds,
    gap,
)
from distributed_llm_simulation.analysis.metrics import (
    interval_union,
    nearest_rank,
)
from distributed_llm_simulation.analysis.search import (
    Evaluator,
    enumerate_placements,
    exact_serial_oracle,
    heft_inspired_plan,
    pareto_frontier,
)
from distributed_llm_simulation.models.calibration import (
    Measurement,
    fit_copy_profile,
    validate_copy_profile,
)
from distributed_llm_simulation.models.network import (
    CapacityProcess,
    independent_rng,
    markov_capacity,
    poisson_arrivals,
)
from distributed_llm_simulation.runtime.arbitration import weighted_max_min
from distributed_llm_simulation.workloads import SyntheticWorkload


def test_parking_lot_progressive_filling():
    rates = weighted_max_min(
        {"a": {"r1": 1}, "b": {"r1": 1, "r2": 1}, "c": {"r2": 1}}, {"r1": 10, "r2": 100}
    )
    assert rates == pytest.approx({"a": 5, "b": 5, "c": 95})
    assert weighted_max_min({"f": {"r": 2}}, {"r": 100}) == {"f": 50}
    with pytest.raises(ValueError):
        weighted_max_min({"f": {}}, {})


def test_bounds_and_exact_small_oracle():
    h = load_hardware("profiles/hardware/uma_workloads.yaml")
    w, p = SyntheticWorkload.fork_join(left="uc0", right="uc0", pool="uma0", duration_s=2)
    e = Lowerer(h).compile(w, p)
    r = Simulator(h).run(e)
    global_bound = critical_path_bound(h, w)
    assert global_bound.value_s == 2
    bounds = execution_bounds(h, e)
    assert max(b.value_s for b in bounds) == pytest.approx(r.makespan_s)
    exact = exact_serial_oracle(h, e)
    assert exact["proven_optimal_in_scope"] is True
    assert exact["best"].makespan_s == pytest.approx(r.makespan_s)
    assert exact_serial_oracle(h, e, max_orders=1)["proven_optimal_in_scope"] is False
    assert gap(r.makespan_s, global_bound.value_s)["ratio"] > 1


def test_search_cache_and_scope():
    h = load_hardware("profiles/hardware/uma_workloads.yaml")
    w, p = SyntheticWorkload.fork_join(left="uc0", right="ug0", pool="uma0", duration_s=0.001)
    evaluator = Evaluator(h, w)
    a, b = evaluator.evaluate(p), evaluator.evaluate(p)
    assert a is b
    result = enumerate_placements(h, w, max_candidates=100)
    assert result["total_placements"] == 8 and result["exhausted"] is True
    assert result["best"].result.status == "SUCCESS"
    assert pareto_frontier(result["evaluations"])
    assert heft_inspired_plan(h, w).result.status == "SUCCESS"


def test_metrics_union_and_quantiles():
    assert interval_union([(0, 2), (1, 3), (4, 5)]) == 4
    assert nearest_rank([1, 2, 3], 0.95) == 3
    assert nearest_rank([], 0.5) is None
    p = CapacityProcess(resource_id="r", points=((0, 100), (5, 0), (8, 100)))
    assert capacity_integral_bound(p, 1000) == 13


def test_independent_reproducible_random_streams():
    assert independent_rng(42, "arrivals").random() != independent_rng(42, "environment").random()
    kwargs = dict(
        resource_id="wan",
        seed=42,
        horizon_s=100,
        capacities=(0, 100),
        transition_rates=((0, 0.1), (0.1, 0)),
    )
    p = markov_capacity(**kwargs)
    assert p == markov_capacity(**kwargs)
    assert all(c >= 0 for _, c in p.points)
    value = p.value_at(20)
    p.value_at(40)
    assert p.value_at(20) == value
    assert poisson_arrivals(seed=42, rate_per_s=2, count=5) == poisson_arrivals(
        seed=42, rate_per_s=2, count=5
    )
    with pytest.raises(ValueError):
        CapacityProcess(resource_id="r", points=((1, 10),))
    with pytest.raises(ValueError):
        CapacityProcess(resource_id="r", points=((0, 10), (0, 20)))


def test_calibration_and_independent_validation():
    def record(i, size, split):
        return Measurement(
            id=str(i),
            device_identity="synthetic",
            software_version="test",
            measured_at="2026-11-13",
            methodology="synchronized synthetic oracle",
            source_uri="synthetic://copy",
            synthetic=True,
            split=split,
            operation="copy",
            size_bytes=size,
            duration_s=0.1 + size / 100,
        )

    profile = fit_copy_profile([record(0, 100, "calibration"), record(1, 1000, "calibration")])
    assert profile.setup_s == pytest.approx(0.1) and profile.payload_Bps == pytest.approx(100)
    report = validate_copy_profile(profile, [record(2, 500, "validation")])
    assert report["p95_relative_error"] == pytest.approx(0)
    with pytest.raises(ValueError):
        validate_copy_profile(profile, [record(0, 100, "calibration")])


def test_compute_shared_controller_coefficients():
    from distributed_llm_simulation.models.compute import compute_cost
    from distributed_llm_simulation.spec import HardwareSpec, OperatorSpec

    h = load_hardware("profiles/hardware/uma_workloads.yaml")
    data = h.model_dump()
    for r in data["resources"]:
        if r["id"] == "uma_bw":
            r["capacity"] = 100
    for c in data["channels"]:
        c["resource_demands"] = (("uma_bw", 2),)
    h = HardwareSpec.model_validate(data)
    compute = next(n for n in h.nodes if n.id == "uc0")
    kernel = next(k for k in h.kernel_profiles if k.id == "uc0_gemm_fp32")
    op = OperatorSpec(
        id="io", op_type="gemm", read_bytes=(("uma0", 100),), write_bytes=(("uma0", 50),)
    )
    _, duration, work = compute_cost(op, compute, kernel, h)
    assert duration == 3 and work == (("uma_bw", 300),)


def test_markov_model_label_and_replication_stats():
    from distributed_llm_simulation import ActivitySpec, ExecutionPlan, SimulationConfig
    from distributed_llm_simulation.analysis.metrics import replication_statistics
    from distributed_llm_simulation.spec import HardwareSpec, ResourceSpec

    h = HardwareSpec(
        name="n",
        resources=(ResourceSpec(id="wan", kind="rate", unit="byte/s", capacity=100),),
        nodes=(),
    )
    e = ExecutionPlan(
        activities=(
            ActivitySpec(
                id="copy",
                kind="transfer",
                work_unit="byte",
                total_work=100,
                rate_demands=(("wan", 1),),
            ),
        )
    )
    results = []
    for seed in range(3):
        process = markov_capacity(
            "wan", seed=seed, horizon_s=10, capacities=(50, 100), transition_rates=((0, 1), (1, 0))
        )
        r = Simulator(
            h,
            config=SimulationConfig(network_mode="fluid", seed=seed, capacity_processes=(process,)),
        ).run(e)
        assert r.status == "SUCCESS" and r.metrics["model_level"] == "D"
        results.append(r)
    stats = replication_statistics(results)
    assert stats["successful_replications"] == 3 and stats["failure_rate"] == 0


def test_parallel_oracle_can_idle_and_prove_conditional_optimum():
    from distributed_llm_simulation import ActivitySpec, ExecutionPlan
    from distributed_llm_simulation.analysis.search import exact_schedule_oracle
    from distributed_llm_simulation.spec import HardwareSpec, ResourceSpec

    h = HardwareSpec(
        name="job_shop",
        resources=tuple(
            ResourceSpec(id=r, kind="slots", unit="count", capacity=1) for r in ("x", "y")
        ),
        nodes=(),
    )
    e = ExecutionPlan(
        activities=(
            ActivitySpec(id="a_long", kind="compute", total_work=10, slot_demands=(("x", 1),)),
            ActivitySpec(id="b_trigger", kind="compute", total_work=1, slot_demands=(("y", 1),)),
            ActivitySpec(
                id="c_urgent",
                kind="compute",
                total_work=1,
                predecessor_ids=("b_trigger",),
                slot_demands=(("x", 1),),
            ),
            ActivitySpec(
                id="d_tail",
                kind="compute",
                total_work=20,
                predecessor_ids=("c_urgent",),
                slot_demands=(("y", 1),),
            ),
        )
    )
    assert Simulator(h).run(e).makespan_s == 31
    report = exact_schedule_oracle(h, e)
    assert report["proven_optimal_in_scope"]
    assert report["incumbent_s"] == 22 and report["best_bound_s"] == 22
    capped = exact_schedule_oracle(h, e, max_orders=1)
    assert not capped["proven_optimal_in_scope"] and capped["best_bound_s"] <= capped["incumbent_s"]


def test_explicit_ring_matches_textbook_when_local_work_ignored():
    from distributed_llm_simulation import SimulationConfig
    from distributed_llm_simulation.compiler.collectives import ring_all_reduce
    from distributed_llm_simulation.spec import HardwareSpec
    from distributed_llm_simulation.workloads import DAGBuilder

    h = load_hardware("profiles/hardware/two_hosts.yaml")
    data = h.model_dump()
    for k in data["kernel_profiles"]:
        if k["op_type"] == "sync":
            k["launch_s"] = 0
    for r in data["resources"]:
        if r["kind"] == "rate":
            r["capacity"] = 100 if r["id"].startswith("wan_") else 1e20
    for c in data["channels"]:
        if c["id"].startswith("hbm_"):
            c.update(payload_Bps=100, latency_s=0.1)
    data["routes"] = tuple(r for r in data["routes"] if r["id"].startswith("hbm_"))
    h = HardwareSpec.model_validate(data)
    b = DAGBuilder("ring")
    inputs = [b.tensor(f"x{i}", 1000, pools=(f"hbm{i}",)) for i in range(2)]
    ring_all_reduce(
        b, "allreduce", inputs, ("gpu0", "gpu1"), ("hbm0", "hbm1"), 1000, model_local_work=False
    )
    w, p = b.build()
    r = Simulator(h, config=SimulationConfig(network_mode="fluid")).run(Lowerer(h).compile(w, p))
    assert r.status == "SUCCESS"
    assert r.makespan_s == pytest.approx(2 * 0.1 + 2 * (2 - 1) / 2 * 1000 / 100)
    assert r.metrics["bytes_transferred"] == 2000
