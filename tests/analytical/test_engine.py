import pytest

from distributed_llm_simulation import (
    ActivitySpec,
    ExecutionPlan,
    Lowerer,
    SimulationConfig,
    Simulator,
    load_hardware,
)
from distributed_llm_simulation.models.network import CapacityProcess
from distributed_llm_simulation.spec import HardwareSpec, ResourceSpec
from distributed_llm_simulation.workloads.synthetic import SyntheticWorkload


def network(capacity=100, resources=None):
    return HardwareSpec(
        name="network",
        resources=tuple(
            resources or [ResourceSpec(id="bw", kind="rate", unit="byte/s", capacity=capacity)]
        ),
        nodes=(),
    )


def flow(aid, size, release=0, demands=(("bw", 1),), weight=1):
    return ActivitySpec(
        id=aid,
        kind="transfer",
        total_work=size,
        work_unit="byte",
        rate_demands=demands,
        release_s=release,
        weight=weight,
    )


def run_flows(flows, *, hardware=None, processes=(), mode="fluid", **config):
    return Simulator(
        hardware or network(),
        config=SimulationConfig(network_mode=mode, capacity_processes=processes, **config),
    ).run(ExecutionPlan(activities=tuple(flows)))


def ends(result):
    return {r["activity_id"]: r["finish_s"] for r in result.timeline}


@pytest.mark.parametrize(("requests", "expected"), [(1, 0.0061), (2, 0.0091)])
def test_pipeline(requests, expected):
    h = load_hardware("profiles/hardware/pipeline_oracle.yaml")
    w, p = SyntheticWorkload.pipeline(requests=requests)
    e = Lowerer(h).compile(w, p)
    result = Simulator(h).run(e)
    assert result.status == "SUCCESS"
    assert result.makespan_s == pytest.approx(expected)
    assert result.metrics["bytes_transferred"] == requests * 8_000_000
    if requests == 2:
        assert [r["latency_s"] for r in result.requests] == pytest.approx([0.0061, 0.0091])


def test_stale_prediction():
    result = run_flows([flow("f1", 1000), flow("f2", 100, 5)])
    assert ends(result) == pytest.approx({"f1": 11, "f2": 7})
    assert result.metrics["bytes_transferred"] == 1100
    assert result.metrics["stale_predictions_discarded"] >= 1
    bw = result.resources[0]
    assert bw["service_integral"] == pytest.approx(1100)
    assert bw["service_utilization"] == pytest.approx(1)


@pytest.mark.parametrize(
    ("points", "expected"), [(((0, 100), (5, 50)), 15), (((0, 100), (5, 0), (8, 100)), 13)]
)
def test_time_varying(points, expected):
    result = run_flows(
        [flow("f1", 1000)], processes=(CapacityProcess(resource_id="bw", points=points),)
    )
    assert result.makespan_s == expected
    assert result.resources[0]["service_integral"] == pytest.approx(1000)
    assert result.resources[0]["service_utilization"] == pytest.approx(1)


def test_paused_forever():
    result = run_flows([flow("f", 10)], hardware=network(0))
    assert result.status == "NO_PROGRESS"
    assert result.failure_snapshot["zero_rate_activities"]["f"]["remaining_work"] == 10
    assert result.resources[0]["service_utilization"] is None


def test_weighted_and_multi_bottleneck():
    result = run_flows([flow("f1", 100, weight=1), flow("f2", 200, weight=2)])
    assert ends(result) == {"f1": 3, "f2": 3}
    rs = [ResourceSpec(id=r, kind="rate", unit="byte/s", capacity=100) for r in ("a", "b")]
    result = run_flows(
        [
            flow("a", 100, demands=(("a", 1),)),
            flow("both", 100, demands=(("a", 1), ("b", 1))),
            flow("b", 100, demands=(("b", 1),)),
        ],
        hardware=network(resources=rs),
    )
    assert ends(result) == {"a": 2, "both": 2, "b": 2}


def test_exclusive_vs_fluid():
    assert ends(run_flows([flow("a", 100), flow("b", 100)], mode="exclusive")) == {"a": 1, "b": 2}
    assert ends(run_flows([flow("a", 100), flow("b", 100)])) == {"a": 2, "b": 2}


def test_full_and_half_duplex():
    rs = [ResourceSpec(id=r, kind="rate", unit="byte/s", capacity=100) for r in ("tx", "rx")]
    assert (
        run_flows(
            [flow("f", 100, demands=(("tx", 1),)), flow("g", 100, demands=(("rx", 1),))],
            hardware=network(resources=rs),
        ).makespan_s
        == 1
    )
    assert run_flows([flow("f", 100), flow("g", 100)]).makespan_s == 2


def test_completion_before_capacity_change():
    result = run_flows(
        [flow("f", 100), flow("g", 100, 1)],
        processes=(CapacityProcess(resource_id="bw", points=((0, 100), (1, 0), (2, 100))),),
    )
    assert ends(result) == {"f": 1, "g": 3}
    events = [e["event_type"] for e in result.trace if e["sim_time"] == 1]
    assert events.index("ACTIVITY_FINISHED") < events.index("CAPACITY_CHANGE")


def test_zero_chain_and_limits():
    activities = tuple(
        ActivitySpec(id=str(i), kind="sync", predecessor_ids=(str(i - 1),) if i else ())
        for i in range(100)
    )
    e = ExecutionPlan(activities=activities)
    assert Simulator(network()).run(e).makespan_s == 0
    assert (
        Simulator(network(), config=SimulationConfig(max_same_time_transitions=5)).run(e).status
        == "TRUNCATED"
    )
    result = run_flows([flow("f", 1000)], max_sim_time=3)
    assert result.status == "TRUNCATED" and result.finish_time_s == 3
    assert result.timeline[0]["served_work"] == 300


def test_replay_and_repeat_instance():
    sim = Simulator(network(), config=SimulationConfig(network_mode="fluid"))
    e = ExecutionPlan(activities=(flow("f1", 1000), flow("f2", 100, 5)))
    a, b = sim.run(e), sim.run(e)
    assert a == b


def test_fork_on_independent_and_shared_slots():
    resources = [ResourceSpec(id=r, kind="slots", unit="count", capacity=1) for r in ("a", "b")]
    h = network(resources=resources)

    def plan(second):
        return ExecutionPlan(
            activities=(
                ActivitySpec(id="left", kind="compute", total_work=2, slot_demands=(("a", 1),)),
                ActivitySpec(id="right", kind="compute", total_work=2, slot_demands=((second, 1),)),
                ActivitySpec(id="join", kind="sync", predecessor_ids=("left", "right")),
            )
        )

    assert Simulator(h).run(plan("b")).makespan_s == 2
    shared = Simulator(h).run(plan("a"))
    assert shared.makespan_s == 4
    from distributed_llm_simulation.analysis.critical_path import explain_critical_path

    explanation = explain_critical_path(plan("a"), shared)
    assert explanation["accounted_path_s"] == 4
    assert any("resource order" in edge["reason"] for edge in explanation["edges"])


def test_shared_uplink_and_startup_sampling():
    result = run_flows([flow("a", 100), flow("b", 100)])
    assert result.makespan_s == 2
    f = flow("a", 100).model_copy(update={"setup_s": 2})
    result = run_flows(
        [f], processes=(CapacityProcess(resource_id="bw", points=((0, 100), (1, 50))),)
    )
    assert result.makespan_s == 4
    assert result.resources[0]["service_integral"] == 100


def test_large_time_and_small_positive_durations():
    e = ExecutionPlan(
        activities=(
            ActivitySpec(id="long", kind="compute", total_work=1e6),
            ActivitySpec(id="short", kind="compute", total_work=1e-12, predecessor_ids=("long",)),
        )
    )
    result = Simulator(network()).run(e)
    assert result.status == "SUCCESS"
    assert ends(result)["short"] > ends(result)["long"]


def test_high_rate_after_large_timestamp_conserves_bytes():
    h = network(capacity=24e9)
    result = run_flows([flow("f", 500, release=10)], hardware=h)
    assert result.status == "SUCCESS" and result.timeline[0]["served_work"] == 500
    assert result.resources[0]["service_integral"] == 500
    assert result.finish_time_s >= 10 + 500 / 24e9


def test_positive_rate_with_unrepresentable_completion_truncates():
    result = run_flows([flow("f", 1e300)], hardware=network(capacity=1e-300), max_sim_time=100)
    assert result.status == "TRUNCATED" and result.finish_time_s == 100
