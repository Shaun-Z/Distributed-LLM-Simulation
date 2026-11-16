import json

import pytest

from distributed_llm_simulation import ActivitySpec, ExecutionPlan, SimulationConfig, Simulator
from distributed_llm_simulation.io import write_json
from distributed_llm_simulation.models.network import CapacityProcess
from distributed_llm_simulation.spec import HardwareSpec, ResourceSpec
from distributed_llm_simulation.web.results import reconstruct_series


def test_shared_and_time_varying_service_integrals(tmp_path):
    h = HardwareSpec(
        name="network",
        resources=(ResourceSpec(id="bw", kind="rate", unit="byte/s", capacity=100),),
        nodes=(),
    )
    a = ActivitySpec(
        id="a", kind="transfer", work_unit="byte", total_work=1000, rate_demands=(("bw", 1),)
    )
    b = ActivitySpec(
        id="b",
        kind="transfer",
        work_unit="byte",
        total_work=100,
        rate_demands=(("bw", 1),),
        release_s=5,
    )
    for name, activities, processes, expected in [
        ("sharing", (a, b), (), 11),
        ("changing", (a,), (CapacityProcess(resource_id="bw", points=((0, 100), (5, 50))),), 15),
        (
            "pause",
            (a,),
            (CapacityProcess(resource_id="bw", points=((0, 100), (5, 0), (8, 100))),),
            13,
        ),
    ]:
        d = tmp_path / name
        d.mkdir()
        e = ExecutionPlan(activities=activities)
        c = SimulationConfig(network_mode="fluid", capacity_processes=processes)
        r = Simulator(h, config=c).run(e)
        assert r.finish_time_s == expected
        for filename, data in [
            ("hardware.json", h.model_dump(mode="json")),
            ("execution.json", e.model_dump(mode="json")),
            ("config.json", c.model_dump(mode="json")),
            ("summary.json", {"finish_time_s": r.finish_time_s}),
        ]:
            write_json(d / filename, data)
        (d / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in r.trace))
        curves = reconstruct_series(d)
        service = next(s for s in curves["series"] if s["kind"] == "service" and s["id"] == "bw")[
            "points"
        ]
        integral = sum((b[0] - a[0]) * a[1] for a, b in zip(service, service[1:], strict=False))
        assert integral == pytest.approx(sum(a.total_work for a in activities))
        assert max(p[1] for p in service) <= 100
