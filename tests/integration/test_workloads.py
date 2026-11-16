import pytest

from distributed_llm_simulation import Lowerer, SimulationConfig, Simulator, load_hardware
from distributed_llm_simulation.workloads import (
    SyntheticWorkload,
    TransformerConfig,
    TransformerWorkloadBuilder,
    image_pipeline,
    llm_metrics,
)


@pytest.mark.parametrize("builder", [image_pipeline, SyntheticWorkload.tiled_matmul])
def test_non_llm_plugins(builder):
    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    w, p = builder()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert r.status == "SUCCESS" and r.makespan_s > 0
    assert len(r.requests) > 0 and all(q["status"] == "completed" for q in r.requests)


def test_transformer_decode_kv_and_markers():
    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    config = TransformerConfig(layers=2, prompt_length=8, decode_steps=3)
    w, p = TransformerWorkloadBuilder(config).build()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    assert r.status == "SUCCESS"
    kv = [t for t in w.tensors if t.id.endswith("kv_append")]
    assert (
        sum(t.storage_bytes for t in kv)
        == 2 * config.layers * (config.prompt_length + config.decode_steps) * config.hidden * 2
    )
    assert all(t.persistent for t in kv)
    row = llm_metrics(r)["requests"][0]
    assert (
        row["delivered_tokens"] == 4 and len(row["itl_s"]) == 3 and all(t > 0 for t in row["itl_s"])
    )
    assert row["ttft_s"] > 0
    # A decode step cannot start before the previous emitted token.
    ops = {row["op_id"]: row for row in r.timeline if row["kind"] == "compute"}
    for step in range(1, 4):
        assert (
            ops[f"request0/step{step}/layer0/rank0/norm1"]["start_s"]
            >= ops[f"request0/emit{step - 1}"]["finish_s"]
        )


@pytest.mark.parametrize("tp,pp", [(2, 1), (1, 2)])
def test_tp_and_pp(tp, pp):
    h = load_hardware("profiles/hardware/two_hosts.yaml")
    stages = (("gpu0", "gpu1"),) if tp == 2 else (("gpu0",), ("gpu1",))
    pools = (("hbm0", "hbm1"),) if tp == 2 else (("hbm0",), ("hbm1",))
    w, p = TransformerWorkloadBuilder(TransformerConfig(microbatches=2, decode_steps=1)).build(
        stages=stages, pools=pools
    )
    e = Lowerer(h).compile(w, p)
    r = Simulator(h, config=SimulationConfig(network_mode="fluid")).run(e)
    assert r.status == "SUCCESS"
    assert r.metrics["bytes_transferred"] > 0
    assert any(a.kind == "transfer" for a in e.activities)
    assert len(r.requests) == 2
    if tp == 2:
        assert any(o.op_type == "reduce" for o in w.operators)


def test_offload_reload_reduces_device_weight_residency():
    h = load_hardware("profiles/hardware/discrete_workloads.yaml")
    c = TransformerConfig(layers=4, decode_steps=1)
    w, p = TransformerWorkloadBuilder(c).build()
    r = Simulator(h).run(Lowerer(h).compile(w, p))
    wo, po = TransformerWorkloadBuilder(c).build(offload_pool="ram0")
    eo = Lowerer(h).compile(wo, po)
    ro = Simulator(h).run(eo)
    assert ro.status == "SUCCESS"
    assert ro.metrics["bytes_transferred"] > r.metrics["bytes_transferred"]
    assert ro.metrics["peak_memory_bytes"]["hbm0"] < r.metrics["peak_memory_bytes"]["hbm0"]
    weight_moves = [
        a for a in eo.activities if a.kind == "transfer" and a.tensor_id.startswith("weights/")
    ]
    assert len(weight_moves) == c.layers * 4 * (c.decode_steps + 1)
    # No future-step weight prefetch before its activation/previous-token dependency.
    assert all(a.predecessor_ids for a in weight_moves)
