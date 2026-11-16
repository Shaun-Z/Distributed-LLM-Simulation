"""Finite decoder-only transformer expansion, isolated from the generic runtime.

This is a synthetic, unfused ordinary/GQA attention model, not a real model loader.
KV is append-only, and all decode steps depend on the previous output token.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from ..compiler.collectives import ring_all_reduce
from ..spec import RequestSpec
from ..spec.base import require
from .synthetic import DAGBuilder


def _costed_op(
    builder,
    prefix,
    device,
    pool,
    dtype,
    request,
    name,
    inputs,
    size,
    flops,
    family="gemm",
    extra_outputs=(),
    controls=(),
):
    tid = builder.tensor(f"{prefix}/{name}", size, dtype=dtype)
    read = sum(builder.tensors[t].storage_bytes for t in dict.fromkeys(inputs))
    written = size + sum(builder.tensors[t].storage_bytes for t in extra_outputs)
    builder.op(
        f"{prefix}/{name}",
        inputs,
        [tid, *extra_outputs],
        device,
        op_type=family,
        dtype=dtype,
        flops=flops,
        read_bytes=((pool, read),),
        write_bytes=((pool, written),),
        control_predecessors=controls,
        request_id=request,
    )
    return tid


@dataclass(frozen=True)
class TransformerConfig:
    layers: int = 2
    hidden: int = 64
    heads: int = 4
    kv_heads: int = 4
    intermediate: int = 128
    prompt_length: int = 8
    decode_steps: int = 2
    batch: int = 1
    microbatches: int = 1
    dtype: str = "bf16"
    gated_mlp: bool = True

    def validate(self, tp):
        values = (
            self.layers,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.intermediate,
            self.prompt_length,
            self.batch,
            self.microbatches,
        )
        require(
            all(type(v) is int and v > 0 for v in values)
            and type(self.decode_steps) is int
            and self.decode_steps >= 0,
            "INVALID_TRANSFORMER",
            "dimensions",
            "positive integer dimensions and nonnegative decode_steps required",
        )
        require(
            self.hidden % self.heads == 0
            and self.heads % self.kv_heads == 0
            and self.heads % tp == 0
            and self.kv_heads % tp == 0
            and self.intermediate % tp == 0,
            "INVALID_TRANSFORMER",
            "shards",
            "head/hidden/intermediate dimensions must divide rank counts",
        )
        require(self.dtype in ("fp32", "bf16", "fp16"), "INVALID_DTYPE", "transformer", self.dtype)


class TransformerWorkloadBuilder:
    def __init__(self, config=None):
        self.config = config or TransformerConfig()

    def build(self, *, stages=(("gpu0",),), pools=(("hbm0",),), offload_pool=None, arrival_s=0):
        c = self.config
        require(
            bool(stages) and len(stages) == len(pools) and bool(stages[0]),
            "INVALID_PLACEMENT",
            "stages",
            "nonempty stages and pools required",
        )
        tp = len(stages[0])
        require(
            all(len(s) == len(p) == tp for s, p in zip(stages, pools, strict=False)),
            "INVALID_PLACEMENT",
            "stages",
            "uniform TP degree required",
        )
        c.validate(tp)
        require(
            len(stages) <= c.layers, "INVALID_PLACEMENT", "stages", "more PP stages than layers"
        )
        b = DAGBuilder("transformer_decoder")
        element = 4 if c.dtype == "fp32" else 2
        h, f, batch = c.hidden, c.intermediate, c.batch
        kv_width = c.kv_heads * (h // c.heads)
        weights, reload = {}, []

        def stage_of(layer):
            return min(layer * len(stages) // c.layers, len(stages) - 1)

        # Non-request weights are shared across microbatches and token steps.
        for layer in range(c.layers):
            stage = stage_of(layer)
            for rank in range(tp):
                counts = {
                    "qkv": h * (h + 2 * kv_width) // tp,
                    "out": h * h // tp,
                    "up": h * f * (2 if c.gated_mlp else 1) // tp,
                    "down": f * h // tp,
                }
                for name, elements in counts.items():
                    tid = b.tensor(
                        f"weights/{layer}/{rank}/{name}",
                        elements * element,
                        dtype=c.dtype,
                        pools=(offload_pool or pools[stage][rank],),
                        persistent=True,
                    )
                    weights[layer, rank, name] = tid
                    if offload_pool:
                        reload.append(tid)
        requests = []
        for micro in range(c.microbatches):
            q = f"request{micro}"
            kv_blocks = {(layer, rank): [] for layer in range(c.layers) for rank in range(tp)}
            previous_token_op = None
            token_outputs = []
            for step in range(c.decode_steps + 1):
                seq = c.prompt_length if step == 0 else 1
                context = c.prompt_length + step
                input_size = batch * seq * h * element
                x = b.tensor(
                    f"{q}/step{step}/input", input_size, dtype=c.dtype, pools=(pools[0][0],)
                )
                current = (x,) * tp
                for layer in range(c.layers):
                    stage = stage_of(layer)
                    devices, memory = stages[stage], pools[stage]
                    prefix = f"{q}/step{step}/layer{layer}"
                    partials = []
                    for rank in range(tp):
                        op = partial(
                            _costed_op,
                            b,
                            f"{prefix}/rank{rank}",
                            devices[rank],
                            memory[rank],
                            c.dtype,
                            q,
                        )

                        controls = (previous_token_op,) if previous_token_op else ()
                        norm = op(
                            "norm1",
                            [current[rank]],
                            input_size,
                            5 * batch * seq * h,
                            "norm",
                            controls=controls,
                        )
                        block_size = 2 * batch * seq * kv_width // tp * element
                        block = b.tensor(
                            f"{prefix}/rank{rank}/kv_append",
                            block_size,
                            dtype=c.dtype,
                            persistent=True,
                        )
                        qkv = op(
                            "qkv",
                            [norm, weights[layer, rank, "qkv"]],
                            batch * seq * (h + 2 * kv_width) // tp * element,
                            2 * batch * seq * h * (h + 2 * kv_width) // tp,
                            extra_outputs=(block,),
                        )
                        kv_blocks[layer, rank].append(block)
                        attention = op(
                            "attention",
                            [qkv, *kv_blocks[layer, rank]],
                            input_size // tp,
                            4 * batch * seq * context * h // tp,
                            "attention",
                        )
                        projected = op(
                            "projection",
                            [attention, weights[layer, rank, "out"]],
                            input_size,
                            2 * batch * seq * h * h // tp,
                        )
                        partials.append(projected)
                    reduced = ring_all_reduce(
                        b,
                        f"{prefix}/attention_ar",
                        partials,
                        devices,
                        memory,
                        input_size,
                        dtype=c.dtype,
                        request_id=q,
                    )
                    mlp = []
                    for rank in range(tp):
                        prefix_rank = f"{prefix}/rank{rank}"

                        local = partial(
                            _costed_op, b, prefix_rank, devices[rank], memory[rank], c.dtype, q
                        )

                        residual = local(
                            "residual1",
                            [current[rank], reduced[rank]],
                            input_size,
                            batch * seq * h,
                            "elementwise",
                        )
                        norm = local("norm2", [residual], input_size, 5 * batch * seq * h, "norm")
                        up = local(
                            "mlp_up",
                            [norm, weights[layer, rank, "up"]],
                            batch * seq * f // tp * element,
                            (4 if c.gated_mlp else 2) * batch * seq * h * f // tp,
                            "gemm",
                        )
                        down = local(
                            "mlp_down",
                            [up, weights[layer, rank, "down"]],
                            input_size,
                            2 * batch * seq * h * f // tp,
                            "gemm",
                        )
                        mlp.append(down)
                    reduced_mlp = ring_all_reduce(
                        b,
                        f"{prefix}/mlp_ar",
                        mlp,
                        devices,
                        memory,
                        input_size,
                        dtype=c.dtype,
                        request_id=q,
                    )
                    current_out = []
                    for rank in range(tp):
                        tid = b.tensor(f"{prefix}/rank{rank}/output", input_size, dtype=c.dtype)
                        b.op(
                            f"{prefix}/rank{rank}/residual2",
                            [reduced[rank], reduced_mlp[rank], current[rank]],
                            [tid],
                            devices[rank],
                            op_type="elementwise",
                            dtype=c.dtype,
                            flops=2 * batch * seq * h,
                            read_bytes=((memory[rank], 3 * input_size),),
                            write_bytes=((memory[rank], input_size),),
                            request_id=q,
                        )
                        current_out.append(tid)
                    current = tuple(current_out)
                token = b.tensor(f"{q}/token{step}", 4, dtype=c.dtype)
                previous_token_op = f"{q}/emit{step}"
                b.op(
                    previous_token_op,
                    [current[0]],
                    [token],
                    stages[-1][0],
                    op_type="sync",
                    dtype=c.dtype,
                    duration_s=0,
                    request_id=q,
                )
                token_outputs.append(token)
            markers = tuple((f"token_{i}", tid) for i, tid in enumerate(token_outputs))
            requests.append(
                RequestSpec(
                    id=q,
                    arrival_s=arrival_s,
                    output_tensor_ids=tuple(token_outputs),
                    markers=markers,
                )
            )
        workload, plan = b.build(requests)
        if reload:
            plan = plan.model_copy(update={"reload_tensor_ids": tuple(reload)})
        return workload, plan


def llm_metrics(result):
    requests = []
    for req in result.requests:
        markers = sorted(
            (
                (int(k.split("_")[1]), v)
                for k, v in req["markers"].items()
                if k.startswith("token_")
            ),
            key=lambda x: x[0],
        )
        times = [t for _, t in markers if t is not None]
        requests.append(
            {
                "request_id": req["request_id"],
                "ttft_s": times[0] - req["arrival_s"] if times else None,
                "itl_s": [b - a for a, b in zip(times, times[1:], strict=False)],
                "tpot_s": (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None,
                "delivered_tokens": len(times),
            }
        )
    tokens = sum(r["delivered_tokens"] for r in requests)
    return {
        "requests": requests,
        "output_tokens_per_s": tokens / result.makespan_s if result.makespan_s else None,
        "kv_policy": "append-only immutable blocks retained through run; no paging/prefix sharing",
    }
