"""Small generic fixtures shared by tests, demos and research experiments."""

from __future__ import annotations

from ..spec import OperatorSpec, PlacementPlan, RequestSpec, TensorSpec, WorkloadSpec


class DAGBuilder:
    def __init__(self, name):
        self.name = name
        self.tensors, self.operators = {}, []
        self.placements = {}

    def tensor(self, tid, size, dtype="fp32", pools=(), persistent=False, shape=()):
        if tid in self.tensors:
            raise ValueError(f"duplicate tensor {tid}")
        self.tensors[tid] = TensorSpec(
            id=tid,
            storage_bytes=size,
            dtype=dtype,
            shape=shape,
            initial_pool_ids=tuple(pools),
            persistent=persistent,
        )
        return tid

    def op(self, oid, inputs, outputs, compute, *, op_type="gemm", dtype="fp32", **cost):
        if oid in self.placements:
            raise ValueError(f"duplicate operator {oid}")
        for tid in dict.fromkeys(inputs):
            tensor = self.tensors[tid]
            self.tensors[tid] = tensor.model_copy(
                update={"consumer_ids": tensor.consumer_ids + (oid,)}
            )
        for tid in outputs:
            tensor = self.tensors[tid]
            if tensor.producer_id or tensor.initial_pool_ids:
                raise ValueError(f"tensor {tid} already has a producer/initial replica")
            self.tensors[tid] = tensor.model_copy(update={"producer_id": oid})
        self.operators.append(
            OperatorSpec(
                id=oid,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                op_type=op_type,
                input_dtype=dtype,
                output_dtype=dtype,
                accumulate_dtype="fp32",
                **cost,
            )
        )
        self.placements[oid] = compute

    def build(self, requests=(), final_output_pools=()):
        # Revalidate copies to keep builder mutation outside immutable specification contracts.
        workload = WorkloadSpec.model_validate(
            {
                "name": self.name,
                "tensors": [t.model_dump() for t in self.tensors.values()],
                "operators": [o.model_dump() for o in self.operators],
                "requests": [r.model_dump() for r in requests],
            }
        )
        return workload, PlacementPlan(
            op_compute=tuple(self.placements.items()), final_output_pools=tuple(final_output_pools)
        )


class SyntheticWorkload:
    @staticmethod
    def pipeline(
        *,
        requests=1,
        arrivals=None,
        cpu="cpu0",
        gpu="gpu0",
        ram="ram0",
        duration_a=0.002,
        duration_b=0.003,
        tensor_bytes=8_000_000,
    ):
        if type(requests) is not int or requests <= 0:
            raise ValueError("requests must be a positive integer")
        arrivals = [0.0] * requests if arrivals is None else arrivals
        if len(arrivals) != requests:
            raise ValueError("arrivals length must match requests")
        builder = DAGBuilder("two_stage_pipeline")
        reqs = []
        for i in range(requests):
            q = f"r{i}"
            x = builder.tensor(f"{q}/input", 4, pools=(ram,))
            y = builder.tensor(f"{q}/activation", tensor_bytes)
            z = builder.tensor(f"{q}/output", 4)
            builder.op(f"{q}/A", [x], [y], cpu, duration_s=duration_a, request_id=q)
            builder.op(f"{q}/B", [y], [z], gpu, duration_s=duration_b, request_id=q)
            reqs.append(RequestSpec(id=q, arrival_s=arrivals[i], output_tensor_ids=(z,)))
        return builder.build(reqs)

    @staticmethod
    def fork_join(*, left="cpu0", right="gpu0", pool="ram0", duration_s=2, tensor_bytes=16):
        b = DAGBuilder("fork_join")
        x = b.tensor("input", tensor_bytes, pools=(pool,))
        a, c, out = (b.tensor(t, tensor_bytes) for t in ("left", "right", "output"))
        b.op("left", [x], [a], left, duration_s=duration_s, request_id="r0")
        b.op("right", [x], [c], right, duration_s=duration_s, request_id="r0")
        b.op("join", [a, c], [out], left, duration_s=0, request_id="r0")
        return b.build([RequestSpec(id="r0", output_tensor_ids=(out,))])

    @staticmethod
    def tiled_matmul(*, tiles=2, tile_size=16, compute="cpu0", pool="ram0"):
        if tiles < 1 or tile_size < 1:
            raise ValueError("positive tile count and tile size required")
        b = DAGBuilder("tiled_matmul")
        size = tile_size**2 * 4
        a = {
            (i, k): b.tensor(f"A/{i}/{k}", size, pools=(pool,))
            for i in range(tiles)
            for k in range(tiles)
        }
        c = {
            (k, j): b.tensor(f"B/{k}/{j}", size, pools=(pool,))
            for k in range(tiles)
            for j in range(tiles)
        }
        outputs = []
        for i in range(tiles):
            for j in range(tiles):
                products = []
                for k in range(tiles):
                    out = b.tensor(f"P/{i}/{j}/{k}", size)
                    b.op(
                        f"gemm/{i}/{j}/{k}",
                        [a[i, k], c[k, j]],
                        [out],
                        compute,
                        flops=2 * tile_size**3,
                        read_bytes=((pool, 2 * size),),
                        write_bytes=((pool, size),),
                        request_id="r0",
                    )
                    products.append(out)
                out = b.tensor(f"C/{i}/{j}", size)
                b.op(
                    f"sum/{i}/{j}",
                    products,
                    [out],
                    compute,
                    op_type="elementwise",
                    flops=(tiles - 1) * tile_size**2,
                    read_bytes=((pool, len(products) * size),),
                    write_bytes=((pool, size),),
                    request_id="r0",
                )
                outputs.append(out)
        return b.build([RequestSpec(id="r0", output_tensor_ids=tuple(outputs))])
