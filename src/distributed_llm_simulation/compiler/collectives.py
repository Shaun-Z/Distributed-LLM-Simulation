"""Collectives expand into generic operations and explicitly routed messages."""

from __future__ import annotations

import math

from ..spec.base import require
from ..spec.workload import DTYPE_BITS


def _local(
    builder,
    oid,
    inputs,
    outputs,
    device,
    pool,
    dtype,
    request_id,
    model_local_work,
    *,
    reduction=False,
):
    if model_local_work:
        read = sum(builder.tensors[t].storage_bytes for t in dict.fromkeys(inputs))
        write = sum(builder.tensors[t].storage_bytes for t in outputs)
        builder.op(
            oid,
            inputs,
            outputs,
            device,
            op_type="reduce" if reduction else "elementwise",
            dtype=dtype,
            flops=write * 8 / DTYPE_BITS[dtype] if reduction else 0,
            read_bytes=((pool, read),),
            write_bytes=((pool, write),),
            request_id=request_id,
        )
    else:
        # Deliberate idealization for the textbook ring oracle, declared by caller.
        builder.op(
            oid,
            inputs,
            outputs,
            device,
            op_type="sync",
            dtype=dtype,
            duration_s=0,
            request_id=request_id,
        )


def ring_reduce_scatter(
    builder,
    prefix,
    inputs,
    devices,
    pools,
    size_bytes,
    *,
    dtype="fp32",
    request_id=None,
    model_local_work=True,
):
    p = len(devices)
    require(
        p > 0 and len(inputs) == len(pools) == p,
        "INVALID_COLLECTIVE",
        prefix,
        "one input/device/pool per rank",
    )
    if p == 1:
        return tuple(inputs)
    element = max(1, math.ceil(DTYPE_BITS[dtype] / 8))
    size = math.ceil(size_bytes / (p * element)) * element
    chunks = {}
    for rank in range(p):
        outputs = [
            builder.tensor(f"{prefix}/split/{rank}/{chunk}", size, dtype=dtype)
            for chunk in range(p)
        ]
        _local(
            builder,
            f"{prefix}/split/{rank}",
            [inputs[rank]],
            outputs,
            devices[rank],
            pools[rank],
            dtype,
            request_id,
            model_local_work,
        )
        chunks.update({(rank, chunk): tid for chunk, tid in enumerate(outputs)})
    for step in range(p - 1):
        updates = {}
        for rank in range(p):
            prev, chunk = (rank - 1) % p, (rank - step - 1) % p
            tid = builder.tensor(f"{prefix}/rs/{step}/{rank}/{chunk}", size, dtype=dtype)
            _local(
                builder,
                f"{prefix}/rs/{step}/{rank}",
                [chunks[rank, chunk], chunks[prev, chunk]],
                [tid],
                devices[rank],
                pools[rank],
                dtype,
                request_id,
                model_local_work,
                reduction=True,
            )
            updates[rank, chunk] = tid
        chunks.update(updates)
    return tuple(chunks[rank, (rank + 1) % p] for rank in range(p))


def ring_all_gather(
    builder,
    prefix,
    inputs,
    devices,
    pools,
    *,
    dtype="fp32",
    request_id=None,
    chunk_indices=None,
    output_bytes=None,
    model_local_work=True,
):
    p = len(devices)
    require(
        p > 0 and len(inputs) == len(pools) == p,
        "INVALID_COLLECTIVE",
        prefix,
        "one input/device/pool per rank",
    )
    indices = tuple(range(p)) if chunk_indices is None else tuple(chunk_indices)
    require(
        sorted(indices) == list(range(p)),
        "INVALID_COLLECTIVE",
        prefix,
        "chunk ownership must be a permutation",
    )
    sizes = {builder.tensors[t].storage_bytes for t in inputs}
    require(len(sizes) == 1, "INVALID_COLLECTIVE", prefix, "uniform chunk sizes required")
    size = sizes.pop()
    owned = {(rank, indices[rank]): inputs[rank] for rank in range(p)}
    for step in range(p - 1):
        updates = {}
        for rank in range(p):
            prev = (rank - 1) % p
            chunk = indices[(rank - step - 1) % p]
            tid = builder.tensor(f"{prefix}/ag/{step}/{rank}/{chunk}", size, dtype=dtype)
            # Receive into this rank's pool; forwarding messages are distinct tensors.
            builder.op(
                f"{prefix}/ag/{step}/{rank}",
                [owned[prev, chunk]],
                [tid],
                devices[rank],
                op_type="sync",
                dtype=dtype,
                duration_s=0,
                request_id=request_id,
            )
            updates[rank, chunk] = tid
        owned.update(updates)
    outputs = []
    for rank in range(p):
        tid = builder.tensor(
            f"{prefix}/result/{rank}",
            output_bytes if output_bytes is not None else p * size,
            dtype=dtype,
        )
        _local(
            builder,
            f"{prefix}/pack/{rank}",
            [owned[rank, chunk] for chunk in range(p)],
            [tid],
            devices[rank],
            pools[rank],
            dtype,
            request_id,
            model_local_work,
        )
        outputs.append(tid)
    return tuple(outputs)


def ring_all_reduce(
    builder,
    prefix,
    inputs,
    devices,
    pools,
    size_bytes,
    *,
    dtype="fp32",
    request_id=None,
    model_local_work=True,
):
    """Ring reduce-scatter plus all-gather, including explicit local IO/reductions."""
    if len(devices) == 1:
        return tuple(inputs)
    partials = ring_reduce_scatter(
        builder,
        f"{prefix}/scatter",
        inputs,
        devices,
        pools,
        size_bytes,
        dtype=dtype,
        request_id=request_id,
        model_local_work=model_local_work,
    )
    return ring_all_gather(
        builder,
        f"{prefix}/gather",
        partials,
        devices,
        pools,
        dtype=dtype,
        request_id=request_id,
        chunk_indices=tuple((rank + 1) % len(devices) for rank in range(len(devices))),
        output_bytes=size_bytes,
        model_local_work=model_local_work,
    )


def broadcast(builder, prefix, source, devices, size_bytes, *, dtype="fp32", request_id=None):
    outputs = []
    for rank, device in enumerate(devices):
        tid = builder.tensor(f"{prefix}/{rank}", size_bytes, dtype=dtype)
        builder.op(
            f"{prefix}/{rank}",
            [source],
            [tid],
            device,
            op_type="sync",
            dtype=dtype,
            duration_s=0,
            request_id=request_id,
        )
        outputs.append(tid)
    return tuple(outputs)
