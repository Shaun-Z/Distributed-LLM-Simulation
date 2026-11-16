"""Deterministic kernel timing and physical IO accounting."""

from collections import defaultdict

from ..spec.base import SpecError, require


def compatible(op, kernel):
    signature = (op.op_type, op.input_dtype, op.accumulate_dtype, op.output_dtype)
    expected = (kernel.op_type, kernel.input_dtype, kernel.accumulate_dtype, kernel.output_dtype)
    shape = dict(op.shape_params)
    return (
        signature == expected
        and (not kernel.layouts or op.layout in kernel.layouts)
        and (not kernel.algorithms or op.algorithm in kernel.algorithms)
        and all(
            name in shape and lo <= shape[name] <= hi for name, lo, hi in kernel.shape_constraints
        )
    )


def compute_cost(op, compute, kernel, hardware):
    """Launch plus roofline/fixed body; local reads/writes are never extra transfers."""
    resources = {r.id: r for r in hardware.resources}
    nodes = {n.id: n for n in hardware.nodes}
    work, channel_work = defaultdict(float), defaultdict(float)
    for operation, entries in (("local_load", op.read_bytes), ("local_store", op.write_bytes)):
        for location, size in entries:
            if location in resources:
                require(
                    resources[location].unit == "byte/s", "INVALID_IO_RESOURCE", op.id, location
                )
                demands = ((location, 1),)
            else:
                pools = {
                    hardware.pool(p)
                    for p in compute.accessible_pool_ids
                    if p == location
                    or nodes[p].tier == location
                    or (
                        location in nodes
                        and nodes[location].kind == "memory"
                        and hardware.pool(p) == hardware.pool(location)
                    )
                }
                require(
                    len(pools) == 1,
                    "INVALID_IO_RESOURCE",
                    op.id,
                    f"ambiguous/inaccessible pool: {location}",
                )
                pool = next(iter(pools))
                channels = [
                    c
                    for c in hardware.channels
                    if operation in c.allowed_operations
                    and (
                        (
                            operation == "local_load"
                            and c.dst == compute.id
                            and nodes[c.src].kind == "memory"
                            and hardware.pool(c.src) == pool
                        )
                        or (
                            operation == "local_store"
                            and c.src == compute.id
                            and nodes[c.dst].kind == "memory"
                            and hardware.pool(c.dst) == pool
                        )
                    )
                ]
                require(
                    bool(channels),
                    "ILLEGAL_ACCESS",
                    op.id,
                    f"no {operation} channel for {location}",
                )
                # Parallel access implementations require an explicit profile choice;
                # use one deterministic path rather than charging every alternative.
                channel = min(channels, key=lambda c: c.id)
                demands = channel.resource_demands
                channel_work[channel.id] += size
            require(
                bool(demands) or not size,
                "INVALID_IO_RESOURCE",
                op.id,
                "local IO needs physical service budget",
            )
            for rid, coefficient in demands:
                require(rid in kernel.memory_rate_resource_ids, "INVALID_IO_RESOURCE", op.id, rid)
                work[rid] += size * coefficient
    if op.duration_s is not None:
        duration = op.duration_s
    elif kernel.model == "calibrated_duration":
        duration = kernel.duration_s
    else:
        arithmetic = (op.integer_ops or op.flops) / kernel.effective_ops_per_s
        memory = []
        for rid, amount in work.items():
            if resources[rid].capacity == 0 and amount:
                raise SpecError("UNAVAILABLE_RATE", op.id, rid)
            memory.append(amount / resources[rid].capacity if amount else 0)
        channels = {c.id: c for c in hardware.channels}
        memory.extend(amount / channels[cid].payload_Bps for cid, amount in channel_work.items())
        duration = max(arithmetic, *memory, 0)
    return compute.launch_s + kernel.launch_s, duration, tuple(sorted(work.items()))
