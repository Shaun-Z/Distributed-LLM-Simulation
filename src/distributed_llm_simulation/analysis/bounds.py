"""Bounds carry scope and assumptions. Heuristic observations never become proofs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import networkx as nx

from ..models.compute import compatible
from ..spec.base import require


@dataclass(frozen=True)
class Bound:
    name: str
    value_s: float
    scope: str
    assumptions: tuple[str, ...]
    fixed_decisions: tuple[str, ...] = ()
    derivation_version: str = "1"

    def to_dict(self):
        return asdict(self)


def critical_path_bound(hardware, workload):
    """Optimistic compute-only critical path; omit launch, IO and all communication."""
    kernels = {k.id: k for k in hardware.kernel_profiles}
    times = {}
    for op in workload.operators:
        candidates = []
        for device in hardware.nodes:
            if device.kind != "compute":
                continue
            for kid in device.kernel_profile_ids:
                k = kernels[kid]
                if not compatible(op, k):
                    continue
                duration = (
                    op.duration_s
                    if op.duration_s is not None
                    else k.duration_s
                    if k.model == "calibrated_duration"
                    else (op.flops or op.integer_ops) / k.effective_ops_per_s
                )
                candidates.append(duration)
        require(bool(candidates), "UNSUPPORTED_KERNEL", op.id, "no compatible implementation")
        times[op.id] = min(candidates)
    g = workload.graph()
    finish = {}
    for node in nx.topological_sort(g):
        finish[node] = max((finish[p] for p in g.predecessors(node)), default=0) + (
            times[node[1]] if node[0] == "op" else 0
        )
    return Bound(
        "compute_dependency_path",
        max(finish.values(), default=0),
        "global lower bound",
        (
            "Fixed workload with no recomputation/fusion/algorithm substitution.",
            "Declared kernel durations/throughputs are optimistic within this model.",
            "Ignore launch, memory IO, resource contention, arrivals and all communication.",
        ),
    )


def execution_bounds(hardware, execution):
    """Conditional bounds for this lowered fixed-time plan with constant capacities."""
    resources = {r.id: r for r in hardware.resources + execution.extra_resources}
    durations, load = {}, defaultdict(float)
    for a in execution.activities:
        if a.work_unit == "s":
            duration = a.total_work
        else:
            terms = [
                a.total_work * coefficient / resources[rid].capacity
                if resources[rid].capacity
                else float("inf")
                for rid, coefficient in a.rate_demands
            ]
            if a.max_rate:
                terms.append(a.total_work / a.max_rate)
            duration = max(terms, default=0)
            for rid, coefficient in a.rate_demands:
                load[rid] += a.total_work * coefficient
        durations[a.id] = a.setup_s + duration
        for rid, count in a.slot_demands:
            load[rid] += count * durations[a.id]
    finish = {}
    g = execution.graph()
    for aid in nx.topological_sort(g):
        finish[aid] = durations[aid] + max((finish[p] for p in g.predecessors(aid)), default=0)
    common = (
        "Fixed placement, routes, output contract and transformations.",
        "Constant capacities; no stochastic or time-varying environment.",
        "Startup is serial with body and slots remain reserved through startup.",
    )
    output = [
        Bound(
            "execution_dependency_path",
            max(finish.values(), default=0),
            "conditional lower bound",
            common,
            (execution.digest(),),
        )
    ]
    for rid, work in sorted(load.items()):
        cap = resources[rid].capacity
        value = work / cap if cap else float("inf") if work else 0
        output.append(
            Bound(
                f"resource_load:{rid}",
                value,
                "conditional lower bound",
                common,
                (execution.digest(),),
            )
        )
    return output


def gap(upper, lower):
    require(upper >= lower - 1e-10, "INVALID_BOUND", "gap", "upper below lower")
    return {
        "upper_bound_s": upper,
        "lower_bound_s": lower,
        "absolute_gap_s": upper - lower,
        "ratio": upper / lower if lower > 0 else None,
        "normalized_gap": (upper - lower) / upper if upper > 0 else None,
    }


def capacity_integral_bound(process, necessary_bytes, earliest_s=0):
    """Conditional integral bound for traffic proved necessary on this resource."""
    remaining = necessary_bytes
    for i, (start, capacity) in enumerate(process.points):
        end = process.points[i + 1][0] if i + 1 < len(process.points) else float("inf")
        start = max(start, earliest_s)
        if end <= start:
            continue
        if remaining <= 0:
            return earliest_s
        if capacity and remaining <= capacity * (end - start):
            return start + remaining / capacity
        if capacity:
            remaining -= capacity * (end - start)
    return float("inf")
