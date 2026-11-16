"""External evaluation/search with explicit finite-search scope and replayable winners."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from ..compiler import ExecutionPlan, Lowerer
from ..models.compute import compatible
from ..runtime import SimulationConfig, Simulator
from ..scheduling.policies import FIFOListScheduler
from ..spec import PlacementPlan
from ..spec.base import require


@dataclass(frozen=True)
class Evaluation:
    placement: PlacementPlan
    result: object | None
    error: str | None
    key: str
    hardware: object | None = None
    execution: ExecutionPlan | None = None

    @property
    def objective(self):
        return (
            self.result.makespan_s if self.result and self.result.status == "SUCCESS" else math.inf
        )


class Evaluator:
    def __init__(self, hardware, workload, config=None):
        self.hardware, self.workload = hardware, workload
        self.config = config or SimulationConfig()
        self.cache = {}

    def evaluate(self, placement):
        from hashlib import sha256

        from .. import __version__

        key = sha256(
            ":".join(
                (
                    __version__,
                    self.hardware.digest(),
                    self.workload.digest(),
                    self.config.digest(),
                    placement.digest(),
                )
            ).encode()
        ).hexdigest()
        if key not in self.cache:
            try:
                execution = Lowerer(self.hardware).compile(self.workload, placement)
                result = Simulator(self.hardware, config=self.config).run(execution)
                value = Evaluation(
                    placement, result, result.failure_reason, key, self.hardware, execution
                )
            except ValueError as e:
                value = Evaluation(placement, None, str(e), key, self.hardware)
            self.cache[key] = value
        return self.cache[key]


def candidate_devices(hardware, workload):
    profiles = {k.id: k for k in hardware.kernel_profiles}
    return {
        o.id: tuple(
            n.id
            for n in hardware.nodes
            if n.kind == "compute" and any(compatible(o, profiles[k]) for k in n.kernel_profile_ids)
        )
        for o in workload.operators
    }


def enumerate_placements(hardware, workload, *, max_candidates=1000, config=None):
    require(
        type(max_candidates) is int and max_candidates > 0,
        "INVALID_SEARCH",
        "max_candidates",
        "positive integer required",
    )
    choices = candidate_devices(hardware, workload)
    names = sorted(choices)
    count = math.prod(len(choices[n]) for n in names)
    evaluator = Evaluator(hardware, workload, config)
    values = []
    for devices in itertools.islice(
        itertools.product(*(choices[n] for n in names)), max_candidates
    ):
        values.append(
            evaluator.evaluate(PlacementPlan(op_compute=tuple(zip(names, devices, strict=False))))
        )
    best = min(values, key=lambda e: e.objective, default=None)
    return {
        "scope": "placement enumeration under the configured fixed scheduler; not unrestricted schedule optimum",
        "candidates_evaluated": len(values),
        "total_placements": count,
        "exhausted": len(values) == count,
        "best": best,
        "evaluations": values,
    }


def heft_inspired_plan(hardware, workload):
    """Greedy candidate simulation using upward dependency order (capacity/network aware).

    Evaluates complete feasible plans; unlike classic HEFT it does not promise an
    insertion-scheduling theorem. Kernel choice remains user-configurable in plans.
    """
    choices = candidate_devices(hardware, workload)
    require(all(choices.values()), "UNSUPPORTED_KERNEL", "planner", "operator without device")
    current = {oid: devices[0] for oid, devices in choices.items()}
    evaluator = Evaluator(hardware, workload)
    import networkx as nx

    profiles = {k.id: k for k in hardware.kernel_profiles}
    nodes = {n.id: n for n in hardware.nodes}
    operators = {o.id: o for o in workload.operators}
    ranks = {}
    graph = workload.graph()
    for node in reversed(list(nx.topological_sort(graph))):
        own = 0.0
        if node[0] == "op":
            op = operators[node[1]]
            costs = []
            for device in choices[op.id]:
                for kid in nodes[device].kernel_profile_ids:
                    k = profiles[kid]
                    if compatible(op, k):
                        costs.append(
                            op.duration_s
                            if op.duration_s is not None
                            else k.duration_s
                            if k.model == "calibrated_duration"
                            else (op.flops or op.integer_ops) / k.effective_ops_per_s
                        )
            own = sum(costs) / len(costs)
        ranks[node] = own + max((ranks[n] for n in graph.successors(node)), default=0)
    for oid in sorted(choices, key=lambda oid: (-ranks["op", oid], oid)):
        candidates = [
            evaluator.evaluate(
                PlacementPlan(op_compute=tuple(sorted({**current, oid: device}.items())))
            )
            for device in choices[oid]
        ]
        best = min(candidates, key=lambda e: (e.objective, e.placement.digest()))
        current = dict(best.placement.op_compute)
    return evaluator.evaluate(PlacementPlan(op_compute=tuple(sorted(current.items()))))


def pareto_frontier(evaluations):
    """Empirical nondominated set over time, pool-by-pool peaks and wire traffic."""
    valid = [e for e in evaluations if math.isfinite(e.objective)]
    pools = sorted({p for e in valid for p in e.result.metrics["peak_memory_bytes"]})
    vectors = [
        (
            e,
            (
                e.objective,
                *(e.result.metrics["peak_memory_bytes"].get(p, 0) for p in pools),
                e.result.metrics["bytes_transferred"],
            ),
        )
        for e in valid
    ]
    return [
        e
        for e, v in vectors
        if not any(
            all(x <= y for x, y in zip(other, v, strict=False))
            and any(x < y for x, y in zip(other, v, strict=False))
            for _, other in vectors
        )
    ]


def memory_sweep(hardware, workload, placement, budgets, config=None):
    """Budgets are mappings from physical capacity resource IDs to exact byte limits."""
    output = []
    for budget in budgets:
        require(
            all(
                rid in {r.id for r in hardware.resources if r.kind == "storage"}
                and type(amount) is int
                and amount >= 0
                for rid, amount in budget.items()
            ),
            "INVALID_BUDGET",
            "budgets",
            "exact nonnegative storage budgets required",
        )
        data = hardware.model_dump()
        for r in data["resources"]:
            if r["id"] in budget:
                r["capacity"] = budget[r["id"]]
        h = type(hardware).model_validate(data)
        output.append((budget, Evaluator(h, workload, config).evaluate(placement)))
    return output


def exact_serial_oracle(hardware, execution, *, max_orders=10000):
    """Exact optimum for the explicitly restricted serial-activity subproblem.

    All topological orders are enumerated; a control edge serializes every pair.
    This is useful as a differential oracle for single shared-slot fixtures. It
    is not an unrestricted parallel optimum. Report exhaustion separately.
    """
    import networkx as nx

    require(
        type(max_orders) is int and max_orders > 0,
        "INVALID_SEARCH",
        "max_orders",
        "positive integer required",
    )
    best, best_execution, evaluated = None, None, 0
    exhausted, truncated = True, False
    for order in nx.all_topological_sorts(execution.graph()):
        if evaluated >= max_orders:
            exhausted = False
            break
        acts = {a.id: a for a in execution.activities}
        previous = None
        serialized = []
        for aid in order:
            a = acts[aid]
            preds = set(a.predecessor_ids) | ({previous} if previous else set())
            serialized.append(a.model_copy(update={"predecessor_ids": tuple(sorted(preds))}))
            previous = aid
        data = execution.model_dump()
        data["activities"] = [a.model_dump() for a in serialized]
        plan = ExecutionPlan.model_validate(data)
        result = Simulator(hardware, FIFOListScheduler()).run(plan)
        truncated = truncated or result.status == "TRUNCATED"
        evaluated += 1
        if result.status == "SUCCESS" and (best is None or result.makespan_s < best.makespan_s):
            best, best_execution = result, plan
    return {
        "scope": "serial activity schedules, fixed placement/routes/kernels; not global parallel optimum",
        "orders_evaluated": evaluated,
        "proven_optimal_in_scope": exhausted and not truncated,
        "best": best,
        "execution": best_execution,
    }


def exact_schedule_oracle(hardware, execution, *, max_orders=10000):
    """Enumerate all topological start orders for a fixed deterministic A plan.

    With nonpreemptive constant durations and exclusive bandwidth gates, starting
    the next activity as soon as its declared predecessors/resources/storage allow
    is dominant for a fixed total start order. Every feasible schedule has such an
    order. Deliberate idle time is represented by waiting for a not-yet-ready head.
    This proof does not extend to fluid interference, changing capacities, dynamic
    placements or an optimizer choosing different routes/transformations.
    """
    import networkx as nx

    require(
        type(max_orders) is int and max_orders > 0,
        "INVALID_SEARCH",
        "max_orders",
        "positive integer required",
    )
    best, best_order, evaluated = None, None, 0
    exhausted, truncated = True, False
    for order in nx.all_topological_sorts(execution.graph()):
        if evaluated >= max_orders:
            exhausted = False
            break
        result = Simulator(hardware, config=SimulationConfig(activity_order=tuple(order))).run(
            execution
        )
        evaluated += 1
        truncated = truncated or result.status == "TRUNCATED"
        if result.status == "SUCCESS" and (
            best is None or result.finish_time_s < best.finish_time_s
        ):
            best, best_order = result, tuple(order)
    proven = exhausted and not truncated
    from .bounds import execution_bounds

    optimistic = max((b.value_s for b in execution_bounds(hardware, execution)), default=0)
    return {
        "scope": "conditional optimum of fixed execution plan, A-level constant capacities, all total start orders",
        "assumptions": [
            "No preemption, fluid sharing, dynamic placement, eviction decisions or changes to routes/kernels.",
            "Atomic output/workspace reservations, predefined lifetimes and conservative exclusive gates.",
            "Objective is finish time from simulation t=0; request arrivals are fixed releases.",
        ],
        "orders_evaluated": evaluated,
        "exhausted": exhausted,
        "proven_optimal_in_scope": proven,
        "proven_infeasible_in_scope": proven and best is None,
        "best_bound_s": best.finish_time_s if proven and best is not None else optimistic,
        "incumbent_s": best.finish_time_s if best else None,
        "best_order": best_order,
        "best": best,
    }
