from __future__ import annotations

from ..models.compute import compatible, compute_cost
from ..spec import HardwareSpec, PlacementPlan, ResourceSpec, WorkloadSpec
from ..spec.base import SpecError, indexed, require
from .execution import ActivitySpec, ExecutionPlan, ReplicaSpec
from .routing import RoutingPolicy


class Lowerer:
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware

    def compile(self, workload: WorkloadSpec, plan: PlacementPlan) -> ExecutionPlan:
        h = self.hardware
        ns, ks, ts, ops = (
            {x.id: x for x in seq}
            for seq in (h.nodes, h.kernel_profiles, workload.tensors, workload.operators)
        )
        for name in ("op_compute", "op_kernel", "tensor_target_pool", "final_output_pools"):
            pairs = getattr(plan, name)
            require(len(dict(pairs)) == len(pairs), "DUPLICATE_MAPPING", name, "duplicate key")
        placements, kernels, targets = (
            dict(plan.op_compute),
            dict(plan.op_kernel),
            dict(plan.tensor_target_pool),
        )
        require(
            set(placements) == set(ops),
            "INVALID_PLACEMENT",
            "op_compute",
            "must cover exactly every operator",
        )
        require(
            set(kernels) <= set(ops) and set(targets) <= set(ts),
            "INVALID_PLACEMENT",
            "placement",
            "unknown operator/tensor",
        )
        for cid in placements.values():
            require(cid in ns and ns[cid].kind == "compute", "INVALID_COMPUTE", "op_compute", cid)
        for pid in list(targets.values()) + [p for _, p in plan.final_output_pools]:
            require(pid in ns and ns[pid].kind == "memory", "UNKNOWN_POOL", "placement", pid)
        routes = tuple(indexed(h.routes + plan.routes, "routes").values())
        # Validate supplied routes with exactly the same rules as hardware routes.
        HardwareSpec.model_validate({**h.model_dump(), "routes": routes})
        router = RoutingPolicy(h, routes)
        for tid, dst, route in plan.transfer_routes:
            require(
                tid in ts and dst in ns and ns[dst].kind == "memory" and route in router.routes,
                "INVALID_TRANSFER_ROUTE",
                tid,
                f"{dst}/{route}",
            )
        forced = {(tid, h.pool(dst)): rid for tid, dst, rid in plan.transfer_routes}
        require(
            len(forced) == len(plan.transfer_routes),
            "DUPLICATE_MAPPING",
            "transfer_routes",
            "duplicate tensor/destination",
        )
        activities, replicas, resident, available = [], {}, {}, {}
        reqs = {r.id: r for r in workload.requests}
        releases = {r.id: r.arrival_s for r in workload.requests}
        op_handles = {}
        require(
            set(plan.reload_tensor_ids) <= set(ts),
            "UNKNOWN_TENSOR",
            "reload_tensor_ids",
            "unknown reload tensor",
        )

        def create(tid, pool, initial=False, retain=False):
            pool = h.pool(pool)
            key = (tid, pool)
            if key in resident:
                return resident[key]
            # Opaque identifiers avoid collisions from arbitrary user ID separators.
            rid = f"replica{len(replicas):06d}"
            tensor = ts[tid]
            replicas[rid] = ReplicaSpec(
                id=rid,
                tensor_id=tid,
                pool_id=pool,
                storage_bytes=tensor.storage_bytes,
                version=tensor.version,
                initial=initial,
                retain=retain,
            )
            resident[key] = rid
            available[rid] = None
            return rid

        for t in workload.tensors:
            for pid in t.initial_pool_ids:
                require(pid in ns and ns[pid].kind == "memory", "UNKNOWN_POOL", t.id, pid)
                create(t.id, pid, initial=True, retain=t.persistent)

        def ensure(tid, target, request=None, gates=()):
            target = h.pool(target)
            if (tid, target) in resident:
                return resident[tid, target]
            sources = sorted(
                (pool, rid) for (tensor, pool), rid in resident.items() if tensor == tid
            )
            require(bool(sources), "MISSING_REPLICA", tid, "producer not lowered")
            selected = None
            for source, rid in sources:
                try:
                    path = router.candidates(
                        source, target, ts[tid].storage_bytes, forced.get((tid, target))
                    )
                    selected = rid, path
                    break
                except SpecError:
                    continue
            if selected is None:
                raise SpecError(
                    "NO_ROUTE", tid, f"cannot reach {target} from {[p for p, _ in sources]}"
                )
            source_rid, path = selected
            for route in path:
                _, dst = router.endpoints(route)
                if (tid, dst) in resident:
                    source_rid = resident[tid, dst]
                    continue
                dest_rid = create(tid, dst)
                demands, latency, ceiling = router.cost(route)
                aid = f"move{len(activities):06d}"
                activities.append(
                    ActivitySpec(
                        id=aid,
                        kind="transfer",
                        predecessor_ids=tuple(
                            sorted(
                                set(gates)
                                | (
                                    {available[source_rid]}
                                    if available[source_rid] is not None
                                    else set()
                                )
                            )
                        ),
                        total_work=ts[tid].wire_bytes
                        if ts[tid].wire_bytes is not None
                        else ts[tid].storage_bytes,
                        work_unit="byte",
                        rate_demands=demands,
                        max_rate=ceiling,
                        slot_demands=route.slot_demands,
                        input_replica_ids=(source_rid,),
                        output_replica_ids=(dest_rid,),
                        tensor_id=tid,
                        request_id=request,
                        release_s=releases.get(request, 0),
                        setup_s=latency,
                    )
                )
                available[dest_rid] = aid
                source_rid = dest_rid
            return source_rid

        for oid in workload.order():
            op = ops[oid]
            compute = ns[placements[oid]]
            candidates = [ks[k] for k in compute.kernel_profile_ids if compatible(op, ks[k])]
            if oid in kernels:
                candidates = [k for k in candidates if k.id == kernels[oid]]
            require(bool(candidates), "UNSUPPORTED_KERNEL", oid, f"{op.op_type} on {compute.id}")
            kernel = sorted(candidates, key=lambda k: k.id)[0]
            accessible = sorted({h.pool(p) for p in compute.accessible_pool_ids})
            require(bool(accessible), "NO_ACCESSIBLE_POOL", oid, compute.id)
            # Explicit output pool must be directly accessible to the producing kernel.
            output_pools = {tid: h.pool(targets.get(tid, accessible[0])) for tid in op.outputs}
            require(
                all(p in accessible for p in output_pools.values()),
                "ILLEGAL_ACCESS",
                oid,
                "output pool inaccessible",
            )
            inputs = []
            input_order = sorted(
                dict.fromkeys(op.inputs), key=lambda t: t in plan.reload_tensor_ids
            )
            for tid in input_order:
                existing = sorted(p for t, p in resident if t == tid and p in accessible)
                preferred = h.pool(targets[tid]) if tid in targets else None
                target = (
                    preferred
                    if preferred in accessible
                    else existing[0]
                    if existing
                    else accessible[0]
                )
                gates = (
                    tuple(available[r] for r in inputs if available[r] is not None)
                    + tuple(op_handles[p] for p in op.control_predecessors)
                    if tid in plan.reload_tensor_ids
                    else ()
                )
                inputs.append(ensure(tid, target, op.request_id, gates))
            outputs = tuple(
                create(tid, output_pools[tid], retain=ts[tid].persistent) for tid in op.outputs
            )
            predecessors = {available[r] for r in inputs if available[r] is not None}
            predecessors.update(op_handles[p] for p in op.control_predecessors)
            # Shared memory handover has explicit, profile-defined visibility cost.
            visibility = 0.0
            for rid in inputs:
                producer = ts[replicas[rid].tensor_id].producer_id
                producer_pool = (
                    h.pool(
                        targets.get(
                            replicas[rid].tensor_id,
                            sorted(ns[placements[producer]].accessible_pool_ids)[0],
                        )
                    )
                    if producer
                    else None
                )
                if (
                    producer
                    and placements[producer] != compute.id
                    and producer_pool == replicas[rid].pool_id
                ):
                    rules = [
                        v
                        for v in h.visibility
                        if h.pool(v.pool_id) == replicas[rid].pool_id
                        and {placements[producer], compute.id} <= set(v.device_ids)
                        and v.shared_addressable
                    ]
                    if rules:
                        visibility = max(visibility, max(v.ownership_transition_s for v in rules))
                    elif (
                        len(
                            [
                                n
                                for n in h.nodes
                                if n.kind == "compute"
                                and replicas[rid].pool_id
                                in {h.pool(p) for p in n.accessible_pool_ids}
                            ]
                        )
                        > 1
                    ):
                        raise SpecError(
                            "INVALID_VISIBILITY", oid, "shared pool handover needs visibility rule"
                        )
            launch, duration, work = compute_cost(op, compute, kernel, h)
            aid = f"compute{len(activities):06d}"
            slots = {
                r: 1
                for r in compute.resource_ids
                if next(x for x in h.resources if x.id == r).kind == "slots"
            }
            slots[kernel.slot_resource_id] = 1
            activities.append(
                ActivitySpec(
                    id=aid,
                    kind="compute",
                    predecessor_ids=tuple(sorted(predecessors)),
                    total_work=duration,
                    input_replica_ids=tuple(inputs),
                    output_replica_ids=outputs,
                    slot_demands=tuple(sorted(slots.items())),
                    exclusive_resources=kernel.memory_rate_resource_ids,
                    setup_s=launch + visibility,
                    workspace=((accessible[0], op.workspace_bytes),) if op.workspace_bytes else (),
                    op_id=oid,
                    request_id=op.request_id,
                    release_s=releases.get(op.request_id, 0),
                    service_work=work,
                )
            )
            op_handles[oid] = aid
            for rid in outputs:
                available[rid] = aid
            for rid in inputs:
                replica = replicas[rid]
                if replica.tensor_id in plan.reload_tensor_ids and not replica.initial:
                    resident.pop((replica.tensor_id, replica.pool_id), None)
        final = dict(plan.final_output_pools)
        require(set(final) <= set(ts), "UNKNOWN_TENSOR", "final_output_pools", "unknown output")
        output_ids = {tid for r in workload.requests for tid in r.output_tensor_ids}
        output_ids.update(final)
        if not workload.requests:
            output_ids.update(
                t.id for t in workload.tensors if t.producer_id and not t.consumer_ids
            )
        output_map = {}
        for tid in sorted(output_ids):
            rid = (
                ensure(tid, final[tid])
                if tid in final
                else next(r for (t, _), r in resident.items() if t == tid)
            )
            replicas[rid] = replicas[rid].model_copy(update={"retain": True})
            output_map[tid] = rid
        contracts = tuple(
            (r.id, tuple(output_map[t] for t in r.output_tensor_ids)) for r in workload.requests
        )
        markers = []
        for req in reqs.values():
            for name, tid in req.markers:
                rid = output_map.get(tid) or next(r for (t, _), r in resident.items() if t == tid)
                markers.append((req.id, name, rid))
        return ExecutionPlan(
            activities=tuple(activities),
            replicas=tuple(replicas.values()),
            requests=workload.requests,
            request_outputs=contracts,
            markers=tuple(markers),
            hardware_hash=h.digest(),
            workload_hash=workload.digest(),
            placement_hash=plan.digest(),
            extra_resources=tuple(
                ResourceSpec(
                    id=f"channel:{c.id}",
                    kind="rate",
                    unit="byte/s",
                    capacity=c.payload_Bps,
                    policy="shared",
                )
                for c in h.channels
            ),
        )
