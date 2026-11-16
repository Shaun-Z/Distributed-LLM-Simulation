"""Deterministic heap-driven engine. Every run owns all mutable state."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict

from ..models.memory import MemoryManager
from ..scheduling.policies import (
    Candidate,
    OrderedScheduler,
    Snapshot,
    scheduler_by_name,
    upward_ranks,
)
from ..spec.base import indexed, require
from .arbitration import weighted_max_min
from .events import EventQueue
from .state import ActivityState, RunResult, SimulationConfig


class Simulator:
    def __init__(self, hardware, scheduler=None, config=None):
        self.hardware = hardware
        self.scheduler = scheduler
        self.config = config or SimulationConfig()

    def run(self, execution):
        return self._prepare(execution).run()

    def validate(self, execution):
        """Validate runtime resource/config contracts without advancing simulation time."""
        self._prepare(execution)

    def _prepare(self, execution):
        # No state is retained by Simulator, so repeated runs are independent.
        if self.config.activity_order:
            require(
                len(self.config.activity_order) == len(execution.activities)
                and set(self.config.activity_order) == {a.id for a in execution.activities},
                "INVALID_ORDER",
                "activity_order",
                "must contain every activity exactly once",
            )
        return _Run(
            self.hardware,
            execution,
            self.scheduler
            or (
                OrderedScheduler(self.config.activity_order)
                if self.config.activity_order
                else scheduler_by_name(self.config.scheduler)
            ),
            self.config,
        )


class _Run:
    def __init__(self, hardware, execution, scheduler, config):
        self.h, self.e, self.scheduler, self.config = hardware, execution, scheduler, config
        self.activities = indexed(execution.activities, "activities")
        self.replicas = indexed(execution.replicas, "replicas")
        self.resources = indexed(hardware.resources + execution.extra_resources, "resources")
        self.capacities = {r.id: r.capacity for r in self.resources.values()}
        self.states = {a.id: ActivityState(float(a.total_work)) for a in execution.activities}
        self.memory = MemoryManager(hardware)
        self.queue = EventQueue()
        self.now = 0.0
        self.processed = 0
        self.trace = []
        self.waiters = {}
        self.previous_wait = {}
        self.ranks = upward_ranks(execution)
        self.pending_readers = Counter(r for a in execution.activities for r in a.input_replica_ids)
        graph = execution.graph()
        self.predecessors_left = {aid: graph.in_degree(aid) for aid in self.activities}
        self.successors = {aid: tuple(sorted(graph.successors(aid))) for aid in self.activities}
        self.ready_ids = {aid for aid, count in self.predecessors_left.items() if count == 0}
        self.completed = set()
        self.started = set()
        self.replica_ready = {}
        self.slot_used = defaultdict(int)
        self.exclusive_owner = {}
        self.request_finish = {}
        self.service_integral = defaultdict(float)
        self.capacity_integral = defaultdict(float)
        self.reserved_integral = defaultdict(float)
        self.storage_integral = defaultdict(float)
        self.active_integral = 0.0
        self.stale_events = 0
        require(
            execution.hardware_hash in (None, hardware.digest()),
            "HARDWARE_MISMATCH",
            "execution",
            "compiled for different hardware",
        )
        processes = indexed_processes(config.capacity_processes)
        for rid, process in processes.items():
            require(
                rid in self.resources and self.resources[rid].kind == "rate",
                "INVALID_TRACE",
                rid,
                "only rates may change",
            )
            require(
                not any(rid in a.exclusive_resources for a in execution.activities),
                "UNSUPPORTED_DYNAMIC_COMPUTE",
                rid,
                "fixed-duration kernel IO budget cannot vary; change network resources instead",
            )
            self.capacities[rid] = process.points[0][1]
            for time, capacity in process.points[1:]:
                self.queue.push(time, "CAPACITY_CHANGE", payload=(rid, capacity))
        for a in execution.activities:
            for rid, _ in a.rate_demands:
                require(
                    rid in self.resources and self.resources[rid].unit == "byte/s",
                    "INVALID_RATE",
                    a.id,
                    rid,
                )
            for rid in a.exclusive_resources:
                require(
                    rid in self.resources and self.resources[rid].kind == "rate",
                    "INVALID_RATE",
                    a.id,
                    rid,
                )
            for rid, count in a.slot_demands:
                require(
                    rid in self.resources
                    and self.resources[rid].kind == "slots"
                    and count <= self.resources[rid].capacity,
                    "INVALID_SLOT",
                    a.id,
                    rid,
                )
            for rid in a.input_replica_ids + a.output_replica_ids:
                require(
                    self.replicas[rid].pool_id in self.memory.capacity,
                    "UNKNOWN_POOL",
                    a.id,
                    self.replicas[rid].pool_id,
                )
            for pool, _ in a.workspace:
                require(pool in self.memory.capacity, "UNKNOWN_POOL", a.id, pool)
            for rid, work in a.service_work:
                require(
                    rid in a.exclusive_resources
                    and work <= self.capacities[rid] * a.total_work * (1 + 1e-9),
                    "INVALID_SERVICE_WORK",
                    a.id,
                    "IO exceeds fixed-duration capacity",
                )
            if a.release_s:
                self.queue.push(a.release_s, "ARRIVAL", a.id)
        for r in execution.requests:
            self.queue.push(r.arrival_s, "ARRIVAL", payload=r.id)
        self.queue.push(0, "INIT")

    def emit(self, event_type, aid=None, **fields):
        a = self.activities.get(aid)
        if self.config.trace_level == "summary" and event_type not in (
            "RUN_FINISHED",
            "REQUEST_FINISHED",
        ):
            return
        if self.config.trace_level == "activities" and event_type in ("RATE_CHANGED", "WAITING"):
            return
        self.trace.append(
            {
                "sim_time": self.now,
                "seq": len(self.trace),
                "event_type": event_type,
                "activity_id": aid,
                "op_id": a.op_id if a else None,
                "tensor_id": a.tensor_id if a else fields.get("tensor_id"),
                "request_id": a.request_id if a else fields.get("request_id"),
                "resource_ids": sorted(
                    set(r for r, _ in a.rate_demands + a.slot_demands) | set(a.exclusive_resources)
                )
                if a
                else [],
                "reason": None,
                "generation": self.states[aid].generation if a else 0,
                **fields,
            }
        )

    def valid_event(self, event):
        if event.kind != "PREDICTED":
            return True
        s = self.states[event.activity_id]
        return s.state == "ACTIVE" and s.generation == event.generation

    def run(self):
        initial = [(r.id, r.pool_id, r.storage_bytes) for r in self.e.replicas if r.initial]
        ok, blockers = self.memory.try_reserve_many(initial)
        if not ok:
            return self.finish(
                "NO_PROGRESS", "INITIAL_CAPACITY_BLOCKED", {"initial_allocations": blockers}
            )
        for rid, _, _ in initial:
            self.memory.mark_valid(rid)
            self.replica_ready[rid] = 0.0
            self.emit_allocation("ALLOCATED", rid)
        self.garbage_collect()
        while True:
            while self.queue and not self.valid_event(self.queue.heap[0]):
                self.queue.pop()
                self.stale_events += 1
            if len(self.completed) == len(self.activities) and len(self.request_finish) == len(
                self.e.requests
            ):
                return self.finish("SUCCESS")
            if not self.queue:
                snapshot = self.snapshot()
                reason = (
                    "PERMANENT_ZERO_CAPACITY"
                    if snapshot["zero_rate_activities"]
                    else "CAPACITY_BLOCKED"
                    if any("pools" in b for b in snapshot["waiting"].values())
                    else "NO_PROGRESS"
                )
                return self.finish("NO_PROGRESS", reason, snapshot)
            if self.processed >= self.config.max_events:
                return self.finish("TRUNCATED", "MAX_EVENTS", self.snapshot())
            next_time = self.queue.heap[0].time_s
            if next_time > self.config.max_sim_time:
                self.integrate(self.config.max_sim_time)
                return self.finish("TRUNCATED", "MAX_SIM_TIME", self.snapshot())
            self.integrate(next_time)
            batch = []
            while self.queue and self.queue.heap[0].time_s == self.now:
                event = self.queue.pop()
                if self.valid_event(event):
                    batch.append(event)
                else:
                    self.stale_events += 1
            self.processed += len(batch)
            # Completions precede changes/arrivals at exactly the same time.
            for event in batch:
                if event.kind == "PREDICTED":
                    s = self.states[event.activity_id]
                    # Prediction timestamps are rounded upward. Only integrated
                    # work, never the presence of a prediction, proves completion.
                    if s.remaining_work == 0:
                        self.complete(event.activity_id)
            for event in batch:
                if event.kind == "CAPACITY_CHANGE":
                    rid, capacity = event.payload
                    self.capacities[rid] = capacity
                    self.emit("CAPACITY_CHANGE", resource_ids=[rid], capacity=capacity)
                elif event.kind == "PHASE_READY":
                    aid = event.activity_id
                    if self.states[aid].state == "STARTUP":
                        self.states[aid].state = "ACTIVE"
                        self.emit("PHASE_READY", aid)
            if not self.dispatch_closure():
                return self.finish("TRUNCATED", "MAX_SAME_TIME_TRANSITIONS", self.snapshot())
            self.update_requests()
            self.recompute_rates()
            if self.config.check_invariants:
                self.assert_invariants()

    def integrate(self, time):
        assert time >= self.now
        delta = time - self.now
        for rid, capacity in self.capacities.items():
            if self.resources[rid].kind in ("rate", "slots"):
                self.capacity_integral[rid] += capacity * delta
        for rid, used in self.slot_used.items():
            self.reserved_integral[rid] += used * delta
        for rid in self.exclusive_owner:
            self.reserved_integral[rid] += delta
        active_any = False
        for aid, s in self.states.items():
            if s.state != "ACTIVE":
                continue
            a = self.activities[aid]
            progress = min(s.remaining_work, s.rate * delta)
            s.remaining_work = max(0.0, s.remaining_work - progress)
            s.served += progress
            active_any = active_any or progress > 0
            for rid, coefficient in a.rate_demands:
                self.service_integral[rid] += progress * coefficient
            if a.total_work:
                for rid, work in a.service_work:
                    self.service_integral[rid] += work * progress / a.total_work
        self.active_integral += delta if active_any else 0
        for pool, used in self.memory.used.items():
            self.storage_integral[pool] += used * delta
        self.now = time

    def gates(self, a):
        result = set(a.exclusive_resources)
        if self.config.network_mode == "exclusive":
            result.update(r for r, _ in a.rate_demands)
        return result

    def can_start(self, a):
        blockers = {}
        missing = [p for p in a.predecessor_ids if p not in self.completed]
        if missing:
            blockers["dependencies"] = missing
        if a.release_s > self.now:
            blockers["arrival"] = a.release_s
        invalid = [
            r
            for r in a.input_replica_ids
            if r not in self.memory.allocations or self.memory.allocations[r].state != "VALID"
        ]
        if invalid:
            blockers["inputs"] = invalid
        slots = {
            r: self.slot_used[r]
            for r, n in a.slot_demands
            if self.slot_used[r] + n > self.capacities[r]
        }
        if slots:
            blockers["slots"] = slots
        conflict = {}
        demands = {r for r, _ in a.rate_demands}
        for rid in self.gates(a) | demands:
            if rid in self.exclusive_owner:
                conflict[rid] = [self.exclusive_owner[rid]]
            elif rid in self.gates(a):
                users = [
                    bid
                    for bid in sorted(self.started - self.completed)
                    if rid in dict(self.activities[bid].rate_demands)
                ]
                if users:
                    conflict[rid] = users
        if conflict:
            blockers["rate_gates"] = conflict
        requests = [
            (rid, self.replicas[rid].pool_id, self.replicas[rid].storage_bytes)
            for rid in a.output_replica_ids
        ]
        requests += [(f"workspace:{a.id}:{i}", p, size) for i, (p, size) in enumerate(a.workspace)]
        space = self.memory.blockers(requests)
        if space:
            blockers["pools"] = space
        return blockers, requests

    def dispatch_closure(self):
        transitions = 0
        while True:
            changed = False
            # All zero-work phases finish before another scheduling snapshot.
            for aid in sorted(self.started - self.completed):
                s = self.states[aid]
                if s.state == "ACTIVE" and s.remaining_work == 0:
                    self.complete(aid)
                    changed = True
                    transitions += 1
            candidates = []
            for aid in sorted(self.ready_ids):
                a = self.activities[aid]
                s = self.states[a.id]
                if s.state != "WAITING":
                    continue
                if all(p in self.completed for p in a.predecessor_ids) and a.release_s <= self.now:
                    if s.ready_s is None:
                        s.ready_s = self.now
                        self.emit("LOGIC_READY", a.id)
                    releasable = sum(
                        self.memory.allocations[r].bytes
                        for r in a.input_replica_ids
                        if self.pending_readers[r] == 1
                        and r in self.memory.allocations
                        and not self.replicas[r].retain
                    )
                    candidates.append(Candidate(a.id, s.ready_s, self.ranks[a.id], releasable))
            snapshot = Snapshot(
                self.now,
                tuple(candidates),
                tuple(
                    (r.id, self.capacities[r.id] - self.slot_used[r.id])
                    for r in self.resources.values()
                    if r.kind == "slots"
                ),
                tuple((p, self.memory.capacity[p] - used) for p, used in self.memory.used.items()),
                tuple(sorted(self.started)),
            )
            proposed = self.scheduler.propose(snapshot)
            require(
                len(set(proposed)) == len(proposed) and set(proposed) <= {c.id for c in candidates},
                "INVALID_SCHEDULER_ACTION",
                "scheduler",
                "start only ready activities once",
            )
            for aid in proposed:
                a = self.activities[aid]
                blockers, requests = self.can_start(a)
                if blockers:
                    self.waiters[aid] = blockers
                    if self.previous_wait.get(aid) != blockers:
                        self.emit("WAITING", aid, reason=blockers)
                        self.previous_wait[aid] = blockers
                    continue
                ok, _ = self.memory.try_reserve_many(requests)
                assert ok
                self.waiters.pop(aid, None)
                for rid, _, _ in requests:
                    self.memory.mark_writing(rid)
                    self.emit_allocation("ALLOCATED", rid, aid)
                for rid in a.input_replica_ids:
                    self.memory.pin(rid)
                for rid, count in a.slot_demands:
                    self.slot_used[rid] += count
                for rid in self.gates(a):
                    self.exclusive_owner[rid] = aid
                s = self.states[aid]
                s.start_s = self.now
                s.state = "STARTUP" if a.setup_s else "ACTIVE"
                self.ready_ids.remove(aid)
                self.started.add(aid)
                self.emit(
                    "ACTIVITY_STARTED",
                    aid,
                    phase=s.state,
                    total_work=a.total_work,
                    work_unit=a.work_unit,
                )
                if a.setup_s:
                    self.queue.push(self.now + a.setup_s, "PHASE_READY", aid)
                changed = True
                transitions += 1
            if transitions > self.config.max_same_time_transitions:
                return False
            if not changed:
                break
        return True

    def complete(self, aid):
        a, s = self.activities[aid], self.states[aid]
        assert s.state == "ACTIVE" and aid not in self.completed
        s.state, s.end_s, s.rate = "COMPLETED", self.now, 0.0
        s.remaining_work = 0.0
        s.served = a.total_work
        s.generation += 1
        self.completed.add(aid)
        for successor in self.successors[aid]:
            self.predecessors_left[successor] -= 1
            if self.predecessors_left[successor] == 0:
                self.ready_ids.add(successor)
        for rid in a.output_replica_ids:
            self.memory.mark_valid(rid)
            self.replica_ready[rid] = self.now
            self.emit_allocation("REPLICA_VALID", rid, aid)
        for rid in a.input_replica_ids:
            self.memory.unpin(rid)
            self.pending_readers[rid] -= 1
        for i, _ in enumerate(a.workspace):
            rid = f"workspace:{a.id}:{i}"
            self.emit_allocation("FREED", rid, aid)
            self.memory.release(rid)
        for rid, count in a.slot_demands:
            self.slot_used[rid] -= count
        for rid in self.gates(a):
            assert self.exclusive_owner.pop(rid) == aid
        self.emit("ACTIVITY_FINISHED", aid, served_work=s.served)
        self.garbage_collect()

    def garbage_collect(self):
        for rid in sorted(list(self.memory.allocations)):
            if (
                rid in self.replicas
                and not self.replicas[rid].retain
                and self.pending_readers[rid] == 0
            ):
                allocation = self.memory.allocations[rid]
                if allocation.state == "VALID" and allocation.readers == 0:
                    self.emit_allocation("FREED", rid)
                    self.memory.release(rid)

    def recompute_rates(self):
        flows = {
            aid: dict(self.activities[aid].rate_demands)
            for aid, s in self.states.items()
            if s.state == "ACTIVE"
            and self.activities[aid].work_unit == "byte"
            and s.remaining_work > 0
        }
        rates = (
            weighted_max_min(
                flows,
                self.capacities,
                {aid: self.activities[aid].weight for aid in flows},
                {
                    aid: self.activities[aid].max_rate
                    for aid in flows
                    if self.activities[aid].max_rate is not None
                },
            )
            if flows
            else {}
        )
        for aid in sorted(self.started - self.completed):
            s, a = self.states[aid], self.activities[aid]
            if s.state != "ACTIVE":
                continue
            rate = rates[aid] if a.work_unit == "byte" else 1.0
            s.generation += 1
            if rate != s.rate:
                self.emit("RATE_CHANGED", aid, rate=rate, remaining_work=s.remaining_work)
            s.rate = rate
            if rate > 0:
                predicted = self.now + s.remaining_work / rate
                # Never round a completion down: subtracting nearby timestamps
                # can otherwise leave nontrivial bytes at very high bandwidth.
                if not math.isfinite(predicted):
                    predicted = math.nextafter(self.config.max_sim_time, math.inf)
                elif (predicted - self.now) * rate < s.remaining_work:
                    predicted = math.nextafter(predicted, math.inf)
                self.queue.push(predicted, "PREDICTED", aid, s.generation)

    def update_requests(self):
        contracts = dict(self.e.request_outputs)
        for req in self.e.requests:
            if (
                req.id not in self.request_finish
                and self.now >= req.arrival_s
                and all(r in self.replica_ready for r in contracts[req.id])
            ):
                finish = max(req.arrival_s, *(self.replica_ready[r] for r in contracts[req.id]))
                self.request_finish[req.id] = finish
                self.emit(
                    "REQUEST_FINISHED",
                    request_id=req.id,
                    finish_s=finish,
                    latency_s=finish - req.arrival_s,
                )

    def emit_allocation(self, event_type, rid, aid=None):
        a = self.memory.allocations[rid]
        self.emit(
            event_type,
            aid,
            allocation_id=rid,
            pool_id=a.pool_id,
            allocation_bytes=a.bytes,
            allocation_state="FREED" if event_type == "FREED" else a.state,
            active_readers=a.readers,
            pending_readers=self.pending_readers[rid],
            tensor_id=self.replicas[rid].tensor_id if rid in self.replicas else None,
        )

    def assert_invariants(self):
        self.memory.assert_invariants()
        for rid, used in self.slot_used.items():
            assert 0 <= used <= self.capacities[rid]
        demand = defaultdict(float)
        for aid, s in self.states.items():
            a = self.activities[aid]
            assert math.isclose(
                s.served + s.remaining_work, a.total_work, abs_tol=1e-9, rel_tol=1e-9
            )
            if s.state in ("ACTIVE", "STARTUP"):
                assert all(p in self.completed for p in a.predecessor_ids)
                assert all(
                    self.memory.allocations[r].state == "VALID"
                    and self.memory.allocations[r].readers > 0
                    for r in a.input_replica_ids
                )
            if s.state == "ACTIVE":
                for rid, coefficient in a.rate_demands:
                    demand[rid] += s.rate * coefficient
                if a.total_work:
                    for rid, work in a.service_work:
                        demand[rid] += s.rate * work / a.total_work
        assert all(
            value <= self.capacities[rid] * (1 + 1e-9) + 1e-12 for rid, value in demand.items()
        )

    def snapshot(self):
        waiting = {}
        for aid in self.activities:
            if aid not in self.started:
                waiting[aid] = self.can_start(self.activities[aid])[0]
        paused = {
            aid: {
                "remaining_work": s.remaining_work,
                "resources": list(dict(self.activities[aid].rate_demands)),
            }
            for aid, s in self.states.items()
            if s.state == "ACTIVE" and s.rate == 0
        }
        return {
            "sim_time": self.now,
            "waiting": waiting,
            "zero_rate_activities": paused,
            "allocations": {rid: asdict(a) for rid, a in self.memory.allocations.items()},
            "pending_readers": dict(self.pending_readers),
            "slot_used": dict(self.slot_used),
            "capacities": self.capacities.copy(),
        }

    def finish(self, status, reason=None, snapshot=None):
        self.update_requests()
        requests = []
        for req in self.e.requests:
            finish = self.request_finish.get(req.id)
            markers = {
                name: self.replica_ready.get(rid) for q, name, rid in self.e.markers if q == req.id
            }
            requests.append(
                {
                    "request_id": req.id,
                    "arrival_s": req.arrival_s,
                    "finish_s": finish,
                    "latency_s": None if finish is None else finish - req.arrival_s,
                    "status": "completed"
                    if finish is not None
                    else "unfinished"
                    if self.now >= req.arrival_s
                    else "not_arrived",
                    "markers": markers,
                }
            )
        arrival = min((r.arrival_s for r in self.e.requests), default=0.0)
        finish = max(self.request_finish.values(), default=self.now)
        makespan = (
            (finish - arrival if self.e.requests else self.now) if status == "SUCCESS" else None
        )
        resources = []
        for r in self.resources.values():
            if r.kind == "storage":
                continue
            capacity = self.capacity_integral[r.id]
            used = self.reserved_integral[r.id]
            resources.append(
                {
                    "resource_id": r.id,
                    "kind": r.kind,
                    "unit": r.unit,
                    "capacity_integral": capacity,
                    "service_integral": self.service_integral[r.id],
                    "reservation_integral": used,
                    "service_utilization": self.service_integral[r.id] / capacity
                    if capacity
                    else None,
                    "slot_utilization": used / capacity if r.kind == "slots" and capacity else None,
                    "exclusive_gate_fraction": used / self.now
                    if r.kind == "rate" and self.now
                    else None,
                }
            )
        timeline = tuple(
            {
                "activity_id": a.id,
                "kind": a.kind,
                "op_id": a.op_id,
                "tensor_id": a.tensor_id,
                "request_id": a.request_id,
                "start_s": self.states[a.id].start_s,
                "finish_s": self.states[a.id].end_s,
                "setup_s": a.setup_s,
                "state": self.states[a.id].state,
                "total_work": a.total_work,
                "served_work": self.states[a.id].served,
                "remaining_work": self.states[a.id].remaining_work,
                "work_unit": a.work_unit,
                "generation": self.states[a.id].generation,
            }
            for a in self.e.activities
        )
        latencies = sorted(r["latency_s"] for r in requests if r["latency_s"] is not None)
        from ..analysis.metrics import interval_union, nearest_rank

        compute_intervals = [
            (r["start_s"], r["finish_s"])
            for r in timeline
            if r["kind"] == "compute" and r["finish_s"] is not None
        ]
        transfer_intervals = [
            (r["start_s"], r["finish_s"])
            for r in timeline
            if r["kind"] == "transfer" and r["finish_s"] is not None
        ]
        compute_time, transfer_time = (
            interval_union(compute_intervals),
            interval_union(transfer_intervals),
        )
        union = interval_union(compute_intervals + transfer_intervals)
        metrics = {
            "finish_time_s": self.now,
            "completed_requests": len(self.request_finish),
            "unfinished_requests": sum(r["status"] == "unfinished" for r in requests),
            "not_arrived_requests": sum(r["status"] == "not_arrived" for r in requests),
            "latency_samples": len(latencies),
            "quantile_method": "nearest_rank",
            "p50_latency_s": nearest_rank(latencies, 0.5),
            "p95_latency_s": nearest_rank(latencies, 0.95),
            "p99_latency_s": nearest_rank(latencies, 0.99),
            "throughput_requests_per_s": len(self.request_finish) / makespan if makespan else None,
            "peak_memory_bytes": self.memory.peak.copy(),
            "memory_byte_seconds": dict(self.storage_integral),
            "bytes_transferred": sum(
                s.served
                for aid, s in self.states.items()
                if self.activities[aid].kind == "transfer"
            ),
            "compute_interval_union_s": compute_time,
            "transfer_interval_union_s": transfer_time,
            "compute_transfer_overlap_s": compute_time + transfer_time - union,
            "events_processed": self.processed,
            "stale_predictions_discarded": self.stale_events,
            "model_level": "D"
            if any(p.model == "markov" for p in self.config.capacity_processes)
            else "C"
            if self.config.capacity_processes
            else "B"
            if self.config.network_mode == "fluid"
            else "A",
        }
        self.emit("RUN_FINISHED", status=status, reason=reason)
        return RunResult(
            status,
            makespan,
            self.now,
            metrics,
            tuple(resources),
            tuple(requests),
            tuple(self.trace),
            timeline,
            reason,
            snapshot,
            (
                "Nonpreemptive fixed-duration kernels; exclusive kernel memory groups.",
                "Whole-tensor validity; explicit legal routes; aligned integer physical storage.",
                "Conservative exclusive reservations include startup; transfer fluid sharing uses byte-rate weighted max-min.",
                "Kernel timings and hardware profiles require independent empirical calibration.",
            ),
        )


def indexed_processes(processes):
    result = {}
    for p in processes:
        require(
            p.resource_id not in result, "DUPLICATE_PROCESS", "capacity_processes", p.resource_id
        )
        result[p.resource_id] = p
    return result
