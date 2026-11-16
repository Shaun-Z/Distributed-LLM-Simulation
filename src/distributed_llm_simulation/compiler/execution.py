"""Generic lowered activity graph; no workload-specific terms enter the engine."""

from __future__ import annotations

from typing import Literal

import networkx as nx
from pydantic import model_validator

from ..spec.base import Bytes, Count, NonNegative, Positive, Spec, indexed, require
from ..spec.hardware import ResourceSpec
from ..spec.workload import RequestSpec


class ReplicaSpec(Spec):
    id: str
    tensor_id: str
    pool_id: str
    storage_bytes: Bytes
    version: Bytes = 0
    initial: bool = False
    retain: bool = False


class ActivitySpec(Spec):
    id: str
    kind: Literal["compute", "transfer", "sync"]
    predecessor_ids: tuple[str, ...] = ()
    total_work: NonNegative = 0
    work_unit: Literal["s", "byte"] = "s"
    rate_demands: tuple[tuple[str, Positive], ...] = ()
    slot_demands: tuple[tuple[str, Count], ...] = ()
    exclusive_resources: tuple[str, ...] = ()
    input_replica_ids: tuple[str, ...] = ()
    output_replica_ids: tuple[str, ...] = ()
    workspace: tuple[tuple[str, Bytes], ...] = ()
    op_id: str | None = None
    tensor_id: str | None = None
    request_id: str | None = None
    release_s: NonNegative = 0
    setup_s: NonNegative = 0
    weight: Positive = 1
    max_rate: Positive | None = None
    # Measured/modelled work used only for reporting, not an extra IO charge.
    service_work: tuple[tuple[str, NonNegative], ...] = ()

    @model_validator(mode="after")
    def check(self):
        for name in ("rate_demands", "slot_demands", "service_work"):
            rows = getattr(self, name)
            require(len(dict(rows)) == len(rows), "DUPLICATE_DEMAND", self.id, name)
        for name in (
            "predecessor_ids",
            "input_replica_ids",
            "output_replica_ids",
            "exclusive_resources",
        ):
            ids = getattr(self, name)
            require(len(set(ids)) == len(ids), "DUPLICATE_REFERENCE", self.id, name)
        require(
            not set(self.input_replica_ids) & set(self.output_replica_ids),
            "MUTABLE_ALIAS_UNSUPPORTED",
            self.id,
            "input/output overlap",
        )
        require(
            self.kind != "transfer" or self.work_unit == "byte",
            "INVALID_WORK_UNIT",
            self.id,
            "transfer uses bytes",
        )
        require(
            self.kind != "compute" or self.work_unit == "s",
            "INVALID_WORK_UNIT",
            self.id,
            "fixed compute uses seconds",
        )
        require(
            self.work_unit != "s" or not self.rate_demands,
            "INVALID_DEMAND",
            self.id,
            "fixed time cannot use byte-rate demands",
        )
        require(
            self.work_unit != "byte" or bool(self.rate_demands) or self.max_rate is not None,
            "UNBOUNDED_FLOW",
            self.id,
            "finite limiter required",
        )
        return self


class ExecutionPlan(Spec):
    activities: tuple[ActivitySpec, ...]
    replicas: tuple[ReplicaSpec, ...] = ()
    extra_resources: tuple[ResourceSpec, ...] = ()
    requests: tuple[RequestSpec, ...] = ()
    # Per request, replicas required by its output contract.
    request_outputs: tuple[tuple[str, tuple[str, ...]], ...] = ()
    markers: tuple[tuple[str, str, str], ...] = ()  # request, name, replica
    hardware_hash: str | None = None
    workload_hash: str | None = None
    placement_hash: str | None = None

    @model_validator(mode="after")
    def check(self):
        acts, replicas = indexed(self.activities, "activities"), indexed(self.replicas, "replicas")
        indexed(self.extra_resources, "extra_resources")
        requests = indexed(self.requests, "requests")
        producers = {}
        g = self.graph()
        for a in self.activities:
            for p in a.predecessor_ids:
                require(p in acts, "UNKNOWN_ACTIVITY", a.id, p)
            for r in a.input_replica_ids + a.output_replica_ids:
                require(r in replicas, "UNKNOWN_REPLICA", a.id, r)
            for r in a.output_replica_ids:
                require(r not in producers and not replicas[r].initial, "INVALID_PRODUCER", a.id, r)
                producers[r] = a.id
        require(nx.is_directed_acyclic_graph(g), "DEPENDENCY_CYCLE", "execution", "cycle")
        for a in self.activities:
            for r in a.input_replica_ids:
                require(
                    replicas[r].initial or r in producers and nx.has_path(g, producers[r], a.id),
                    "INVALID_DATA_DEPENDENCY",
                    a.id,
                    r,
                )
        for r in self.replicas:
            require(
                r.initial or r.id in producers, "MISSING_PRODUCER", r.id, "uninitialized replica"
            )
        require(
            len(dict(self.request_outputs)) == len(self.request_outputs),
            "DUPLICATE_OUTPUT",
            "request_outputs",
            "duplicate request",
        )
        for q, name, rid in self.markers:
            require(
                q in requests and rid in replicas, "INVALID_MARKER", name, "unknown request/replica"
            )
        for q, outputs in self.request_outputs:
            require(
                q in requests
                and bool(outputs)
                and all(r in replicas and replicas[r].retain for r in outputs),
                "INVALID_OUTPUT",
                q,
                "unknown request/replica",
            )
        require(
            set(dict(self.request_outputs)) == set(requests),
            "INVALID_OUTPUT",
            "request_outputs",
            "each request needs a contract",
        )
        return self

    def graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(a.id for a in self.activities)
        for a in self.activities:
            g.add_edges_from((p, a.id) for p in a.predecessor_ids)
        return g
