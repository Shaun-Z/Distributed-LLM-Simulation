from __future__ import annotations

from typing import Literal

import networkx as nx
from pydantic import field_validator, model_validator

from .base import Bytes, Count, NonNegative, Positive, Spec, indexed, require


class ResourceSpec(Spec):
    id: str
    kind: Literal["storage", "rate", "slots"]
    unit: Literal["byte", "byte/s", "FLOP/s", "op/s", "count"]
    capacity: int | float
    policy: Literal["exclusive", "shared"] = "exclusive"

    @field_validator("capacity", mode="before")
    @classmethod
    def numeric_capacity(cls, value):
        if type(value) not in (int, float):
            raise ValueError("capacity must be a numeric quantity, not a boolean/string")
        return value

    @model_validator(mode="after")
    def check(self):
        require(self.capacity >= 0, "INVALID_CAPACITY", self.id, "must be nonnegative")
        if self.kind in ("storage", "slots"):
            require(
                type(self.capacity) is int, "INVALID_CAPACITY", self.id, "exact integer required"
            )
        expected = {"storage": {"byte"}, "slots": {"count"}, "rate": {"byte/s", "FLOP/s", "op/s"}}
        require(self.unit in expected[self.kind], "INVALID_UNIT", self.id, self.unit)
        return self


class NodeSpec(Spec):
    id: str
    kind: Literal["compute", "memory", "nic", "switch", "dma"]
    host_id: str | None = None
    resource_ids: tuple[str, ...] = ()
    device_type: str | None = None
    kernel_profile_ids: tuple[str, ...] = ()
    accessible_pool_ids: tuple[str, ...] = ()
    launch_s: NonNegative = 0
    physical_pool_id: str | None = None
    capacity_resource_id: str | None = None
    address_domain: str | None = None
    tier: str | None = None
    technology: str = "unspecified"
    alignment_bytes: Count = 1


class ChannelSpec(Spec):
    id: str
    src: str
    dst: str
    kind: str
    allowed_operations: tuple[
        Literal["local_load", "local_store", "copy", "p2p_copy", "rdma_read", "rdma_write"], ...
    ]
    payload_Bps: Positive
    latency_s: NonNegative = 0
    setup_s: NonNegative = 0
    resource_demands: tuple[tuple[str, Positive], ...] = ()
    pair_id: str | None = None


class RouteSpec(Spec):
    id: str
    channel_ids: tuple[str, ...]
    mode: Literal["fluid_path"] = "fluid_path"
    operation: Literal["copy", "p2p_copy", "rdma_read", "rdma_write"] = "copy"
    slot_demands: tuple[tuple[str, Count], ...] = ()


class KernelProfile(Spec):
    id: str
    op_type: str
    input_dtype: str
    accumulate_dtype: str
    output_dtype: str
    model: Literal["analytical_roofline", "calibrated_duration"]
    effective_ops_per_s: Positive | None = None
    duration_s: NonNegative | None = None
    slot_resource_id: str
    memory_rate_resource_ids: tuple[str, ...] = ()
    launch_s: NonNegative = 0
    layouts: tuple[str, ...] = ()
    algorithms: tuple[str, ...] = ()
    shape_constraints: tuple[tuple[str, Bytes, Bytes], ...] = ()

    @model_validator(mode="after")
    def check(self):
        require(
            all(lo <= hi for _, lo, hi in self.shape_constraints),
            "INVALID_SHAPE",
            self.id,
            "lower shape limit exceeds upper",
        )
        require(
            self.duration_s is not None
            if self.model == "calibrated_duration"
            else self.effective_ops_per_s is not None,
            "INVALID_PROFILE",
            self.id,
            "model needs duration_s or effective_ops_per_s",
        )
        return self


class VisibilitySpec(Spec):
    pool_id: str
    device_ids: tuple[str, ...]
    shared_addressable: bool
    visibility_policy: Literal["explicit_event"] = "explicit_event"
    ownership_transition_s: NonNegative = 0


class HardwareSpec(Spec):
    schema_version: Literal[1] = 1
    name: str
    resources: tuple[ResourceSpec, ...]
    nodes: tuple[NodeSpec, ...]
    channels: tuple[ChannelSpec, ...] = ()
    routes: tuple[RouteSpec, ...] = ()
    kernel_profiles: tuple[KernelProfile, ...] = ()
    visibility: tuple[VisibilitySpec, ...] = ()
    synthetic: bool = True

    @model_validator(mode="after")
    def check(self):
        rs, ns, cs, ks = (
            indexed(getattr(self, name), name)
            for name in ("resources", "nodes", "channels", "kernel_profiles")
        )
        indexed(self.routes, "routes")
        owners = {}
        for n in self.nodes:
            for rid in n.resource_ids:
                require(rid in rs, "UNKNOWN_RESOURCE", n.id, rid)
            if n.kind == "memory":
                require(bool(n.physical_pool_id), "INVALID_POOL", n.id, "physical_pool_id required")
                require(
                    n.capacity_resource_id in rs and rs[n.capacity_resource_id].kind == "storage",
                    "INVALID_POOL",
                    n.id,
                    "storage resource required",
                )
                # Multiple views of one pool share exactly one owner, domain and alignment.
                signature = (n.capacity_resource_id, n.address_domain, n.alignment_bytes)
                require(
                    n.physical_pool_id not in owners or owners[n.physical_pool_id] == signature,
                    "DUPLICATE_POOL",
                    n.id,
                    "inconsistent physical pool views",
                )
                owners[n.physical_pool_id] = signature
            elif n.kind == "compute":
                require(bool(n.device_type), "INVALID_COMPUTE", n.id, "device_type required")
                for pool in n.accessible_pool_ids:
                    require(pool in ns and ns[pool].kind == "memory", "UNKNOWN_POOL", n.id, pool)
                for kid in n.kernel_profile_ids:
                    require(kid in ks, "UNKNOWN_KERNEL", n.id, kid)
        capacity_owners = {}
        for pool, (capacity, _, _) in owners.items():
            require(
                capacity not in capacity_owners,
                "DUPLICATE_POOL",
                pool,
                "storage budget has multiple physical owners",
            )
            capacity_owners[capacity] = pool
        for k in self.kernel_profiles:
            require(
                k.slot_resource_id in rs and rs[k.slot_resource_id].kind == "slots",
                "INVALID_SLOT",
                k.id,
                k.slot_resource_id,
            )
            for r in k.memory_rate_resource_ids:
                require(r in rs and rs[r].unit == "byte/s", "INVALID_RATE", k.id, r)
        for c in self.channels:
            require(c.src in ns and c.dst in ns, "UNKNOWN_ENDPOINT", c.id, f"{c.src}->{c.dst}")
            for rid, _ in c.resource_demands:
                require(rid in rs and rs[rid].unit == "byte/s", "INVALID_RATE", c.id, rid)
        for route in self.routes:
            require(bool(route.channel_ids), "INVALID_ROUTE", route.id, "empty route")
            for cid in route.channel_ids:
                require(cid in cs, "UNKNOWN_CHANNEL", route.id, cid)
                require(
                    route.operation in cs[cid].allowed_operations,
                    "ILLEGAL_OPERATION",
                    route.id,
                    cid,
                )
            edges = [cs[cid] for cid in route.channel_ids]
            require(
                all(a.dst == b.src for a, b in zip(edges, edges[1:], strict=False)),
                "DISCONTINUOUS_ROUTE",
                route.id,
                "channel endpoints differ",
            )
            require(
                ns[edges[0].src].kind == ns[edges[-1].dst].kind == "memory",
                "INVALID_ROUTE",
                route.id,
                "copy endpoints must be pools",
            )
            require(
                all(ns[e.dst].kind != "memory" for e in edges[:-1]),
                "STAGING_REQUIRED",
                route.id,
                "split route at intermediate memory",
            )
            for rid, count in route.slot_demands:
                require(
                    rid in rs and rs[rid].kind == "slots" and count <= rs[rid].capacity,
                    "INVALID_SLOT",
                    route.id,
                    rid,
                )
        for v in self.visibility:
            require(
                v.pool_id in ns and ns[v.pool_id].kind == "memory",
                "UNKNOWN_POOL",
                "visibility",
                v.pool_id,
            )
            require(
                all(d in ns and v.pool_id in ns[d].accessible_pool_ids for d in v.device_ids),
                "INVALID_VISIBILITY",
                v.pool_id,
                "device cannot access pool",
            )
        return self

    def graph(self, operation: str | None = None) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        g.add_nodes_from((n.id, {"spec": n}) for n in self.nodes)
        for c in self.channels:
            if operation is None or operation in c.allowed_operations:
                g.add_edge(c.src, c.dst, key=c.id, spec=c)
        return g

    def pool(self, node_id: str) -> str:
        node = next(n for n in self.nodes if n.id == node_id)
        require(node.kind == "memory", "INVALID_POOL", node_id, "not memory")
        return min(
            n.id
            for n in self.nodes
            if n.kind == "memory" and n.physical_pool_id == node.physical_pool_id
        )
