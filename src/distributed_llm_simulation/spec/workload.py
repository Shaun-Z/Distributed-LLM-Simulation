from __future__ import annotations

import math
from typing import Literal

import networkx as nx
from pydantic import model_validator

from .base import Bytes, NonNegative, Spec, indexed, require

DTYPE_BITS = {
    "fp64": 64,
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "uint8": 8,
    "int4": 4,
    "bool": 1,
}


class TensorSpec(Spec):
    id: str
    shape: tuple[Bytes, ...] = ()
    dtype: str = "fp32"
    storage_bytes: Bytes
    wire_bytes: Bytes | None = None
    producer_id: str | None = None
    consumer_ids: tuple[str, ...] = ()
    initial_pool_ids: tuple[str, ...] = ()
    persistent: bool = False
    version: Bytes = 0

    @model_validator(mode="after")
    def check(self):
        require(self.dtype in DTYPE_BITS, "INVALID_DTYPE", self.id, self.dtype)
        minimum = math.ceil(math.prod(self.shape) * DTYPE_BITS[self.dtype] / 8) if self.shape else 0
        require(
            self.storage_bytes >= minimum,
            "INVALID_SIZE",
            self.id,
            "storage smaller than dense shape; use explicit packed shape",
        )
        return self


class OperatorSpec(Spec):
    id: str
    op_type: str
    algorithm: str = "default"
    layout: str = "dense"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    input_dtype: str = "fp32"
    accumulate_dtype: str = "fp32"
    output_dtype: str = "fp32"
    flops: NonNegative = 0
    integer_ops: NonNegative = 0
    read_bytes: tuple[tuple[str, Bytes], ...] = ()
    write_bytes: tuple[tuple[str, Bytes], ...] = ()
    workspace_bytes: Bytes = 0
    duration_s: NonNegative | None = None
    request_id: str | None = None
    control_predecessors: tuple[str, ...] = ()
    shape_params: tuple[tuple[str, Bytes], ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()


class RequestSpec(Spec):
    id: str
    arrival_s: NonNegative = 0
    output_tensor_ids: tuple[str, ...]
    markers: tuple[tuple[str, str], ...] = ()


class WorkloadSpec(Spec):
    schema_version: Literal[1] = 1
    name: str
    tensors: tuple[TensorSpec, ...]
    operators: tuple[OperatorSpec, ...]
    requests: tuple[RequestSpec, ...] = ()

    @model_validator(mode="after")
    def check(self):
        ts, ops, reqs = (
            indexed(self.tensors, "tensors"),
            indexed(self.operators, "operators"),
            indexed(self.requests, "requests"),
        )
        actual_consumers = {tid: set() for tid in ts}
        actual_producers = {tid: [] for tid in ts}
        for op in self.operators:
            require(
                not (op.flops and op.integer_ops),
                "MIXED_WORK_UNITS",
                op.id,
                "split integer and floating work",
            )
            require(
                op.request_id is None or op.request_id in reqs,
                "UNKNOWN_REQUEST",
                op.id,
                str(op.request_id),
            )
            require(
                len(set(op.outputs)) == len(op.outputs),
                "DUPLICATE_OUTPUT",
                op.id,
                "duplicate tensor",
            )
            for tid in op.inputs + op.outputs:
                require(tid in ts, "UNKNOWN_TENSOR", op.id, tid)
            for tid in op.inputs:
                actual_consumers[tid].add(op.id)
                require(ts[tid].dtype == op.input_dtype, "DTYPE_MISMATCH", op.id, tid)
            for tid in op.outputs:
                actual_producers[tid].append(op.id)
                require(ts[tid].dtype == op.output_dtype, "DTYPE_MISMATCH", op.id, tid)
            for pred in op.control_predecessors:
                require(pred in ops, "UNKNOWN_OPERATOR", op.id, pred)
        for t in self.tensors:
            require(
                actual_producers[t.id] == ([t.producer_id] if t.producer_id else []),
                "INVALID_PRODUCER",
                t.id,
                "exactly one declared producer required",
            )
            require(
                set(t.consumer_ids) == actual_consumers[t.id],
                "INVALID_CONSUMERS",
                t.id,
                "consumer ports differ",
            )
            require(
                t.producer_id is not None or bool(t.initial_pool_ids),
                "MISSING_INITIAL_DATA",
                t.id,
                "no producer or initial pool",
            )
            require(
                not (t.producer_id and t.initial_pool_ids),
                "INVALID_INITIAL_DATA",
                t.id,
                "produced tensor cannot start valid",
            )
        for r in self.requests:
            require(bool(r.output_tensor_ids), "INVALID_REQUEST", r.id, "output required")
            for tid in r.output_tensor_ids + tuple(t for _, t in r.markers):
                require(tid in ts, "UNKNOWN_TENSOR", r.id, tid)
        require(
            nx.is_directed_acyclic_graph(self.graph()),
            "DEPENDENCY_CYCLE",
            "workload",
            "operator/tensor graph must be acyclic",
        )
        return self

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for t in self.tensors:
            g.add_node(("tensor", t.id), kind="tensor")
            if t.producer_id:
                g.add_edge(("op", t.producer_id), ("tensor", t.id))
            for consumer in t.consumer_ids:
                g.add_edge(("tensor", t.id), ("op", consumer))
        for o in self.operators:
            g.add_node(("op", o.id), kind="operator")
            for p in o.control_predecessors:
                g.add_edge(("op", p), ("op", o.id))
        return g

    def order(self) -> tuple[str, ...]:
        return tuple(
            i for kind, i in nx.lexicographical_topological_sort(self.graph()) if kind == "op"
        )
