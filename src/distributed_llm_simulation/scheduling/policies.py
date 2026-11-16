from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import networkx as nx


@dataclass(frozen=True)
class Candidate:
    id: str
    ready_s: float
    rank: float
    releasable_bytes: int


@dataclass(frozen=True)
class Snapshot:
    now_s: float
    candidates: tuple[Candidate, ...]
    free_slots: tuple[tuple[str, int], ...]
    free_storage_bytes: tuple[tuple[str, int], ...]
    started_activity_ids: tuple[str, ...] = ()


class Scheduler(Protocol):
    def propose(self, snapshot: Snapshot) -> tuple[str, ...]: ...


class FIFOListScheduler:
    def propose(self, snapshot):
        return tuple(c.id for c in sorted(snapshot.candidates, key=lambda c: (c.ready_s, c.id)))


class CriticalRankScheduler:
    def propose(self, snapshot):
        return tuple(
            c.id for c in sorted(snapshot.candidates, key=lambda c: (-c.rank, c.ready_s, c.id))
        )


class MemoryAwareScheduler:
    def propose(self, snapshot):
        return tuple(
            c.id
            for c in sorted(
                snapshot.candidates, key=lambda c: (-c.releasable_bytes, c.ready_s, c.id)
            )
        )


class OrderedScheduler:
    """Enforce one total start order, allowing deliberate idle time.

    Stateless: rejected starts remain at the head until the engine accepts them.
    The exhaustive A-level oracle enumerates these orders.
    """

    def __init__(self, order):
        self.order = tuple(order)

    def propose(self, snapshot):
        started = set(snapshot.started_activity_ids)
        ready = {c.id for c in snapshot.candidates}
        head = next((aid for aid in self.order if aid not in started), None)
        return (head,) if head in ready else ()


def upward_ranks(execution):
    g = execution.graph()
    activities = {a.id: a for a in execution.activities}
    ranks = {}
    for aid in reversed(list(nx.topological_sort(g))):
        a = activities[aid]
        duration = a.total_work if a.work_unit == "s" else a.total_work / (a.max_rate or 1)
        ranks[aid] = a.setup_s + duration + max((ranks[n] for n in g.successors(aid)), default=0)
    return ranks


def scheduler_by_name(name):
    return {
        "fifo": FIFOListScheduler,
        "critical_rank": CriticalRankScheduler,
        "memory_aware": MemoryAwareScheduler,
    }[name]()
