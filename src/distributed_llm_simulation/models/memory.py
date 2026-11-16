"""Exact integer storage accounting, with atomic multi-pool reservation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Allocation:
    id: str
    pool_id: str
    bytes: int
    state: str = "RESERVED"
    readers: int = 0


class MemoryManager:
    def __init__(self, hardware):
        self.capacity, self.alignment = {}, {}
        resources = {r.id: r for r in hardware.resources}
        for node in hardware.nodes:
            if node.kind == "memory":
                pool = hardware.pool(node.id)
                self.capacity[pool] = resources[node.capacity_resource_id].capacity
                self.alignment[pool] = node.alignment_bytes
        self.allocations: dict[str, Allocation] = {}
        self.used = {pool: 0 for pool in self.capacity}
        self.peak = dict(self.used)

    def aligned(self, pool, size):
        alignment = self.alignment[pool]
        return ((size + alignment - 1) // alignment) * alignment

    def blockers(self, requests):
        totals = defaultdict(int)
        ids = set()
        for aid, pool, size in requests:
            if aid in self.allocations or aid in ids:
                raise ValueError(f"duplicate allocation {aid}")
            ids.add(aid)
            totals[pool] += self.aligned(pool, size)
        return {
            pool: {
                "required_bytes": size,
                "free_bytes": self.capacity[pool] - self.used[pool],
                "holders": sorted(a.id for a in self.allocations.values() if a.pool_id == pool),
            }
            for pool, size in totals.items()
            if self.used[pool] + size > self.capacity[pool]
        }

    def try_reserve_many(self, requests):
        requests = tuple(requests)
        blocked = self.blockers(requests)
        if blocked:
            return False, blocked
        for aid, pool, size in requests:
            size = self.aligned(pool, size)
            self.allocations[aid] = Allocation(aid, pool, size)
            self.used[pool] += size
            self.peak[pool] = max(self.peak[pool], self.used[pool])
        return True, {}

    def mark_writing(self, aid):
        assert self.allocations[aid].state == "RESERVED"
        self.allocations[aid].state = "WRITING"

    def mark_valid(self, aid):
        assert self.allocations[aid].state in ("RESERVED", "WRITING")
        self.allocations[aid].state = "VALID"

    def pin(self, aid):
        assert self.allocations[aid].state == "VALID"
        self.allocations[aid].readers += 1

    def unpin(self, aid):
        assert self.allocations[aid].readers > 0
        self.allocations[aid].readers -= 1

    def release(self, aid):
        allocation = self.allocations[aid]
        assert allocation.readers == 0
        allocation.state = "FREED"
        self.used[allocation.pool_id] -= allocation.bytes
        del self.allocations[aid]

    def assert_invariants(self):
        for pool, capacity in self.capacity.items():
            assert 0 <= self.used[pool] <= capacity
            assert self.used[pool] == sum(
                a.bytes for a in self.allocations.values() if a.pool_id == pool
            )
        assert all(a.readers >= 0 for a in self.allocations.values())
