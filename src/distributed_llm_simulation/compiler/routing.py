"""Select only explicitly declared legal routes; memory endpoints create staging."""

from __future__ import annotations

import heapq
from collections import defaultdict

from ..spec.base import SpecError


class RoutingPolicy:
    def __init__(self, hardware, routes):
        self.hardware = hardware
        self.routes = {r.id: r for r in routes}
        self.channels = {c.id: c for c in hardware.channels}

    def endpoints(self, route):
        return (
            self.hardware.pool(self.channels[route.channel_ids[0]].src),
            self.hardware.pool(self.channels[route.channel_ids[-1]].dst),
        )

    def candidates(self, source, target, size, forced=None):
        if forced:
            if forced not in self.routes:
                raise SpecError("NO_ROUTE", "transfer_routes", forced)
            route = self.routes[forced]
            if self.endpoints(route) != (source, target):
                raise SpecError("NO_ROUTE", forced, "wrong endpoints")
            return (route,)
        adjacency = defaultdict(list)
        capacities = {r.id: r.capacity for r in self.hardware.resources}
        capacities.update({f"channel:{c.id}": c.payload_Bps for c in self.hardware.channels})
        for route in self.routes.values():
            src, dst = self.endpoints(route)
            demands, latency, ceiling = self.cost(route)
            limits = [ceiling] + [capacities[r] / coefficient for r, coefficient in demands]
            bottleneck = min(limits)
            cost = latency + size / bottleneck if bottleneck else float("inf")
            adjacency[src].append((dst, route.id, cost))
        # A zero-capacity route is still legal and may recover later.
        queue = [(0.0, (), source)]
        visited = set()
        while queue:
            cost, path, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == target:
                return tuple(self.routes[rid] for rid in path)
            for dst, rid, edge_cost in sorted(adjacency[node]):
                heapq.heappush(queue, (cost + edge_cost, path + (rid,), dst))
        raise SpecError("NO_ROUTE", f"{source}->{target}", "no legal declared copy or staging path")

    def cost(self, route):
        demands = defaultdict(float)
        latency = 0.0
        ceiling = float("inf")
        for cid in route.channel_ids:
            channel = self.channels[cid]
            latency += channel.setup_s + channel.latency_s
            ceiling = min(ceiling, channel.payload_Bps)
            # Repeated physical controller accesses really add byte demand.
            for rid, coefficient in channel.resource_demands:
                demands[rid] += coefficient
            # Channel capacity also constrains concurrent paths sharing this segment.
            demands[f"channel:{cid}"] += 1
        return tuple(sorted(demands.items())), latency, ceiling
