"""Pure capacity traces and reproducible, independently seeded scenario generators."""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from typing import Literal

from pydantic import model_validator

from ..spec.base import NonNegative, Spec, require


class CapacityProcess(Spec):
    resource_id: str
    model: Literal["piecewise_constant", "markov"] = "piecewise_constant"
    seed: int | None = None
    source: str = "specified"
    points: tuple[tuple[NonNegative, NonNegative], ...]

    @model_validator(mode="after")
    def check(self):
        require(
            bool(self.points) and self.points[0][0] == 0,
            "INVALID_TRACE",
            self.resource_id,
            "must start at t=0",
        )
        require(
            all(a[0] < b[0] for a, b in zip(self.points, self.points[1:], strict=False)),
            "INVALID_TRACE",
            self.resource_id,
            "times must strictly increase",
        )
        return self

    def value_at(self, now_s):
        return self.points[bisect.bisect_right([p[0] for p in self.points], now_s) - 1][1]

    def next_change_after(self, now_s):
        index = bisect.bisect_right([p[0] for p in self.points], now_s)
        return self.points[index][0] if index < len(self.points) else None


def independent_rng(seed, stream):
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def markov_capacity(resource_id, *, seed, horizon_s, capacities, transition_rates, initial_state=0):
    """Continuous-time finite-state Markov chain; rates[i][j] are transition hazards."""
    n = len(capacities)
    require(
        n > 0 and len(transition_rates) == n and all(len(row) == n for row in transition_rates),
        "INVALID_PROCESS",
        resource_id,
        "square rates required",
    )
    require(
        all(math.isfinite(c) and c >= 0 for c in capacities)
        and math.isfinite(horizon_s)
        and horizon_s >= 0,
        "INVALID_PROCESS",
        resource_id,
        "nonnegative finite capacity/horizon required",
    )
    require(
        0 <= initial_state < n
        and all(math.isfinite(x) and x >= 0 for row in transition_rates for x in row),
        "INVALID_PROCESS",
        resource_id,
        "invalid state/rate",
    )
    rng = independent_rng(seed, f"environment:{resource_id}")
    state, now = initial_state, 0.0
    points = [(0.0, capacities[state])]
    while now < horizon_s:
        rates = [0 if j == state else r for j, r in enumerate(transition_rates[state])]
        total = sum(rates)
        if not total:
            break
        now += rng.expovariate(total)
        if now > horizon_s:
            break
        draw, cumulative = rng.random() * total, 0.0
        for j, rate in enumerate(rates):
            cumulative += rate
            if draw < cumulative:
                state = j
                break
        points.append((now, capacities[state]))
    return CapacityProcess(
        resource_id=resource_id,
        points=tuple(points),
        model="markov",
        seed=seed,
        source="synthetic continuous-time Markov chain",
    )


def poisson_arrivals(*, seed, rate_per_s, count):
    require(
        rate_per_s > 0 and math.isfinite(rate_per_s) and type(count) is int and count >= 0,
        "INVALID_ARRIVALS",
        "arrivals",
        "positive rate and nonnegative integer count required",
    )
    rng = independent_rng(seed, "arrivals")
    now = 0.0
    result = []
    for _ in range(count):
        now += rng.expovariate(rate_per_s)
        result.append(now)
    return tuple(result)
