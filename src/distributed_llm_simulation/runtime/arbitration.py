"""Weighted max-min progressive filling for linear, nonnegative byte-rate constraints."""

from __future__ import annotations

import math


def weighted_max_min(demands, capacities, weights=None, ceilings=None):
    weights, ceilings = weights or {}, ceilings or {}
    matrix = {f: dict(rows) for f, rows in demands.items()}
    capacity = dict(capacities)
    for f, ceiling in ceilings.items():
        rid = ("flow_ceiling", f)
        capacity[rid] = ceiling
        matrix[f][rid] = 1.0
    for f, row in matrix.items():
        if not row or not math.isfinite(weights.get(f, 1)) or weights.get(f, 1) <= 0:
            raise ValueError(f"invalid flow/weight: {f}")
        if any(r not in capacity or not math.isfinite(a) or a <= 0 for r, a in row.items()):
            raise ValueError(f"invalid constraint: {f}")
    if any(not math.isfinite(c) or c < 0 for c in capacity.values()):
        raise ValueError("capacity must be finite and nonnegative")
    residual, rates, pending = dict(capacity), dict.fromkeys(matrix, 0.0), set(matrix)
    while pending:
        slopes = {
            r: sum(matrix[f].get(r, 0) * weights.get(f, 1) for f in sorted(pending))
            for r in capacity
        }
        ratios = {r: max(0.0, residual[r]) / slope for r, slope in slopes.items() if slope > 0}
        delta = min(ratios.values())
        for f in sorted(pending):
            rates[f] += weights.get(f, 1) * delta
        for r, slope in slopes.items():
            residual[r] = max(0.0, residual[r] - slope * delta)
        # Freeze from the minimizing ratios, rather than an absolute byte/s epsilon.
        saturated = {
            r for r, ratio in ratios.items() if math.isclose(ratio, delta, rel_tol=1e-12, abs_tol=0)
        }
        frozen = {f for f in pending if any(r in saturated for r in matrix[f])}
        if not frozen:
            raise ArithmeticError("progressive filling made no progress")
        pending -= frozen
    return rates
