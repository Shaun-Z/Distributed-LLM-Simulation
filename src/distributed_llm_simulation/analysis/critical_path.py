"""Explain a realized trace with dependency and observed exclusive resource order."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx


def explain_critical_path(execution, result):
    rows = {r["activity_id"]: r for r in result.timeline if r["finish_s"] is not None}
    g = execution.graph().subgraph(rows).copy()
    reasons = {(a, b): "data/control" for a, b in g.edges}
    uses = defaultdict(list)
    for a in execution.activities:
        if a.id in rows:
            for rid in set(a.exclusive_resources) | {r for r, count in a.slot_demands}:
                uses[rid].append(a.id)
    for rid, activities in uses.items():
        ordered = sorted(
            activities, key=lambda aid: (rows[aid]["start_s"], rows[aid]["finish_s"], aid)
        )
        for a, b in zip(ordered, ordered[1:], strict=False):
            if rows[a]["finish_s"] <= rows[b]["start_s"] and a != b and not nx.has_path(g, b, a):
                g.add_edge(a, b)
                reasons[a, b] = f"observed resource order:{rid}"
    lengths, paths = {}, {}
    for aid in nx.topological_sort(g):
        parents = list(g.predecessors(aid))
        parent = max(parents, key=lambda p: lengths[p], default=None)
        duration = rows[aid]["finish_s"] - rows[aid]["start_s"]
        lengths[aid] = duration + (lengths[parent] if parent else 0)
        paths[aid] = (paths[parent] if parent else []) + [aid]
    last = max(lengths, key=lengths.get, default=None)
    path = paths[last] if last else []
    return {
        "scope": "realized schedule explanation; not a lower bound",
        "accounted_path_s": lengths[last] if last else 0,
        "activity_ids": path,
        "edges": [
            {"from": a, "to": b, "reason": reasons[a, b]}
            for a, b in zip(path, path[1:], strict=False)
        ],
        "limitations": "Observed exclusive orders explain serialization; fluid slowdown is in realized durations. Arrival, memory pressure and multi-slot causality may leave unexplained gaps; consult WAITING trace.",
    }
