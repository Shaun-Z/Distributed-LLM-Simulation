from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..models.network import CapacityProcess
from ..spec.base import Count, NonNegative, Spec


class SimulationConfig(Spec):
    network_mode: Literal["exclusive", "fluid"] = "exclusive"
    scheduler: Literal["fifo", "critical_rank", "memory_aware"] = "fifo"
    seed: int = 0
    activity_order: tuple[str, ...] = ()
    capacity_processes: tuple[CapacityProcess, ...] = ()
    max_sim_time: NonNegative = 1e9
    max_events: Count = 1000000
    max_same_time_transitions: Count = 100000
    check_invariants: bool = True
    trace_level: Literal["summary", "activities", "resource_changes", "debug"] = "resource_changes"


@dataclass
class ActivityState:
    remaining_work: float
    state: str = "WAITING"
    rate: float = 0
    generation: int = 0
    ready_s: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    served: float = 0


@dataclass(frozen=True)
class RunResult:
    status: str
    makespan_s: float | None
    finish_time_s: float
    metrics: dict
    resources: tuple[dict, ...]
    requests: tuple[dict, ...]
    trace: tuple[dict, ...]
    timeline: tuple[dict, ...]
    failure_reason: str | None = None
    failure_snapshot: dict | None = None
    assumptions: tuple[str, ...] = ()
