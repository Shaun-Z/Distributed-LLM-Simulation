"""Public API for the heterogeneous distributed workload simulator."""

from .compiler import ActivitySpec, ExecutionPlan, Lowerer, ReplicaSpec
from .runtime import RunResult, SimulationConfig, Simulator
from .scheduling import CriticalRankScheduler, FIFOListScheduler, MemoryAwareScheduler
from .spec import (
    HardwareSpec,
    PlacementPlan,
    WorkloadSpec,
    load_hardware,
    load_placement,
    load_workload,
)

__version__ = "0.1.0"
__all__ = [
    "ActivitySpec",
    "ExecutionPlan",
    "Lowerer",
    "ReplicaSpec",
    "Simulator",
    "SimulationConfig",
    "RunResult",
    "FIFOListScheduler",
    "CriticalRankScheduler",
    "MemoryAwareScheduler",
    "HardwareSpec",
    "WorkloadSpec",
    "PlacementPlan",
    "load_hardware",
    "load_workload",
    "load_placement",
]


def main():
    from .cli import main as cli_main

    raise SystemExit(cli_main())
