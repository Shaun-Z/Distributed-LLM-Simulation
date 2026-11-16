"""Shared single-experiment entry point for CLI and Web workers."""

from pathlib import Path

from . import Lowerer, Simulator
from .io import export_graphs, write_json, write_results


def run_experiment(hardware, workload, placement, config, output, *, overwrite=False):
    execution = Lowerer(hardware).compile(workload, placement)
    result = Simulator(hardware, config=config).run(execution)
    write_results(
        result,
        output,
        hardware=hardware,
        workload=workload,
        placement=placement,
        config=config,
        execution=execution,
        overwrite=overwrite,
    )
    export_graphs(hardware, workload, output)
    if not config.capacity_processes:
        from .analysis.bounds import critical_path_bound, execution_bounds, gap

        bounds = [critical_path_bound(hardware, workload), *execution_bounds(hardware, execution)]
        report = {
            "bound_target": "finish time from simulation t=0, rather than request-window makespan",
            "bounds": [b.to_dict() for b in bounds],
        }
        if result.status == "SUCCESS":
            report["conditional_gap"] = gap(result.finish_time_s, max(b.value_s for b in bounds))
        write_json(Path(output) / "bounds.json", report)
    if any(r.markers for r in workload.requests):
        from .workloads.transformer import llm_metrics

        write_json(Path(output) / "llm_metrics.json", llm_metrics(result))
    return result
