"""Command-line experiments with complete, replayable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import Lowerer, Simulator, __version__, load_hardware, load_placement, load_workload
from .io import export_graphs, write_json, write_results
from .runtime import SimulationConfig
from .spec.schema import load, save_spec


def _inputs(args):
    return (
        load_hardware(args.hardware),
        load_workload(args.workload),
        load_placement(args.placement),
    )


def _config(args):
    config = (
        load(args.config, SimulationConfig) if getattr(args, "config", None) else SimulationConfig()
    )
    overrides = {
        k: getattr(args, k)
        for k in ("network_mode", "scheduler", "seed")
        if getattr(args, k, None) is not None
    }
    return SimulationConfig.model_validate({**config.model_dump(), **overrides})


def _run(hardware, workload, placement, config, args):
    from .application import run_experiment

    result = run_experiment(
        hardware, workload, placement, config, args.output, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "makespan_s": result.makespan_s,
                "output": str(Path(args.output)),
                "failure_reason": result.failure_reason,
            }
        )
    )
    return 0 if result.status == "SUCCESS" else 2


def _write_evaluation(evaluation, output, workload, config):
    if evaluation.result is not None:
        write_results(
            evaluation.result,
            output,
            hardware=evaluation.hardware,
            workload=workload,
            placement=evaluation.placement,
            config=config,
            execution=evaluation.execution,
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "invalid.json",
            {"status": "INVALID", "reason": evaluation.error, "key": evaluation.key},
        )
        for name, spec in (
            ("hardware", evaluation.hardware),
            ("workload", workload),
            ("placement", evaluation.placement),
            ("config", config),
        ):
            save_spec(spec, output / f"{name}.json")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="heterosim", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate and hash hardware/workload/placement")
    validate.add_argument("--hardware", required=True)
    validate.add_argument("--workload")
    validate.add_argument("--placement")
    validate.add_argument(
        "--export", help="directory for normalized inputs and readable graph JSON"
    )
    run = commands.add_parser("run", help="run a fixed placement")
    for key in ("hardware", "workload", "placement"):
        run.add_argument(f"--{key}", required=True)
    demo = commands.add_parser("demo", help="run a bundled synthetic workload")
    demo.add_argument(
        "scenario", choices=("pipeline", "fork_join", "image", "matmul", "transformer")
    )
    demo.add_argument("--hardware")
    demo.add_argument("--requests", type=int, default=2)
    demo.add_argument("--offload-pool")
    for p in (run, demo):
        p.add_argument("--config")
        p.add_argument("--network-mode", choices=("exclusive", "fluid"))
        p.add_argument("--scheduler", choices=("fifo", "critical_rank", "memory_aware"))
        p.add_argument("--seed", type=int)
        p.add_argument("--output", required=True)
        p.add_argument("--overwrite", action="store_true")
    inspect = commands.add_parser("inspect", help="read an exported run summary and timeline")
    inspect.add_argument("directory")
    inspect.add_argument("--timeline", action="store_true")
    replay = commands.add_parser("replay", help="rerun saved inputs and compare semantic outputs")
    replay.add_argument("directory")
    sweep = commands.add_parser("sweep", help="enumerate placement or explicit storage budgets")
    for key in ("hardware", "workload"):
        sweep.add_argument(f"--{key}", required=True)
    sweep.add_argument("--placement")
    sweep.add_argument("--budgets", help="JSON list of capacity resource -> byte-budget mappings")
    sweep.add_argument("--max-candidates", type=int, default=100)
    sweep.add_argument("--config")
    sweep.add_argument("--output", required=True)
    oracle = commands.add_parser("oracle", help="enumerate tiny fixed-plan A-level start schedules")
    for key in ("hardware", "workload", "placement"):
        oracle.add_argument(f"--{key}", required=True)
    oracle.add_argument("--max-orders", type=int, default=10000)
    oracle.add_argument("--output", required=True)
    ui = commands.add_parser("ui", help="serve the local experiment workbench")
    ui.add_argument("--runs-dir", default="runs")
    ui.add_argument("--profiles-dir")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "ui":
            try:
                import uvicorn

                from .web.app import create_app
            except ImportError as error:
                raise ValueError("Install Web dependencies with: uv sync --extra web") from error
            uvicorn.run(
                create_app(args.runs_dir, args.profiles_dir), host=args.host, port=args.port
            )
            return 0
        if args.command == "validate":
            h = load_hardware(args.hardware)
            result = {
                "hardware_hash": h.digest(),
                "physical_pools": sorted({h.pool(n.id) for n in h.nodes if n.kind == "memory"}),
            }
            if args.workload:
                w = load_workload(args.workload)
                result["workload_hash"] = w.digest()
                if args.placement:
                    e = Lowerer(h).compile(w, load_placement(args.placement))
                    result["execution_hash"] = e.digest()
                if args.export:
                    export_graphs(h, w, args.export)
                    save_spec(w, Path(args.export) / "workload.json")
                    save_spec(h, Path(args.export) / "hardware.json")
            elif args.placement or args.export:
                parser.error("--placement/--export require --workload")
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "run":
            return _run(*_inputs(args), _config(args), args)
        if args.command == "demo":
            from .workloads import SyntheticWorkload, TransformerWorkloadBuilder, image_pipeline

            default = (
                "pipeline_oracle"
                if args.scenario == "pipeline"
                else "uma_workloads"
                if args.scenario == "fork_join"
                else "discrete_workloads"
            )
            h = load_hardware(args.hardware or f"profiles/hardware/{default}.yaml")
            builders = {
                "pipeline": lambda: SyntheticWorkload.pipeline(requests=args.requests),
                "fork_join": lambda: SyntheticWorkload.fork_join(
                    left="uc0", right="ug0", pool="uma0"
                ),
                "image": lambda: image_pipeline(images=args.requests),
                "matmul": SyntheticWorkload.tiled_matmul,
                "transformer": lambda: TransformerWorkloadBuilder().build(
                    offload_pool=args.offload_pool
                ),
            }
            w, p = builders[args.scenario]()
            return _run(h, w, p, _config(args), args)
        if args.command == "inspect":
            d = Path(args.directory)
            print((d / "summary.json").read_text())
            if args.timeline:
                for row in json.loads((d / "timeline.json").read_text()):
                    print(
                        f"{row['activity_id']:24s} {str(row['start_s']):>14s} -> {str(row['finish_s']):>14s} {row['op_id'] or row['tensor_id'] or ''}"
                    )
            return 0
        if args.command == "replay":
            d = Path(args.directory)
            h, w, p = (
                load_hardware(d / "hardware.resolved.yaml"),
                load_workload(d / "workload.json"),
                load_placement(d / "placement.json"),
            )
            c = load(d / "config.json", SimulationConfig)
            execution = Lowerer(h).compile(w, p)
            result = Simulator(h, config=c).run(execution)
            expected = json.loads((d / "summary.json").read_text())
            semantic = {
                "status": result.status,
                "makespan_s": result.makespan_s,
                "finish_time_s": result.finish_time_s,
                "failure_reason": result.failure_reason,
                "metrics": result.metrics,
                "assumptions": list(result.assumptions),
            }
            from .io.results import json_safe

            expected_trace = [
                json.loads(line) for line in (d / "trace.jsonl").read_text().splitlines()
            ]
            manifest = json.loads((d / "manifest.json").read_text())
            hashes = {
                "hardware": h.digest(),
                "workload": w.digest(),
                "placement": p.digest(),
                "config": c.digest(),
                "execution": execution.digest(),
            }
            hashes_match = hashes == manifest["hashes"]
            matches = (
                hashes_match and semantic == expected and json_safe(result.trace) == expected_trace
            )
            print(
                json.dumps(
                    {
                        "replay_matches": matches,
                        "input_hashes_match": hashes_match,
                        "status": result.status,
                    }
                )
            )
            return 0 if matches else 2
        if args.command == "oracle":
            from .analysis.search import exact_schedule_oracle

            h, w, p = _inputs(args)
            output = Path(args.output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"nonempty oracle directory: {output}")
            e = Lowerer(h).compile(w, p)
            report = exact_schedule_oracle(h, e, max_orders=args.max_orders)
            best = report.pop("best")
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "oracle.json", report)
            if best:
                config = SimulationConfig(activity_order=report["best_order"])
                write_results(
                    best,
                    output / "best",
                    hardware=h,
                    workload=w,
                    placement=p,
                    config=config,
                    execution=e,
                )
            print(
                json.dumps(
                    {
                        "output": str(output),
                        "proven_optimal_in_scope": report["proven_optimal_in_scope"],
                        "incumbent_s": report["incumbent_s"],
                    }
                )
            )
            return 0
        if args.command == "sweep":
            from .analysis.search import enumerate_placements, memory_sweep, pareto_frontier

            h, w, config = load_hardware(args.hardware), load_workload(args.workload), _config(args)
            output = Path(args.output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"nonempty sweep directory: {output}")
            output.mkdir(parents=True, exist_ok=True)
            if args.budgets:
                if not args.placement:
                    parser.error("--budgets requires --placement")
                values = memory_sweep(
                    h,
                    w,
                    load_placement(args.placement),
                    json.loads(Path(args.budgets).read_text()),
                    config,
                )
                rows = []
                for i, (budget, e) in enumerate(values):
                    _write_evaluation(e, output / f"point-{i:04d}", w, config)
                    rows.append(
                        {
                            "budget": budget,
                            "status": e.result.status if e.result else "INVALID",
                            "error": e.error,
                            "makespan_s": e.objective,
                            "peak_memory_bytes": e.result.metrics["peak_memory_bytes"]
                            if e.result
                            else None,
                            "placement": e.placement.model_dump(),
                        }
                    )
                write_json(
                    output / "sweep.json",
                    {
                        "scope": "fixed placement, configured heuristic scheduler",
                        "evaluations": rows,
                    },
                )
            else:
                report = enumerate_placements(
                    h, w, max_candidates=args.max_candidates, config=config
                )
                values = report.pop("evaluations")
                best = report.pop("best")
                for i, evaluation in enumerate(values):
                    _write_evaluation(evaluation, output / "candidates" / f"{i:04d}", w, config)
                report["evaluations"] = [
                    {
                        "key": e.key,
                        "makespan_s": e.objective,
                        "error": e.error,
                        "placement": e.placement.model_dump(),
                        "peak_memory_bytes": e.result.metrics["peak_memory_bytes"]
                        if e.result
                        else None,
                    }
                    for e in values
                ]
                report["empirical_frontier_keys"] = [e.key for e in pareto_frontier(values)]
                write_json(output / "sweep.json", report)
                if best and best.result and best.result.status == "SUCCESS":
                    e = Lowerer(h).compile(w, best.placement)
                    write_results(
                        best.result,
                        output / "best",
                        hardware=h,
                        workload=w,
                        placement=best.placement,
                        config=config,
                        execution=e,
                    )
            print(json.dumps({"output": str(output), "evaluated": len(values)}))
            return 0
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"status": "INVALID_INPUT", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
