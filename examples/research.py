"""Run TP/PP and common-random-number scenarios; execute from repository root.

uv run python examples/research.py --output runs/research
"""

from __future__ import annotations

import argparse
from pathlib import Path

from distributed_llm_simulation import Lowerer, SimulationConfig, Simulator, load_hardware
from distributed_llm_simulation.analysis.metrics import replication_statistics
from distributed_llm_simulation.io import write_json, write_results
from distributed_llm_simulation.models.network import markov_capacity
from distributed_llm_simulation.workloads import (
    TransformerConfig,
    TransformerWorkloadBuilder,
    llm_metrics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/research")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error("choose a new output directory")
    h = load_hardware("profiles/hardware/two_hosts.yaml")
    builder = TransformerWorkloadBuilder(TransformerConfig(microbatches=2, decode_steps=2))
    variants = {
        "tp": builder.build(stages=(("gpu0", "gpu1"),), pools=(("hbm0", "hbm1"),)),
        "pp": builder.build(stages=(("gpu0",), ("gpu1",)), pools=(("hbm0",), ("hbm1",))),
    }
    results = {name: [] for name in variants}
    for seed in (10, 20, 30):
        # Each strategy sees the same external capacity at each simulation time.
        environment = tuple(
            markov_capacity(
                rid,
                seed=seed,
                horizon_s=0.02,
                capacities=(12.5e9, 1e9, 0),
                transition_rates=((0, 500, 50), (500, 0, 50), (500, 500, 0)),
            )
            for rid in ("wan_01", "wan_10")
        )
        config = SimulationConfig(network_mode="fluid", seed=seed, capacity_processes=environment)
        for name, (w, p) in variants.items():
            execution = Lowerer(h).compile(w, p)
            result = Simulator(h, config=config).run(execution)
            output = root / f"{name}-seed{seed}"
            write_results(
                result,
                output,
                hardware=h,
                workload=w,
                placement=p,
                config=config,
                execution=execution,
            )
            write_json(output / "llm_metrics.json", llm_metrics(result))
            results[name].append(result)
    write_json(
        root / "replications.json",
        {name: replication_statistics(runs) for name, runs in results.items()},
    )
    print(root / "replications.json")


if __name__ == "__main__":
    main()
