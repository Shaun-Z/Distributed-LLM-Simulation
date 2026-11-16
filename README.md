# Distributed LLM Simulation

English | [简体中文](README_CN.md)

A research prototype for heterogeneous distributed computing, implemented from the [engineering specification](docs/heterogeneous-distributed-simulator-design.md) (in Chinese). The core simulates general computation through a **hardware resource graph, workload DAG, placement, scheduler, and discrete-event engine**. A separate plugin expands LLM workloads. Requires Python 3.12+. Dependencies and development tools are managed with `uv`, with exact versions recorded in `uv.lock`.

## Installation and demos

Run these commands from the repository root:

```sh
uv sync --locked
uv run heterosim demo pipeline --requests 2 --output runs/pipeline
uv run heterosim inspect runs/pipeline --timeline
uv run heterosim replay runs/pipeline
```

The pipeline completes in **6.1 ms** for one request and **9.1 ms** for two requests, matching Section 14 of the design document. For repeated runs, choose a new output directory or explicitly pass `--overwrite`.

```sh
uv run heterosim demo fork_join --output runs/uma
uv run heterosim demo image --output runs/image
uv run heterosim demo matmul --output runs/matmul
uv run heterosim demo transformer --output runs/transformer
uv run heterosim demo transformer --offload-pool ram0 --output runs/offload
uv run python examples/research.py --output runs/tp-pp
```

`distributed-llm-simulation` and `python -m distributed_llm_simulation` are also available as entry points. `demo` uses the repository's `profiles/` directory. After installing the wheel, use `run` to supply your own configuration files.

## Visual workbench

An English browser workbench supports experiment configuration, isolated execution and cancellation, hardware/workload/execution graphs, replayable timelines, physical memory and service curves, trace diagnostics, and comparison of up to four runs.

```sh
uv sync --extra web --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --extra web heterosim ui --runs-dir runs --port 8765
```

Open **http://127.0.0.1:8765**. Start with a real built-in scenario, edit forms or YAML/JSON sources, validate, then run. Existing CLI results are discovered without modification. Complete traces are read after execution; task status updates are live. Missing trace data is labeled unavailable. Node is needed only for development/building; release wheels bundle the UI and demo profiles.

See the [workbench guide](docs/workbench.md) for TP/PP examples, development, curve interpretation, tests, and packaging.

## Configuration and CLI

```sh
uv run heterosim validate --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --export runs/validated
uv run heterosim run --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --network-mode fluid --scheduler memory_aware --output runs/custom
uv run heterosim run --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --config examples/environment.json --output runs/varying
uv run heterosim sweep --hardware profiles/hardware/uma_workloads.yaml \
  --workload examples/fork_join/workload.json --max-candidates 100 --output runs/search
uv run heterosim oracle --hardware profiles/hardware/pipeline_oracle.yaml \
  --workload examples/pipeline/workload.json --placement examples/pipeline/placement.json \
  --max-orders 10000 --output runs/oracle
```

`validate` rejects duplicate IDs or keys, invalid units, noninteger capacities, NaN/Inf, discontinuous or illegal routes, DAG cycles, inconsistent data ports, and capability mismatches. Inputs support YAML and JSON. Mappings can be dictionaries or lists of key-value pairs and are normalized into immutable tuples. Storage capacities and slot counts use exact integers. Quantities such as `32 GiB`, `100 Gbit/s`, and `0.1 ms` are explicitly normalized; ambiguous values such as `100G` are rejected. Default fields appear in the resolved configuration, and hashes are stable for the same normalized representation.

The exit code is 0 on success and 2 for invalid input, lack of progress, or truncation. Failed simulations still save diagnostic snapshots. Schema and lowering errors are rejected before entering the engine.

## Python API

```python
from distributed_llm_simulation import (
    Lowerer, SimulationConfig, Simulator, load_hardware,
)
from distributed_llm_simulation.workloads import SyntheticWorkload

hardware = load_hardware("profiles/hardware/pipeline_oracle.yaml")
workload, placement = SyntheticWorkload.pipeline(requests=2)
execution = Lowerer(hardware).compile(workload, placement)
config = SimulationConfig(network_mode="exclusive", scheduler="fifo")
result = Simulator(hardware, config=config).run(execution)
assert result.status == "SUCCESS"
assert abs(result.makespan_s - 0.0091) < 1e-12
```

`PlacementPlan.op_compute` assigns a device to each operator; `op_kernel` can fix the kernel implementation. `tensor_target_pool` specifies the target pool for a producer's output or an accessible input. `transfer_routes` fixes the route for a tensor to a destination pool, and `final_output_pools` defines final output returns. `reload_tensor_ids` reloads offloaded tensors for each use and releases temporary device replicas. Reloads respect current input and control dependencies.

Kernel matching checks the operation, all three dtypes, optional layout and algorithm constraints, and shape ranges. The roofline model sums reads and writes that use the same physical controller. Calibrated durations do not charge local IO time again. `duration_s` overrides the fixed duration; declared IO must fit within the resource capacity available during that duration.

## Execution semantics

- Static specifications are immutable, and each run creates independent state. The hardware graph preserves parallel directed channels, and routes store channel IDs. Shared resource budgets are independent of channels.
- Multiple UMA views resolve to one physical pool. Outputs and workspace are reserved atomically. Only VALID replicas can be read, and inputs stay pinned until computation or copying finishes. Fan-out consumers reuse the same destination replica, including a transfer already in progress.
- Only legal predefined routes are used. Without a P2P/RDMA route, transfers can stage through RAM across multiple routes. Each hop allocates storage, transfers data, and publishes a valid replica separately. Missing routes cause an explicit failure.
- Level A uses exclusive bandwidth gates. Level B applies weighted max-min fairness in byte rates to transfers only; kernels retain fixed durations and exclusively reserve their declared memory bandwidth groups. Reservations include startup, making this a conservative model.
- Level C replays piecewise capacity traces. Level D uses Markov scenarios generated and saved in advance. The engine integrates old rates and processes completions before applying changes at the same timestamp. Generation counters invalidate old predictions. Zero bandwidth waits for a future recovery; a permanent stall returns a diagnostic snapshot.
- The IO budgets of fixed-duration kernels cannot vary during execution; such configurations are rejected. Fluid streaming compute kernels are not implemented.

## Results and research tools

Each run saves resolved hardware, workload, placement, execution, configuration, a manifest, a JSON summary, resource and request CSV files, an activity timeline, a JSONL trace, and exogenous environment traces. The manifest records input hashes, seeds, versions, model assumptions, and the source revision and dirty state. Failures also save snapshots of allocations, pins, slots, waiting reasons, and activities with zero service rates.

`replay` reruns the simulation and compares the summary and trace. Metrics include nearest-rank request latency quantiles, throughput, per-pool peak memory and byte-seconds, integrated resource service, slot and gate reservation utilization, and the union and overlap of compute and transfer intervals. `critical_path.json` explains elapsed time using actual dependencies and observed exclusive resource ordering. Memory contention and contention across multiple slots still require inspecting WAITING records.

`analysis.search` provides a cached evaluator, placement enumeration, greedy planning based on upward ranks, memory budget sweeps, and empirical Pareto sets. Cache keys include the full specifications, configuration, scenario, seed, and version. The best feasible heuristic result from enumeration is an upper bound. `oracle` enumerates topological start orders for a fixed Level A execution plan and allows deliberate idle time. It reports proven optimality within that scope only when enumeration finishes without truncation. The proof fixes placement, kernels, routes, capacities, lifetimes, and transformations; it does not establish a global limit for heterogeneous hardware.

`analysis.bounds` provides global compute dependency lower bounds, conditional dependency and resource-load lower bounds for a fixed plan, and capacity integral bounds. Each includes scope, assumptions, and fixed decisions. Bounds are combined with `max`, and gaps are reported. The module does not automatically apply a general IO lower bound or report unmodeled power and monetary costs.

## Workloads and model boundaries

General plugins include fork-join, an image pipeline, and tiled matrix multiplication. The Transformer plugin expands normalization, projection, attention, residual, and MLP operations. It supports GQA, finite prefill and decode sequences, append-only KV storage, TP ring collectives, PP microbatches, and explicit offload reloads. The collective module also provides reduce-scatter, all-gather, and broadcast. Message routes, local reduction and packing IO, and buffer occupancy all enter the generic graph. KV storage is retained until the run ends. The LLM module derives TTFT, ITL, TPOT, and token throughput from generic markers.

All bundled hardware profiles and measurement data are **synthetic**. Calibration interfaces support provenance, linear fitting for copies, and independent held-out validation; real device measurements are not yet included. Research extensions such as dynamic continuous batching, paged or shared-prefix KV storage, automatic eviction and recomputation, fluid compute, failure recovery, and general IO theorems are not implemented. See [implementation and acceptance status](docs/implementation-status.md) (in Chinese).

## Development checks

```sh
uv run pytest -q
uv run ruff check src tests examples
uv run ruff format --check src tests examples
uv build
```

Tests cover analytical oracles, conservation, atomic rollback on OOM, aliasing and lifetimes, staging, replay, plugins, search, and upper/lower bound labels. Passing tests validates the implementation of the declared model; it does not establish predictive accuracy on real hardware.
