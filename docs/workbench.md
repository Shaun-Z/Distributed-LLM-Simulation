# Heterosim experiment workbench

The workbench is an English, local, single-user application. It uses the same specification parser, lowerer, simulator, and result exporter as the CLI. It does not change simulation semantics.

## Start from source

Python 3.12+, uv, and Node.js 22.12+ (or Node.js 24) are required for development. Node is only needed to build the frontend.

```sh
uv sync --extra web --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run --extra web heterosim ui --runs-dir runs --port 8765
```

Open **http://127.0.0.1:8765**. The default listener is loopback. The application loads built frontend assets from the Python package, not a CDN. Fonts are bundled locally. To use additional hardware profiles, pass `--profiles-dir /path/to/profiles`.

For frontend hot reload, keep the Python server running and execute:

```sh
npm --prefix frontend run dev
```

Open the URL printed by Vite. Its `/api` proxy preserves the request host, so same-origin mutation checks work in development. Restart the Python service after changing backend files, or use Uvicorn's factory/reload mode for backend development.

## Appearance

Use the **Light theme / Dark theme** button in the header. Dark is the default; the browser remembers an explicit selection across page navigation and refresh. Theme changes also update React Flow, chart colors and the source editor without changing the experiment draft. Graph selection and viewport position are preserved.

DAGs use brighter, thicker edges and matching arrowheads, opaque high-contrast edge labels, distinct tensor/memory/transfer nodes, and a visible selection outline. Node labels wrap to two lines, and initial fitting avoids shrinking large graphs below a readable scale. Use **Readable view** to focus on the start of a large graph, **Fit all** for an overview, or pan and navigate with the minimap and zoom controls.

## Workflow

1. **New experiment**: choose a workload and compatible hardware profile. Scenario fields regenerate specs only when you click **Build scenario**. Rebuilding asks before replacing manual edits. The default templates use the same known-compatible device IDs as the CLI demos.
2. **Configure**: change resource capacities, channel bandwidth/setup/latency, scheduler, transfer model, seed, and run limits. Exact byte counts and supported unit strings are parsed by Python. The placement table supports operator filters, bulk device assignments, and explicit kernel selection.
3. **Advanced sources**: edit any of the four source documents as YAML or JSON. Import/export an experiment bundle, or load a file into the selected document. The bundle stores source text, retaining integers beyond JavaScript's exact range. Validate advanced edits before trusting the form or graph preview.
4. **Run**: a validated snapshot is submitted to one isolated worker. Additional jobs queue. The task list shows live job states; it does not claim to stream simulator progress. Completed artifacts unlock result charts.
5. **Explore**: inspect physical topology, logical/execution DAGs, startup/body timelines, resource curves, trace events, scoped bounds, model assumptions, and LLM metrics. Graphs show up to 400 matching nodes; filter large graphs. Timeline filtering and zoom are independent of the replay cursor. Playback replays recorded simulation time.
6. **Compare**: select two to four runs. Choose a baseline for percentage differences and a physical pool for memory overlays. Hardware/config hashes, model levels and seeds remain visible. A pool with the same name in two different profiles is not necessarily equivalent hardware.

Transformer template fields include TP stage/rank devices, pool mappings, model dimensions, prefill/decode sizes, microbatches, and an optional offload pool. For two-host TP use the `two_hosts` profile with stages `[["gpu0", "gpu1"]]` and pools `[["hbm0", "hbm1"]]`. For two-stage PP use `[["gpu0"], ["gpu1"]]` and `[["hbm0"], ["hbm1"]]`. The core validates divisibility, kernel support and routes.

## Storage and task lifecycle

Every new run gets an independent `ui-<job-id>` directory inside `--runs-dir`. Existing runs, including nested CLI sweep/oracle results, remain read-only. The catalog discovers `summary.json`, `manifest.json`, and `invalid.json`; missing optional files do not prevent browsing the overview. Inputs reconstruct graphs without requiring graph exports.

Queue metadata and submitted drafts are stored in `<runs-dir>/.ui/jobs/`. There is one active worker. Cancelled queued jobs never launch. A running cancellation sends terminate, waits up to five seconds, then kills if necessary. Shutdown terminates active workers; queued/running jobs become interrupted after restart and never restart automatically. Failed/cancelled/interrupted job configurations can be copied into a new draft. Cancelling cannot create a complete trace if the simulator had not exported one yet; partial files are retained.

Job states (`queued`, `running`, `finished`, `failed`, `cancelled`, `interrupted`) are separate from simulation outcomes (`SUCCESS`, `NO_PROGRESS`, `TRUNCATED`). A `finished` job can therefore contain an infeasible simulation with an auditable failure snapshot.

## Curve interpretation and limits

- Memory is reconstructed from aligned ALLOCATED/FREED events by allocation ID and canonical physical pool. Shared UMA views do not add capacity. Persistent weights, outputs, and KV may remain resident at run end.
- Service rates combine RATE_CHANGED/ACTIVITY_FINISHED events with execution demand coefficients. Fixed-kernel declared IO is included as normalized service; unmodeled physical IO is not inferred. Dashed lines show capacity, while the CSV table separately reports slot/gate reservations.
- Requests for a time window preserve the immediately preceding value. Downsampling retains bucket endpoints and extrema, including instantaneous memory peaks. Summary metrics are never computed from downsampled curves.
- A summary-only trace cannot reconstruct memory/rate curves. Activity-only traces lack dynamic rate changes. The UI labels unavailable curves rather than inventing zeros or extrapolating from aggregate utilization. Legacy runs missing required inputs or artifacts show a diagnostic.
- Raw trace pages contain at most 1,000 matching events per request (100 per UI page). Derived curves are lazily reconstructed from the saved trace and cached for up to eight runs; the first query still scans the source trace. Graph and timeline payloads are intended for prototype-scale workloads, not unbounded distributed telemetry.
- The frontend does not add fluid compute, live engine streaming, topology creation by dragging, authentication, collaborative editing, or launch controls for sweeps and oracles. Use the CLI for search and load the resulting run directories for analysis.

## Verification and packaging

```sh
uv run --extra web pytest -q
uv run --extra web ruff check src tests examples
uv run --extra web ruff format --check src tests examples
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
# With the local server running:
cd frontend
npx playwright install chromium
npm run test:e2e
```

Playwright exercises real execution, duplicate/import state, graph selection, trace filters, result comparison, Transformer token metrics, OOM diagnostics, worker cancellation and narrow-screen layout. It creates named browser-test runs in the server's results directory. Use a separate `--runs-dir` for acceptance testing. Screenshots and Playwright failure traces are test outputs; no experiment data is uploaded.

Build the frontend **before** `uv build`. The frontend build copies the canonical repository hardware profiles into package data; generated static/data folders are ignored by Git but included in wheel/sdist as Python package data. Release builds therefore contain both the UI and demo profiles. A packaged installation needs the `web` extra and no Node runtime:

```sh
uv pip install 'distributed_llm_simulation-0.1.0-py3-none-any.whl[web]'
heterosim ui --runs-dir ./runs
```

Without compiled assets, `heterosim ui` exposes an actionable build instruction instead of a blank page. Backend OpenAPI documentation is available at `/docs`; the typed contracts are versioned separately from core simulation specs. File access is limited to discovered run IDs and a fixed artifact allowlist inside the configured root; cross-origin mutations are rejected.
