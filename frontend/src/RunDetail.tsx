import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  Copy,
  Download,
  Play,
  Pause,
  SkipBack,
  Search,
  ArrowLeft,
} from "lucide-react";
import { api } from "./api";
import type {
  Detail,
  Draft,
  Series,
  Specs,
  TimelineRow,
  TracePage,
} from "./types";
import {
  Badge,
  Card,
  Empty,
  ErrorBox,
  JsonView,
  Loading,
  Metric,
  PageTitle,
  Status,
} from "./ui";
import { bytes, duration, number, percent } from "./format";
import { Graphs } from "./Graphs";
import { ResourceChart, TimelineChart } from "./Charts";

export function RunDetail() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [tab, setTab] = useState("overview"),
    [selected, setSelected] = useState(""),
    [search, setSearch] = useState(""),
    [request, setRequest] = useState(""),
    [start, setStart] = useState(""),
    [end, setEnd] = useState(""),
    [cursor, setCursor] = useState(0),
    [playing, setPlaying] = useState(false),
    [pool, setPool] = useState(""),
    [resource, setResource] = useState(""),
    [error, setError] = useState<unknown>(null);
  const detail = useQuery({
    queryKey: ["run", id],
    queryFn: () => api<Detail>("/runs/" + id),
  });
  const specs = useQuery({
    queryKey: ["specs", id],
    queryFn: () => api<Specs>("/runs/" + id + "/specs"),
  });
  const query = new URLSearchParams();
  if (search) query.set("search", search);
  if (request) query.set("request", request);
  if (start !== "") query.set("start", start);
  if (end !== "") query.set("end", end);
  const timeline = useQuery({
    queryKey: ["timeline", id, query.toString()],
    queryFn: () => api<TimelineRow[]>("/runs/" + id + "/timeline?" + query),
    enabled: tab === "timeline",
  });
  const range = new URLSearchParams();
  if (start !== "") range.set("start", start);
  if (end !== "") range.set("end", end);
  const series = useQuery({
    queryKey: ["series", id, range.toString()],
    queryFn: () => api<Series>("/runs/" + id + "/series?" + range),
    enabled: tab === "resources",
  });
  const finish = detail.data?.summary.finish_time_s ?? 0;
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(
      () =>
        setCursor((t) => {
          const next = Math.min(finish, t + finish / 120);
          if (next >= finish) setPlaying(false);
          return next;
        }),
      80,
    );
    return () => clearInterval(timer);
  }, [playing, finish]);
  useEffect(() => {
    setTab("overview");
    setSelected("");
    setSearch("");
    setStart("");
    setEnd("");
    setCursor(0);
    setPlaying(false);
  }, [id]);
  async function copy() {
    try {
      const d = await api<Draft>("/runs/" + id + "/draft");
      sessionStorage.setItem("copy-draft", JSON.stringify(d));
      navigate("/new?copy=1");
    } catch (e) {
      setError(e);
    }
  }
  function selectActivity(identifier: string) {
    const replicas = new Set(
      (specs.data?.execution?.replicas ?? [])
        .filter((r) => r.tensor_id === identifier)
        .map((r) => r.id),
    );
    const a =
      specs.data?.execution?.activities.find(
        (a) =>
          a.id === identifier ||
          a.op_id === identifier ||
          a.tensor_id === identifier,
      ) ??
      specs.data?.execution?.activities.find((a) =>
        [...a.input_replica_ids, ...a.output_replica_ids].some((id) =>
          replicas.has(id),
        ),
      );
    setSelected(a?.id ?? identifier);
    setSearch(a?.id ?? identifier);
    setTab("timeline");
  }
  if (detail.isPending) return <Loading />;
  if (detail.error) return <ErrorBox error={detail.error} />;
  const d = detail.data!;
  const metrics = d.summary.metrics ?? {};
  return (
    <>
      <button className="back-button" onClick={() => navigate("/")}>
        <ArrowLeft size={14} /> Experiments
      </button>
      <PageTitle
        eyebrow={"Run / " + id.slice(0, 10)}
        title={specs.data?.workload.name.replaceAll("_", " ") ?? d.directory}
        description={d.directory}
        actions={
          <>
            <Status value={d.summary.status ?? "INCOMPLETE"} />
            <button onClick={() => void copy()}>
              <Copy size={15} /> Duplicate configuration
            </button>
          </>
        }
      />
      <div className="run-meta">
        <Badge tone="active">
          Model {String(d.manifest.model_level ?? "N/A")}
        </Badge>
        <Badge>{specs.data?.config.scheduler ?? "N/A"}</Badge>
        <Badge>{specs.data?.config.network_mode ?? "N/A"} network</Badge>
        <Badge tone={d.manifest.synthetic_hardware ? "warning" : "neutral"}>
          {d.manifest.synthetic_hardware
            ? "Synthetic hardware"
            : "Hardware profile"}
        </Badge>
        <span>
          Seed <code>{String(d.manifest.seed ?? "N/A")}</code>
        </span>
      </div>
      <ErrorBox error={error} />
      <div className="tabs">
        {[
          "overview",
          "graphs",
          "timeline",
          "resources",
          "trace",
          "configuration",
        ].map((t) => (
          <button
            key={t}
            className={tab === t ? "active" : ""}
            onClick={() => setTab(t)}
          >
            {t === "trace"
              ? "Trace & diagnostics"
              : t[0].toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      {tab === "overview" && (
        <>
          <div className="metrics four">
            <Metric
              label="MAKESPAN"
              value={duration(d.summary.makespan_s)}
              foot={"Finish from t=0: " + duration(d.summary.finish_time_s)}
              tone="teal"
            />
            <Metric
              label="THROUGHPUT"
              value={number(metrics.throughput_requests_per_s)}
              foot="requests / second"
            />
            <Metric
              label="P95 REQUEST LATENCY"
              value={duration(metrics.p95_latency_s)}
              foot="Nearest-rank · inspect sample count"
            />
            <Metric
              label="DATA TRANSFERRED"
              value={bytes(metrics.bytes_transferred)}
              foot="Modeled wire payload across all copies"
            />
          </div>
          {(d.summary.failure_reason || d.failure || d.invalid) && (
            <Card
              title="Execution diagnostics"
              subtitle={
                d.summary.failure_reason ?? "Input or execution failed."
              }
            >
              <JsonView value={d.failure ?? d.invalid} />
              <button onClick={() => setTab("trace")}>
                Inspect trace and waiting reasons →
              </button>
            </Card>
          )}
          <div className="builder-grid">
            <Card
              title="Memory footprint"
              subtitle="One budget for each physical pool."
            >
              {Object.entries(metrics.peak_memory_bytes ?? {}).map(
                ([key, value]) => (
                  <div className="stat-line" key={key}>
                    <span className="mono">{key}</span>
                    <strong>{bytes(value)}</strong>
                  </div>
                ),
              )}
              {!Object.keys(metrics.peak_memory_bytes ?? {}).length && (
                <Empty title="No memory metrics" />
              )}
            </Card>
            <Card
              title="Request performance"
              subtitle="Finite-batch metrics; sample counts and unfinished requests matter."
            >
              {[
                ["Completed", number(metrics.completed_requests)],
                ["Unfinished", number(metrics.unfinished_requests)],
                ["Latency samples", number(metrics.latency_samples)],
                ["P50 latency", duration(metrics.p50_latency_s)],
                ["P99 latency", duration(metrics.p99_latency_s)],
                [
                  "Compute / transfer overlap",
                  duration(metrics.compute_transfer_overlap_s),
                ],
              ].map(([key, value]) => (
                <div className="stat-line" key={key}>
                  <span>{key}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </Card>
          </div>
          {d.llm_metrics && (
            <Card
              title="Decoder performance"
              subtitle="Token delivery markers, interpreted by the LLM plugin."
            >
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>REQUEST</th>
                      <th>TTFT</th>
                      <th>TPOT</th>
                      <th>TOKENS DELIVERED</th>
                      <th>INTER-TOKEN LATENCIES</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.llm_metrics.requests.map((r) => (
                      <tr key={r.request_id}>
                        <td>{r.request_id}</td>
                        <td>{duration(r.ttft_s)}</td>
                        <td>{duration(r.tpot_s)}</td>
                        <td>{r.delivered_tokens}</td>
                        <td>{r.itl_s.map(duration).join(" · ") || "N/A"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint">
                Output token throughput:{" "}
                {number(d.llm_metrics.output_tokens_per_s)} tokens / s
              </p>
            </Card>
          )}
          <Card
            title="Model assumptions"
            subtitle="A simulation result is evidence within its declared model."
          >
            <ul className="assumptions">
              {(d.summary.assumptions ?? []).map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
            {d.critical_path && (
              <details>
                <summary>
                  Critical path explanation ·{" "}
                  {duration(d.critical_path.accounted_path_s)}
                </summary>
                <p className="hint">{d.critical_path.limitations}</p>
                <JsonView value={d.critical_path} />
              </details>
            )}
            {d.bounds != null && (
              <details>
                <summary>Scoped lower bounds and gaps</summary>
                <JsonView value={d.bounds} />
              </details>
            )}
          </Card>
        </>
      )}
      {tab === "graphs" &&
        (specs.isPending ? (
          <Loading />
        ) : specs.data ? (
          <Card
            title="Resource and dependency graphs"
            subtitle="Select a node to inspect its identity, costs, and relationships."
          >
            <Graphs specs={specs.data} onActivity={selectActivity} />
          </Card>
        ) : (
          <ErrorBox error={specs.error} />
        ))}
      {tab === "timeline" && (
        <Card
          title="Execution timeline"
          subtitle="A replay of recorded simulation time, not live execution progress."
          actions={
            <div className="legend-inline">
              <span className="dot teal" /> Compute{" "}
              <span className="dot purple" /> Transfer{" "}
              <span className="dot amber" /> Startup / critical path
            </div>
          }
        >
          <div className="toolbar">
            <div className="search">
              <Search size={15} />
              <input
                aria-label="Filter activities"
                placeholder="Find an activity, operator, tensor…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
            <select
              aria-label="Request filter"
              value={request}
              onChange={(e) => setRequest(e.target.value)}
            >
              <option value="">All requests</option>
              {specs.data?.workload.requests.map((r) => (
                <option key={r.id}>{r.id}</option>
              ))}
            </select>
            <TimeRange
              start={start}
              end={end}
              setStart={setStart}
              setEnd={setEnd}
            />
            <button
              onClick={() => {
                setSearch("");
                setSelected("");
                setRequest("");
                setStart("");
                setEnd("");
              }}
            >
              Reset
            </button>
          </div>
          <div className="replay-controls">
            <button
              className="icon-button"
              aria-label="Reset replay"
              onClick={() => {
                setCursor(0);
                setPlaying(false);
              }}
            >
              <SkipBack size={15} />
            </button>
            <button
              className="icon-button"
              aria-label={playing ? "Pause replay" : "Play replay"}
              onClick={() => {
                if (cursor >= finish) setCursor(0);
                setPlaying((p) => !p);
              }}
            >
              {playing ? <Pause size={15} /> : <Play size={15} />}
            </button>
            <input
              aria-label="Replay simulation time"
              type="range"
              min={0}
              max={finish || 1}
              step={(finish || 1) / 1000}
              value={cursor}
              onChange={(e) => setCursor(Number(e.target.value))}
            />
            <code>{duration(cursor)}</code>
            <span className="muted">/ {duration(finish)}</span>
          </div>
          <ErrorBox error={timeline.error ?? specs.error} />
          {timeline.isPending || specs.isPending ? (
            <Loading />
          ) : (
            specs.data &&
            timeline.data && (
              <TimelineChart
                rows={timeline.data}
                specs={specs.data}
                selected={selected}
                onSelect={setSelected}
                critical={d.critical_path?.activity_ids ?? []}
                cursor={cursor}
              />
            )
          )}
          {selected && (
            <div className="selection-details">
              <div className="toolbar">
                <strong className="mono">{selected}</strong>
                <button onClick={() => setTab("trace")}>
                  Related trace events →
                </button>
                <button onClick={() => setSelected("")}>Clear selection</button>
              </div>
              <JsonView
                value={{
                  activity: specs.data?.execution?.activities.find(
                    (a) => a.id === selected,
                  ),
                  timing: timeline.data?.find(
                    (a) => a.activity_id === selected,
                  ),
                }}
              />
            </div>
          )}
        </Card>
      )}
      {tab === "resources" && (
        <>
          <Card
            title="Memory residency"
            subtitle="Aligned allocation bytes, deduplicated by physical pool."
          >
            <div className="toolbar">
              <select
                aria-label="Memory pool"
                value={pool}
                onChange={(e) => setPool(e.target.value)}
              >
                <option value="">All physical pools</option>
                {series.data?.series
                  .filter((s) => s.kind === "memory")
                  .map((s) => (
                    <option key={s.id}>{s.id}</option>
                  ))}
              </select>
              <TimeRange
                start={start}
                end={end}
                setStart={setStart}
                setEnd={setEnd}
              />
            </div>
            <ErrorBox error={series.error} />
            {series.isPending ? (
              <Loading />
            ) : (
              series.data && (
                <ResourceChart
                  series={series.data}
                  kind="memory"
                  resource={pool || undefined}
                />
              )
            )}
          </Card>
          <Card
            title="Service and capacity"
            subtitle="Solid: declared service rate. Dashed: available capacity. Reservations are reported separately."
          >
            <div className="toolbar">
              <select
                aria-label="Service resource"
                value={resource}
                onChange={(e) => setResource(e.target.value)}
              >
                <option value="">Active bandwidth resources</option>
                {series.data?.series
                  .filter((s) => s.kind === "service")
                  .map((s) => (
                    <option key={s.id}>{s.id}</option>
                  ))}
              </select>
              {series.data?.downsampled && (
                <Badge tone="warning">Peak-preserving downsampling</Badge>
              )}
            </div>
            {series.isPending ? (
              <Loading />
            ) : (
              series.data && (
                <ResourceChart
                  series={series.data}
                  kind="service"
                  resource={resource || undefined}
                />
              )
            )}
          </Card>
          <Card
            title="Integrated utilization"
            subtitle="Unchanged values from resources.csv; zero-capacity denominators remain N/A."
          >
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>RESOURCE</th>
                    <th>DECLARED SERVICE</th>
                    <th>SERVICE UTILIZATION</th>
                    <th>SLOT RESERVATION</th>
                    <th>EXCLUSIVE GATE</th>
                  </tr>
                </thead>
                <tbody>
                  {d.resources.map((r) => (
                    <tr key={r.resource_id}>
                      <td className="mono">{r.resource_id}</td>
                      <td>
                        {r.unit === "byte/s"
                          ? bytes(r.service_integral)
                          : number(r.service_integral)}
                      </td>
                      <td>{percent(r.service_utilization)}</td>
                      <td>{percent(r.slot_utilization)}</td>
                      <td>{percent(r.exclusive_gate_fraction)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
      {tab === "trace" && (
        <TracePanel
          id={id}
          selected={selected}
          onSelect={setSelected}
          detail={d}
        />
      )}
      {tab === "configuration" && (
        <>
          <Card
            title="Run provenance"
            subtitle="Resolved inputs, versions, hashes, and assumptions."
          >
            <JsonView value={d.manifest} />
          </Card>
          <Card
            title="Resolved specifications"
            actions={
              <button onClick={() => void copy()}>
                <Copy size={14} /> Edit a copy
              </button>
            }
          >
            {specs.data ? (
              <>
                <details open>
                  <summary>Simulation configuration</summary>
                  <JsonView value={specs.data.config} />
                </details>
                <details>
                  <summary>Placement</summary>
                  <JsonView value={specs.data.placement} />
                </details>
                <details>
                  <summary>Hardware</summary>
                  <JsonView value={specs.data.hardware} />
                </details>
                <details>
                  <summary>Workload</summary>
                  <JsonView value={specs.data.workload} />
                </details>
              </>
            ) : (
              <ErrorBox error={specs.error} />
            )}
          </Card>
          <Artifacts id={id} names={d.artifacts} />
        </>
      )}
    </>
  );
}
function TimeRange({
  start,
  end,
  setStart,
  setEnd,
}: {
  start: string;
  end: string;
  setStart: (value: string) => void;
  setEnd: (value: string) => void;
}) {
  return (
    <div className="time-range">
      <input
        type="number"
        min="0"
        step="any"
        aria-label="Start time seconds"
        placeholder="Start (s)"
        value={start}
        onChange={(e) => setStart(e.target.value)}
      />
      <span>—</span>
      <input
        type="number"
        min="0"
        step="any"
        aria-label="End time seconds"
        placeholder="End (s)"
        value={end}
        onChange={(e) => setEnd(e.target.value)}
      />
    </div>
  );
}
function Artifacts({ id, names }: { id: string; names: string[] }) {
  return (
    <Card
      title="Experiment artifacts"
      subtitle="Download original files without modifying the run."
    >
      <div className="artifact-grid">
        {names.map((name) => (
          <a
            key={name}
            href={"/api/runs/" + id + "/artifacts/" + name}
            download
          >
            <Download size={14} />
            {name}
          </a>
        ))}
      </div>
    </Card>
  );
}
function TracePanel({
  id,
  selected,
  onSelect,
  detail,
}: {
  id: string;
  selected: string;
  onSelect: (id: string) => void;
  detail: Detail;
}) {
  const [offset, setOffset] = useState(0),
    [event, setEvent] = useState(""),
    [focus, setFocus] = useState<unknown>(null);
  const query = new URLSearchParams({ offset: String(offset), limit: "100" });
  if (event) query.set("event_type", event);
  if (selected) query.set("activity", selected);
  const trace = useQuery({
    queryKey: ["trace", id, query.toString()],
    queryFn: () => api<TracePage>("/runs/" + id + "/trace?" + query),
    enabled: detail.artifacts.includes("trace.jsonl"),
  });
  useEffect(() => setOffset(0), [event, selected]);
  return (
    <>
      <Card
        title="Event trace"
        subtitle="Stable simulation ordering. Select an event to inspect its complete payload."
      >
        <div className="toolbar">
          <select
            aria-label="Event type"
            value={event}
            onChange={(e) => {
              setEvent(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">All event types</option>
            {[
              "ACTIVITY_STARTED",
              "ACTIVITY_FINISHED",
              "RATE_CHANGED",
              "CAPACITY_CHANGE",
              "WAITING",
              "ALLOCATED",
              "FREED",
              "REPLICA_VALID",
              "REQUEST_FINISHED",
            ].map((e) => (
              <option key={e}>{e}</option>
            ))}
          </select>
          <input
            aria-label="Trace activity ID"
            placeholder="Activity ID…"
            value={selected}
            onChange={(e) => onSelect(e.target.value)}
          />
          <button
            onClick={() => {
              setEvent("");
              onSelect("");
              setOffset(0);
            }}
          >
            Reset filters
          </button>
        </div>
        <ErrorBox error={trace.error} />
        {!detail.artifacts.includes("trace.jsonl") ? (
          <Empty title="Trace artifact unavailable" />
        ) : trace.isPending ? (
          <Loading />
        ) : (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>SEQ</th>
                    <th>SIMULATION TIME</th>
                    <th>EVENT</th>
                    <th>ACTIVITY / REQUEST</th>
                    <th>REASON</th>
                  </tr>
                </thead>
                <tbody>
                  {trace.data?.rows.map((row) => (
                    <tr
                      key={row.seq}
                      className="clickable"
                      onClick={() => setFocus(row)}
                    >
                      <td className="mono muted">{row.seq}</td>
                      <td className="mono">{duration(row.sim_time)}</td>
                      <td>
                        <Badge
                          tone={
                            row.event_type === "WAITING"
                              ? "warning"
                              : row.event_type === "ACTIVITY_FINISHED"
                                ? "success"
                                : "neutral"
                          }
                        >
                          {row.event_type}
                        </Badge>
                      </td>
                      <td className="mono">
                        {row.activity_id ?? "—"}
                        <small>{row.request_id}</small>
                      </td>
                      <td className="truncate">
                        {row.reason ? JSON.stringify(row.reason) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pagination">
              <button
                disabled={offset === 0}
                onClick={() => setOffset((o) => Math.max(0, o - 100))}
              >
                Previous
              </button>
              <span>
                {offset + 1}–{Math.min(offset + 100, trace.data?.total ?? 0)} of{" "}
                {trace.data?.total ?? 0}
              </span>
              <button
                disabled={offset + 100 >= (trace.data?.total ?? 0)}
                onClick={() => setOffset((o) => o + 100)}
              >
                Next
              </button>
            </div>
          </>
        )}
        {focus !== null && <JsonView value={focus} />}
      </Card>
      {detail.failure && (
        <Card
          title="Failure snapshot"
          subtitle="Allocations, active readers, unavailable capacity, and blocked activities."
        >
          <JsonView value={detail.failure} />
        </Card>
      )}
      <Artifacts id={id} names={detail.artifacts} />
    </>
  );
}
