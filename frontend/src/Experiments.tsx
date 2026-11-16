import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import {
  Plus,
  RefreshCw,
  Search,
  ArrowRight,
  GitCompareArrows,
  Clock3,
  X,
  Activity,
  Network,
  Layers3,
} from "lucide-react";
import { api } from "./api";
import type { Catalog, Job, Run } from "./types";
import {
  Badge,
  Card,
  Empty,
  ErrorBox,
  Loading,
  Metric,
  OpenRun,
  PageTitle,
  Status,
} from "./ui";
import { bytes, duration } from "./format";

export function Experiments() {
  const client = useQueryClient(),
    navigate = useNavigate();
  const [search, setSearch] = useState(""),
    [status, setStatus] = useState("all"),
    [selected, setSelected] = useState<string[]>([]),
    [error, setError] = useState<unknown>(null);
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<Run[]>("/runs"),
    refetchInterval: 3000,
  });
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<Job[]>("/jobs"),
    refetchInterval: 1000,
  });
  const catalog = useQuery({
    queryKey: ["catalog"],
    queryFn: () => api<Catalog>("/catalog"),
  });
  const active = (jobs.data ?? []).filter((j) =>
    ["queued", "running"].includes(j.state),
  );
  const visible = (runs.data ?? []).filter(
    (r) =>
      (status === "all" || r.status === status) &&
      `${r.label} ${r.workload} ${r.id}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  );
  const latest = (runs.data ?? [])[0];
  const poolCount = Object.keys(latest?.metrics.peak_memory_bytes ?? {}).length;
  async function cancel(id: string) {
    try {
      await api("/jobs/" + id + "/cancel", {});
      await client.invalidateQueries({ queryKey: ["jobs"] });
    } catch (e) {
      setError(e);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="Experiment workspace"
        title="A clearer view of distributed compute."
        description="Configure your system. Follow every dependency. Understand the trade-offs."
        actions={
          <Link className="button primary" to="/new">
            <Plus size={16} /> New experiment
          </Link>
        }
      />
      <div className="metrics four">
        <Metric
          label="EXPERIMENTS"
          value={runs.data?.length ?? "—"}
          foot="Persistent, reproducible artifacts"
        />
        <Metric
          label="ACTIVE JOBS"
          value={active.length.toString().padStart(2, "0")}
          foot="One isolated worker · queued execution"
          tone="teal"
        />
        <Metric
          label="LATEST MAKESPAN"
          value={duration(latest?.makespan_s)}
          foot={latest?.workload ?? "Run an experiment to get started"}
        />
        <Metric
          label="MEMORY POOLS · LATEST RUN"
          value={latest ? poolCount : "—"}
          foot="Physical budgets, counted once"
        />
      </div>
      <ErrorBox error={error ?? runs.error ?? jobs.error} />
      {active.length > 0 && (
        <Card
          title="In flight"
          subtitle="Task status is live. Simulation charts become available after the run finishes."
        >
          <div className="jobs">
            {active.map((j) => (
              <div className="job" key={j.id}>
                <Clock3 size={18} />
                <div>
                  <strong>{j.label}</strong>
                  <small>
                    {j.id.slice(0, 10)} ·{" "}
                    {new Date(j.created_at).toLocaleTimeString()}
                  </small>
                </div>
                <Status value={j.state} />
                <button
                  className="icon-button"
                  title="Cancel job"
                  aria-label={"Cancel " + j.label}
                  onClick={() => cancel(j.id)}
                >
                  <X size={16} />
                </button>
              </div>
            ))}
          </div>
        </Card>
      )}
      <Card
        title="Experiments"
        subtitle="Inspect execution, compare outcomes, and build on previous runs."
        actions={
          <button
            className="icon-button"
            aria-label="Refresh experiments"
            onClick={() => {
              client.invalidateQueries({ queryKey: ["runs"] });
              client.invalidateQueries({ queryKey: ["jobs"] });
            }}
          >
            <RefreshCw size={16} />
          </button>
        }
      >
        <div className="toolbar">
          <div className="search">
            <Search size={16} />
            <input
              aria-label="Search experiments"
              placeholder="Search experiments or workloads…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <select
            aria-label="Filter experiment status"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
          >
            <option value="all">All outcomes</option>
            {[
              "SUCCESS",
              "NO_PROGRESS",
              "TRUNCATED",
              "INVALID",
              "INCOMPLETE",
            ].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <button
            disabled={selected.length < 2}
            onClick={() => navigate("/compare?runs=" + selected.join(","))}
          >
            <GitCompareArrows size={15} /> Compare{" "}
            {selected.length ? `(${selected.length}/4)` : ""}
          </button>
        </div>
        {runs.isPending ? (
          <Loading />
        ) : visible.length ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th aria-label="Select" />
                  <th>EXPERIMENT / WORKLOAD</th>
                  <th>OUTCOME</th>
                  <th>MODEL / POLICY</th>
                  <th>MAKESPAN</th>
                  <th>MEMORY PEAKS</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {visible.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={"Compare " + r.label}
                        checked={selected.includes(r.id)}
                        disabled={
                          !selected.includes(r.id) && selected.length === 4
                        }
                        onChange={(e) =>
                          setSelected(
                            e.target.checked
                              ? [...selected, r.id]
                              : selected.filter((id) => id !== r.id),
                          )
                        }
                      />
                    </td>
                    <td>
                      <Link className="run-name" to={"/runs/" + r.id}>
                        {r.label}
                      </Link>
                      <small>{r.workload}</small>
                    </td>
                    <td>
                      <Status value={r.status} />
                    </td>
                    <td>
                      <Badge>{r.model_level ?? "—"}</Badge>{" "}
                      <span className="muted">{r.scheduler ?? "N/A"}</span>
                    </td>
                    <td className="mono">{duration(r.makespan_s)}</td>
                    <td>
                      {Object.entries(r.metrics.peak_memory_bytes ?? {}).map(
                        ([pool, n]) => (
                          <small key={pool}>
                            {pool} <span className="mono">{bytes(n)}</span>
                          </small>
                        ),
                      )}
                    </td>
                    <td>
                      <OpenRun id={r.id} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty
            title={
              search
                ? "No matching experiments"
                : "Your next experiment starts here"
            }
          >
            Choose a template below or create an experiment. Results will appear
            here after execution.
          </Empty>
        )}
      </Card>
      <div className="section-heading">
        <div>
          <div className="eyebrow">Start with a question</div>
          <h2>From simple pipelines to parallel decoders.</h2>
        </div>
        <Badge>5 built-in scenarios</Badge>
      </div>
      <div className="template-grid">
        {catalog.data?.templates.map((t, i) => (
          <Link
            to={"/new?template=" + t.id}
            className="template-card"
            key={t.id}
          >
            <div className={"template-icon tint" + i}>
              {i % 3 === 0 ? (
                <Network size={21} />
              ) : i % 3 === 1 ? (
                <Layers3 size={21} />
              ) : (
                <Activity size={21} />
              )}
            </div>
            <h3>{t.title}</h3>
            <p>{t.description}</p>
            <span>
              Configure scenario <ArrowRight size={14} />
            </span>
          </Link>
        ))}
      </div>
      {(jobs.data ?? []).some((j) =>
        ["failed", "cancelled", "interrupted"].includes(j.state),
      ) && (
        <Card
          title="Stopped jobs"
          subtitle="Cancelled, interrupted, and worker failures remain inspectable."
        >
          {(jobs.data ?? [])
            .filter((j) =>
              ["failed", "cancelled", "interrupted"].includes(j.state),
            )
            .slice(0, 10)
            .map((j) => (
              <details className="stopped-job" key={j.id}>
                <summary>
                  <span>{j.label}</span>
                  <Status value={j.state} />
                </summary>
                <pre>
                  {j.error ?? "This job has no completed simulation result."}
                </pre>
                <button
                  onClick={async () => {
                    try {
                      const draft = await api("/jobs/" + j.id + "/draft");
                      sessionStorage.setItem(
                        "copy-draft",
                        JSON.stringify(draft),
                      );
                      navigate("/new?copy=1");
                    } catch (e) {
                      setError(e);
                    }
                  }}
                >
                  Copy configuration
                </button>
              </details>
            ))}
        </Card>
      )}
    </>
  );
}
