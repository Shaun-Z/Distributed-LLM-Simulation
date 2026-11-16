import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "./api";
import type { Detail, Series } from "./types";
import { Badge, Card, Empty, ErrorBox, Loading, PageTitle, Status } from "./ui";
import { bytes, delta, duration, number } from "./format";
import { OverlayChart } from "./Charts";

export function Compare() {
  const [params] = useSearchParams();
  const ids = [
    ...new Set((params.get("runs") ?? "").split(",").filter(Boolean)),
  ].slice(0, 4);
  const [baseline, setBaseline] = useState(ids[0]),
    [pool, setPool] = useState("");
  const results = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => api<Detail>("/runs/" + id),
    })),
  });
  const series = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["compare-series", id],
      queryFn: () => api<Series>("/runs/" + id + "/series"),
    })),
  });
  const data = results.map((r) => r.data).filter((d): d is Detail => !!d);
  const base = data.find((d) => d.id === baseline);
  const pools = [
    ...new Set(
      data.flatMap((d) =>
        Object.keys(d.summary.metrics?.peak_memory_bytes ?? {}),
      ),
    ),
  ];
  const metrics: {
    label: string;
    value: (d: Detail) => unknown;
    format: (value: unknown) => string;
  }[] = [
    { label: "Makespan", value: (d) => d.summary.makespan_s, format: duration },
    {
      label: "Throughput · req/s",
      value: (d) => d.summary.metrics?.throughput_requests_per_s,
      format: number,
    },
    {
      label: "P95 request latency",
      value: (d) => d.summary.metrics?.p95_latency_s,
      format: duration,
    },
    {
      label: "Transferred payload",
      value: (d) => d.summary.metrics?.bytes_transferred,
      format: bytes,
    },
    ...pools.map((p) => ({
      label: p + " peak",
      value: (d: Detail) => d.summary.metrics?.peak_memory_bytes?.[p],
      format: bytes,
    })),
  ];
  return (
    <>
      <PageTitle
        eyebrow="Cross-experiment analysis"
        title="Every trade-off, side by side."
        description="Compare recorded outcomes. Different model assumptions remain visible."
        actions={
          <Link className="button" to="/">
            Choose experiments
          </Link>
        }
      />
      {ids.length < 2 ? (
        <Empty title="Select at least two experiments">
          Use the checkboxes on the experiments page to compare up to four runs.
        </Empty>
      ) : results.some((r) => r.isPending) ? (
        <Loading />
      ) : (
        <>
          <ErrorBox error={results.find((r) => r.error)?.error} />
          <Card
            title="Outcome comparison"
            subtitle="Percentage changes are relative to the selected baseline; lower is not always better."
            actions={
              <select
                aria-label="Comparison baseline"
                value={baseline}
                onChange={(e) => setBaseline(e.target.value)}
              >
                {data.map((d) => (
                  <option value={d.id} key={d.id}>
                    {d.directory}
                  </option>
                ))}
              </select>
            }
          >
            <div className="table-wrap">
              <table className="compare-table">
                <thead>
                  <tr>
                    <th>METRIC</th>
                    {data.map((d) => (
                      <th key={d.id}>
                        <Link to={"/runs/" + d.id}>{d.directory}</Link>
                        <div className="comparison-meta">
                          <Status value={d.summary.status} />
                          <Badge>
                            Model {String(d.manifest.model_level ?? "N/A")}
                          </Badge>
                          <span>seed {String(d.manifest.seed ?? "N/A")}</span>
                        </div>
                        <small>
                          hardware{" "}
                          {d.manifest.hashes?.hardware?.slice(0, 10) ?? "N/A"}
                        </small>
                        <small>
                          config{" "}
                          {d.manifest.hashes?.config?.slice(0, 10) ?? "N/A"}
                        </small>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m) => (
                    <tr key={m.label}>
                      <td>{m.label}</td>
                      {data.map((d) => (
                        <td key={d.id}>
                          <strong>{m.format(m.value(d))}</strong>
                          <small
                            className={d.id === baseline ? "teal-text" : ""}
                          >
                            {d.id === baseline
                              ? "Baseline"
                              : delta(
                                  m.value(d),
                                  base ? m.value(base) : undefined,
                                )}
                          </small>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card
            title="Memory residency overlay"
            subtitle="Pools are compared by ID; RAM and VRAM are never added together."
            actions={
              <select
                aria-label="Comparison memory pool"
                value={pool || pools[0] || ""}
                onChange={(e) => setPool(e.target.value)}
              >
                {pools.map((p) => (
                  <option key={p}>{p}</option>
                ))}
              </select>
            }
          >
            <ErrorBox error={series.find((r) => r.error)?.error} />
            {series.some((r) => r.isPending) ? (
              <Loading />
            ) : (
              <OverlayChart
                pool={pool || pools[0]}
                entries={series.flatMap((r, i) =>
                  r.data
                    ? [
                        {
                          name:
                            data.find((d) => d.id === ids[i])?.directory ??
                            ids[i],
                          data: r.data,
                        },
                      ]
                    : [],
                )}
              />
            )}
          </Card>
          <div className="notice">
            These are empirical outcomes within each run's declared model.
            Matching pool IDs do not imply equivalent hardware; inspect the
            configuration and hardware hashes.
          </div>
        </>
      )}
    </>
  );
}
