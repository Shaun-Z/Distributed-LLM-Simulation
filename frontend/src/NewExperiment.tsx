import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import CodeMirror from "@uiw/react-codemirror";
import { yaml } from "@codemirror/lang-yaml";
import {
  Play,
  CheckCheck,
  Download,
  Upload,
  ArrowRight,
  SlidersHorizontal,
  Code2,
} from "lucide-react";
import { api, download } from "./api";
import type { Catalog, Draft, Edit, Preview, Section, Specs } from "./types";
import { Badge, Card, ErrorBox, Loading, PageTitle, Validated } from "./ui";
import { Graphs } from "./Graphs";

export function NewExperiment() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const catalog = useQuery({
    queryKey: ["catalog"],
    queryFn: () => api<Catalog>("/catalog"),
  });
  const [template, setTemplate] = useState(
      params.get("template") ?? "pipeline",
    ),
    [profile, setProfile] = useState(""),
    [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [draft, setDraftState] = useState<Draft | null>(null),
    [pv, setPv] = useState<Preview | null>(null),
    [formSpecs, setFormSpecs] = useState<Specs | null>(null),
    [applying, setApplying] = useState(0),
    [validated, setValidated] = useState(""),
    [error, setError] = useState<unknown>(null),
    [tab, setTab] = useState("setup"),
    [section, setSection] = useState<Section>("hardware"),
    [filter, setFilter] = useState(""),
    [page, setPage] = useState(0),
    [bulk, setBulk] = useState("");
  const current = useRef<Draft | null>(null);
  const queue = useRef(Promise.resolve());
  const initialized = useRef(false);
  const validationRequest = useRef(0);
  function setDraft(d: Draft) {
    current.current = d;
    setDraftState(d);
    try {
      localStorage.setItem("heterosim-draft", JSON.stringify(d));
    } catch {
      setError(
        new Error(
          "Browser draft storage is full. Export this configuration to keep a copy.",
        ),
      );
    }
  }
  const key = (d: Draft) => JSON.stringify({ ...d, label: "" });
  async function validate(d: Draft) {
    const request = ++validationRequest.current;
    setError(null);
    try {
      try {
        const description = await api<Preview["specs"]>("/describe", d);
        if (
          request === validationRequest.current &&
          current.current &&
          key(current.current) === key(d)
        )
          setFormSpecs(description);
      } catch {
        /* Full validation below supplies the diagnostic. */
      }
      const result = await api<Preview>("/preview", d);
      if (
        request !== validationRequest.current ||
        (current.current && key(current.current) !== key(d))
      )
        return null;
      result.draft.label = current.current?.label ?? result.draft.label;
      setDraft(result.draft);
      setFormSpecs(result.specs);
      setPv({
        ...result,
        specs: { ...result.specs, execution: result.execution },
      });
      setValidated(key(result.draft));
      return result;
    } catch (e) {
      if (request === validationRequest.current) setError(e);
      return null;
    }
  }
  useEffect(() => {
    if (!catalog.data || initialized.current) return;
    initialized.current = true;
    const selected = catalog.data.templates.find((t) => t.id === template)!;
    setProfile(selected.profile);
    setParameters(selected.parameters);
    const copied = params.get("copy") && sessionStorage.getItem("copy-draft");
    const stored =
      !params.get("template") &&
      !copied &&
      localStorage.getItem("heterosim-draft");
    if (copied || stored) {
      try {
        const saved = JSON.parse(copied || stored || "");
        if (
          ["hardware", "workload", "placement", "config"].every(
            (k) => typeof saved[k] === "string",
          )
        ) {
          setDraft(saved);
          void validate(saved);
          return;
        }
      } catch {
        /* Ignore obsolete browser drafts. */
      }
    }
    void generate(selected.id, selected.profile, selected.parameters, false);
  }, [catalog.data]);
  async function generate(
    id = template,
    hardware = profile,
    values = parameters,
    confirm = true,
  ) {
    if (
      confirm &&
      draft &&
      !window.confirm(
        "Rebuild the workload and placement? Manual edits to the current draft will be replaced.",
      )
    )
      return;
    setError(null);
    try {
      const d = await api<Draft>("/templates", {
        template: id,
        profile: hardware,
        parameters: values,
      });
      setDraft(d);
      await validate(d);
    } catch (e) {
      setError(e);
    }
  }
  function commit(edits: Edit[]) {
    setApplying((n) => n + 1);
    queue.current = queue.current
      .then(async () => {
        if (!current.current) return;
        try {
          const snapshot = current.current;
          const next = await api<Draft>("/draft/patch", {
            draft: snapshot,
            edits,
          });
          if (current.current && key(current.current) !== key(snapshot))
            throw new Error(
              "The draft changed while an edit was applying. Review it and apply the edit again.",
            );
          next.label = current.current?.label ?? next.label;
          setDraft(next);
          await validate(next);
        } catch (e) {
          setError(e);
        }
      })
      .finally(() => setApplying((n) => n - 1));
    return queue.current;
  }
  const launch = useMutation({
    mutationFn: async () => {
      await queue.current;
      if (!current.current) throw Error("No configuration");
      const result = await validate(current.current);
      if (!result) throw Error("Resolve validation errors before running.");
      return api("/jobs", result.draft);
    },
    onSuccess: () => navigate("/"),
    onError: setError,
  });
  const formPreview = pv ? { ...pv, specs: formSpecs ?? pv.specs } : null;
  const selectedTemplate = catalog.data?.templates.find(
    (t) => t.id === template,
  );
  const matching =
    formPreview?.specs.workload.operators.filter((o) =>
      `${o.id} ${o.op_type}`.toLowerCase().includes(filter.toLowerCase()),
    ) ?? [];
  const valid = draft && key(draft) === validated && applying === 0;
  async function importBundle(file: File) {
    try {
      const parsed = JSON.parse(await file.text());
      if (
        !["hardware", "workload", "placement", "config"].every(
          (k) => typeof parsed[k] === "string",
        )
      )
        throw Error(
          "Import an exported experiment bundle containing all four source documents.",
        );
      const d = { ...parsed, label: parsed.label ?? file.name };
      setDraft(d);
      await validate(d);
    } catch (e) {
      setError(e);
    }
  }
  return (
    <>
      <PageTitle
        eyebrow="Experiment builder"
        title="Make the system your variable."
        description="Start with a scenario, tune the resources, and validate the execution before you run."
        actions={
          <>
            <label className="button">
              <Upload size={15} /> Import
              <input
                hidden
                type="file"
                accept=".json"
                aria-label="Import experiment bundle"
                onChange={(e) => {
                  if (e.target.files?.[0]) void importBundle(e.target.files[0]);
                  e.target.value = "";
                }}
              />
            </label>
            <button
              disabled={!draft}
              onClick={() =>
                draft &&
                download("experiment.json", JSON.stringify(draft, null, 2))
              }
            >
              <Download size={15} /> Export
            </button>
            <button
              className="primary"
              disabled={!draft || launch.isPending}
              onClick={() => launch.mutate()}
            >
              <Play size={15} />
              {launch.isPending ? "Submitting…" : "Run experiment"}
            </button>
          </>
        }
      />
      <ErrorBox error={error ?? catalog.error} />
      <div className="builder-status">
        <span className="step-index">01</span>
        <span>Configure</span>
        <ArrowRight size={14} />
        <span className="step-index">02</span>
        <span>Validate</span>
        <ArrowRight size={14} />
        <span className="step-index">03</span>
        <span>Run & explore</span>
        <div className="grow" />
        {valid ? (
          <Validated />
        ) : (
          <Badge tone="warning">Unvalidated changes</Badge>
        )}
      </div>
      <div className="tabs">
        {[
          ["setup", "Scenario & resources"],
          ["placement", "Placement"],
          ["advanced", "Advanced sources"],
          ["preview", "Graph preview"],
        ].map(([id, label]) => (
          <button
            key={id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {id === "advanced" ? (
              <Code2 size={15} />
            ) : id === "setup" ? (
              <SlidersHorizontal size={15} />
            ) : null}
            {label}
          </button>
        ))}
        <button
          className="validate-button"
          disabled={!draft}
          onClick={() => draft && void validate(draft)}
        >
          <CheckCheck size={15} /> Validate draft
        </button>
      </div>
      {!draft || !catalog.data ? (
        <Loading />
      ) : (
        <>
          {tab === "setup" && (
            <>
              <div className="builder-grid">
                <Card
                  title="Workload scenario"
                  subtitle="Templates generate real specs and an explicit placement."
                >
                  <label className="field-label">
                    Scenario
                    <select
                      value={template}
                      onChange={(e) => {
                        const t = catalog.data.templates.find(
                          (t) => t.id === e.target.value,
                        )!;
                        setTemplate(t.id);
                        setProfile(t.profile);
                        setParameters(t.parameters);
                      }}
                    >
                      {catalog.data.templates.map((t) => (
                        <option value={t.id} key={t.id}>
                          {t.title}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field-label">
                    Hardware profile
                    <select
                      value={profile}
                      onChange={(e) => setProfile(e.target.value)}
                    >
                      {catalog.data.profiles.map((p) => (
                        <option value={p.id} key={p.id}>
                          {p.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <p className="muted">{selectedTemplate?.description}</p>
                  <div className="parameter-grid">
                    {selectedTemplate &&
                      Object.keys(selectedTemplate.parameters).map((name) => (
                        <Parameter
                          key={template + name}
                          name={name}
                          value={parameters[name]}
                          onChange={(value) =>
                            setParameters((p) => ({ ...p, [name]: value }))
                          }
                        />
                      ))}
                  </div>
                  <button className="wide" onClick={() => void generate()}>
                    Build scenario <ArrowRight size={15} />
                  </button>
                  <p className="hint">
                    Device and pool IDs must match the selected hardware.
                    Rebuilding replaces manual placement edits.
                  </p>
                </Card>
                <Card
                  title="Run configuration"
                  subtitle="All durations use seconds internally."
                >
                  <label className="field-label">
                    Experiment name
                    <input
                      aria-label="Experiment name"
                      value={draft.label}
                      onChange={(e) =>
                        setDraft({ ...draft, label: e.target.value })
                      }
                    />
                  </label>
                  <ConfigFields pv={formPreview} commit={commit} />
                  <div className="info-panel">
                    <strong>Whole-tensor, auditable execution</strong>
                    <p>
                      Fixed-duration kernels. Explicit copies. Finite physical
                      storage. Fluid sharing applies to transfers.
                    </p>
                  </div>
                </Card>
              </div>
              {pv && (
                <>
                  <Card
                    title="Physical resource budgets"
                    subtitle="Exact capacities are parsed in Python. Inputs also accept units such as 8 GiB or 100 Gbit/s."
                  >
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>RESOURCE</th>
                            <th>KIND</th>
                            <th>CAPACITY</th>
                            <th>UNIT</th>
                          </tr>
                        </thead>
                        <tbody>
                          {formPreview!.specs.hardware.resources.map((r, i) => (
                            <tr key={r.id}>
                              <td className="mono">{r.id}</td>
                              <td>
                                <Badge>{r.kind}</Badge>
                              </td>
                              <td>
                                <Editable
                                  label={"Capacity " + r.id}
                                  value={r.capacity}
                                  onCommit={(value) =>
                                    commit([
                                      {
                                        section: "hardware",
                                        path: ["resources", i, "capacity"],
                                        value,
                                        kind:
                                          r.kind === "rate"
                                            ? "number"
                                            : "integer",
                                        unit: r.unit,
                                      },
                                    ])
                                  }
                                />
                              </td>
                              <td>{r.unit}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                  <Card
                    title="Channels"
                    subtitle="Latency and payload bandwidth remain separate from shared physical budgets."
                  >
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>CHANNEL</th>
                            <th>ENDPOINTS</th>
                            <th>PAYLOAD · BYTE/S</th>
                            <th>LATENCY · S</th>
                            <th>SETUP · S</th>
                          </tr>
                        </thead>
                        <tbody>
                          {formPreview!.specs.hardware.channels.map((c, i) => (
                            <tr key={c.id}>
                              <td className="mono">{c.id}</td>
                              <td>
                                {c.src} → {c.dst}
                              </td>
                              {(
                                ["payload_Bps", "latency_s", "setup_s"] as const
                              ).map((k) => (
                                <td key={k}>
                                  <Editable
                                    label={k + " " + c.id}
                                    value={c[k]}
                                    onCommit={(value) =>
                                      commit([
                                        {
                                          section: "hardware",
                                          path: ["channels", i, k],
                                          value,
                                          kind: "number",
                                          unit:
                                            k === "payload_Bps"
                                              ? "byte/s"
                                              : "s",
                                        },
                                      ])
                                    }
                                  />
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </Card>
                </>
              )}
            </>
          )}
          {tab === "placement" && pv && (
            <Card
              title="Operator placement"
              subtitle="Assign legal devices and kernels. The backend validates routes, dtype, shape, and capacity semantics."
            >
              <div className="toolbar">
                <input
                  aria-label="Filter operators"
                  placeholder="Filter operators…"
                  value={filter}
                  onChange={(e) => {
                    setFilter(e.target.value);
                    setPage(0);
                  }}
                />
                <select
                  aria-label="Bulk target device"
                  value={bulk}
                  onChange={(e) => setBulk(e.target.value)}
                >
                  <option value="">Choose a device…</option>
                  {formPreview!.specs.hardware.nodes
                    .filter((n) => n.kind === "compute")
                    .map((n) => (
                      <option key={n.id}>{n.id}</option>
                    ))}
                </select>
                <button
                  disabled={!bulk}
                  onClick={() => {
                    const ids = new Set(matching.map((o) => o.id));
                    void commit(
                      formPreview!.specs.placement.op_compute.flatMap(
                        ([id], i) =>
                          ids.has(id)
                            ? [
                                {
                                  section: "placement" as const,
                                  path: ["op_compute", i, 1],
                                  value: bulk,
                                },
                              ]
                            : [],
                      ),
                    );
                  }}
                >
                  Map {matching.length} matching operators
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>OPERATOR</th>
                      <th>FAMILY / DTYPE</th>
                      <th>DEVICE</th>
                      <th>KERNEL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {matching.slice(page * 40, (page + 1) * 40).map((o) => {
                      const idx =
                        formPreview!.specs.placement.op_compute.findIndex(
                          ([id]) => id === o.id,
                        );
                      const device =
                        formPreview!.specs.placement.op_compute[idx]?.[1];
                      const ki =
                        formPreview!.specs.placement.op_kernel.findIndex(
                          ([id]) => id === o.id,
                        );
                      const kernel =
                        formPreview!.specs.placement.op_kernel[ki]?.[1] ?? "";
                      return (
                        <tr key={o.id}>
                          <td className="mono">{o.id}</td>
                          <td>
                            {o.op_type}
                            <small>
                              {o.input_dtype} → {o.accumulate_dtype} →{" "}
                              {o.output_dtype}
                            </small>
                          </td>
                          <td>
                            <select
                              aria-label={"Device " + o.id}
                              value={device}
                              onChange={(e) =>
                                void commit([
                                  {
                                    section: "placement",
                                    path: ["op_compute", idx, 1],
                                    value: e.target.value,
                                  },
                                ])
                              }
                            >
                              {formPreview!.specs.hardware.nodes
                                .filter((n) => n.kind === "compute")
                                .map((n) => (
                                  <option key={n.id}>{n.id}</option>
                                ))}
                            </select>
                          </td>
                          <td>
                            <select
                              aria-label={"Kernel " + o.id}
                              value={kernel}
                              onChange={(e) => {
                                const rows =
                                  formPreview!.specs.placement.op_kernel.filter(
                                    ([id]) => id !== o.id,
                                  );
                                if (e.target.value)
                                  rows.push([o.id, e.target.value]);
                                void commit([
                                  {
                                    section: "placement",
                                    path: ["op_kernel"],
                                    value: JSON.stringify(rows),
                                    kind: "json",
                                  },
                                ]);
                              }}
                            >
                              <option value="">
                                Automatic compatible kernel
                              </option>
                              {formPreview!.specs.hardware.kernel_profiles
                                .filter(
                                  (k) =>
                                    formPreview!.specs.hardware.nodes
                                      .find((n) => n.id === device)
                                      ?.kernel_profile_ids.includes(k.id) &&
                                    k.op_type === o.op_type &&
                                    k.input_dtype === o.input_dtype &&
                                    k.output_dtype === o.output_dtype &&
                                    k.accumulate_dtype === o.accumulate_dtype,
                                )
                                .map((k) => (
                                  <option key={k.id}>{k.id}</option>
                                ))}
                            </select>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div className="pagination">
                <button
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  Previous
                </button>
                <span>
                  {page + 1} / {Math.max(1, Math.ceil(matching.length / 40))}
                </span>
                <button
                  disabled={(page + 1) * 40 >= matching.length}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                </button>
              </div>
            </Card>
          )}
          {tab === "advanced" && (
            <Card
              title="Canonical source documents"
              subtitle="YAML and JSON are both accepted. Validate after editing to update the forms and graph preview."
              actions={
                <label className="button">
                  <Upload size={14} /> Load {section}
                  <input
                    hidden
                    type="file"
                    aria-label="Load source document"
                    onChange={async (e) => {
                      if (e.target.files?.[0])
                        setDraft({
                          ...current.current!,
                          [section]: await e.target.files[0].text(),
                        });
                      e.target.value = "";
                    }}
                  />
                </label>
              }
            >
              <div className="segmented source-tabs">
                {(
                  ["hardware", "workload", "placement", "config"] as Section[]
                ).map((key) => (
                  <button
                    key={key}
                    className={key === section ? "selected" : ""}
                    onClick={() => setSection(key)}
                  >
                    {key}.yaml
                  </button>
                ))}
              </div>
              <CodeMirror
                value={draft[section]}
                height="540px"
                theme="dark"
                extensions={[yaml()]}
                onChange={(value) =>
                  setDraft({ ...current.current!, [section]: value })
                }
              />
            </Card>
          )}
          {tab === "preview" && pv && (
            <Card
              title="Validated execution preview"
              subtitle="Logical dependencies, physical resources, and lowered activities."
            >
              <div className="count-row">
                {Object.entries(pv.counts).map(([name, value]) => (
                  <Badge key={name}>
                    {value} {name}
                  </Badge>
                ))}
                {!valid && <Badge tone="warning">Last validated draft</Badge>}
              </div>
              <Graphs specs={pv.specs} />
            </Card>
          )}
        </>
      )}
    </>
  );
}
function Parameter({
  name,
  value,
  onChange,
}: {
  name: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const label = name.replaceAll("_", " ");
  if (typeof value === "boolean")
    return (
      <label className="field-label">
        {label}
        <select
          value={String(value)}
          onChange={(e) => onChange(e.target.value === "true")}
        >
          <option>true</option>
          <option>false</option>
        </select>
      </label>
    );
  if (typeof value === "object")
    return (
      <StructuredParameter label={label} value={value} onChange={onChange} />
    );
  return (
    <label className="field-label">
      {label}
      <input
        aria-label={label}
        value={String(value ?? "")}
        onChange={(e) =>
          onChange(
            typeof value === "number"
              ? e.target.value === ""
                ? ""
                : Number(e.target.value)
              : e.target.value,
          )
        }
      />
    </label>
  );
}
export function Editable({
  label,
  value,
  onCommit,
}: {
  label: string;
  value: unknown;
  onCommit: (value: string) => unknown;
}) {
  const [text, setText] = useState(String(value ?? ""));
  useEffect(() => setText(String(value ?? "")), [value]);
  return (
    <input
      aria-label={label}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        if (text !== String(value ?? "")) void onCommit(text);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
      }}
    />
  );
}
function ConfigFields({
  pv,
  commit,
}: {
  pv: Preview | null;
  commit: (edits: Edit[]) => unknown;
}) {
  if (!pv) return null;
  return (
    <>
      <div className="parameter-grid">
        {[
          ["scheduler", ["fifo", "critical_rank", "memory_aware"]],
          ["network_mode", ["exclusive", "fluid"]],
          [
            "trace_level",
            ["summary", "activities", "resource_changes", "debug"],
          ],
        ].map(([name, choices]) => (
          <label className="field-label" key={name as string}>
            {(name as string).replaceAll("_", " ")}
            <select
              aria-label={name as string}
              value={pv.specs.config[name as keyof typeof pv.specs.config]}
              onChange={(e) =>
                void commit([
                  {
                    section: "config",
                    path: [name as string],
                    value: e.target.value,
                  },
                ])
              }
            >
              {(choices as string[]).map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        ))}
        {(["seed", "max_events", "max_sim_time"] as const).map((name) => (
          <label className="field-label" key={name}>
            {name.replaceAll("_", " ")}
            <Editable
              label={name}
              value={pv.specs.config[name]}
              onCommit={(value) =>
                commit([
                  {
                    section: "config",
                    path: [name],
                    value,
                    kind: name === "max_sim_time" ? "number" : "integer",
                  },
                ])
              }
            />
          </label>
        ))}
      </div>
      <p className="hint">
        Use resource_changes or debug traces for memory and bandwidth analysis.
        No streaming compute model is implied.
      </p>
    </>
  );
}

function StructuredParameter({
  label,
  value,
  onChange,
}: {
  label: string;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [text, setText] = useState(JSON.stringify(value));
  const [error, setError] = useState(false);
  useEffect(() => setText(JSON.stringify(value)), [value]);
  return (
    <label className="field-label">
      {label}
      <textarea
        aria-label={label}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => {
          try {
            onChange(JSON.parse(text));
            setError(false);
          } catch {
            setError(true);
          }
        }}
      />
      {error && <small>Enter a valid JSON array.</small>}
    </label>
  );
}
