import { useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  BaseEdge,
  EdgeLabelRenderer,
  type EdgeProps,
  type Node,
  type Edge,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import "@xyflow/react/dist/style.css";
import type { Specs } from "./types";
import { bytes } from "./format";
import { JsonView } from "./ui";

type Entity = { id: string; label: string; kind: string; data: unknown };
export function Graphs({
  specs,
  onActivity,
}: {
  specs: Specs;
  onActivity?: (id: string) => void;
}) {
  const [mode, setMode] = useState("hardware");
  const [search, setSearch] = useState("");
  const [tensors, setTensors] = useState(false);
  const [selected, setSelected] = useState<Entity | null>(null);
  const graph = useMemo(() => {
    let entities: Entity[] = [];
    let edges: Edge[] = [];
    if (mode === "hardware") {
      entities = specs.hardware.nodes.map((n) => ({
        id: n.id,
        label: n.id,
        kind: n.kind,
        data: n,
      }));
      const siblings = new Map<string, number>();
      const totals = new Map<string, number>();
      for (const c of specs.hardware.channels) {
        const key = c.src + "→" + c.dst;
        totals.set(key, (totals.get(key) ?? 0) + 1);
      }
      edges = specs.hardware.channels.map((c) => {
        const key = c.src + "→" + c.dst;
        const index = siblings.get(key) ?? 0;
        siblings.set(key, index + 1);
        return {
          id: c.id,
          source: c.src,
          target: c.dst,
          label: `${c.id} · ${bytes(c.payload_Bps)}/s`,
          type: "channel",
          data: {
            channel: c,
            offset: (index - ((totals.get(key) ?? 1) - 1) / 2) * 55,
          },
          style: { stroke: c.kind === "memory_access" ? "#344755" : "#298f8a" },
          labelStyle: { fill: "#8a9ba9", fontSize: 10 },
          labelBgStyle: { fill: "#111821" },
        };
      });
    } else if (mode === "workload") {
      entities = specs.workload.operators.map((o) => ({
        id: "op:" + o.id,
        label: o.id,
        kind: o.op_type,
        data: o,
      }));
      if (tensors) {
        entities.push(
          ...specs.workload.tensors.map((t) => ({
            id: "tensor:" + t.id,
            label: t.id,
            kind: "tensor",
            data: t,
          })),
        );
        for (const t of specs.workload.tensors) {
          if (t.producer_id)
            edges.push({
              id: "produce:" + t.id,
              source: "op:" + t.producer_id,
              target: "tensor:" + t.id,
            });
          for (const c of t.consumer_ids)
            edges.push({
              id: t.id + "→" + c,
              source: "tensor:" + t.id,
              target: "op:" + c,
            });
        }
      } else {
        for (const t of specs.workload.tensors)
          if (t.producer_id)
            for (const c of t.consumer_ids)
              edges.push({
                id: t.id + "→" + c,
                source: "op:" + t.producer_id,
                target: "op:" + c,
                label: bytes(t.storage_bytes),
                data: t,
              });
      }
      for (const o of specs.workload.operators)
        for (const pred of o.control_predecessors ?? [])
          edges.push({
            id: "control:" + pred + "→" + o.id,
            source: "op:" + pred,
            target: "op:" + o.id,
            label: "control",
            style: { stroke: "#a88955", strokeDasharray: "5 4" },
          });
    } else {
      entities = (specs.execution?.activities ?? []).map((a) => ({
        id: a.id,
        label: a.op_id ?? a.tensor_id ?? a.id,
        kind: a.kind,
        data: a,
      }));
      edges = (specs.execution?.activities ?? []).flatMap((a) =>
        a.predecessor_ids.map((p) => ({
          id: p + "→" + a.id,
          source: p,
          target: a.id,
        })),
      );
    }
    const matches = entities.filter(
      (e) =>
        !search ||
        JSON.stringify(e).toLowerCase().includes(search.toLowerCase()),
    );
    const visible = matches.slice(0, 400);
    const ids = new Set(visible.map((e) => e.id));
    edges = edges.filter((e) => ids.has(e.source) && ids.has(e.target));
    const layout = new dagre.graphlib.Graph({ multigraph: true });
    layout.setGraph({
      rankdir: mode === "hardware" ? "LR" : "TB",
      ranksep: 70,
      nodesep: 30,
    });
    layout.setDefaultEdgeLabel(() => ({}));
    visible.forEach((e) => layout.setNode(e.id, { width: 195, height: 66 }));
    edges.forEach((e) => layout.setEdge(e.source, e.target, {}, e.id));
    dagre.layout(layout);
    const nodes: Node[] = visible.map(
      (e) =>
        ({
          id: e.id,
          data: {
            label: (
              <>
                <span className="graph-kind">
                  {e.kind}
                  {mode === "hardware" &&
                  (e.data as { host_id?: string }).host_id
                    ? " · " + (e.data as { host_id?: string }).host_id
                    : ""}
                </span>
                <strong title={e.label}>{e.label}</strong>
              </>
            ),
          },
          position: {
            x: layout.node(e.id).x - 97.5,
            y: layout.node(e.id).y - 33,
          },
          className:
            "graph-node " +
            (e.kind === "transfer"
              ? "transfer"
              : e.kind === "memory"
                ? "memory"
                : ""),
          style: { width: 195 },
          sourcePosition: mode === "hardware" ? "right" : "bottom",
          targetPosition: mode === "hardware" ? "left" : "top",
        }) as Node,
    );
    return {
      entities: visible,
      nodes,
      edges: edges.map((e) => ({
        ...e,
        markerEnd: { type: MarkerType.ArrowClosed, color: "#476574" },
        style: e.style ?? { stroke: "#3c646d" },
        labelStyle: e.labelStyle ?? { fill: "#8c9eab", fontSize: 10 },
        labelBgStyle: { fill: "#111821" },
      })),
      total: matches.length,
    };
  }, [specs, mode, search, tensors]);
  return (
    <>
      <div className="toolbar">
        <div className="segmented">
          {["hardware", "workload", "execution"].map((m) => (
            <button
              key={m}
              className={mode === m ? "selected" : ""}
              onClick={() => {
                setMode(m);
                setSelected(null);
              }}
            >
              {m === "hardware"
                ? "Hardware topology"
                : m === "workload"
                  ? "Workload DAG"
                  : "Execution DAG"}
            </button>
          ))}
        </div>
        <input
          aria-label="Filter graph"
          placeholder="Find a node…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {mode === "workload" && (
          <label className="checkbox">
            <input
              type="checkbox"
              checked={tensors}
              onChange={(e) => setTensors(e.target.checked)}
            />{" "}
            Show tensors
          </label>
        )}
      </div>
      {graph.total > 400 && (
        <div className="notice">
          Showing 400 of {graph.total} matching nodes. Filter to inspect the
          rest.
        </div>
      )}
      <div className={"graph-layout " + (selected ? "has-inspector" : "")}>
        <div className="graph-canvas">
          <ReactFlow
            key={mode + search + tensors}
            nodes={graph.nodes}
            edges={graph.edges}
            edgeTypes={edgeTypes}
            onEdgeClick={(_, edge) => {
              if (edge.data)
                setSelected({
                  id: edge.id,
                  label: edge.id,
                  kind: "edge",
                  data: edge.data.channel ?? edge.data,
                });
            }}
            nodesDraggable={false}
            nodesConnectable={false}
            deleteKeyCode={null}
            fitView
            minZoom={0.05}
            maxZoom={2}
            colorMode="dark"
            onNodeClick={(_, n) =>
              setSelected(graph.entities.find((e) => e.id === n.id) ?? null)
            }
          >
            <Background color="#26313d" gap={24} />
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              nodeColor="#246a70"
              maskColor="#0b1018aa"
            />
          </ReactFlow>
          <div className="graph-caption">
            {graph.nodes.length} nodes · {graph.edges.length} edges · scroll to
            zoom
          </div>
        </div>
        {selected && (
          <aside className="inspector">
            <div className="eyebrow">Selected node</div>
            <h3>{selected.label}</h3>
            {onActivity && mode !== "hardware" && (
              <button
                onClick={() =>
                  onActivity(
                    mode === "execution"
                      ? selected.id
                      : (selected.data as { id: string }).id,
                  )
                }
              >
                Find in timeline →
              </button>
            )}
            <JsonView value={selected.data} />
          </aside>
        )}
      </div>
    </>
  );
}

function ChannelEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  label,
  style,
  markerEnd,
}: EdgeProps) {
  const offset = Number(data?.offset ?? 0);
  const mx = (sourceX + targetX) / 2;
  const my = (sourceY + targetY) / 2 + offset;
  const path = `M ${sourceX},${sourceY} Q ${mx},${my} ${targetX},${targetY}`;
  return (
    <>
      <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd} />
      <EdgeLabelRenderer>
        <div
          className="channel-label nodrag nopan"
          style={{
            transform: `translate(-50%,-50%) translate(${mx}px,${(sourceY + targetY) / 2 + offset / 2}px)`,
          }}
        >
          {label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
const edgeTypes = { channel: ChannelEdge };
