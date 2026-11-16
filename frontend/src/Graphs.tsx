import { useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  useReactFlow,
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
import { palettes, useTheme } from "./theme";

type Entity = { id: string; label: string; kind: string; data: unknown };
export function Graphs({
  specs,
  onActivity,
}: {
  specs: Specs;
  onActivity?: (id: string) => void;
}) {
  const { theme } = useTheme();
  const palette = palettes[theme];
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
          style: {
            stroke:
              c.kind === "memory_access" ? palette.graphEdge : palette.accent,
          },
          labelStyle: { fill: palette.graphLabel, fontSize: 11 },
          labelBgStyle: { fill: palette.graphLabelBg },
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
            style: { stroke: palette.graphControl, strokeDasharray: "6 4" },
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
      ranksep: 85,
      nodesep: 38,
    });
    layout.setDefaultEdgeLabel(() => ({}));
    visible.forEach((e) => layout.setNode(e.id, { width: 220, height: 82 }));
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
            x: layout.node(e.id).x - 110,
            y: layout.node(e.id).y - 41,
          },
          className:
            "graph-node " +
            (e.kind === "transfer"
              ? "transfer"
              : e.kind === "memory"
                ? "memory"
                : e.kind === "tensor"
                  ? "tensor"
                  : ""),
          width: 220,
          height: 82,
          style: { width: 220, height: 82 },
          sourcePosition: mode === "hardware" ? "right" : "bottom",
          targetPosition: mode === "hardware" ? "left" : "top",
        }) as Node,
    );
    return {
      entities: visible,
      nodes,
      edges: edges.map((e) => ({
        ...e,
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: e.style?.stroke ?? palette.graphEdge,
          width: 20,
          height: 20,
        },
        style: { stroke: palette.graphEdge, ...e.style, strokeWidth: 2 },
        labelStyle: { fill: palette.graphLabel, fontSize: 11, fontWeight: 600 },
        labelShowBg: true,
        labelBgPadding: [7, 4] as [number, number],
        labelBgBorderRadius: 4,
        labelBgStyle: { fill: palette.graphLabelBg, fillOpacity: 1 },
      })),
      total: matches.length,
    };
  }, [specs, mode, search, tensors, palette]);
  const nodes = useMemo(
    () => graph.nodes.map((n) => ({ ...n, selected: selected?.id === n.id })),
    [graph.nodes, selected?.id],
  );
  const edges = useMemo(
    () =>
      graph.edges.map((e) => ({
        ...e,
        selected: selected?.kind === "edge" && selected.id === e.id,
      })),
    [graph.edges, selected],
  );
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
            nodes={nodes}
            edges={edges}
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
            fitViewOptions={{
              padding: 0.2,
              minZoom: mode === "hardware" ? 0.05 : 0.8,
              maxZoom: 1,
              ...(mode !== "hardware" && graph.nodes.length > 8
                ? { nodes: graph.nodes.slice(0, 1) }
                : {}),
            }}
            minZoom={0.05}
            maxZoom={2}
            colorMode={theme}
            onPaneClick={() => setSelected(null)}
            onNodeClick={(_, n) =>
              setSelected(graph.entities.find((e) => e.id === n.id) ?? null)
            }
          >
            <Background
              color={palette.graphGrid}
              bgColor={palette.graphCanvas}
              gap={24}
            />
            <Panel position="top-left">
              <ViewButtons firstId={graph.nodes[0]?.id} />
            </Panel>
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              style={{ width: 140, height: 95 }}
              nodeColor={palette.graphComputeBorder}
              maskColor={palette.graphMask}
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

function ViewButtons({ firstId }: { firstId?: string }) {
  const { fitView } = useReactFlow();
  return (
    <div className="graph-view-actions">
      <button
        disabled={!firstId}
        onClick={() =>
          void fitView({
            nodes: [{ id: firstId! }],
            padding: 0.2,
            minZoom: 0.8,
            maxZoom: 1,
            duration: 150,
          })
        }
      >
        Readable view
      </button>
      <button
        onClick={() =>
          void fitView({
            padding: 0.2,
            minZoom: 0.05,
            maxZoom: 1,
            duration: 150,
          })
        }
      >
        Fit all
      </button>
    </div>
  );
}
