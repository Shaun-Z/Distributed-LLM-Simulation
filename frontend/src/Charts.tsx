import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { CustomChart, LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
echarts.use([
  CustomChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
  CanvasRenderer,
]);
import type { EChartsOption } from "echarts";
import type { Series, Specs, TimelineRow } from "./types";
import { bytes, colors, duration } from "./format";
import { Empty } from "./ui";

export function Chart({
  option,
  height = 320,
  onClick,
}: {
  option: EChartsOption;
  height?: number;
  onClick?: (value: unknown) => void;
}) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.ECharts | null>(null);
  const callback = useRef(onClick);
  callback.current = onClick;
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, null, { renderer: "canvas" });
    instance.current = chart;
    chart.on("click", (p) => callback.current?.(p.data));
    const resize = new ResizeObserver(() => chart.resize());
    resize.observe(element.current);
    return () => {
      resize.disconnect();
      chart.dispose();
      instance.current = null;
    };
  }, []);
  useEffect(() => {
    instance.current?.setOption(option, true);
  }, [option]);
  return (
    <div
      role="img"
      aria-label="Interactive experiment chart"
      ref={element}
      style={{ height, width: "100%" }}
    />
  );
}
const base: EChartsOption = {
  backgroundColor: "transparent",
  color: colors,
  textStyle: {
    color: "#9baab9",
    fontFamily: "DM Sans, system-ui, sans-serif",
    fontSize: 11,
  },
  grid: { left: 65, right: 28, top: 38, bottom: 85 },
  tooltip: {
    trigger: "axis",
    backgroundColor: "#16202d",
    borderColor: "#334454",
    textStyle: { color: "#d9e8ef" },
  },
  legend: { type: "scroll", textStyle: { color: "#91a3b5" }, top: 0 },
  xAxis: {
    type: "value",
    name: "Simulation time",
    nameLocation: "middle",
    nameGap: 38,
    axisLine: { lineStyle: { color: "#283441" } },
    splitLine: { lineStyle: { color: "#1d2732" } },
    axisLabel: { formatter: (v: number) => duration(v) },
  },
  yAxis: {
    type: "value",
    splitLine: { lineStyle: { color: "#1d2732" } },
    axisLabel: { color: "#91a3b5" },
  },
  dataZoom: [
    { type: "inside" },
    {
      type: "slider",
      height: 16,
      bottom: 8,
      borderColor: "#26313d",
      fillerColor: "#32d5c71c",
      handleStyle: { color: "#32d5c7" },
    },
  ],
};
export function ResourceChart({
  series,
  kind,
  resource,
  titlePrefix = "",
}: {
  series: Series;
  kind: "memory" | "service";
  resource?: string;
  titlePrefix?: string;
}) {
  const [showCapacity, setShowCapacity] = useState(false);
  let items = series.series.filter(
    (s) => s.kind === kind && (!resource || s.id === resource),
  );
  if (kind === "service" && !resource) {
    const active = items.filter((s) => s.points.some((p) => p[1] > 0));
    if (active.length) items = active;
  }
  if (!items.length)
    return (
      <Empty title="Curve unavailable">
        {series.reason ?? "Choose a resource with recorded events."}
      </Empty>
    );
  const lines: any[] = items.map((s) => ({
    name: titlePrefix + s.id,
    type: "line",
    step: "end",
    showSymbol: false,
    data: s.points,
    lineStyle: { width: 2 },
    areaStyle: kind === "memory" ? { opacity: 0.06 } : undefined,
  }));
  if (kind === "memory" && showCapacity)
    items.forEach((s) => {
      if (s.capacity != null)
        lines.push({
          name: s.id + " capacity",
          type: "line",
          showSymbol: false,
          data: [
            [0, Number(s.capacity)],
            [s.points.at(-1)?.[0] ?? 0, Number(s.capacity)],
          ],
          lineStyle: { type: "dashed", width: 1, opacity: 0.5 },
        });
    });
  else if (kind === "service")
    series.series
      .filter((s) => s.kind === "capacity" && items.some((i) => i.id === s.id))
      .forEach((s) =>
        lines.push({
          name: s.id + " capacity",
          type: "line",
          step: "end",
          showSymbol: false,
          data: s.points,
          lineStyle: { type: "dashed", width: 1, opacity: 0.6 },
        }),
      );
  const option: EChartsOption = {
    ...base,
    yAxis: {
      ...(base.yAxis as object),
      axisLabel: {
        formatter: (v: number) => bytes(v) + (kind === "service" ? "/s" : ""),
      },
    },
    series: lines,
  };
  return (
    <>
      {kind === "memory" && (
        <div className="curve-controls">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={showCapacity}
              onChange={(e) => setShowCapacity(e.target.checked)}
            />
            Show capacity bounds
          </label>
          <span>
            {items
              .map((s) => `${s.id}: ${bytes(s.capacity)} capacity`)
              .join(" · ")}
          </span>
        </div>
      )}
      <Chart option={option} />
    </>
  );
}
export function OverlayChart({
  entries,
  pool,
}: {
  entries: { name: string; data: Series }[];
  pool: string;
}) {
  const lines = entries.flatMap((e) =>
    e.data.series
      .filter((s) => s.kind === "memory" && s.id === pool)
      .map((s) => ({
        name: e.name,
        type: "line" as const,
        step: "end" as const,
        showSymbol: false,
        data: s.points,
        lineStyle: { width: 2 },
      })),
  );
  return lines.length ? (
    <Chart
      option={{
        ...base,
        yAxis: {
          ...(base.yAxis as object),
          axisLabel: { formatter: (v: number) => bytes(v) },
        },
        series: lines,
      }}
    />
  ) : (
    <Empty title="No compatible memory curves" />
  );
}
export function TimelineChart({
  rows,
  specs,
  critical,
  selected,
  onSelect,
  cursor,
}: {
  rows: TimelineRow[];
  specs: Specs;
  critical: string[];
  selected: string;
  onSelect: (id: string) => void;
  cursor?: number;
}) {
  const option = useMemo(() => {
    const assignments = Object.fromEntries(specs.placement.op_compute);
    const acts = Object.fromEntries(
      (specs.execution?.activities ?? []).map((a) => [a.id, a]),
    );
    const replicas = Object.fromEntries(
      (specs.execution?.replicas ?? []).map((r) => [r.id, r]),
    );
    const lanes: string[] = [];
    const ends: Record<string, number[]> = {};
    const data: any[] = [];
    for (const row of rows) {
      if (row.start_s == null) continue;
      const a = acts[row.activity_id];
      const source = replicas[a?.input_replica_ids[0]];
      const target = replicas[a?.output_replica_ids[0]];
      const baseLane =
        row.kind === "transfer"
          ? `${source?.pool_id ?? "source"} → ${target?.pool_id ?? "target"}`
          : (assignments[row.op_id ?? ""] ?? "control");
      ends[baseLane] ??= [];
      let sub = ends[baseLane].findIndex((end) => end <= row.start_s!);
      if (sub < 0) sub = ends[baseLane].length;
      ends[baseLane][sub] = row.finish_s ?? Infinity;
      const lane = baseLane + (sub ? " · " + (sub + 1) : "");
      if (!lanes.includes(lane)) lanes.push(lane);
      const finish = row.finish_s ?? row.start_s;
      const setupEnd = Math.min(finish, row.start_s + row.setup_s);
      const color =
        row.activity_id === selected
          ? "#f5f9fc"
          : row.kind === "transfer"
            ? "#a69aff"
            : "#32d5c7";
      if (setupEnd > row.start_s)
        data.push({
          value: [
            lanes.indexOf(lane),
            row.start_s,
            setupEnd,
            row.activity_id,
            "startup",
          ],
          itemStyle: { color: "#b38949", opacity: 0.75 },
        });
      data.push({
        value: [
          lanes.indexOf(lane),
          setupEnd,
          finish,
          row.activity_id,
          row.op_id ?? row.tensor_id ?? row.activity_id,
        ],
        itemStyle: {
          color,
          borderColor: critical.includes(row.activity_id) ? "#f7c867" : color,
          borderWidth: critical.includes(row.activity_id) ? 2 : 0,
          opacity:
            selected &&
            !row.activity_id.includes(selected) &&
            !row.op_id?.includes(selected)
              ? 0.55
              : 0.9,
        },
      });
    }
    const opt: EChartsOption = {
      ...base,
      legend: { show: false },
      grid: { left: 155, right: 25, top: 15, bottom: 66 },
      tooltip: {
        trigger: "item",
        backgroundColor: "#16202d",
        borderColor: "#334454",
        textStyle: { color: "#d9e8ef" },
        formatter: (p: any) =>
          `${echarts.format.encodeHTML(String(p.value[4]))}<br/>${duration(p.value[1])} → ${duration(p.value[2])}<br/>${echarts.format.encodeHTML(String(p.value[3]))}`,
      },
      yAxis: {
        type: "category",
        data: lanes,
        inverse: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: "#94a5b7", width: 140, overflow: "truncate" },
        splitLine: { show: true, lineStyle: { color: "#1b2530" } },
      },
      series: [
        {
          type: "custom",
          renderItem: (params: any, api: any) => {
            const start = api.coord([api.value(1), api.value(0)]),
              end = api.coord([api.value(2), api.value(0)]);
            const height = Math.min(26, api.size([0, 1])[1] * 0.6);
            const shape = echarts.graphic.clipRectByRect(
              {
                x: start[0],
                y: start[1] - height / 2,
                width: Math.max(3, end[0] - start[0]),
                height,
              },
              {
                x: params.coordSys.x,
                y: params.coordSys.y,
                width: params.coordSys.width,
                height: params.coordSys.height,
              },
            );
            return shape
              ? { type: "rect", shape: { ...shape, r: 4 }, style: api.style() }
              : undefined;
          },
          encode: { x: [1, 2], y: 0 },
          data,
          markLine:
            cursor === undefined
              ? undefined
              : {
                  silent: true,
                  symbol: "none",
                  label: { show: false },
                  lineStyle: { color: "#f7c867", width: 1 },
                  data: [{ xAxis: cursor }],
                },
        },
      ],
    };
    return { opt, lanes };
  }, [rows, specs, critical, selected, cursor]);
  return rows.length ? (
    <Chart
      option={option.opt}
      height={Math.min(900, Math.max(310, option.lanes.length * 46 + 90))}
      onClick={(value) => {
        const row = value as { value: unknown[] };
        if (row?.value) onSelect(String(row.value[3]));
      }}
    />
  ) : (
    <Empty title="No activities in this view">
      Adjust the filters or select another time range.
    </Empty>
  );
}
