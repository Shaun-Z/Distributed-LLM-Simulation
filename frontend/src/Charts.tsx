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
import { palettes, useTheme, type Theme } from "./theme";

export function Chart({
  option,
  height = 320,
  onClick,
}: {
  option: EChartsOption;
  height?: number;
  onClick?: (value: unknown) => void;
}) {
  const { theme } = useTheme();
  const previousTheme = useRef(theme);
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
    const chart = instance.current;
    if (!chart) return;
    const changingTheme = previousTheme.current !== theme;
    const old = changingTheme ? chart.getOption() : null;
    chart.setOption(option, true);
    if (old) {
      const zoom = (
        old.dataZoom as { start?: number; end?: number }[] | undefined
      )?.map(({ start, end }) => ({ start, end }));
      const legend = (
        old.legend as { selected?: Record<string, boolean> }[] | undefined
      )?.map(({ selected }) => ({ selected }));
      chart.setOption({
        ...(zoom ? { dataZoom: zoom } : {}),
        ...(legend ? { legend } : {}),
      });
    }
    previousTheme.current = theme;
  }, [option, theme]);
  return (
    <div
      role="img"
      aria-label="Interactive experiment chart"
      ref={element}
      style={{ height, width: "100%" }}
    />
  );
}
function chartBase(theme: Theme): EChartsOption {
  const palette = palettes[theme];
  return {
    backgroundColor: "transparent",
    color:
      theme === "dark"
        ? colors
        : [
            palette.accent,
            palette.transfer,
            palette.startup,
            "#b24163",
            "#2866bb",
            "#56812d",
          ],
    textStyle: {
      color: palette.text,
      fontFamily: "DM Sans, system-ui, sans-serif",
      fontSize: 11,
    },
    grid: { left: 65, right: 28, top: 38, bottom: 85 },
    tooltip: {
      trigger: "axis",
      backgroundColor: palette.tooltip,
      borderColor: palette.border,
      textStyle: { color: palette.text },
    },
    legend: { type: "scroll", textStyle: { color: palette.text }, top: 0 },
    xAxis: {
      type: "value",
      name: "Simulation time",
      nameLocation: "middle",
      nameGap: 38,
      axisLine: { lineStyle: { color: palette.border } },
      splitLine: { lineStyle: { color: palette.grid } },
      axisLabel: { color: palette.text, formatter: (v: number) => duration(v) },
      nameTextStyle: { color: palette.text },
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: palette.grid } },
      axisLabel: { color: palette.text },
    },
    dataZoom: [
      { type: "inside" },
      {
        type: "slider",
        height: 16,
        bottom: 8,
        borderColor: palette.border,
        fillerColor: "#32d5c71c",
        handleStyle: { color: palette.accent },
      },
    ],
  };
}
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
  const { theme } = useTheme();
  const base = chartBase(theme);
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
        color: palettes[theme].text,
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
  const { theme } = useTheme();
  const base = chartBase(theme);
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
          axisLabel: {
            color: palettes[theme].text,
            formatter: (v: number) => bytes(v),
          },
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
  const { theme } = useTheme();
  const option = useMemo(() => {
    const palette = palettes[theme];
    const base = chartBase(theme);
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
          ? palette.graphLabel
          : row.kind === "transfer"
            ? palette.transfer
            : palette.accent;
      if (setupEnd > row.start_s)
        data.push({
          value: [
            lanes.indexOf(lane),
            row.start_s,
            setupEnd,
            row.activity_id,
            "startup",
          ],
          itemStyle: { color: palette.startup, opacity: 0.75 },
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
          borderColor: critical.includes(row.activity_id)
            ? palette.selected
            : color,
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
        backgroundColor: palette.tooltip,
        borderColor: palette.border,
        textStyle: { color: palette.text },
        formatter: (p: any) =>
          `${echarts.format.encodeHTML(String(p.value[4]))}<br/>${duration(p.value[1])} → ${duration(p.value[2])}<br/>${echarts.format.encodeHTML(String(p.value[3]))}`,
      },
      yAxis: {
        type: "category",
        data: lanes,
        inverse: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: palette.text, width: 140, overflow: "truncate" },
        splitLine: { show: true, lineStyle: { color: palette.grid } },
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
                  lineStyle: { color: palette.selected, width: 1 },
                  data: [{ xAxis: cursor }],
                },
        },
      ],
    };
    return { opt, lanes };
  }, [rows, specs, critical, selected, cursor, theme]);
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
