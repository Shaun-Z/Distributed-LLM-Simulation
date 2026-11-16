export function duration(value: unknown): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value)))
    return "N/A";
  const n = Number(value);
  return n === 0
    ? "0 s"
    : Math.abs(n) < 0.001
      ? `${(n * 1e6).toLocaleString("en", { maximumFractionDigits: 2 })} µs`
      : Math.abs(n) < 1
        ? `${(n * 1e3).toLocaleString("en", { maximumFractionDigits: 3 })} ms`
        : `${n.toLocaleString("en", { maximumFractionDigits: 3 })} s`;
}
export function bytes(value: unknown): string {
  if (value === null || value === undefined || !Number.isFinite(Number(value)))
    return "N/A";
  const n = Number(value);
  const i =
    n === 0
      ? 0
      : Math.max(
          0,
          Math.min(4, Math.floor(Math.log(Math.abs(n)) / Math.log(1024))),
        );
  return `${(n / 1024 ** i).toLocaleString("en", { maximumFractionDigits: 2 })} ${["B", "KiB", "MiB", "GiB", "TiB"][i]}`;
}
export function number(value: unknown): string {
  return value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
    ? "N/A"
    : Number(value).toLocaleString("en", { maximumFractionDigits: 2 });
}
export function percent(value: unknown): string {
  return value === null || value === undefined || value === ""
    ? "N/A"
    : `${(Number(value) * 100).toFixed(1)}%`;
}
export function delta(value: unknown, baseline: unknown): string {
  if (
    value === null ||
    value === undefined ||
    baseline === null ||
    baseline === undefined ||
    Number(baseline) === 0
  )
    return "N/A";
  const p = (Number(value) / Number(baseline) - 1) * 100;
  return `${p > 0 ? "+" : ""}${p.toFixed(1)}%`;
}
export const colors = [
  "#32d5c7",
  "#9b9dff",
  "#f6b65b",
  "#fb8297",
  "#79aeea",
  "#9bd98c",
];
