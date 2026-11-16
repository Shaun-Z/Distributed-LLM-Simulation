import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  initialTheme,
  palettes,
  THEME_STORAGE_KEY,
  ThemeProvider,
  ThemeToggle,
} from "./theme";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("theme preference", () => {
  it("defaults to dark and rejects unknown saved values", () => {
    expect(initialTheme()).toBe("dark");
    localStorage.setItem(THEME_STORAGE_KEY, "invalid");
    expect(initialTheme()).toBe("dark");
  });
  it("switches, persists and restores the chosen theme", () => {
    const view = render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Switch to light theme" }),
    );
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(
      document.documentElement.style.getPropertyValue("--viz-graph-canvas"),
    ).toBe(palettes.light.graphCanvas);
    view.unmount();
    render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>,
    );
    expect(
      screen.getByRole("button", { name: "Switch to dark theme" }),
    ).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(
      screen.getByRole("button", { name: "Switch to dark theme" }),
    );
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("dark");
  });
  it("still switches when browser storage is blocked", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw Error("blocked");
    });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw Error("blocked");
    });
    render(
      <ThemeProvider>
        <ThemeToggle />
      </ThemeProvider>,
    );
    act(() =>
      fireEvent.click(
        screen.getByRole("button", { name: "Switch to light theme" }),
      ),
    );
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
  });
});

function luminance(hex: string) {
  const channels = hex
    .slice(1)
    .match(/../g)!
    .map((c) => parseInt(c, 16) / 255)
    .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
}
function contrast(a: string, b: string) {
  const x = luminance(a),
    y = luminance(b);
  return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
}
describe("DAG readability", () => {
  for (const theme of ["dark", "light"] as const)
    it(`${theme}: text and edges remain distinguishable`, () => {
      const p = palettes[theme];
      expect(contrast(p.graphLabel, p.graphLabelBg)).toBeGreaterThanOrEqual(
        4.5,
      );
      for (const bg of [
        p.graphComputeBg,
        p.graphMemoryBg,
        p.graphTensorBg,
        p.graphTransferBg,
      ]) {
        expect(contrast(p.graphLabel, bg)).toBeGreaterThanOrEqual(4.5);
        expect(contrast(p.graphMuted, bg)).toBeGreaterThanOrEqual(4.5);
      }
      expect(contrast(p.graphEdge, p.graphCanvas)).toBeGreaterThanOrEqual(3);
      expect(contrast(p.graphControl, p.graphCanvas)).toBeGreaterThanOrEqual(3);
    });
});
