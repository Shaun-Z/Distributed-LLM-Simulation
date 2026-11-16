import {
  createContext,
  useContext,
  useLayoutEffect,
  useState,
  type ReactNode,
} from "react";
import { Moon, Sun } from "lucide-react";

export type Theme = "dark" | "light";
export const THEME_STORAGE_KEY = "heterosim-theme";
export const palettes = {
  dark: {
    text: "#c7d6e5",
    border: "#3b4e61",
    grid: "#293848",
    tooltip: "#162332",
    accent: "#55e0ce",
    transfer: "#b7a6ff",
    startup: "#d5a953",
    selected: "#f8d378",
    graphCanvas: "#0c1420",
    graphGrid: "#293b4d",
    graphEdge: "#adc8df",
    graphControl: "#f3c575",
    graphLabel: "#f0f6ff",
    graphLabelBg: "#172638",
    graphMuted: "#b6cee2",
    graphComputeBg: "#12373d",
    graphComputeBorder: "#68dccb",
    graphMemoryBg: "#302444",
    graphMemoryBorder: "#c9acf8",
    graphTensorBg: "#1b304c",
    graphTensorBorder: "#91bdff",
    graphTransferBg: "#332748",
    graphTransferBorder: "#bda3ff",
    graphMask: "#0c1420b8",
    graphShadow: "#00000055",
  },
  light: {
    text: "#374f67",
    border: "#a8bacb",
    grid: "#dce5ee",
    tooltip: "#ffffff",
    accent: "#087e73",
    transfer: "#7450c6",
    startup: "#a66b0b",
    selected: "#a26300",
    graphCanvas: "#f7faff",
    graphGrid: "#d7e1ec",
    graphEdge: "#4c6887",
    graphControl: "#986113",
    graphLabel: "#203c56",
    graphLabelBg: "#ffffff",
    graphMuted: "#3c5c75",
    graphComputeBg: "#e6f7f2",
    graphComputeBorder: "#198b78",
    graphMemoryBg: "#f1eafa",
    graphMemoryBorder: "#8961b5",
    graphTensorBg: "#e8f1ff",
    graphTensorBorder: "#568bc6",
    graphTransferBg: "#eee8ff",
    graphTransferBorder: "#8a65c5",
    graphMask: "#f7faffb8",
    graphShadow: "#314b671a",
  },
};
export type Palette = typeof palettes.dark;
export function initialTheme(): Theme {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) === "light"
      ? "light"
      : "dark";
  } catch {
    return "dark";
  }
}
const ThemeContext = createContext<{ theme: Theme; toggle: () => void }>({
  theme: "dark",
  toggle: () => {},
});
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useLayoutEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    for (const [key, value] of Object.entries(palettes[theme])) {
      root.style.setProperty(
        "--viz-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase()),
        value,
      );
    }
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#0b0e14" : "#f3f6fa");
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      /* Switching still works when browser storage is unavailable. */
    }
  }, [theme]);
  return (
    <ThemeContext.Provider
      value={{
        theme,
        toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
}
export function useTheme() {
  return useContext(ThemeContext);
}
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const label =
    theme === "dark" ? "Switch to light theme" : "Switch to dark theme";
  return (
    <button
      type="button"
      className="theme-toggle"
      aria-label={label}
      aria-pressed={theme === "light"}
      title={label}
      onClick={toggle}
    >
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
      <span>{theme === "dark" ? "Light theme" : "Dark theme"}</span>
    </button>
  );
}
