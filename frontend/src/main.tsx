import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Activity,
  ArrowUpRight,
  FlaskConical,
  LayoutDashboard,
  Plus,
  SquareStack,
  Terminal,
} from "lucide-react";
const Experiments = lazy(() =>
  import("./Experiments").then((m) => ({ default: m.Experiments })),
);
const NewExperiment = lazy(() =>
  import("./NewExperiment").then((m) => ({ default: m.NewExperiment })),
);
const RunDetail = lazy(() =>
  import("./RunDetail").then((m) => ({ default: m.RunDetail })),
);
const Compare = lazy(() =>
  import("./Compare").then((m) => ({ default: m.Compare })),
);
import "@fontsource/dm-sans/latin-400.css";
import "@fontsource/dm-sans/latin-500.css";
import "@fontsource/dm-sans/latin-600.css";
import "@fontsource/jetbrains-mono/latin-400.css";
import "./style.css";
import { ThemeProvider, ThemeToggle } from "./theme";
const client = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 1500 } },
});
function App() {
  return (
    <BrowserRouter>
      <div className="shell">
        <aside className="sidebar">
          <NavLink to="/" className="brand">
            <div className="brand-mark">
              <SquareStack size={23} />
            </div>
            <div>
              heterosim<span>EXPERIMENT WORKBENCH</span>
            </div>
          </NavLink>
          <div className="workspace-tag">
            <span className="live-dot" /> Local workspace <span>v0.1</span>
          </div>
          <div className="nav-label">WORKSPACE</div>
          <nav>
            <NavLink to="/" end>
              <LayoutDashboard size={17} /> Experiments
            </NavLink>
            <NavLink to="/new">
              <Plus size={18} /> New experiment
            </NavLink>
            <NavLink to="/compare">
              <FlaskConical size={17} /> Compare runs
            </NavLink>
          </nav>
          <div className="sidebar-bottom">
            <div className="engine-status">
              <Activity size={17} />
              <div>
                Discrete-event engine
                <small>Simulation time. Explicit resources.</small>
              </div>
            </div>
            <a href="/docs" target="_blank" rel="noreferrer">
              <Terminal size={15} /> API reference <ArrowUpRight size={14} />
            </a>
            <div className="local-label">
              <span className="live-dot" /> Runs stay on this machine
            </div>
          </div>
        </aside>
        <div className="workspace">
          <header className="topbar">
            <span>
              <span className="topbar-project">Distributed LLM Simulation</span>
              <span className="breadcrumb">/</span>Workbench
            </span>
            <div>
              <ThemeToggle />
              <span className="live-dot" />
              <span>LOCAL SESSION</span>
              <div className="avatar">HS</div>
            </div>
          </header>
          <main>
            <Suspense
              fallback={<div className="loading">Loading workspace…</div>}
            >
              <Routes>
                <Route path="/" element={<Experiments />} />
                <Route path="/new" element={<NewExperiment />} />
                <Route path="/runs/:id" element={<RunDetail />} />
                <Route path="/compare" element={<Compare />} />
                <Route
                  path="*"
                  element={
                    <div className="empty">
                      <h1>Page not found</h1>
                      <NavLink to="/">Return to experiments</NavLink>
                    </div>
                  }
                />
              </Routes>
            </Suspense>
            <footer className="page-footer">
              <span>HETEROSIM · REPRODUCIBLE BY DESIGN</span>
              <span>Resource graphs → execution evidence</span>
            </footer>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
);
