import type { ReactNode } from "react";
import { AlertCircle, LoaderCircle, ArrowUpRight, Check } from "lucide-react";
import { Link } from "react-router-dom";
export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return (
    <span className={"badge " + tone}>
      <i />
      {children}
    </span>
  );
}
export function Status({ value }: { value?: string }) {
  const tone =
    value === "SUCCESS" || value === "finished"
      ? "success"
      : value === "running" || value === "queued"
        ? "active"
        : value === "NO_PROGRESS" ||
            value === "TRUNCATED" ||
            value === "interrupted"
          ? "warning"
          : value === "failed" || value === "INVALID"
            ? "danger"
            : "neutral";
  return <Badge tone={tone}>{value ?? "Unavailable"}</Badge>;
}
export function Loading() {
  return (
    <div className="loading">
      <LoaderCircle className="spin" size={20} /> Loading experiment data…
    </div>
  );
}
export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div role="alert" className="error-box">
      <AlertCircle size={18} />
      <div>
        <strong>Something needs attention</strong>
        <pre>{error instanceof Error ? error.message : String(error)}</pre>
      </div>
    </div>
  );
}
export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">◇</div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={"card " + className}>
      {title && (
        <div className="card-head">
          <div>
            <h3>{title}</h3>
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}
export function PageTitle({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-title">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="page-actions">{actions}</div>
    </header>
  );
}
export function Metric({
  label,
  value,
  foot,
  tone,
}: {
  label: string;
  value: ReactNode;
  foot?: ReactNode;
  tone?: string;
}) {
  return (
    <div className={"metric " + (tone ?? "")}>
      <span>{label}</span>
      <strong>{value}</strong>
      {foot && <small>{foot}</small>}
    </div>
  );
}
export function JsonView({ value }: { value: unknown }) {
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>;
}
export function OpenRun({ id }: { id: string }) {
  return (
    <Link className="text-link" to={"/runs/" + id}>
      Explore <ArrowUpRight size={14} />
    </Link>
  );
}
export function Validated() {
  return (
    <span className="validated">
      <Check size={14} /> Validated
    </span>
  );
}
