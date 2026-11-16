import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Status, ErrorBox, Metric } from "./ui";
describe("observable result states", () => {
  it("distinguishes model failure from worker state", () => {
    render(
      <>
        <Status value="NO_PROGRESS" />
        <Status value="running" />
      </>,
    );
    expect(screen.getByText("NO_PROGRESS")).toHaveClass("warning");
    expect(screen.getByText("running")).toHaveClass("active");
  });
  it("surfaces diagnostics accessibly", () => {
    render(<ErrorBox error={new Error("NO_ROUTE at x")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("NO_ROUTE at x");
  });
  it("does not hide unavailable metrics", () => {
    render(<Metric label="LATENCY" value="N/A" />);
    expect(screen.getByText("N/A")).toBeVisible();
  });
});
