import { expect, test } from "@playwright/test";

test("theme persists and keeps hardware, DAG, editor and resource charts readable", async ({
  page,
}, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Switch to dark theme" }),
  ).toBeVisible();

  await page.goto("/new?template=fork_join");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  const source = await page.evaluate(() =>
    localStorage.getItem("heterosim-draft"),
  );
  await page
    .getByRole("button", { name: "Advanced sources", exact: true })
    .click();
  await expect(page.locator(".cm-theme-light .cm-editor")).toBeVisible();
  await expect(page.locator(".cm-editor")).toHaveCSS(
    "background-color",
    "rgb(247, 249, 252)",
  );
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator(".cm-theme-dark .cm-editor")).toBeVisible();
  await expect(page.locator(".cm-editor")).toHaveCSS(
    "background-color",
    "rgb(12, 19, 29)",
  );
  expect(
    await page.evaluate(() => localStorage.getItem("heterosim-draft")),
  ).toBe(source);

  await page
    .getByRole("button", { name: "Graph preview", exact: true })
    .click();
  await expect(page.locator(".react-flow")).toHaveClass(/dark/);
  await page.getByRole("button", { name: "Workload DAG", exact: true }).click();
  await expect(page.locator(".react-flow__edge-path").first()).toHaveCSS(
    "stroke",
    "rgb(173, 200, 223)",
  );
  await expect(page.locator(".react-flow__edge-path").first()).toHaveCSS(
    "stroke-width",
    "2px",
  );
  await expect(page.locator(".react-flow__edge-text").first()).toHaveCSS(
    "fill",
    "rgb(240, 246, 255)",
  );
  await expect(page.locator(".graph-node strong").first()).toHaveCSS(
    "font-size",
    "12px",
  );
  await expect(page.locator(".react-flow__minimap-node")).toHaveCount(3);
  await expect(page.locator(".react-flow__background")).toHaveCSS(
    "background-color",
    "rgb(12, 20, 32)",
  );
  await page.screenshot({
    path: testInfo.outputPath("dag-dark.png"),
    fullPage: true,
  });

  await page.locator(".react-flow__node").first().click();
  await expect(page.locator(".react-flow__node.selected")).toHaveCount(1);
  await expect(page.locator(".react-flow__node.selected")).toHaveCSS(
    "border-color",
    "rgb(248, 211, 120)",
  );
  const viewport = await page
    .locator(".react-flow__viewport")
    .getAttribute("style");
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  await expect(page.locator(".react-flow")).toHaveClass(/light/);
  await expect(page.locator(".react-flow__node.selected")).toHaveCount(1);
  await expect(page.locator(".react-flow__edge-path").first()).toHaveCSS(
    "stroke",
    "rgb(76, 104, 135)",
  );
  await expect(page.locator(".graph-canvas")).toHaveCSS(
    "background-color",
    "rgb(247, 250, 255)",
  );
  await expect(page.locator(".react-flow__background")).toHaveCSS(
    "background-color",
    "rgb(247, 250, 255)",
  );
  expect(
    await page.locator(".react-flow__viewport").getAttribute("style"),
  ).toBe(viewport);
  await page.screenshot({
    path: testInfo.outputPath("dag-light.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Hardware topology", exact: true })
    .click();
  await expect(page.locator(".channel-label").first()).toHaveCSS(
    "color",
    "rgb(32, 60, 86)",
  );

  // Use a real CLI/UI artifact when available. The test still works on an empty test workspace.
  let runs = await (await page.request.get("/api/runs")).json();
  if (!runs.some((r: { status: string }) => r.status === "SUCCESS")) {
    const draft = await (
      await page.request.post("/api/templates", {
        data: { template: "pipeline" },
      })
    ).json();
    draft.label = "Theme regression " + Date.now();
    const job = await (
      await page.request.post("/api/jobs", { data: draft })
    ).json();
    await expect
      .poll(
        async () =>
          (await (await page.request.get("/api/jobs/" + job.id)).json()).state,
      )
      .toBe("finished");
    runs = await (await page.request.get("/api/runs")).json();
  }
  const run = runs.find((r: { status: string }) => r.status === "SUCCESS");
  await page.goto("/runs/" + run.id);
  await page.getByRole("button", { name: "Resources", exact: true }).click();
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("resources-light.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByRole("button", { name: "Graphs", exact: true }).click();
  await page
    .getByRole("button", { name: "Execution DAG", exact: true })
    .click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await expect(page.locator(".react-flow")).toHaveClass(/dark/);
  await expect(
    page.getByRole("button", { name: "Readable view", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Fit all", exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("button", { name: "Switch to light theme" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Switch to light theme" }).click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.getByRole("button", { name: "Fit all", exact: true }).click();
  await expect
    .poll(async () =>
      Number(
        (
          await page.locator(".react-flow__viewport").getAttribute("style")
        )?.match(/scale\(([^)]+)\)/)?.[1],
      ),
    )
    .toBeLessThan(0.8);
  await page
    .getByRole("button", { name: "Readable view", exact: true })
    .click();
  await expect
    .poll(async () =>
      Number(
        (
          await page.locator(".react-flow__viewport").getAttribute("style")
        )?.match(/scale\(([^)]+)\)/)?.[1],
      ),
    )
    .toBeGreaterThanOrEqual(0.8);
  await page.locator(".graph-canvas").scrollIntoViewIfNeeded();
  await expect(page.locator(".react-flow__node").first()).toBeInViewport();
  await page.screenshot({
    path: testInfo.outputPath("theme-mobile.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});
