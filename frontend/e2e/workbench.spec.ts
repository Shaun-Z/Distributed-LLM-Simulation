import { test, expect } from "@playwright/test";
const suffix = Date.now().toString(36);
const labels = {
  pipeline: "Browser pipeline " + suffix,
  duplicate: "Browser duplicate " + suffix,
  transformer: "Browser transformer " + suffix,
  oom: "Browser OOM " + suffix,
  cancel: "Browser cancellation " + suffix,
};

test("configure, run, inspect, duplicate, compare and render responsive layouts", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", {
      name: "A clearer view of distributed compute.",
    }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "New experiment", exact: true })
    .first()
    .click();
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page.getByLabel("Experiment name").fill(labels.pipeline);
  await page
    .getByLabel("Capacity hbm_capacity", { exact: true })
    .fill("32000000");
  await page.getByLabel("Capacity hbm_capacity", { exact: true }).press("Tab");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Placement", exact: true }).click();
  await page.getByLabel("Filter operators").fill("r0/A");
  await expect(page.getByLabel("Device r0/A", { exact: true })).toHaveValue(
    "cpu0",
  );
  await page
    .getByRole("button", { name: "Advanced sources", exact: true })
    .click();
  await expect(page.locator(".cm-content")).toContainText(
    "synthetic_pipeline_oracle",
  );
  const original = await page.evaluate(
    () =>
      JSON.parse(localStorage.getItem("heterosim-draft")!).hardware as string,
  );
  await page.locator(".cm-content").fill("{invalid");
  await page
    .getByRole("button", { name: "Validate draft", exact: true })
    .click();
  await expect(page.getByRole("alert")).toBeVisible();
  await page.locator(".cm-content").fill(original);
  await page
    .getByRole("button", { name: "Validate draft", exact: true })
    .click();
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Graph preview", exact: true })
    .click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await page
    .getByRole("button", { name: "Run experiment", exact: true })
    .click();
  const row = page.getByRole("row").filter({
    has: page.getByRole("link", { name: labels.pipeline, exact: true }),
  });
  await expect(row).toContainText("SUCCESS");
  await row.getByRole("link", { name: "Explore" }).click();
  await expect(page.getByText("9.1 ms", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Graphs", exact: true }).click();
  await expect(page.locator(".react-flow__node").first()).toBeVisible();
  await page.locator(".react-flow__node").first().click();
  await expect(page.getByText("Selected node")).toBeVisible();
  await page.getByRole("button", { name: "Timeline", exact: true }).click();
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByLabel("Filter activities").fill("r0/");
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.getByRole("button", { name: "Play replay", exact: true }).click();
  await page.getByRole("button", { name: "Pause replay", exact: true }).click();
  await page.getByRole("button", { name: "Resources", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Memory residency", exact: true }),
  ).toBeVisible();
  await expect(page.locator("canvas").first()).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("resources-desktop.png"),
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Trace & diagnostics", exact: true })
    .click();
  await page.getByLabel("Event type").selectOption("WAITING");
  await expect(
    page.getByRole("heading", { name: "Event trace" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Duplicate configuration", exact: true })
    .click();
  await page.getByLabel("Experiment name").fill(labels.duplicate);
  await page
    .getByRole("button", { name: "Run experiment", exact: true })
    .click();
  const duplicate = page.getByRole("row").filter({
    has: page.getByRole("link", { name: labels.duplicate, exact: true }),
  });
  await expect(duplicate).toContainText("SUCCESS");
  await page
    .getByRole("checkbox", { name: "Compare " + labels.pipeline, exact: true })
    .check();
  await page
    .getByRole("checkbox", { name: "Compare " + labels.duplicate, exact: true })
    .check();
  await page
    .getByRole("button", { name: "Compare (2/4)", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Outcome comparison", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Baseline", { exact: true }).first(),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("compare-desktop.png"),
    fullPage: true,
  });
  await page.goto("/");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", {
      name: "A clearer view of distributed compute.",
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: testInfo.outputPath("mobile.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({
    path: testInfo.outputPath("experiments-desktop.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("run transformer and inspect a real capacity failure", async ({
  page,
}) => {
  await page.goto("/new?template=transformer");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page.getByLabel("Experiment name").fill(labels.transformer);
  await page
    .getByRole("button", { name: "Run experiment", exact: true })
    .click();
  const row = page.getByRole("row").filter({
    has: page.getByRole("link", { name: labels.transformer, exact: true }),
  });
  await expect(row).toContainText("SUCCESS");
  await row.getByRole("link", { name: "Explore" }).click();
  await expect(
    page.getByRole("heading", { name: "Decoder performance", exact: true }),
  ).toBeVisible();
  await page.goto("/new?template=pipeline");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page.getByLabel("Experiment name").fill(labels.oom);
  await page.getByLabel("Capacity ram_capacity", { exact: true }).fill("0");
  await page.getByLabel("Capacity ram_capacity", { exact: true }).press("Tab");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page
    .getByRole("button", { name: "Run experiment", exact: true })
    .click();
  const failed = page.getByRole("row").filter({
    has: page.getByRole("link", { name: labels.oom, exact: true }),
  });
  await expect(failed).toContainText("NO_PROGRESS");
  await failed.getByRole("link", { name: "Explore" }).click();
  await expect(
    page.getByRole("heading", { name: "Execution diagnostics", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("INITIAL_CAPACITY_BLOCKED", { exact: true }),
  ).toBeVisible();
});

test("cancel an actual worker from the interface", async ({ page }) => {
  await page.goto("/new?template=pipeline");
  await expect(page.getByText("Validated", { exact: true })).toBeVisible();
  await page.getByLabel("Experiment name").fill(labels.cancel);
  // A long body is model time only; enough operators create a cancellable worker.
  await page
    .getByRole("button", { name: "Advanced sources", exact: true })
    .click();
  const draft = await page.request.post("/api/templates", {
    data: { template: "pipeline", parameters: { requests: 700 } },
  });
  const body = await draft.json();
  body.label = labels.cancel;
  await page.request.post("/api/jobs", { data: body });
  await page.goto("/");
  await expect(
    page.getByRole("button", {
      name: "Cancel " + labels.cancel,
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Cancel " + labels.cancel, exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: "Cancel " + labels.cancel,
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(
    page.getByText("cancelled", { exact: true }).first(),
  ).toBeVisible();
});
