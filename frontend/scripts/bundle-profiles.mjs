import { cp, mkdir, rm } from "node:fs/promises";
const target = new URL(
  "../../src/distributed_llm_simulation/web/data/hardware/",
  import.meta.url,
);
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
await cp(new URL("../../profiles/hardware/", import.meta.url), target, {
  recursive: true,
});
