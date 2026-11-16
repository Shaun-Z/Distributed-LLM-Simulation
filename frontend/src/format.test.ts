import { describe, expect, it } from "vitest";
import { duration, bytes, number, delta, percent } from "./format";
describe("scientific display semantics", () => {
  it("keeps missing values unavailable instead of inventing zero", () => {
    expect(duration(null)).toBe("N/A");
    expect(bytes(undefined)).toBe("N/A");
    expect(number(null)).toBe("N/A");
    expect(delta(undefined, 2)).toBe("N/A");
    expect(percent("")).toBe("N/A");
  });
  it("preserves meaningful scale and zero", () => {
    expect(duration(0.0061)).toBe("6.1 ms");
    expect(duration(0.000001)).toBe("1 µs");
    expect(duration(0)).toBe("0 s");
    expect(bytes(1024)).toBe("1 KiB");
    expect(bytes(0)).toBe("0 B");
    expect(delta(3, 2)).toBe("+50.0%");
    expect(delta(1, 0)).toBe("N/A");
  });
});
