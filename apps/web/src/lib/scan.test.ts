import { describe, expect, it } from "vitest";
import { pose } from "../capture/test-utils";
import type { ViewCapture, ViewName } from "../capture/types";
import { buildScan } from "./scan";

const view = (name: ViewName, size = 1000): ViewCapture => ({
  name,
  maskPngB64: "A".repeat(size),
  keypoints: pose(),
  sharpness: 1,
  issues: [],
  warnings: [],
});

describe("buildScan", () => {
  it("accepts front + left, with or without back/right", () => {
    expect(Object.keys(buildScan([view("front"), view("left")]).views)).toEqual(["front", "left"]);
    expect(Object.keys(buildScan(["front", "left", "back", "right"].map((n) => view(n as ViewName))).views)).toHaveLength(4);
  });

  it("rejects a missing required view", () => {
    expect(() => buildScan([view("front")])).toThrow("left");
    expect(() => buildScan([view("left")])).toThrow("front");
  });

  it("rejects an oversized view, an oversized total, and wrong keypoint count", () => {
    expect(() => buildScan([view("front", 350_001), view("left")])).toThrow("too large");
    expect(() => buildScan([view("front", 300_000), view("left", 300_000), view("back", 300_000)])).toThrow("512 KB");
    expect(() => buildScan([{ ...view("front"), keypoints: pose().slice(1) }, view("left")])).toThrow("33");
  });
});
