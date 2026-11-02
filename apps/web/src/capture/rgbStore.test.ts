import { afterEach, describe, expect, it, vi } from "vitest";
import { rgbStore } from "./rgbStore";

const frame = { data: new Uint8ClampedArray(4), width: 1, height: 1 } as unknown as ImageData;

afterEach(() => {
  rgbStore.enable(false);
  vi.unstubAllGlobals();
});

describe("rgbStore frame lifecycle (ADR-051)", () => {
  it("stores nothing while the definition model is off (the default)", () => {
    rgbStore.set("front", frame);
    expect(rgbStore.get("front")).toBeUndefined();
  });

  it("keeps a frame only while enabled, and clear releases it and blanks the scratch canvas", () => {
    const scratch = { width: 384, height: 512 };
    rgbStore.attachScratch(scratch);
    rgbStore.enable(true);
    rgbStore.set("front", frame);
    expect(rgbStore.get("front")).toBe(frame);

    rgbStore.clear();
    expect(rgbStore.get("front")).toBeUndefined();
    expect(scratch).toEqual({ width: 0, height: 0 });
  });

  it("releaseScratch blanks the last drawn frame without touching stored frames", () => {
    const scratch = { width: 384, height: 512 };
    rgbStore.attachScratch(scratch);
    rgbStore.enable(true);
    rgbStore.set("left", frame);
    rgbStore.releaseScratch();
    expect(scratch).toEqual({ width: 0, height: 0 });
    expect(rgbStore.get("left")).toBe(frame);
  });

  it("keeps the person mask with its frame, and clear drops both", () => {
    const mask = new Float32Array(1);
    rgbStore.enable(true);
    rgbStore.set("front", frame, mask);
    expect(rgbStore.getMask("front")).toBe(mask);
    rgbStore.clear();
    expect(rgbStore.getMask("front")).toBeUndefined();
  });

  it("stores no mask while the model is off", () => {
    rgbStore.set("front", frame, new Float32Array(1));
    expect(rgbStore.getMask("front")).toBeUndefined();
  });

  it("turning the model off drops every frame", () => {
    rgbStore.enable(true);
    rgbStore.set("back", frame);
    rgbStore.enable(false);
    expect(rgbStore.get("back")).toBeUndefined();
  });
});

describe("storage check: a mocked capture-and-process cycle writes no image data anywhere persistent", () => {
  it("touches no localStorage, sessionStorage, IndexedDB or Cache API", () => {
    const touched: string[] = [];
    const spy = (name: string) =>
      new Proxy({}, { get: (_t, prop) => (..._a: unknown[]) => void touched.push(`${name}.${String(prop)}`) });
    for (const name of ["localStorage", "sessionStorage", "indexedDB", "caches"]) vi.stubGlobal(name, spy(name));

    const scratch = { width: 384, height: 512 };
    rgbStore.attachScratch(scratch);
    rgbStore.enable(true); // the strictest case: frames held for the on-device model
    for (const view of ["front", "left", "back", "right"] as const) rgbStore.set(view, frame);
    rgbStore.releaseScratch();
    rgbStore.clear();

    expect(touched).toEqual([]);
    expect(rgbStore.get("front")).toBeUndefined();
  });
});
