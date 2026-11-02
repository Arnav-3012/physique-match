import { afterEach, describe, expect, it, vi } from "vitest";
import { rgbStore } from "../capture/rgbStore";
import { bandOf } from "./bands";
import { frontTensor, INPUT_H, INPUT_W, type Frame } from "./preprocess";
import { classOf, scoreFrontAndClear, scoreTensor, type Runners } from "./score";

/** A w x h grey frame with a solid red figure in the rectangle [x0, x1) x [y0, y1) and its person mask. */
function figure(w: number, h: number, [x0, y0, x1, y1]: number[]) {
  const data = new Uint8ClampedArray(w * h * 4);
  const prob = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const on = x >= x0 && x < x1 && y >= y0 && y < y1;
      data.set(on ? [255, 0, 0, 255] : [10, 200, 10, 255], (y * w + x) * 4);
      prob[y * w + x] = on ? 1 : 0;
    }
  }
  return { img: { data, width: w, height: h } as Frame, prob };
}
const R = (t: Float32Array, y: number, x: number) => t[y * INPUT_W + x];

afterEach(() => rgbStore.enable(false));

describe("frontTensor: the same framing as the training renders", () => {
  const { img, prob } = figure(100, 200, [40, 10, 60, 190]);
  const t = frontTensor(img, prob)!;

  it("is a CHW float tensor in [0, 1]", () => {
    expect(t).toHaveLength(3 * INPUT_H * INPUT_W);
    expect(t.every((v) => v >= 0 && v <= 1)).toBe(true);
  });

  it("puts the body at about 91% of the height with a 5% margin, like the renders", () => {
    // 180 px body + 9 px margin each side -> scale 256 / 198; the body spans rows 12..243 (hand-computed)
    const rows = [...Array(INPUT_H).keys()].filter((y) => R(t, y, INPUT_W / 2) > 0.9);
    expect(rows[0]).toBeGreaterThanOrEqual(10);
    expect(rows[0]).toBeLessThanOrEqual(14);
    expect(rows.at(-1)).toBeGreaterThanOrEqual(241);
    expect(rows.at(-1)).toBeLessThanOrEqual(245);
  });

  it("replaces everything outside the person with the grey the models were scored on", () => {
    const plane = INPUT_H * INPUT_W;
    for (const c of [0, 1, 2]) expect(t[c * plane]).toBeCloseTo(128 / 255, 5); // top-left corner
    const mid = (INPUT_H / 2) * INPUT_W + INPUT_W / 2;
    expect([t[mid], t[plane + mid], t[2 * plane + mid]]).toEqual([1, 0, 0]); // the person's red, untouched
  });

  it("returns null when the mask finds no person, and refuses a mask of another size", () => {
    expect(frontTensor(img, new Float32Array(100 * 200))).toBeNull();
    expect(() => frontTensor(img, new Float32Array(10))).toThrow(/differ in size/);
  });

  it("does not depend on where the person stands or how large the frame is", () => {
    const moved = figure(300, 400, [140, 20, 160, 380]); // same shape rule, other place and scale
    const t2 = frontTensor(moved.img, moved.prob)!;
    expect(R(t2, 20, INPUT_W / 2)).toBe(R(t, 20, INPUT_W / 2));
    expect(R(t2, 128, INPUT_W / 2)).toBe(1);
  });
});

describe("scoring maps each output to the right model", () => {
  const logits = (n: number) => Array.from({ length: 4 }, (_, i) => (i < n ? 1 : -1));
  const runners = (def: number, adip: number): Runners & { calls: Float32Array[] } => {
    const calls: Float32Array[] = [];
    return {
      calls,
      definition: async (x) => (calls.push(x), logits(def)),
      adiposity: async (x) => (calls.push(x), logits(adip)),
    };
  };

  it("CORAL class is 1 + the number of logits above 0", () => {
    expect([0, 1, 2, 3, 4].map((n) => classOf(logits(n)))).toEqual([1, 2, 3, 4, 5]);
  });

  it("definition comes from the definition runner (the 8-head model), adiposity from the adiposity runner (the 2-head model)", async () => {
    const r = runners(3, 1);
    expect(await scoreTensor(r, new Float32Array(3))).toEqual({ definition: 4, adiposity: 2 });
    const swapped = runners(1, 3);
    expect(await scoreTensor(swapped, new Float32Array(3))).toEqual({ definition: 2, adiposity: 4 });
  });

  it("both graphs read the same preprocessed tensor, and only two values exist to show", async () => {
    const r = runners(2, 2);
    const x = new Float32Array(3);
    const scores = await scoreTensor(r, x);
    expect(r.calls).toEqual([x, x]);
    expect(r.calls[0]).toBe(r.calls[1]);
    expect(Object.keys(scores)).toEqual(["definition", "adiposity"]); // no regional field can reach the UI or the payload
  });
});

describe("scoreFrontAndClear: the frame is released whatever happens (ADR-051)", () => {
  const store = () => {
    const { img, prob } = figure(100, 200, [40, 10, 60, 190]);
    rgbStore.enable(true);
    rgbStore.set("front", img as unknown as ImageData, prob);
  };
  const ok: Runners = { definition: async () => [1, 1, 1, -1], adiposity: async () => [-1, -1, -1, -1] };

  it("scores, then clears the frame and the mask", async () => {
    store();
    expect(await scoreFrontAndClear(async () => ok)).toEqual({ definition: 4, adiposity: 1 });
    expect(rgbStore.get("front")).toBeUndefined();
    expect(rgbStore.getMask("front")).toBeUndefined();
  });

  it("clears and returns null when the models fail to load", async () => {
    store();
    expect(await scoreFrontAndClear(async () => Promise.reject(new Error("no model")))).toBeNull();
    expect(rgbStore.get("front")).toBeUndefined();
  });

  it("returns null and never loads a model when the opt-in is off (nothing stored)", async () => {
    const load = vi.fn(async () => ok);
    rgbStore.enable(false);
    expect(await scoreFrontAndClear(load)).toBeNull();
    expect(load).not.toHaveBeenCalled();
  });
});

describe("storage check: scoring a frame persists no image data anywhere", () => {
  it("touches no localStorage, sessionStorage, IndexedDB or Cache API during a capture-and-score cycle", async () => {
    const touched: string[] = [];
    const spy = (name: string) =>
      new Proxy({}, { get: (_t, prop) => (..._a: unknown[]) => void touched.push(`${name}.${String(prop)}`) });
    for (const name of ["localStorage", "sessionStorage", "indexedDB", "caches"]) vi.stubGlobal(name, spy(name));

    const { img, prob } = figure(100, 200, [40, 10, 60, 190]);
    rgbStore.enable(true);
    rgbStore.set("front", img as unknown as ImageData, prob);
    const ok: Runners = { definition: async () => [1, -1, -1, -1], adiposity: async () => [1, 1, -1, -1] };
    expect(await scoreFrontAndClear(async () => ok)).toEqual({ definition: 2, adiposity: 3 });

    vi.unstubAllGlobals();
    expect(touched).toEqual([]);
    expect(rgbStore.get("front")).toBeUndefined();
  });
});

describe("bands", () => {
  it("low = 1-2, mid = 3, high = 4-5", () => {
    expect([1, 2, 3, 4, 5].map(bandOf)).toEqual(["low", "low", "mid", "high", "high"]);
  });
});
