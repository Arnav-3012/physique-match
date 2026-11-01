import { expect, it } from "vitest";
import { laplacianVariance, toGray } from "./sharpness";

const N = 16;
const checker = Float32Array.from({ length: N * N }, (_, i) => ((i % N) + Math.floor(i / N)) % 2 === 0 ? 255 : 0);
const flat = new Float32Array(N * N).fill(128);

it("scores a sharp pattern above a flat one", () => {
  expect(laplacianVariance(flat, N, N)).toBe(0);
  expect(laplacianVariance(checker, N, N)).toBeGreaterThan(1000);
});

it("restricts to the mask and handles an empty one", () => {
  expect(laplacianVariance(checker, N, N, new Uint8Array(N * N))).toBe(0);
});

it("converts RGBA to luma", () => {
  expect(toGray(new Uint8ClampedArray([255, 255, 255, 255]))[0]).toBeCloseTo(255, 3);
});
