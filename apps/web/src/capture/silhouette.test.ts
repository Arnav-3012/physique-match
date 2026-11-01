import { decode } from "fast-png";
import { expect, it } from "vitest";
import { CFG } from "./config";
import { maskToPngB64 } from "./encode";
import { pose } from "./test-utils";
import { buildSilhouette } from "./silhouette";

const W = 100, H = 200;
const person = new Float32Array(W * H);
for (let y = 20; y < 180; y++) for (let x = 40; x < 60; x++) person[y * W + x] = 1;

const rowHasMask = (m: Uint8Array, row: number) =>
  m.subarray(row * CFG.maskW, (row + 1) * CFG.maskW).some((v) => v === 255);

it("returns null when nothing is segmented", () => {
  expect(buildSilhouette(new Float32Array(W * H), W, H, pose())).toBeNull();
});

it("outputs a fixed-size {0,255} mask with the head removed", () => {
  const s = buildSilhouette(person, W, H, pose())!;
  expect(s.mask).toHaveLength(CFG.maskW * CFG.maskH);
  expect(new Set(s.mask)).toEqual(new Set([0, 255]));

  // cut row = shoulderY - 0.4 * (shoulderY - earY), in output coordinates
  const sh = s.keypoints[11].y * CFG.maskH;
  const ear = (s.keypoints[7].y + s.keypoints[8].y) / 2 * CFG.maskH;
  const cut = Math.round(sh - CFG.headCut * (sh - ear));
  expect(rowHasMask(s.mask, cut - 2)).toBe(false);
  expect(rowHasMask(s.mask, cut + 3)).toBe(true);
});

it("keeps keypoints in [0,1] and the head-top above the shoulders", () => {
  const s = buildSilhouette(person, W, H, pose({ 27: { x: 1.4, y: 1.2 } }))!;
  for (const p of s.keypoints) {
    expect(p.x).toBeGreaterThanOrEqual(0);
    expect(p.x).toBeLessThanOrEqual(1);
    expect(p.y).toBeGreaterThanOrEqual(0);
    expect(p.y).toBeLessThanOrEqual(1);
  }
  expect(s.keypoints[0].y).toBeLessThan(s.keypoints[11].y);
});

it("encodes to a 1-channel PNG that decodes back to the same {0,255} mask", () => {
  const s = buildSilhouette(person, W, H, pose())!;
  const bin = Uint8Array.from(atob(maskToPngB64(s.mask, CFG.maskW, CFG.maskH)), (c) => c.charCodeAt(0));
  const png = decode(bin);
  expect([png.width, png.height, png.channels, png.depth]).toEqual([CFG.maskW, CFG.maskH, 1, 8]);
  expect(Array.from(png.data as Uint8Array)).toEqual(Array.from(s.mask));
});
