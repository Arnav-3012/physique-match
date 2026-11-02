import { describe, expect, it } from "vitest";
import { CFG } from "./config";
import { torsoLooksCovered, torsoSkinShare } from "./torso";
import { LM, type Kp } from "./types";

const W = 200, H = 400;
const SKIN = [200, 150, 120], SHIRT = [90, 90, 100];

/** Skin-coloured frame with a shirt over the torso rectangle (or none), keypoints for a front-facing person. */
function scene(shirt: number[] | null, visibleRef = true) {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const torso = x >= 60 && x < 140 && y >= 100 && y < 260;
      data.set([...(shirt && torso ? shirt : SKIN), 255], (y * W + x) * 4);
    }
  }
  const k: Kp[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  const put = (i: number, x: number, y: number) => (k[i] = { x, y, z: 0, visibility: 1 });
  put(LM.nose, 0.5, 0.08);
  put(LM.lShoulder, 0.35, 0.25);
  put(LM.rShoulder, 0.65, 0.25);
  put(LM.lHip, 0.4, 0.6);
  put(LM.rHip, 0.6, 0.6);
  put(LM.lIndex, 0.18, 0.52);
  put(LM.rIndex, 0.82, 0.52);
  if (!visibleRef) for (const i of [LM.nose, LM.lIndex, LM.rIndex]) k[i].visibility = 0;
  return { img: { data, width: W, height: H }, prob: new Float32Array(W * H).fill(1), k };
}

describe("torsoSkinShare: does the torso look bare, judged against the person's own skin", () => {
  it("is high for a bare torso", () => {
    const s = scene(null);
    expect(torsoSkinShare(s.img, s.prob, s.k)!).toBeGreaterThan(0.9);
  });

  it("is low for a shirt of another colour", () => {
    const s = scene(SHIRT);
    expect(torsoSkinShare(s.img, s.prob, s.k)!).toBeLessThan(0.1);
  });

  it("does not depend on the skin tone: a darker person with a bare torso still reads bare", () => {
    const s = scene(null);
    for (let i = 0; i < s.img.data.length; i += 4) for (let c = 0; c < 3; c++) s.img.data[i + c] *= 0.4;
    expect(torsoSkinShare(s.img, s.prob, s.k)!).toBeGreaterThan(0.9);
  });

  it("returns null, so no warning, when no skin reference or no torso can be read", () => {
    const s = scene(SHIRT, false);
    expect(torsoSkinShare(s.img, s.prob, s.k)).toBeNull();
    const t = scene(SHIRT);
    t.k[LM.lHip].visibility = 0;
    expect(torsoSkinShare(t.img, t.prob, t.k)).toBeNull();
    expect(torsoSkinShare(t.img, new Float32Array(W * H), scene(null).k)).toBeNull(); // no person pixels
  });
});

describe("torsoLooksCovered: only the full-body scope asks for a bare torso", () => {
  const covered = CFG.torsoBareMin - 0.1;
  it("warns for full body with a low share", () => {
    expect(torsoLooksCovered("full_body", covered)).toBe(true);
  });
  it("never warns for the other scopes, an unreadable share, a bare torso, or after the user confirms", () => {
    expect(torsoLooksCovered("arms_shoulders_legs", covered)).toBe(false);
    expect(torsoLooksCovered("none", covered)).toBe(false);
    expect(torsoLooksCovered("full_body", null)).toBe(false);
    expect(torsoLooksCovered("full_body", undefined)).toBe(false);
    expect(torsoLooksCovered("full_body", CFG.torsoBareMin + 0.1)).toBe(false);
    expect(torsoLooksCovered("full_body", covered, true)).toBe(false);
  });
});
