import { CFG } from "./config";
import { LM, type Kp } from "./types";
import type { Scope } from "./scope";

interface Frame {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

const chroma = (r: number, g: number, b: number): [number, number] => {
  const s = r + g + b + 1e-6;
  return [r / s, g / s]; // brightness drops out, so shading does not read as a different colour
};
const median = (v: number[]) => [...v].sort((a, b) => a - b)[v.length >> 1];

/** The person's own exposed skin: median colour in small discs at the nose and both index fingers. null with fewer than two. */
function skinReference(img: Frame, k: Kp[]): [number, number, number] | null {
  const r = Math.max(2, Math.round(0.015 * img.height));
  const px: number[][] = [[], [], []];
  let points = 0;
  for (const i of [LM.nose, LM.lIndex, LM.rIndex]) {
    const cx = k[i].x * img.width, cy = k[i].y * img.height;
    if (k[i].visibility < CFG.minVisibility || cx < r || cy < r || cx >= img.width - r || cy >= img.height - r) continue;
    points++;
    for (let y = Math.round(cy) - r; y <= Math.round(cy) + r; y++) {
      for (let x = Math.round(cx) - r; x <= Math.round(cx) + r; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 > r * r) continue;
        for (let c = 0; c < 3; c++) px[c].push(img.data[(y * img.width + x) * 4 + c]);
      }
    }
  }
  return points >= 2 ? [median(px[0]), median(px[1]), median(px[2])] : null;
}

/**
 * How bare the torso looks (front view): the share of torso pixels whose colour is close to the person's own skin
 * (face and hands, so it needs no assumption about skin tone). The torso is the shoulder-hip quadrilateral without
 * the neck, armpits and waistband. null when the reference or the torso cannot be read. A starting guess like the other
 * checks in CFG, not validated: it can be fooled by a shirt close to the skin colour, so it only warns.
 */
export function torsoSkinShare(img: Frame, prob: Float32Array, k: Kp[]): number | null {
  const corners = [LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip];
  if (corners.some((i) => k[i].visibility < CFG.minVisibility)) return null;
  const ref = skinReference(img, k);
  if (!ref) return null;
  const [rr, rg] = chroma(...ref);

  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  let near = 0, total = 0;
  for (let t = 0.2; t <= 0.7; t += 0.05) {
    for (let u = 0.15; u <= 0.85; u += 0.05) {
      const at = (top: number, bottom: number, side: "x" | "y") => lerp(k[top][side], k[bottom][side], t);
      const [lx, ly, rx, ry] = [at(LM.lShoulder, LM.lHip, "x"), at(LM.lShoulder, LM.lHip, "y"), at(LM.rShoulder, LM.rHip, "x"), at(LM.rShoulder, LM.rHip, "y")];
      const x = Math.round(lerp(lx, rx, u) * img.width), y = Math.round(lerp(ly, ry, u) * img.height);
      if (x < 0 || y < 0 || x >= img.width || y >= img.height || prob[y * img.width + x] <= CFG.maskThreshold) continue;
      const o = (y * img.width + x) * 4;
      const [cr, cg] = chroma(img.data[o], img.data[o + 1], img.data[o + 2]);
      total++;
      if (Math.hypot(cr - rr, cg - rg) <= CFG.torsoSkinChroma) near++;
    }
  }
  return total >= 20 ? near / total : null;
}

/** The chosen scope asks for a bare torso (or a sports bra) and the front view looks covered, and the user has not said otherwise. */
export const torsoLooksCovered = (scope: Scope | undefined, share: number | null | undefined, confirmedBare = false) =>
  scope === "full_body" && !confirmedBare && share != null && share < CFG.torsoBareMin;
