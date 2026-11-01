import { CFG } from "./config";
import { LM, type Kp } from "./types";

export interface Box {
  x0: number;
  y0: number;
  x1: number; // exclusive
  y1: number; // exclusive
}

/** Bounding box of pixels > thr, or null if none. */
export function boundingBox(m: ArrayLike<number>, w: number, h: number, thr: number): Box | null {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (m[y * w + x] > thr) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1: x1 + 1, y1: y1 + 1 };
}

export function coverage(m: ArrayLike<number>, thr: number): number {
  let n = 0;
  for (let i = 0; i < m.length; i++) if (m[i] > thr) n++;
  return n / m.length;
}

export function meanLuma(gray: Float32Array): number {
  return gray.reduce((a, v) => a + v, 0) / gray.length;
}

const PAIRS = [
  ["shoulders", LM.lShoulder, LM.rShoulder],
  ["hips", LM.lHip, LM.rHip],
  ["knees", LM.lKnee, LM.rKnee],
  ["ankles", LM.lAnkle, LM.rAnkle],
] as const;

/**
 * Full body in frame: for each left/right pair the better-visible landmark must be seen and inside the frame.
 * Judging the pair (not both) keeps side and back poses valid, where the far side or the face is hidden.
 */
export function checkFullBody(k: Kp[]): string[] {
  const missing = PAIRS.filter(([, a, b]) => {
    const p = k[a].visibility >= k[b].visibility ? k[a] : k[b];
    return p.visibility < CFG.minBodyVisibility || p.x < 0.01 || p.x > 0.99 || p.y < 0.01 || p.y > 0.99;
  }).map(([name]) => name);
  return missing.length ? [`Can't see your ${missing.join(", ")} - step back until head to feet is in frame`] : [];
}

export function checkLighting(meanLumaValue: number): string[] {
  const [lo, hi] = CFG.luminance;
  if (meanLumaValue < lo) return ["Too dark - add light in front of you"];
  if (meanLumaValue > hi) return ["Too bright - avoid a bright window behind you"];
  return [];
}

/** Width (px) of the contiguous silhouette run through column `cx` on row `y`. */
function runWidth(mask: ArrayLike<number>, w: number, y: number, cx: number): number {
  const at = (x: number) => mask[y * w + x] > CFG.maskThreshold;
  if (!at(cx)) return 0;
  let l = cx, r = cx;
  while (l > 0 && at(l - 1)) l--;
  while (r < w - 1 && at(r + 1)) r++;
  return r - l + 1;
}

/** Warn-only: mid-torso silhouette much wider than the shoulder keypoints. Front view only. */
export function checkLooseClothing(prob: ArrayLike<number>, w: number, h: number, k: Kp[]): string[] {
  const [l, r] = [k[LM.lShoulder], k[LM.rShoulder]];
  const [hl, hr] = [k[LM.lHip], k[LM.rHip]];
  const row = Math.round(((l.y + r.y + hl.y + hr.y) / 4) * h);
  const col = Math.round(((l.x + r.x + hl.x + hr.x) / 4) * w);
  const shoulderPx = Math.abs(l.x - r.x) * w;
  if (shoulderPx <= 0 || row < 0 || row >= h) return [];
  return runWidth(prob, w, row, col) / shoulderPx > CFG.looseClothingRatio
    ? ["Loose clothing detected - results will be less accurate"]
    : [];
}

/** Non-empty = this view is not usable. `prob` is the raw mask at frame size. */
export function viewIssues(prob: ArrayLike<number>, w: number, h: number, k: Kp[]): string[] {
  const issues = checkFullBody(k);
  const cov = coverage(prob, CFG.maskThreshold);
  if (cov < CFG.coverage[0] || cov > CFG.coverage[1]) issues.push("Silhouette not found - check lighting and background");
  const edges = touchedEdges(prob, w, h);
  if (edges.length) issues.push(`Body touches the ${edges.join(" and ")} of the frame - step back`);
  return issues;
}

const EDGE_FRACTION = 0.03; // of the edge length: a cut-off limb is far wider than a stray blob

/** Which frame edges the silhouette runs into (cut-off feet, head or hands), with the pixel run found. */
export function touchedEdges(m: ArrayLike<number>, w: number, h: number): string[] {
  const edges: [string, (i: number) => number, number][] = [
    ["top", (x) => x, w],
    ["bottom", (x) => (h - 1) * w + x, w],
    ["left side", (y) => y * w, h],
    ["right side", (y) => y * w + w - 1, h],
  ];
  return edges.flatMap(([name, idx, n]) => {
    let count = 0;
    for (let i = 0; i < n; i++) if (m[idx(i)] > CFG.maskThreshold) count++;
    return count >= Math.max(3, Math.round(EDGE_FRACTION * n)) ? [`${name} (${count} px)`] : [];
  });
}
