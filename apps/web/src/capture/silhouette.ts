import { CFG } from "./config";
import { boundingBox } from "./quality";
import { LM, type Kp } from "./types";

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function bilinear(p: Float32Array, w: number, h: number, x: number, y: number): number {
  if (x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5) return 0;
  const x0 = Math.max(0, Math.floor(x)), y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = x - x0, fy = y - y0;
  return (
    p[y0 * w + x0] * (1 - fx) * (1 - fy) + p[y0 * w + x1] * fx * (1 - fy) +
    p[y1 * w + x0] * (1 - fx) * fy + p[y1 * w + x1] * fx * fy
  );
}

/**
 * Person-probability mask -> head-less binary silhouette (0/255, maskW x maskH) + keypoints in that frame.
 * Crop is taken from the FULL body (head included) so the head-top stays a height reference
 * in the keypoints; only then are the head pixels removed.
 */
export function buildSilhouette(prob: Float32Array, w: number, h: number, k: Kp[]) {
  const box = boundingBox(prob, w, h, CFG.maskThreshold);
  if (!box) return null;

  const pad = (box.y1 - box.y0) * CFG.bboxMargin;
  const [sx0, sy0] = [box.x0 - pad, box.y0 - pad];
  const [sw, sh] = [box.x1 - box.x0 + 2 * pad, box.y1 - box.y0 + 2 * pad];
  const scale = Math.min(CFG.maskW / sw, CFG.maskH / sh);
  const [offX, offY] = [(CFG.maskW - sw * scale) / 2, (CFG.maskH - sh * scale) / 2];
  const outY = (ny: number) => (ny * h - sy0) * scale + offY;

  const shoulderY = (k[LM.lShoulder].y + k[LM.rShoulder].y) / 2;
  const earY = (k[LM.lEar].y + k[LM.rEar].y) / 2;
  const cutRow = outY(shoulderY - CFG.headCut * Math.max(shoulderY - earY, 0));

  const mask = new Uint8Array(CFG.maskW * CFG.maskH);
  for (let oy = Math.max(0, Math.ceil(cutRow)); oy < CFG.maskH; oy++) {
    const sy = (oy + 0.5 - offY) / scale + sy0 - 0.5;
    for (let ox = 0; ox < CFG.maskW; ox++) {
      const sx = (ox + 0.5 - offX) / scale + sx0 - 0.5;
      if (bilinear(prob, w, h, sx, sy) > CFG.maskThreshold) mask[oy * CFG.maskW + ox] = 255;
    }
  }

  const keypoints = k.map((p) => ({
    ...p,
    x: clamp01(((p.x * w - sx0) * scale + offX) / CFG.maskW),
    y: clamp01(outY(p.y) / CFG.maskH),
  }));
  return { mask, keypoints };
}
