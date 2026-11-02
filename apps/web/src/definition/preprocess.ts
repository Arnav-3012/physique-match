import { CFG } from "../capture/config";
import { boundingBox } from "../capture/quality";
import { bilinear, cropFit } from "../capture/silhouette";

/**
 * The one preprocessing pass for both definition models (ADR-053 Amendment 6): the front frame cropped like the
 * silhouette (same box and margin, so the framing matches the training renders), the person kept and everything else
 * set to the grey the models were scored on (ml/src/definition/data.py EVAL_BACKGROUND), resized to 192 x 256 and
 * scaled to [0, 1] as a CHW float tensor. ImageNet normalisation is inside the ONNX graphs.
 */
export const INPUT_W = 192;
export const INPUT_H = 256;
const BACKGROUND = 128;

export interface Frame {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

const px = new Float32Array(3);

/** Bilinear RGB at (x, y) into `px`; the caller has checked the point is inside the person mask. */
function sampleRgb(img: Frame, x: number, y: number) {
  const { width: w, height: h, data } = img;
  const x0 = Math.min(w - 1, Math.max(0, Math.floor(x))), y0 = Math.min(h - 1, Math.max(0, Math.floor(y)));
  const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
  const fx = Math.min(1, Math.max(0, x - x0)), fy = Math.min(1, Math.max(0, y - y0));
  for (let c = 0; c < 3; c++) {
    px[c] =
      data[(y0 * w + x0) * 4 + c] * (1 - fx) * (1 - fy) + data[(y0 * w + x1) * 4 + c] * fx * (1 - fy) +
      data[(y1 * w + x0) * 4 + c] * (1 - fx) * fy + data[(y1 * w + x1) * 4 + c] * fx * fy;
  }
}

/** null when the mask finds no person. `prob` must be the person mask of this same frame. */
export function frontTensor(img: Frame, prob: Float32Array): Float32Array | null {
  const { width: w, height: h } = img;
  if (prob.length !== w * h) throw new Error("the mask and the frame differ in size");
  const box = boundingBox(prob, w, h, CFG.maskThreshold);
  if (!box) return null;
  const { sx0, sy0, scale, offX, offY } = cropFit(box, INPUT_W, INPUT_H);

  // The frame is larger than the 192 x 256 target: average n x n sub-samples per output pixel, like the area-aware
  // bilinear downscale (PIL) the training images went through, so aliasing is not a train / inference mismatch.
  const n = Math.max(1, Math.min(4, Math.ceil(1 / scale)));
  const plane = INPUT_W * INPUT_H;
  const out = new Float32Array(3 * plane);
  const norm = n * n * 255;
  for (let oy = 0; oy < INPUT_H; oy++) {
    for (let ox = 0; ox < INPUT_W; ox++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const sx = (ox + (i + 0.5) / n - offX) / scale + sx0 - 0.5;
          const sy = (oy + (j + 0.5) / n - offY) / scale + sy0 - 0.5;
          if (bilinear(prob, w, h, sx, sy) > CFG.maskThreshold) {
            sampleRgb(img, sx, sy);
            r += px[0]; g += px[1]; b += px[2];
          } else {
            r += BACKGROUND; g += BACKGROUND; b += BACKGROUND;
          }
        }
      }
      const k = oy * INPUT_W + ox;
      out[k] = r / norm;
      out[plane + k] = g / norm;
      out[2 * plane + k] = b / norm;
    }
  }
  return out;
}
