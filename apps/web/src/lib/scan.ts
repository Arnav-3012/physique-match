import type { Scan, ViewData } from "./types.gen";
import type { ViewCapture } from "../capture/types";

// Mirrors packages/schemas MAX_VIEW_B64 / MAX_PAYLOAD_B64
const MAX_VIEW_B64 = 350_000;
const MAX_PAYLOAD_B64 = 700_000;

/** Throws if the views would be rejected by the API's `Scan` schema. */
export function buildScan(views: ViewCapture[]): Scan {
  const byName = Object.fromEntries(views.map((v) => [v.name, v]));
  for (const need of ["front", "left"]) if (!byName[need]) throw new Error(`missing required view: ${need}`);

  let total = 0;
  const out: Record<string, ViewData> = {};
  for (const v of views) {
    if (v.keypoints.length !== 33) throw new Error(`${v.name}: expected 33 keypoints`);
    if (v.maskPngB64.length > MAX_VIEW_B64) throw new Error(`${v.name}: mask too large`);
    total += v.maskPngB64.length;
    out[v.name] = { mask_png_b64: v.maskPngB64, keypoints: v.keypoints as ViewData["keypoints"] };
  }
  if (total > MAX_PAYLOAD_B64) throw new Error("scan payload exceeds 512 KB");
  return { views: out };
}
