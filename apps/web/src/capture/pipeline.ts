import { CFG } from "./config";
import { maskToPngB64 } from "./encode";
import { detectPose, type Detection } from "./landmarker";
import { checkLighting, checkLooseClothing, meanLuma, viewIssues } from "./quality";
import { rgbStore } from "./rgbStore";
import { buildSilhouette } from "./silhouette";
import { laplacianVariance, toGray } from "./sharpness";
import type { ViewCapture, ViewName } from "./types";

export interface Candidate {
  det: Detection;
  img: ImageData; // RGB: memory only, handed to rgbStore for the definition model
  gray: Float32Array;
  sharpness: number;
}

const scratch = document.createElement("canvas");

/** Draw into one shared canvas, long side capped at CFG.frameMaxSide. */
export function drawScaled(src: CanvasImageSource, sw: number, sh: number): HTMLCanvasElement {
  const s = Math.min(1, CFG.frameMaxSide / Math.max(sw, sh));
  scratch.width = Math.round(sw * s);
  scratch.height = Math.round(sh * s);
  scratch.getContext("2d", { willReadFrequently: true })!.drawImage(src, 0, 0, scratch.width, scratch.height);
  return scratch;
}

export function scoreFrame(canvas: HTMLCanvasElement, det: Detection): Candidate {
  const img = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
  const gray = toGray(img.data);
  const body = Uint8Array.from(det.prob, (v) => (v > CFG.maskThreshold ? 1 : 0));
  return { det, img, gray, sharpness: laplacianVariance(gray, canvas.width, canvas.height, body) };
}

/** Silhouette + checks for the chosen frame; also parks its RGB in rgbStore. */
export function finalizeView(name: ViewName, c: Candidate): ViewCapture {
  const { det } = c;
  const issues = viewIssues(det.prob, det.w, det.h, det.keypoints);
  const sil = buildSilhouette(det.prob, det.w, det.h, det.keypoints);
  rgbStore.set(name, c.img);
  if (!sil) return { name, maskPngB64: "", keypoints: det.keypoints, sharpness: c.sharpness, issues: [...issues, "No silhouette found"], warnings: [] };
  return {
    name,
    maskPngB64: maskToPngB64(sil.mask, CFG.maskW, CFG.maskH),
    keypoints: sil.keypoints,
    sharpness: c.sharpness,
    issues,
    warnings: [
      ...checkLighting(meanLuma(c.gray)),
      ...(name === "front" ? checkLooseClothing(det.prob, det.w, det.h, det.keypoints) : []),
    ],
  };
}

/** One still image -> one view (file upload path; also reused for the goal photo in P7). */
export async function processUpload(file: File, name: ViewName): Promise<ViewCapture | null> {
  const bmp = await createImageBitmap(file);
  const canvas = drawScaled(bmp, bmp.width, bmp.height);
  bmp.close();
  const det = await detectPose(canvas, "IMAGE");
  return det ? finalizeView(name, scoreFrame(canvas, det)) : null;
}

const GRAY = 0xffebebeb;
const DARK = 0xff181818;

/** Live preview: the silhouette by default, the camera frame only when the user asks. */
export function paintPreview(target: HTMLCanvasElement, frame: HTMLCanvasElement, det: Detection | null, showFeed: boolean) {
  target.width = frame.width;
  target.height = frame.height;
  const ctx = target.getContext("2d")!;
  if (showFeed) return void ctx.drawImage(frame, 0, 0);
  const img = ctx.createImageData(frame.width, frame.height);
  const px = new Uint32Array(img.data.buffer);
  for (let i = 0; i < px.length; i++) px[i] = det && det.prob[i] > CFG.maskThreshold ? GRAY : DARK;
  ctx.putImageData(img, 0, 0);
}
