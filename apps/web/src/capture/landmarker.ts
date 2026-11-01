import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Kp } from "./types";

export interface Detection {
  keypoints: Kp[];
  prob: Float32Array; // person probability per pixel, w x h
  w: number;
  h: number;
}

type Mode = "IMAGE" | "VIDEO";

let landmarker: Promise<PoseLandmarker> | undefined;
let mode: Mode = "VIDEO";

const create = async (delegate: "GPU" | "CPU") =>
  PoseLandmarker.createFromOptions(await FilesetResolver.forVisionTasks("/wasm"), {
    baseOptions: { modelAssetPath: "/models/pose_landmarker_lite.task", delegate },
    runningMode: mode,
    numPoses: 1,
    outputSegmentationMasks: true,
  });

/** Same-origin model + wasm only: nothing third-party is fetched. GPU first, CPU fallback (iOS Safari). */
export function getLandmarker(): Promise<PoseLandmarker> {
  landmarker ??= create("GPU")
    .catch(() => create("CPU"))
    .catch((e) => {
      landmarker = undefined;
      throw e;
    });
  return landmarker;
}

export async function detectPose(src: HTMLCanvasElement, runningMode: Mode, timestampMs = 0): Promise<Detection | null> {
  const lm = await getLandmarker();
  if (runningMode !== mode) {
    await lm.setOptions({ runningMode });
    mode = runningMode;
  }
  const r = runningMode === "VIDEO" ? lm.detectForVideo(src, timestampMs) : lm.detect(src);
  try {
    const pts = r.landmarks[0];
    const mask = r.segmentationMasks?.[0];
    if (!pts || !mask) return null;
    return {
      keypoints: pts.map(({ x, y, z, visibility }) => ({ x, y, z, visibility })),
      prob: mask.getAsFloat32Array().slice(), // the backing buffer is reused by the next frame
      w: mask.width,
      h: mask.height,
    };
  } finally {
    r.close();
  }
}
