export const VIEWS = ["front", "left", "back", "right"] as const;
export type ViewName = (typeof VIEWS)[number];

export interface Kp {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

// MediaPipe Pose Landmarker indices
export const LM = {
  nose: 0,
  lEar: 7,
  rEar: 8,
  lShoulder: 11,
  rShoulder: 12,
  lIndex: 19,
  rIndex: 20,
  lHip: 23,
  rHip: 24,
  lKnee: 25,
  rKnee: 26,
  lAnkle: 27,
  rAnkle: 28,
} as const;

/** Only masks + keypoints; the RGB frame lives in rgbStore (memory only). */
export interface ViewCapture {
  name: ViewName;
  maskPngB64: string;
  keypoints: Kp[];
  sharpness: number;
  issues: string[]; // non-empty = invalid view
  warnings: string[];
  torsoSkinShare?: number | null; // front view only: how bare the torso looks, 0-1; null = could not tell. Never sent.
}
