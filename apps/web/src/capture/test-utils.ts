import { LM, type Kp } from "./types";

/** 33 keypoints, standing square to the camera, everything visible. Override by landmark index. */
export function pose(over: Record<number, Partial<Kp>> = {}): Kp[] {
  const k: Kp[] = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 1 }));
  const set = (i: number, p: Partial<Kp>) => Object.assign(k[i], p);
  set(LM.nose, { y: 0.15, z: -0.2 });
  set(LM.lEar, { x: 0.53, y: 0.17 });
  set(LM.rEar, { x: 0.47, y: 0.17 });
  set(LM.lShoulder, { x: 0.6, y: 0.3 });
  set(LM.rShoulder, { x: 0.4, y: 0.3 });
  set(LM.lHip, { x: 0.57, y: 0.6 });
  set(LM.rHip, { x: 0.43, y: 0.6 });
  set(LM.lKnee, { x: 0.57, y: 0.78 });
  set(LM.rKnee, { x: 0.43, y: 0.78 });
  set(LM.lAnkle, { x: 0.57, y: 0.95 });
  set(LM.rAnkle, { x: 0.43, y: 0.95 });
  for (const [i, p] of Object.entries(over)) set(Number(i), p);
  return k;
}

/** Seen from the camera's side: shoulders collapse onto each other, far shoulder less visible. */
export function sidePose(side: "left" | "right"): Kp[] {
  const near = side === "left" ? LM.lShoulder : LM.rShoulder;
  const far = side === "left" ? LM.rShoulder : LM.lShoulder;
  return pose({
    [near]: { x: 0.51, z: -0.1, visibility: 0.95 },
    [far]: { x: 0.49, z: 0.1, visibility: 0.5 },
    [LM.nose]: { z: 0 },
  });
}
