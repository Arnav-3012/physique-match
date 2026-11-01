import { CFG } from "./config";
import { LM, type Kp, type ViewName } from "./types";

const mid = (a: Kp, b: Kp) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });

/** Shoulder width / torso length in pixel units. `aspect` = frame width / height. */
export function shoulderRatio(k: Kp[], aspect: number): number {
  const [l, r] = [k[LM.lShoulder], k[LM.rShoulder]];
  const hips = mid(k[LM.lHip], k[LM.rHip]);
  const sh = mid(l, r);
  const width = Math.hypot((l.x - r.x) * aspect, l.y - r.y);
  const torso = Math.hypot((sh.x - hips.x) * aspect, sh.y - hips.y);
  return width / torso;
}

/** 0° = square to the camera, 90° = side-on. `ref` = shoulderRatio when facing the camera. */
export function absYawDeg(k: Kp[], aspect: number, ref: number): number {
  const c = Math.min(1, shoulderRatio(k, aspect) / ref);
  return (Math.acos(c) * 180) / Math.PI;
}

/** Nose closer to the camera than the shoulders (smaller z) and visible. */
export function facingCamera(k: Kp[]): boolean {
  const nose = k[LM.nose];
  return nose.visibility > CFG.minVisibility && nose.z < mid(k[LM.lShoulder], k[LM.rShoulder]).z;
}

/** Which side the camera sees, from the far shoulder being occluded. */
export function visibleSide(k: Kp[]): "left" | "right" | null {
  const gap = k[LM.lShoulder].visibility - k[LM.rShoulder].visibility;
  if (gap > CFG.sideVisibilityGap) return "left";
  if (gap < -CFG.sideVisibilityGap) return "right";
  return null;
}

/**
 * Second yaw estimate from depth: shoulder z gap vs their x gap (z is roughly in x units).
 * ~0 square to the camera, ~90 side-on. Noisy, so only used together with the ratio estimate.
 */
export function zYawDeg(k: Kp[]): number {
  const dz = Math.abs(k[LM.lShoulder].z - k[LM.rShoulder].z);
  const dx = Math.abs(k[LM.lShoulder].x - k[LM.rShoulder].x);
  return (Math.atan2(dz, dx) * 180) / Math.PI;
}

/**
 * Does this frame look like the view the turn is currently at?
 * The turn order (front, left, back, right) already separates front from back and left from right,
 * so only the angle is checked; the flaky face/far-shoulder visibility cues are not gating.
 * The ratio angle decides; the noisy depth angle (zYawDeg) only has to agree loosely, so a wrong
 * `ref` can never pass a frontal frame as a side view (or the reverse). Depth alone let mid-turn frames through.
 */
export function matchesView(view: ViewName, k: Kp[], aspect: number, ref: number): boolean {
  const yaw = absYawDeg(k, aspect, ref);
  const z = zYawDeg(k);
  if (view === "front" || view === "back") return yaw <= CFG.angleTolDeg && z <= CFG.frontMaxZYawDeg;
  return yaw >= CFG.sideMinYawDeg && z >= CFG.sideMinZYawDeg;
}
