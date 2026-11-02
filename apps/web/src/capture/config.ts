// Starting guesses, tuned over the 10 gate captures (docs/phase-plan.md P1). Not facts.
export const CFG = {
  maskW: 384,
  maskH: 512,
  maskThreshold: 0.5,
  bboxMargin: 0.05, // of body height, added on every side before the crop
  headCut: 0.4, // cut row = shoulderY - headCut * (shoulderY - earY)
  angleTolDeg: 25,
  frameMaxSide: 720, // frames are downscaled before pose+mask; keeps the mask read-back cheap
  countdownMs: 3000,
  holdMs: 600, // pose must match this long before a view is captured
  sideHoldMs: 1000, // side views need a longer, steadier hold than front/back
  sideMinYawDeg: 72, // a side view only counts once the body is at least this far turned
  sideMinZYawDeg: 35, // depth must agree loosely (guards against a bad frontal reference)
  frontMaxZYawDeg: 50, // ...and front/back must not look side-on in depth
  flashMs: 400, // "captured" confirmation before the next view
  matchGraceMs: 300, // tolerated mismatch gap (jitter) before the hold restarts
  viewTimeoutMs: 25000, // give up on one view (it can be retaken)
  minVisibility: 0.5,
  minBodyVisibility: 0.3, // better-visible landmark of each left/right pair; far-side limbs are hidden in profile
  coverage: [0.03, 0.5], // silhouette area / frame area
  luminance: [60, 200], // mean 0-255
  looseClothingRatio: 1.6, // torso-row width / shoulder-keypoint width (warn only); arms hugging the torso inflate it
  sideVisibilityGap: 0.1, // far-side shoulder is less visible by at least this
  torsoSkinChroma: 0.04, // torso pixel counts as skin if its r/g chromaticity is this close to the person's own face / hands
  torsoBareMin: 0.5, // share of torso pixels that look like skin; below it the torso looks covered (warn, user can override)
} as const;
