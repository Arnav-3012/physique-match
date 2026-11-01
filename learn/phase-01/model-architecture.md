# P1 · MediaPipe Pose Landmarker: architecture and how it works

The only model in P1. It runs **in the browser** (WASM + WebGL/CPU), never on a server. Our wrapper: [landmarker.ts](../../apps/web/src/capture/landmarker.ts).

> Source note: the file size (5.5 MB), the API, the input sizes (224 / 256) and the option names were checked against the installed package typings and Google's docs page. The internals in section 2 come from the BlazePose paper (Bazarevsky et al., 2020) and are not verified against the `.task` bundle itself. Parameter counts are deliberately not quoted: we did not measure them.

## 1. What it gives us

One call per frame returns, for the single person found:

| Output | Shape | We use it for |
|---|---|---|
| `landmarks` | 33 x {x, y, z, visibility, presence} | joint positions, body angle, "full body in frame" |
| `worldLandmarks` | 33 x {x, y, z, ...} in metres, origin at the hips | not used in P1 |
| `segmentationMasks[0]` | W x H float32, probability a pixel is the person | the silhouette |

- `x`, `y` are fractions of the image width/height (0..1). They are **not in the same units**: 0.1 in x is not 0.1 in y unless the image is square. That is why [yaw.ts](../../apps/web/src/capture/yaw.ts) multiplies x by the aspect ratio before measuring any distance.
- `z` is depth relative to the hips (smaller = closer to the camera), on roughly the same scale as `x`. It is the noisiest of the four coordinates.
- `visibility` is the model's belief the point is *not occluded*, `presence` that it is in frame. For occluded points (far arm in profile) the model still **guesses a position**, so a low visibility is not the same as "outside the frame".

### The 33 landmarks
`0` nose · `1-6` eyes · `7/8` ears · `9/10` mouth · `11/12` shoulders · `13/14` elbows · `15/16` wrists · `17-22` hand points · `23/24` hips · `25/26` knees · `27/28` ankles · `29/30` heels · `31/32` foot index. Even numbers are the subject's right, odd the left. We name the ones we use in `LM` in [types.ts](../../apps/web/src/capture/types.ts).

## 2. Two-stage architecture (detector, then landmark tracker)

```
frame ──► [1] person DETECTOR ──► body ROI (centre, scale, rotation)
              (224x224 input)            │
                                         ▼
                          crop + rotate so the body is upright
                                         │
                                         ▼
                        [2] LANDMARK model (256x256 input)
                          ├─► 33 landmarks (x, y, z, visibility, presence)
                          └─► segmentation mask (person probability)
                                         │
              VIDEO mode: landmarks ─────┘──► ROI for the NEXT frame
                          (detector only re-runs if tracking is lost)
```

1. **Detector.** A small CNN finds the person. BlazePose's key trick is that it detects the *face*, then infers the body ROI from two virtual points: the hip midpoint and a circle that encloses the whole body. This is why it needs a face-ish anchor and why it can struggle for people facing fully away or heavily occluded.
2. **Landmark model.** Runs on the cropped, rotated ROI (256x256). It regresses 33 keypoints and a segmentation mask from a shared encoder. The paper trains it with a heatmap branch for supervision and uses a regression head at inference, which keeps it fast on mobile.
3. **Tracking (VIDEO mode only).** The next frame's ROI is derived from this frame's landmarks, so the detector is skipped. That is the speed win, and it is why timestamps passed to `detectForVideo` **must increase** (we pass `performance.now()`).
4. **IMAGE mode** runs the detector every call. Used for uploaded photos.

### Model variants
`lite`, `full`, `heavy` (same API, larger = more accurate and slower). We ship **lite**: `pose_landmarker_lite.task`, **5.5 MB measured**, stored in [public/models/](../../apps/web/public/models/). One `.task` file bundles the detector and the landmark model.

## 3. How we run it ([landmarker.ts](../../apps/web/src/capture/landmarker.ts))

| Decision | Why |
|---|---|
| `FilesetResolver.forVisionTasks("/wasm")` and a same-origin `.task` path | no third-party request during capture (privacy, and the network-tab gate). `public/wasm` is copied from `node_modules` by the `wasm` npm script |
| `delegate: "GPU"`, fall back to `"CPU"` | GPU is much faster; some browsers (iOS Safari) fail to init it |
| `numPoses: 1`, `outputSegmentationMasks: true` | one person, and we need the mask (ADR-023: no separate Image Segmenter) |
| Lazy singleton promise | the model loads once; a failed load clears the promise so "Start" can retry |
| `setOptions({ runningMode })` only when the mode changes | switching IMAGE <-> VIDEO reconfigures the graph |
| `mask.getAsFloat32Array().slice()` | the buffer is reused by the next frame, so we copy |
| `result.close()` in `finally` | frees the mask resources owned by the result |
| Frames downscaled to long side 720 first ([pipeline.ts](../../apps/web/src/capture/pipeline.ts) `drawScaled`) | the mask comes back at *input* size; a 1280x720 float mask read back from the GPU every frame is slow |

## 4. What it does NOT give us (and what we built on top)

| Gap | Our answer |
|---|---|
| No body angle | `yaw.ts`: shoulder width / torso length vs a learned frontal reference, cross-checked with the depth gap |
| No "is the full body in frame" | `quality.ts` `checkFullBody` (pair-wise) + `touchedEdges` (mask touches the border) |
| Mask includes the head, wrong size, not binary | `silhouette.ts`: threshold, crop from full-body bbox, letterbox to 384x512, head cut |
| Left/right can't be trusted from visibility | order of the turn (front, left, back, right) decides which side |

## 5. What breaks it (seen or expected)

| Situation | Effect | Seen? |
|---|---|---|
| Profile / back view | far limbs and face hidden: low `visibility`, guessed positions | yes, caused the false "not in frame" |
| Feet cut off at the frame bottom | ankles extrapolated off-frame, mask runs into the border | yes, "touches the bottom" |
| Loose clothing / bag | mask follows the outline, not the body | warn-only heuristic |
| Busy or similar-colour background | holes or blobs in the mask | not yet; trigger to add the Image Segmenter |
| Very close to the camera | torso cropped; ratios unreliable | step back 2-3 m |
| Several people in frame | `numPoses: 1` picks one, possibly the wrong one | not handled |

## 6. Self-check

1. Why does `matchesView` not trust `visibility` to tell left from right? *(The model guesses occluded points, so the far shoulder often still reads as visible.)*
2. Why must the `detectForVideo` timestamp increase? *(Tracking state is carried frame to frame.)*
3. The mask has the size of the frame you pass in. What breaks if you pass the raw 1280x720 video? *(GPU-to-CPU read-back of a large float array every frame: low fps.)*
