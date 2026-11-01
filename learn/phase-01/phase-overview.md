# P1 · Private 4-view capture: overview

**One line:** a browser flow that turns a webcam (or four uploaded photos) into four head-less 1-bit-style silhouettes plus 33 keypoints each, matching the `Scan` schema, with nothing leaving the device. The review screen shows exactly what *would* be sent.

## 1. Data flow

```
 camera frame (or uploaded photo)
        │  drawScaled: long side <= 720            pipeline.ts
        ▼
 PoseLandmarker (WASM, on-device)                  landmarker.ts
   ├─ 33 keypoints ─────────────┐
   └─ person-probability mask   │
        │                       │
        │          yaw.ts: body angle vs "square to camera" reference
        │          quality.ts: full body? edges? lighting?
        │                       │
        ▼                       ▼
 useTurnCapture: hold the pose 0.6 s (side views 1 s) ──► keep the SHARPEST frame (sharpness.ts)
        │
        ▼  finalizeView (pipeline.ts)
 silhouette.ts: threshold ► crop from FULL-body box ► letterbox 384x512 ► remove head ► remap keypoints
        │
        ├─ encode.ts: 8-bit 1-channel PNG {0,255} ► base64 ──► ViewCapture ──► lib/scan.ts ──► Scan
        └─ RGB frame ──► rgbStore.ts (memory only, for the P4 definition model; cleared on leave)
```

Privacy invariants (ADR-003/013): the RGB frame is never written to disk, IndexedDB or localStorage, and P1 makes **no request carrying image data** (it makes none at all except the `/health` ping). The model and WASM are same-origin.

## 2. What each file does

### App shell
| File | Role |
|---|---|
| [App.tsx](../../apps/web/src/App.tsx) | step state machine: consent, capture, upload, review; `/health` line kept as a footer |
| [lib/scan.ts](../../apps/web/src/lib/scan.ts) | `buildScan`: needs front + left, 33 keypoints, per-view <= 350k and total <= 700k base64 chars (mirrors the Python schema) |
| [lib/types.gen.ts](../../apps/web/src/lib/types.gen.ts) | generated from `packages/schemas`; never edited by hand |

### `apps/web/src/capture/` (logic)
| File | Role | Unit-tested |
|---|---|---|
| [config.ts](../../apps/web/src/capture/config.ts) | every tunable number in one place (mask size, thresholds, hold times) | |
| [types.ts](../../apps/web/src/capture/types.ts) | `ViewName`, `Kp`, `LM` landmark indices, `ViewCapture` | |
| [landmarker.ts](../../apps/web/src/capture/landmarker.ts) | loads and runs the Pose Landmarker (GPU then CPU), returns keypoints + float mask | manual |
| [pipeline.ts](../../apps/web/src/capture/pipeline.ts) | frame downscale, sharpness scoring, `finalizeView`, upload path, live silhouette preview | manual |
| [yaw.ts](../../apps/web/src/capture/yaw.ts) | body angle from shoulder/torso ratio and depth; `matchesView` | yes |
| [sharpness.ts](../../apps/web/src/capture/sharpness.ts) | grayscale + variance of the Laplacian (blur detector) | yes |
| [quality.ts](../../apps/web/src/capture/quality.ts) | bounding box, coverage, full-body, border touch, lighting, loose-clothing warning, `viewIssues` | yes |
| [silhouette.ts](../../apps/web/src/capture/silhouette.ts) | mask to final silhouette, keypoint remap | yes |
| [encode.ts](../../apps/web/src/capture/encode.ts) | mask to base64 PNG | yes (decode round-trip) |
| [useTurnCapture.ts](../../apps/web/src/capture/useTurnCapture.ts) | the guided-turn state machine | manual |
| [rgbStore.ts](../../apps/web/src/capture/rgbStore.ts) | in-memory holder of the 4 chosen RGB frames | |
| [test-utils.ts](../../apps/web/src/capture/test-utils.ts) | synthetic poses (`pose`, `sidePose`) for tests | |

### `apps/web/src/capture/` (screens)
| File | Role |
|---|---|
| [Consent.tsx](../../apps/web/src/capture/Consent.tsx) | itemised notice (captured / sent / never sent), 18+ checkbox, clothing note; state in memory only |
| [CaptureScreen.tsx](../../apps/web/src/capture/CaptureScreen.tsx) | live silhouette (feed hidden by default), pose guide, prompts, hold bar, beep + flash, "Capture now", Debug readout |
| [UploadCapture.tsx](../../apps/web/src/capture/UploadCapture.tsx) | four file pickers (front + left required), same pipeline; reused for the goal photo in P7 |
| [ReviewScreen.tsx](../../apps/web/src/capture/ReviewScreen.tsx) | the actual PNGs that would be sent, size, valid/invalid, issues, per-view retake |

### Config / infra touched
`package.json` (`wasm`, `predev`, `prebuild`, `test` scripts), `vite.config.ts` (vitest), `.gitignore` (`apps/web/public/wasm/`), `.github/workflows/ci.yml` (adds `npm test`), `public/models/pose_landmarker_lite.task` (5.5 MB, committed).

## 3. The maths worth knowing

**Body angle from shoulders.** A shoulder span of real width `W` turned by `θ` projects to `W·cos θ`. Dividing by torso length (shoulder-mid to hip-mid) cancels distance from the camera; converting x to pixels first cancels the aspect ratio:

```
ratio = hypot((Lx-Rx)·aspect, Ly-Ry) / hypot((Sx-Hx)·aspect, Sy-Hy)
yaw   = acos( clamp(ratio / ref, 0, 1) )        ref = median ratio while square to the camera
```
`cos θ = cos(-θ)`, so the sign and front-vs-back are invisible to this formula: the fixed turn order supplies them. A second estimate from depth, `atan2(|ΔzL-R|, |ΔxL-R|)`, only has to agree loosely; it stops a wrong `ref` from passing a frontal frame as a side view.

**Sharpest frame.** Variance of the Laplacian over body pixels: `var( p[x-1]+p[x+1]+p[y-1]+p[y+1] - 4·p )`. Blur removes high frequencies, so the variance drops.

**Letterbox and keypoint remap.** Box with margin `(sx0, sy0, sw, sh)`, `scale = min(384/sw, 512/sh)`, offsets centre the box; each output pixel is bilinearly sampled from the probability map and thresholded at 0.5; a keypoint `(x, y)` maps to `((x·w − sx0)·scale + offX) / 384`, clamped to [0,1].

**Head cut.** Rows above `shoulderY − 0.4·(shoulderY − earY)` are zeroed *after* cropping, so the head-top stays available as a height reference in the keypoints.

## 4. Decisions (see ADR-023 in [decisions.md](../../docs/decisions.md))
Pose-Landmarker mask only (no Image Segmenter, until masks prove bad) · crop then head-cut · 8-bit 1-channel {0,255} PNG · self-hosted model and WASM · user-paced capture with a manual fallback · turn order, not face visibility, separates views.

## 5. Results vs gate

Gate: *10 captures (laptop + phone): 10/10 give 4 valid masks; network tab shows no image except masks; capture <= 12 s.*

| Check | Result |
|---|---|
| Unit tests + strict TS build | build green (`tsc --noEmit`, 63 modules, 407 kB JS / 129 kB gzip); the build runs only after `npm test` passes in the same command |
| Laptop scan, 4/4 valid | yes; turn 7.6 s on the run with all four valid |
| Phone | user-reported working (4 captures) |
| Captures done | ~8 reported, **10 not yet reached** |
| Network tab | **not yet reported** |
| Earlier runs | 19.6 s (before the hold/flash shortening), 5.9 s (with views failing), 52.9 s (reference bug, 2 views missing) |

The final numbers for the gate go in [dev-log.md](../../docs/logs/dev-log.md) once the two missing checks are in; this table is not a gate pass.

## 6. What went wrong (details in [mistakes.md](../../docs/mistakes.md))
1. Time-slotted capture: no "turn now" cue, silent skipped views.
2. "Body not in frame" with the whole body visible: required both sides and the nose, which profile and back poses hide.
3. A one-way "frontal reference" ratcheted too high; frontal poses were captured as left/right and retakes had no reference.
4. Edge check tripped on 3 stray pixels; side views accepted at ~50 deg of turn.

Pattern: every fix came from showing the number behind the failing check (Debug box, pixel counts in messages).

## 7. Known limits
Loose-clothing check is crude (arms against the torso inflate it; warn-only). One person only. Thresholds were tuned on one person and one room. Mask quality on busy backgrounds is untested. Safari/iOS GPU path untested beyond the CPU fallback.

## 8. Interview Q&A

1. **Why do the masks leave the device but the photos don't?** The model needs outline and joints, not skin or face; the server rejects anything that is not a 1-channel {0,255} PNG (ADR-003), and the RGB frame lives only in memory.
2. **How do you know the user turned to a side view?** Shoulder width / torso length against a median frontal reference gives `acos`; it must exceed 72 deg and the depth gap must agree; held for 1 s; the sharpest frame in the hold is kept.
3. **Why not trust landmark visibility?** The model guesses occluded joints, so the far shoulder reads visible, and in profile the far limbs read hidden even when the body is fully in frame.
4. **What was your hardest bug?** The frontal reference updated only upward, so one early bad frame made every frame look turned: frontal poses were stored as side views. Fix: rolling median, kept across retakes, plus an independent depth cross-check.
5. **Why crop before removing the head?** The head-top is the best height reference; cropping from the full-body box and cutting afterwards keeps it in the keypoints for P2's height normalisation.
