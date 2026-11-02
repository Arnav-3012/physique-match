# Privacy & body-image safety (hard requirements)

## The core answer: nobody's photo ever leaves their device
"Device" = whatever runs the browser: laptop webcam, phone camera, or an uploaded file (ADR-019).
The model never needs skin, faces or texture. It needs **body outline + joint positions**. So:

1. Camera frame -> MediaPipe segmentation + pose **in the browser (WASM)**.
2. Browser keeps only: a **1-bit silhouette** (white shape on black) + 33 keypoints.
3. Head region is cropped out of the silhouette (above shoulder-keypoint line).
4. Raw frame is dropped from memory immediately. Never written to disk, IndexedDB or localStorage.
5. Server rejects any upload that isn't a binary mask (checks: 1 channel, only values {0,255}, size ≤ 512 KB).

What the user sees during capture: their **live silhouette preview**, not their camera feed (toggle to show feed for alignment, off by default). This is the single biggest comfort feature — you watch a shadow, not yourself.

## On-device muscle-definition model (ADR-013)
- Opt-in toggle (a checkbox on the scope step, full-body option only), off by default. When on, two small CNNs run **inside the browser** on the front RGB frame and output two numbers: one overall "leanness & muscularity (estimate)" and one adiposity (a waist-girth proxy), each 1–5, shown as low / mid / high. **There is no per-region definition anywhere in the app, the request or the report** (ADR-053 Amendment 6: the regional heads did not beat the overall score). Only those two numbers are sent.
- **Torso check.** Under the full-body option, if the front photo looks like the torso is covered, the app warns, does not show the estimate, and offers retake, a different option, or "my torso is bare, score it". The check compares the torso's colour with the person's own face and hands (a rough guess, unvalidated, never blocks size gaps).
- The "what we send" preview shows those numbers too.
- Free hosted LLMs never receive any image (several free hosts may train on prompts).

## Clothing guidance (no undressing required)
- Shown in UI: "Wear fitted clothing — gym wear, a sports bra / fitted tee, leggings or shorts. You do not need to remove clothes."
- Loose clothing detector: silhouette area vs. keypoint-implied body width; if the outline balloons > threshold at torso/thigh, prompt "loose clothing detected — results will be less accurate" (warn, don't block).
- Region mode: scan only arms, or only legs, etc. No torso exposure needed.

## Capture scope (ADR-051)
The muscle-definition model is trained on bare-body renders, and clothing hides exactly what it scores. So the app asks once, before capture, "What do you want to compare?" and scores definition only where the scan shows the body.
- **A. Arms, shoulders and legs:** sleeveless or tank top, shorts. No shirtless scan. Definition: not scored, "this scan does not show the abdomen" (ADR-053 Amendment 4; regional definition is TBD after demo, original wording: arms, shoulders, legs).
- **B. Full body:** men shirtless + shorts; women a sports bra (training bra) + shorts; if sex is unset or the user prefers, "bare torso" or "sports bra / crop top" is accepted. For women in a bra the chest estimate is labelled approximate. Definition: one overall "leanness & muscularity (estimate)" score, carried by the abdomen (ADR-053 Amendment 4; the back is not scored in the demo; original wording: A plus chest, abs, back, TBD after demo).
- **C. Clothed scan** (always available): fitted clothes; size gaps and body-fat range only; every definition score is N/A, with the reason "clothed scan".
- **Gates.** B is available only when the age field is 18 or more; under 18 or unset it is hidden with "This option is for adults", and the server rejects a request claiming B with an age under 18. Nothing is forced: every scope can be skipped, the clothed scan always works, and a bare-torso scan is never the default or required. The capture screen states the requirement and the privacy facts **before the camera starts**, with a "use clothed scan instead" button.
- **N/A always has a reason** (covered by clothing in this scan option, age gate, goal photo does not show the region); it is never a number and never read as zero.

**Privacy copy (exact; mirrored in `apps/web/src/capture/privacyCopy.ts`, a test checks they match).**
> Photos are processed on this device. No photo is uploaded or saved. By design, this app writes no photo, thumbnail or video frame to your browser's storage or to a file, and frames are released from memory once the outlines are made. The server receives only a black-and-white outline (silhouette), body joint positions and the numbers you enter. It uses the outline to calculate your result and does not store it. The hosting service keeps its own technical request logs (time, address, status), not your outline. The muscle-definition model runs in your browser and is off unless you turn it on.

Each sentence was checked against the code on 2026-11-02 (ADR-051 lists what was read and what was not). The hosting sentence describes the planned Cloud Run deployment and is not yet an observation. Not to be replaced by "nothing leaves your device" or "never stored".

**"By design" is the honest word for the browser-storage sentence.** Part of it rests on a source scan, not a runtime observation: six capture files (`config`, `encode`, `quality`, `sharpness`, `silhouette`, `yaw`) were not read, and `tests/test_privacy_copy.py` greps every web source file for storage APIs instead. A grep cannot see a library that stores things on its own. The manual check below is what turns the claim into an observation.

**The server and the outline (answered, with one open part).** *Does the server store the outline?* Not in our code: the analyze route holds the request in memory, computes and returns; it opens no file, database or cache for the body (read 2026-11-02; `tests/test_privacy_copy.py` fails if the route gains `open(`, `write_text`, `sqlite`, `redis`, `pickle` and similar). The in-memory rate limiter keeps a client key and timestamps for 60 s, no body. **Open question:** whether the hosting platform's request logs ever hold a request body. Cloud Run is not deployed, so this is unobserved; at the deploy spike, read the service's logs after a real analyze call and confirm they contain no outline, then log the result. Until then "does not store it" in the copy is a statement about our code, with the hosting sentence kept separate and limited to "time, address, status".

**Manual DevTools check (user-run, before the demo, and again after any capture-code change).** Run one full scan (scope B, four views, then the review screen), then in DevTools, Application tab: *Local storage*, *Session storage*, *IndexedDB* and *Cache storage* for the app's origin hold no photo, thumbnail or frame data (empty is expected; any entry from a library is named in the dev-log with what it holds). Also *Network*: the only requests are same-origin files, the MediaPipe model files and the analyze call, and the analyze payload holds silhouettes, keypoints and numbers only. Log the result, with the date and browser, in `docs/logs/dev-log.md`. Result so far: **not run**.

**Data lifecycle.** Frames exist only in memory (canvas, `ImageData`, `ImageBitmap`) and are released after the masks (and, with the opt-in model on, the definition scores) are computed: bitmaps closed, canvases cleared, camera tracks stopped. No image data in `localStorage`, `sessionStorage`, `IndexedDB`, the Cache API or object URLs, including any later progress tracking (P9 stores numbers only). Logs and error reports never contain image data or free text. The report shows no photo of the user.

## Consent & trust UX
- Plain-language screen before camera: what is captured, what is sent (show the actual silhouette that will be sent), what is never sent.
- "What the server received" button on the report: shows the exact mask + numbers.
- No accounts in v1. No storage of scans server-side. Report lives in browser memory; user downloads it if they want.
- 18+ gate before the camera opens.
- Goal image: user uploads; same pipeline runs on it (silhouette only). Note in UI: use a photo you have rights to; we don't store it.

## Body-image safety (report + plan)
- Language: "development opportunities", never "flaws/lacking/fat". Banned-word list checked in `safety` eval suite.
- Realism flag: if gap per region exceeds a plausible range for natural training in the plan horizon, say so plainly and set an intermediate target. Goal photos are often edited, posed, pumped or enhanced; the report says this once.
- No supplement/drug advice.

## Body-fat % and calorie rules (ADR-012/015)
- Body-fat % is always shown as a **range with confidence** (e.g. "18–23%"), never a single number, plus the method used.
- Height, weight, age, sex are typed in the browser; they go to the server as numbers only (needed for the BF model). Not stored.
- **Sex (ADR-037).** Two optional fields, both typed by the user, never inferred from a photo and never defaulted: "your sex" and "the goal physique's sex" (asked beside the goal photo upload, and never copied from the first). Wording beside the first: "Needed for body-fat %, calories, back width and the V-taper ceiling." Consent screen line: "Sex is only used for those four things. It is sent as a choice, not stored, and never read from your photo." Leaving it unset is allowed: no body-fat %, no calorie or protein targets and no back width (each with a "set your sex" message), while gaps and the training plan still work. Cut / recomp / bulk is always the user's choice, never derived from body-fat %.
- **Different-sex goal.** If both sexes are set and differ, back width is not estimated and every report carries: "The goal physique is the other sex from yours, so absolute size gaps may not be reachable." The other regions still compute.
- **Back width is an estimate.** Shown as "back width (estimate)", unranked, at low reliability. It is inferred from chest, shoulder and waist ratios (ANSUR II fit), not measured, and it is not a measure of lat muscle. `upper_back` size is never shown: "Not measurable from a front photo; a back silhouette adds no size information."
- Calorie targets: deficit capped at 20% of TDEE and at a loss rate of ~1% bodyweight/week; never below estimated BMR; surplus capped at 10% of TDEE.
- No cut plan if BMI < 18.5 or estimated BF below sex-specific essential-fat floor + margin (men < 8%, women < 18%): show maintenance + strength focus and a note to see a professional.
- Screening: one optional question before nutrition targets ("Do you have a history of an eating disorder?" -> yes = training plan only, no calorie numbers, supportive note).
- Goal-photo BF is labelled "appearance-based estimate"; photos are often posed, lit, flexed or edited.
- **No photos or numbers about a user ever go to free LLM endpoints except the numeric gap/BF profile.** Several free hosts may train on prompts (ADR-017).
- Intent router sends anything about extreme restriction, purging, or self-harm to a supportive fixed response with helpline info; no plan generated.
- Every report ends with: not medical advice; see a professional for injuries or health conditions.

## Logging
- Logs: request id, latency, cache hit, region deltas (numbers only). Never masks, never free-text queries beyond the intent label. Never secrets.
- Cloud Run (ADR-028) writes its own request logs (URL, status, latency, client IP, user agent) outside our code. Set the log retention as short as the console allows, and say in the consent screen that the host sees the requester's IP address. Per-IP rate-limit counters in Upstash expire with the window.
