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
- Opt-in toggle, off by default. When on, a small CNN runs **inside the browser** on the RGB frames and outputs numbers (definition 1–5 per region). Only those numbers are sent.
- The "what we send" preview shows those numbers too.
- Free hosted LLMs never receive any image (several free hosts may train on prompts).

## Clothing guidance (no undressing required)
- Shown in UI: "Wear fitted clothing — gym wear, a sports bra / fitted tee, leggings or shorts. You do not need to remove clothes."
- Loose clothing detector: silhouette area vs. keypoint-implied body width; if the outline balloons > threshold at torso/thigh, prompt "loose clothing detected — results will be less accurate" (warn, don't block).
- Region mode: scan only arms, or only legs, etc. No torso exposure needed.

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
