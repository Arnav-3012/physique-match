# Decisions (ADR log)
Format: ID · status · decision · why · alternatives rejected. Append only; supersede, don't edit.

**ADR-001 · Accepted · Ship geometric analyzer v1 before Siamese v2**
- Why: no public dataset labels "muscle development per region". Geometry from silhouettes + keypoints works on day one and gives a baseline the Siamese net must beat.
- Rejected: Siamese-first (demo blocked on training), VLM-only analysis (non-deterministic, sends images out).

**ADR-002 · Accepted · BodyM as training data**
- 2,505 subjects, frontal+lateral silhouettes, height, weight, 14 measurements (bicep, chest, waist, hip, thigh, calf, forearm, shoulder breadth...). Silhouettes match our privacy design exactly.
- Licence: CC BY-NC 4.0 -> fine for portfolio/research, **not commercial**. Must be stated in model card and README. Commercial version would need own data or synthetic SMPL renders.
- Pairs for Siamese: any two subjects (A, B) -> target = meas(A) − meas(B), normalised by height. ~6M possible pairs; sample per epoch.

**ADR-003 · Accepted · Silhouette-only upload**
- Raw RGB never leaves the device; server rejects non-binary images. Answers the "people feel uncomfortable undressing" problem and removes most data-protection risk.

**ADR-004 · Accepted · Siamese backbone = ResNet-18 (ImageNet), fine-tune**
- Silhouettes are low-information; 11M params is plenty; trains on M4 Pro MPS in minutes per epoch; ONNX on free CPU is fast.
- Ladder: linear probe on frozen features -> unfreeze last 2 blocks -> full fine-tune. Stop at the first rung that hits the gate.
- Rejected: DINOv2/ViT (overkill for binary shapes, slower CPU inference), EfficientNet (no clear gain, more fiddly on MPS).

**ADR-005 · Accepted · One agent + tools, no LangChain**
- Tools: `retrieve(query, filters)`, `get_gap_profile()`, `compute_volume(region, level, days)`, `check_safety(text)`. Direct OpenAI-compatible SDK so provider swaps are env-only.

**ADR-006 · Accepted · Siamese must beat "single regressor × 2 then subtract" baseline**
- If it doesn't beat it by a meaningful margin on held-out subjects, keep the simpler model and record the negative result. Honest negative results go in the README.

**ADR-007 · Accepted (amended by ADR-028: embeddings via ONNX Runtime int8, no torch in the API image) · Hybrid RAG = BM25 + bge-small-en-v1.5, RRF, numpy/FAISS-flat**
- Corpus is small (< 5k chunks). No vector DB. Exercise names need keyword match (BM25), intent needs semantics (dense).

**ADR-008 · Accepted (amended by ADR-028: cache state lives in Upstash Redis, not process memory) · Semantic cache keyed on sub-queries, not whole reports**
- Personalised reports are near-unique; caching them would serve wrong plans. Retrieval results per (normalised sub-query, namespace) are highly reusable.

**ADR-009 · Superseded by ADR-022, then ADR-028 · Deploy: Vercel (web) + HF Space Docker (API) + HF Hub (model)**
- All free. GitHub Pages rejected for web: no env vars / preview deploys; Vercel gives both.

**ADR-010 · Superseded by ADR-011 · Front view required, side view optional in v1**
- Front silhouette gives widths only; girths (arms, thighs) need side view. v1 marks girth-type regions as lower confidence without side view.

**ADR-011 · Accepted · 4-view guided turn capture (front, left, back, right) from video**
- User does one slow turn (~8 s) in front of the camera. Browser samples frames, estimates body yaw from keypoints (shoulder-width ratio + left/right landmark visibility), keeps the sharpest frame per view, and converts each to a silhouette. Still only silhouettes leave the device.
- Back/chest split: in a side silhouette, pixels in front of the shoulder–hip keypoint line = chest projection, behind = upper-back projection. Front/back width at armpit level = lat width (V-taper). Two sides -> left/right asymmetry.
- Limits: a silhouette sees outline, not definition. Trap/rhomboid/rear-delt detail and muscle separation are invisible; report says so. BodyM has front + side only, so the Siamese model uses those 2 views; back + second side feed the geometric analyzer (back width, asymmetry).
- Rejected: sending RGB for muscle-definition analysis (breaks ADR-003; no labelled data anyway).

**ADR-012 · Accepted · Body-fat % v1 = RFM from silhouette waist, shown as a range**
- RFM = 64 − 20·(height/waist) + 12·sex; validated vs DXA on NHANES (R² 0.75 men / 0.69 women). Waist girth comes from front+side silhouettes (calibrated on BodyM); a tape-measured waist, if entered, overrides it.
- Output a range (estimate ± measured error), never a single number. Known weak spot: lean people and older adults.
- Rejected: LLM estimating BF% (no grounding, hallucinates); scraped Reddit BF images (no consent, ToS risk, perceived labels, men only).
- See `docs/research/body-composition-research.md`.

**ADR-013 · Accepted (amends ADR-003) · On-device RGB model allowed; only scores leave the device**
- The muscle-definition model runs in the browser (ONNX Runtime Web, WebGPU -> WASM fallback) on the RGB frame. It outputs a 1–5 score per region; only those numbers are sent. The server still rejects any image that isn't a binary mask.
- Opt-in toggle ("analyse muscle definition on my device"); off -> report uses shape-only.

**ADR-014 · Accepted · Definition model = MobileNetV3-Small trained on MakeHuman synthetic renders**
- No public labelled real dataset exists. MakeHuman assets are CC0; muscle/weight sliders become labels. Render a grid (sex × muscle × fat × height × skin tone × lighting × fitted clothing) in Blender headless, ~30k images, 4 views.
- Labels: per-region definition 1–5 derived from (muscle slider, fat slider) via a fixed mapping. Real fine-tune + validation on a consented set (ADR-016).
- Gate: Spearman ρ ≥ 0.5 vs blinded human ratings on the real set. Below that -> feature ships as "experimental" or is cut; negative result documented.
- Rejected: ResNet-50/ViT (too heavy for phones), cloud VLM (breaks privacy; free hosts may train on prompts).

**ADR-015 · Accepted · Energy targets: Mifflin-St Jeor × activity factor, with safety floors**
- Most accurate common RMR equation (≤ 10% error for 78% of subjects). Katch-McArdle not used: needs measured lean mass.
- Goals: cut / recomp / bulk as ranges. Exact floors and caps live in `privacy-and-safety.md` (single source of truth).
- Supersedes the "no calorie targets in v1" line in `privacy-and-safety.md`.

**ADR-016 · Accepted · Real validation set = consented volunteers with tape measurements**
- 20–40 adults (mixed sex/BMI), signed consent, 4-view capture + tape (waist, neck, hip, arm, thigh) + 3 blinded raters scoring definition 1–5. Stored offline, encrypted, deleted after the project. Used for evaluation and a small fine-tune only.
- Optional: a gym BIA/DXA reading where available as a stronger BF% reference.

**ADR-018 · Accepted (amends ADR-016) · No scraping; real set is not a demo blocker**
- The demo (P0–P7) needs no real labelled photos: BF% is RFM, definition model trains on synthetic data (ADR-020).
- P8 real data: own scans + tape measurements, consenting friends, openly licensed images (Wikimedia Commons CC0/CC BY, licence logged per image).
- Rejected: scraping r/guessmybf etc. — no consent for semi-nude body photos, Reddit's terms restrict ML use of its data, labels are guesses, almost all male.

**ADR-019 · Accepted · "On-device" = the user's browser on any device**
- Laptop webcam, phone camera or file upload all work; MediaPipe and ONNX Runtime Web run in Chrome/Safari/Edge on desktop and mobile. Live webcam preview is supported.

**ADR-020 · Accepted (amends ADR-014, ADR-018) · Definition model is part of the demo, trained on synthetic data only**
- Muscle definition is a core feature, so it ships in Track A (P4), before the demo deploy.
- Demo training data: MakeHuman synthetic renders only (+ BodyM for silhouettes). Real data (own measurements, Wikimedia Commons) arrives in P8.
- Demo gate is on synthetic held-out data + a logged sanity check on own scans; UI carries an "experimental — trained on synthetic data" badge until real-data ρ ≥ 0.5 (P8).
- Risk accepted knowingly: synthetic-to-real gap. Mitigation: same browser preprocessing on renders, background swap, heavy augmentation.

**ADR-021 · Accepted (supersedes ADR-001 ordering) · Siamese CNN before muscle definition, both before demo deploy**
- Order: P2 BodyM + geometric baseline -> P3 Siamese CNN -> P4 definition model -> P5 RAG -> P6 agent -> P7 deploy.
- The Siamese CNN is the core model and ships in the demo. Geometric analyzer stays as baseline A, fallback, and for back view + asymmetry (BodyM has no back view).

**ADR-017 · Accepted · Free LLM providers, text only**
- Primary: Groq `openai/gpt-oss-120b` (~30 RPM, 1,000 req/day free). Fallback: OpenRouter free endpoints with `data_collection: "deny"` / no-train hosts. Provider + model in env; one OpenAI-compatible client.
- LLMs never receive images; they receive the GapProfile, BodyComp numbers and retrieved text.
- Budget: ≤ 3 LLM calls per report (route only if embedding router is unsure, write report, safety rewrite if flagged). Semantic cache keeps repeat traffic off the quota.
- Embeddings run locally (bge-small) — no API dependency.

**ADR-022 · Superseded by ADR-028 (had superseded ADR-009) · No HF Space / HF Hub; API and models run locally**
- HF Docker and Gradio Spaces are not free for this account (static Spaces cannot run FastAPI). Decision (user, 2026-11-01): drop HF Spaces and HF Hub for now. API (`uvicorn`) and ONNX/torch models run on the laptop; model cards live in `ml/`.
- Web stays on Vercel (free). Calling `http://localhost:8000` from the Vercel page works in Chrome/Edge, not Safari. Day-to-day dev runs the web locally.
- Consequence: the P7 gate "stranger completes the flow from a public URL" is blocked until public API hosting is chosen (candidates: Cloudflare Tunnel to the laptop, a free Render/Fly service). Decide at P7 start; the `Dockerfile` is kept for that.
- No HF token needed; `deploy-space.yml` removed. Privacy model unchanged (silhouettes still never include RGB; the API now just runs on the user's own machine).
- Rejected: Gradio-wrapper workaround (not free either), paid hardware.

**ADR-023 · Accepted (amends P1 task list, refines privacy-and-safety.md) · Browser capture design**
- Segmentation: the **Pose Landmarker's own mask** (`outputSegmentationMasks`), one model and one pass. No separate Image Segmenter. Trigger to add it: masks with holes, background blobs or missing limbs on the review screen.
- Mask frame: crop from the **full-body** bbox (+5% margin), letterboxed into 384x512 (W x H), then the head is removed, so the head-top stays a height reference in the keypoints. Keypoints are remapped to that frame and clamped to [0,1]. Head cut row = `shoulderY - 0.4 * (shoulderY - earY)` (tunable in `capture/config.ts`); the privacy doc's "above the shoulder keypoint line" would shave deltoid/trap tops, which sit above the joint.
- Mask PNG: 8-bit, 1-channel, values {0,255} via `fast-png` (`canvas.toBlob` can only emit RGBA). Meets the server check "1 channel, only {0,255}"; not packed 1-bit, size is irrelevant at this resolution (2-4 KB).
- Assets self-hosted (`public/models/pose_landmarker_lite.task`, `public/wasm` copied from `node_modules` at `predev`/`prebuild`): no third-party request during capture.
- View matching is **user-paced**: body angle from shoulder-width / torso-length (pixel units, so distance and aspect cancel) against a median "square to the camera" reference learned in the countdown, cross-checked with the shoulder depth gap. Fixed turn order (front, left, back, right) separates front from back and left from right; face visibility and far-shoulder visibility are debug-only. A hold bar fills while the pose matches; **Capture now** is the manual fallback.
- Rejected: fixed time slots per view (no cue, silent misses); visibility-based "full body" and left/right tests (far limbs and the face are legitimately hidden in profile and back views).

**ADR-024 · Accepted · Geometric analyzer: silhouette-only landmark rows, two scales never mixed, null-model gate**
- Rows: BodyM has no keypoints, so rows come from the mask alone, one code path for BodyM and browser masks. Run-length analysis: central run = torso; crotch = where the central run splits; armpit = where arm runs merge into the torso; neck = narrowest row in the top ~25%; waist = narrowest torso row between armpit and crotch; hip = widest row between waist and crotch. Side-view rows where arm and torso runs merge are skipped. Search windows are tuned on the BodyM train split only. Keypoints are used for view selection and a logged sanity check only.
- Two scales: (1) the gap (`GapProfile.delta`) uses scale-free silhouette features divided by silhouette height, for both goal and current; a goal photo has no weight, so no weight-aware model touches it. (2) Absolute cm (waist for RFM, `BodyComp`) uses a regressor on silhouette features + height + weight + sex, for the current person's own scan only. Test: the gap is identical when only the profile weight changes.
- Null model: every silhouette claim is reported next to a ridge on [height, weight, sex] only. A gate counts only where the silhouette model beats it (waist and hip correlate strongly with weight alone, so a bare ρ gate can pass without the silhouette adding anything). Applies to the per-region ρ gate and to waist MAE.
- Labels: only 8 regions have BodyM ground truth (shoulders, chest, waist, hips, upper_arm, forearm, thigh, calf). `upper_back` and `lats` are geometric-only and outside the ρ gate.
- Rejected: MediaPipe on BodyM silhouettes (unreliable on binary masks, a second code path); a weight-aware regressor inside the gap (goal has no weight, mixes scales).

**ADR-029 · Accepted (amends ADR-024 gate) · Gate 2 criterion changed after the first result**
- Numbering note: first written as a second "ADR-025" (the goal-photo ADR landed in parallel); renumbered 2026-11-01. Nothing else refers to it by number.
- Original gate (ADR-024): silhouette-only feature must beat the weight-aware null ridge on >= 6 of 8 regions. First run, testA (87 subjects): **1/8** (waist only, interval includes 0). Null rho 0.80-0.93 on every region: height, weight and sex alone predict BodyM measurements/height very well. The comparison was unlike-for-like (weight-free vs weight-aware), so it could not pass.
- New criterion, decided by the user on 2026-11-01 after seeing val and testA: (a) silhouette-only rho >= 0.6 on >= 6 of 8 labelled regions (the gap is weight-free, ADR-024), and (b) the silhouette adds signal beyond the null: (combined - null) rho with a 95% subject-bootstrap interval excluding 0 on >= 6 of 8.
- First-run state for the record (testA): (a) 5/8 (shoulders .79, chest .85, waist .91, hips .83, upper arm .76; forearm .15, thigh .40, calf .46); (b) not yet bootstrapped.
- Because the criterion moved after seeing testA, all feature changes are tuned on val only, and the final numbers are reported on testA and testB together.
- Known weak spots: forearm, thigh, calf (arm width depends on A-pose angle; a hand in front of the thigh inflates side depth; 26% of side masks are clipped, so girths are missing for 12-29% of subjects).

**ADR-025 · Accepted (amends ADR-011 for the goal input; revives ADR-010's idea for the goal side only; extends ADR-013) · Mode 1: goal photo, front-only by default, priority for P7**
- Goal photos are usually front-only; the current scan keeps all 4 views. Goal side view is optional.
- Siamese (P3): train with **side-view dropout**, randomly blanking the goal's side input, so a front-only goal is an in-distribution input. Girth regions (upper_arm, forearm, thigh, calf, chest depth) get lower confidence when the goal side view is missing (on top of MC dropout, calibrated on the slice below). The P3 gate report adds a **front-only slice on BodyM Test-B**, with baselines A and B run on the same slice and full-input numbers beside it.
- Definition (P4): the in-browser model also scores the goal photo (RGB stays on the device, ADR-013, only scores leave). The report compares definition and adiposity, goal vs current. Regions a front photo cannot show (back) are `N/A` for the goal.
- Labelling: every goal-photo-derived score is shown as **"appearance-based"** (pose, pump, lighting, edits). Warn, don't block, if the full body is not in frame: a cropped body breaks the height normalisation behind every scale-free feature (ADR-024). Reuse the P1 mask edge-contact check and show its number (see mistakes.md, 2026-11-01).
- Realism flag: compare the goal's **shoulder-to-waist ratio** with the ratio the user's own frame allows, because bone width cannot change. Ratios, not widths, so the goal person's size does not matter. Formula and threshold are fixed in P6 from `evals/safety.jsonl`, not guessed here.
- Rejected: requiring a side-view goal (most goal photos don't have one, and the flow would stall); a separate front-only model (two models to ship; dropout gets it from one); hiding the girth-confidence drop (users should see why arms and thighs are less certain).

**ADR-026 · Accepted (amends ADR-005, ADR-017) · Mode 2: "describe your goal" via text -> archetype table, built after the deploy**
- Scope and timing: **P9, after P7 is live; must not delay P7.** The one pre-deploy requirement lands in P4: save every render's parameters (sex, muscle, fat, height, proportions) in `synth/params.csv` and its 4-view masks next to the images, so P9 reuses them and nothing is re-rendered. Masks come from the same segmentation + crop + head removal as real input, run on the clean render, so archetype proportions are comparable with user scans.
- P9 builds an offline **archetype table** from those renders: region proportions via the geometric analyzer (scale-free, ADR-024) + definition labels from `labels.csv`.
- New agent tool `goal_from_text` (**text only, never images**): extracts a pydantic-validated goal (sex, build, muscle level, fat level, region emphasis), confirms the interpretation with the user, applies the same safety floors (no target BF below the sex floor, `privacy-and-safety.md`), then looks up the table -> target `GapProfile` + target definition scores -> the same report and plan flow as Mode 1 (so the realism flag and safety layer apply unchanged).
- Named people become descriptors only ("lean, athletic, defined"). The agent never states a real person's body stats.
- Consequences: (1) a Mode 2 gap is geometric-analyzer quality (baseline A), not Siamese, and its confidence is set accordingly; (2) the extraction is an extra LLM call, so Mode 2 reports may use up to 4 calls where ADR-017 allows 3; (3) the user's goal text reaches the LLM provider, so it gets the same PII normalisation as plan queries and is never logged beyond the intent label (`privacy-and-safety.md` needs a one-line note when P9 starts).
- Rejected: an LLM stating target measurements directly (ungrounded, may quote real people's stats); re-rendering in P9 (cost, delays); sending any image to the LLM.

**ADR-027 · Accepted · Goal step UI: two tabs; Mode 2 preview is a static pre-rendered library**
- The goal step has two tabs: "Upload a goal photo" (default) and "Describe your goal". P7 ships the tab structure with only the upload tab live; P9 turns the second on.
- Preview (P9): a library of ~300–500 pre-rendered front+side images (neutral grey, fitted clothing, no face detail, compressed), served as static assets. The UI shows the render nearest to the parsed goal, and "more muscle / leaner" sliders swap previews client-side. No live rendering, no server cost. Labelled as an illustration of the described goal, not a prediction of how the user will look.
- Stretch, optional and only after the P9 core passes: live 3D preview in three.js with muscle/fat morphs.
- Rejected: live server-side rendering (cost, latency, nothing free to host it, ADR-022); making the three.js preview part of the core (large scope for a nice-to-have).

**ADR-028 · Accepted (supersedes ADR-022 and, through it, ADR-009; amends ADR-007, ADR-008, ADR-017) · API on Google Cloud Run free tier, web on Vercel, models on HF Hub, billing safety first**
- Note: the repo's previous deployment ADR was ADR-022 (local API, HF Spaces dropped). No "Vercel Python Functions" ADR exists; that option survives only as the fallback spike below.
- **Topology:** web = Vercel (`apps/web`, definition ONNX stays a static asset there). API = FastAPI in a Docker container on Cloud Run. Models = public HF Hub model repos (storage + model cards, linked from README and UI footer). The HF Space sync Action and any Vercel API project are out of the plan; CI deploys the API image to Cloud Run (P7, keyless auth via Workload Identity Federation, no JSON key in GitHub).
- **Service config:** region `asia-south1` (Mumbai) if the console offers it, recorded in `dev-log.md`; 1 vCPU / 2 GiB; request-based billing (CPU only during requests); `min-instances=0`; `max-instances` 1–2; concurrency a small number (start at 4, tune in the spike: inference is CPU-bound, LLM waits are not); CORS = the production Vercel origin only (preview-deploy origins cannot call the API unless added on purpose).
- **Free-tier budget (per month, per billing account):** 2M requests, 180k vCPU-s, 360k GiB-s. At 1 vCPU / 2 GiB both compute quotas run out at the same point: 180,000 billed instance-seconds (50 h). That is the number to watch, not requests. Billed time includes startup and I/O waits (an LLM call inside `/plan` bills its whole duration). The spike replaces this estimate with measured numbers.
- **Image (hard budget):** `python:3.12-slim`, no torch. onnxruntime, numpy, Pillow; bge-small via ONNX Runtime int8 plus the `tokenizers` package (assumption, measure it); BM25 via rank-bm25; corpus embeddings precomputed offline, only the query is embedded at runtime. `ml/src/preprocess.py` also imports `scipy.ndimage` today; the spike measures what scipy costs and the decision (keep, or replace `label` with a numpy routine) is made from that number. Design target: image <= ~250 MB as stored, so two versions fit Artifact Registry's 0.5 GiB free allowance (above that ~$0.10/GiB-month). Old versions are removed after each deploy (cleanup policy or CI step, chosen in the spike).
- **Model delivery:** Siamese ONNX baked into the image vs downloaded from HF Hub at container start: decided in the spike by measuring cold start. A download adds startup time and counts as in-memory /tmp.
- **State:** with `min-instances=0` the in-process semantic cache and rate limiter reset on scale-to-zero and differ per instance (ADR-008 amended). Both go behind small interfaces: in-memory implementation for local dev, **Upstash Redis (free tier)** implementation for deployed use. Cache arrives in P5, rate limiter in the P2 `/analyze` skeleton (in-memory) and is switched to Upstash before the service is public (P7). If Upstash proves unusable, the fallback is to document the limitation in the README, not to build another store. Per-IP limiting is not relied on until the spike shows how the client IP arrives (`X-Forwarded-For`, via `/whoami`).
- **Secrets:** Groq/OpenRouter/Upstash keys via Secret Manager (mounted as env vars) or Cloud Run env vars; never in the repo, the image or logs. Cloud Run's own request logs include the client IP; `privacy-and-safety.md` states it.
- **Billing safety (hard requirements, set up before the first deploy):**
  1. Start on the **$300 / 90-day Free Trial** billing account: it is capped at the credit and closes (30-day grace, then workloads are deleted) instead of charging if not upgraded. Trial end date goes in `dev-log.md`; **decide on day 85** whether to upgrade (the free tier continues on a paid account, overage billed).
  2. One project only. No extra projects or billing accounts to dodge limits (Google's terms prohibit it).
  3. Budget of ~INR 100 scoped to that project, alerts at 50/90/100%, publishing to a Pub/Sub topic. Open check: whether the budget nets off trial credits; it must count gross usage, otherwise it never fires during the trial.
  4. **Kill switch:** budget -> Pub/Sub -> Cloud Run function that detaches billing from the project (`infra/kill_switch/`). `SIMULATE_DEACTIVATION` defaults to true; tested in simulation first. It also stops **every** service in the project, including free-tier ones, and resources may be irrecoverably lost. Cost reporting is delayed (notifications arrive several times a day, first one can take hours, delivery is at-least-once), so the cap is **not exact**; Google says the pattern does not guarantee spend stays under the budget.
  5. `max-instances` is set on the service **before** it is made public (deploy private, set the cap, then allow unauthenticated).
  6. Flipping the kill switch from simulation to live is a deliberate, separate step before P7 goes public (it needs a real trip test; billing is re-linked by hand afterwards).
- **Gates revised:** Siamese ONNX inference < 150 ms measured **on the deployed container (1 vCPU)**; Siamese model file <= 50 MB (int8 if needed); P7 deploy = Vercel (web) + Cloud Run (API) + HF Hub (models); P7 "stranger from a public URL" gate is no longer blocked.
- **Deploy spike (gate before P3):** FastAPI container on Cloud Run importing onnxruntime + numpy + Pillow, loading a ~50 MB dummy ONNX, answering a POST of a ~100 KB PNG. Logged in `dev-log.md`: image size, cold start, warm latency, usage and billing report showing $0, budget alert created, kill switch tested in simulation, how `X-Forwarded-For` arrives. If signup or billing fails (card/UPI), stop and tell the user; fallback is a Vercel Python Functions spike.
- **Verified 2026-11-01** (provider pages via search; re-check in the console): free-tier amounts above are the Tier 1 figures and Mumbai is listed as Tier 1 on the [Cloud Run locations page](https://docs.cloud.google.com/run/docs/locations) (Tier 2 regions get 128,571 vCPU-s / 257,142 GiB-s; free tier is aggregated per billing account across projects) · [Cloud Run pricing](https://cloud.google.com/run/pricing) · Free Trial 90 days / $300, no action needed to avoid charges, an INR billing account is required in India, **UPI is not mentioned anywhere I found** ([FAQ](https://cloud.google.com/signup-faqs?hl=en)) · [disable-billing example](https://docs.cloud.google.com/billing/docs/how-to/disable-billing-with-notifications) (Cloud Run functions Gen 2, sample has a simulation flag, grants `roles/billing.admin` on the billing account to the function's service account) · [Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing) (0.5 GB free, shared across the billing account's projects) · [HF Hub storage](https://huggingface.co/docs/hub/storage-limits) (public repos free, "best effort").
- **Not yet verified (check at the step that needs it):** Upstash free-tier limits (before P5); Secret Manager free allowance; a narrower billing role than `billing.admin` for the kill-switch service account; Artifact Registry cleanup-policy syntax; Cloud Run egress for our response sizes.
- Rejected: Vercel Python Functions for the API (user's choice of Cloud Run; I did not verify its limits; kept only as the fallback spike); HF Spaces (ADR-022); always-on CPU or `min-instances>=1` (idle billing breaks the free tier); creating extra projects/accounts for more quota (terms); torch in the image (size).

**ADR-030 · Accepted (refines ADR-023 and ADR-024; builds on ADR-029) · Canonical frame, row finder and features as built in P2**
- Canonical frame (`ml/src/preprocess.py`): browser masks (384x512, head-less) pass through unchanged (`from_browser`: validate, largest component), because re-cropping a head-less mask would rescale the body. BodyM full-body masks go through `from_bodym`: the browser's exact crop and letterbox, then a neck cut. Body height in the frame is `2 * (feet_row - 256)` (the letterbox is centred), so no keypoints are needed. `NECK_CUT_OFFSET = 0.0`: parity on 3 captures, |z| <= 1.9 on the head cut; phone captures cut higher, left to P3 cut-row jitter augmentation.
- Row finder (`analyzer/rows.py`), no centre column: armpit = first row with >= 3 runs held 5 rows; crotch = first row in 0.45-0.65 of body height where the widest run drops below 65% of its maximum since the armpit, held 8 rows. This replaces "where the central run splits" from ADR-024 (the centre column drifts with lean: found 66.8% of BodyM crotches, now 98%). Outside 0.45-0.65 the crotch counts as not found and a prior is used: shorts hide it (hem gap measured 0.667-0.694 vs BodyM 0.583 +/- 0.028). Waist and hip come from the widest-run profile.
- Features (`analyzer/features.py`), all divided by the view's own body height: girth = ellipse from front width and side depth (chest, waist, hip); arm thickness = area / path length, independent of the A-pose angle (upper arm window 0.00-0.07 below the armpit); thigh and calf use the peak front width (side depth hurt them: hands in front of the thigh, clipped feet); shoulder row = armpit - 0.07. Windows swept on train (`ml/src/sweep_windows.py`), checked on val, reported on testA and testB.
- Forearm: the upper-arm window also scores rho 0.78 against forearm circumference, but that is general arm size. Rejected as the forearm feature; the forearm keeps its own window (rho ~0.3) and near-zero confidence.
- Clipped BodyM side masks (26% touch the bottom edge, random in sex and height): excluded from every side-view fit and eval.
- Confidence per region = clip((rho_val - 0.3) / 0.6, 0, 1), times 0.7 per row-finder fallback; `upper_back` and `lats` (back view, no label) are fixed at 0.3. Left/right asymmetry is not implemented (no ground truth to validate it).
- Gap = per-region slope x (goal feature - current feature), slopes fit on train (`gap_coef.json`). Waist model `waist_coef.json`. Both derive from BodyM (CC BY-NC 4.0).
- For the deploy spike: the API imports `ml/src/preprocess.py` (so the image must carry the `ml` package files it needs) and `scipy.ndimage` (ADR-028 asks the spike to measure its cost).

**ADR-031 · Accepted (amends ADR-028: spike timing and the P3 gate) · Deploy spike and billing safety deferred to deploy time**
- Decision (user, 2026-11-01): the Cloud Run deploy spike, the budget and the kill switch are done when we deploy (start of P7), not now. P3 is no longer blocked by the spike.
- Unchanged and still hard: nothing is deployed to Cloud Run before the budget, the kill switch (simulation first) and `max-instances` exist; one project; trial billing account. The runbook is the "Deploy spike + billing safety" block in `phase-plan.md` P2; the first P7 item points to it.
- Done already: project `physique-gap-3012` with the trial billing account linked, `gcloud` logged in, APIs `billingbudgets`, `pubsub`, `cloudbilling` enabled, Pub/Sub topic `budget-alerts`. No budget, no function, no service. The free trial clock is running meanwhile (90 days from signup), so P7 should start with enough of the trial left; the day-85 decision date is logged in `dev-log.md` once the signup date is known.
- Consequence for P3: the "ONNX < 150 ms on the deployed 1 vCPU container" gate is measured locally on one CPU thread as a proxy and re-confirmed at the P7 spike. The Siamese file limit (<= 50 MB) is checked in P3 as before. If the deployed number misses, the model is slimmed then (int8 or smaller input), at the cost of a late surprise; that risk is accepted.
- Rejected: dropping the safety steps with the spike (a public service without a cap is the one thing ADR-028 forbids).
