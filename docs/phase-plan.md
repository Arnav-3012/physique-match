# Phase plan
Status values: `TODO` · `IN PROGRESS` · `DONE`. Exactly one phase is `IN PROGRESS`.
Each phase ends with the end-of-phase protocol in `CLAUDE.md` (logs, ADRs, mistakes, tag). `/sensei` -> `learn/phase-NN/` only for the major phases: P3, P4, P5, P6.

**Track A — demo (P0–P7):** capture -> data + baseline -> **Siamese CNN** -> **muscle definition** -> RAG -> agent -> deploy.
**Track B — best (P8–P10):** real data, product depth, hardening. Do not start B until A is live.
Data rule (ADR-020): demo models train on **BodyM + synthetic only**. Real photos come in P8.
Hosting (ADR-028): web on Vercel, API on **Google Cloud Run free tier**, models on HF Hub. The deploy spike and billing safety (budget + kill switch) are deferred to deploy time (ADR-031) but come before the first deploy; see the block inside P2 and the first P7 item.
**Goal input has two modes (ADR-025..027).** Mode 1, goal photo (usually front-only): **priority, ships in P7** (touches P3, P4, P7). Mode 2, "describe your goal" (text -> archetype table): **P9 only, must not delay P7**; its single pre-deploy cost is the P4 render-params + masks save.

| Phase | Outcome | ~Time |
|---|---|---|
| P0 | Repo, schemas, empty deploys live | 1 d |
| P1 | Private 4-view capture in browser | 4 d |
| P2 | Deploy spike (Cloud Run) + BodyM pipeline + geometric baseline + body comp | 4.5 d |
| **P3** | **Siamese CNN (core model)** | 1 wk |
| **P4** | **Muscle-definition model (synthetic, on-device)** | 1.5 wk |
| P5 | Hybrid RAG + query manager + semantic cache | 4 d |
| P6 | Agent + planner + safety | 4 d |
| **P7** | **Demo deploy** | 2 d |
| P8 | Real data + fine-tune + validation | 1 wk |
| P9 | Product depth + describe-your-goal (Mode 2) | 2 wk |
| P10 | Hardening & polish | 4 d |

---

## P0 · Foundation — `DONE` (~1 day)
- [x] Monorepo scaffold per README folder tree; `uv` for Python, Vite+React+TS for web
- [x] `.gitignore` (ml/data, synth/out, *.onnx, .env, validation-set/), pre-commit (ruff; prettier hook dropped, see mistakes.md)
- [x] Pydantic schemas in `packages/schemas`: `Scan`, `GapProfile`, `BodyComp`, `DefinitionScores`, `UserProfile`, `Report` (+ generated TS types)
- [x] GitHub repo, push; Vercel project linked (web only). At P0 the API and models stayed local (ADR-022); superseded by ADR-028: API on Cloud Run, models on HF Hub, from the P2 deploy spike on
- [x] Groq + OpenRouter keys in local `.env`; `/health` pings both
**Gate:** `pytest -q` green on schema tests; hello-world web on Vercel URL; `/health` on localhost:8000 shows both providers `ok`.

## P1 · Private capture in the browser — `DONE` (~4 days)
- [x] MediaPipe Pose Landmarker (mask only, ADR-023; no Image Segmenter) on live webcam (laptop/phone) + file upload
- [x] Silhouette preview mode (camera feed hidden by default), pose outline guide
- [x] Guided turn capture (ADR-011): user-paced hold, yaw from keypoints, sharpest frame per view (front / left / back / right), retake per view
- [x] Quality checks: full body in frame, lighting, loose-clothing warning
- [x] Head removal + binarise + resize -> `Scan {views[mask, keypoints]}`; keep the 4 RGB frames **in memory only** for the definition model (P4), then discard
- [x] Consent screen (itemised notice), 18+ gate, "what we send" preview
**Gate:** 10 test captures (laptop + phone): 10/10 produce 4 valid masks; network tab shows no image except 1-bit masks; capture ≤ 12 s.
**Result (2026-11-01):** ~8/10 captures valid, accepted by the user; best turn 7.6 s; network tab all localhost (user-reported). See `docs/logs/dev-log.md`.

## P2 · Deploy spike + BodyM data pipeline + geometric baseline + body comp — `DONE` (BodyM part; spike deferred to P7, ADR-031) (~4.5 days)
### Deploy spike + billing safety (ADR-028) — `DEFERRED` to deploy time by the user (ADR-031); does not block P3
Run this whole block at the start of P7 (or earlier if the user says so), before anything is deployed. Already done: APIs `billingbudgets`, `pubsub`, `cloudbilling` enabled; Pub/Sub topic `budget-alerts` created in project `physique-gap-3012`. Nothing is deployed and no budget exists yet.
Order matters: account and billing safety, then code. The user does account/billing setup; Claude writes files only and gives one exact command at a time.
- [x] (done 2026-11-01; trial dates still to log) User checklist: Google account -> Free Trial billing account (INR; if card/UPI signup fails **stop and tell Claude**, fallback = Vercel Python Functions spike) -> one project -> link billing -> install + log in to `gcloud` -> enable only the APIs the steps need (Claude names each at the step that needs it)
- [ ] Log trial start/end dates and the **day-85 upgrade decision** date in `dev-log.md`
- [ ] Budget (~INR 100, project-scoped, alerts 50/90/100%, Pub/Sub topic attached); check it counts gross cost, not net of trial credits; budget alert created
- [ ] Kill switch (`infra/kill_switch/`): function deployed, `SIMULATE_DEACTIVATION=true`, tested with a synthetic over-budget Pub/Sub message; log shows "(Simulated)", billing still linked
- [ ] Confirm region in the console (`asia-south1` if offered) and record it
- [ ] Spike service (`spikes/cloudrun/`): FastAPI + onnxruntime + numpy + Pillow, loads a ~50 MB dummy ONNX, `POST /infer` takes a ~100 KB PNG, `GET /whoami` echoes how the client IP arrives; Artifact Registry repo with old-version cleanup
- [ ] Deploy **private** with `--max-instances` set (and 1 vCPU / 2 GiB, request-based billing, min 0); only then allow public access
- [ ] Measure: image size (as stored), cold start (baked model vs fetched at start; scipy in/out), warm latency (p50/p95 over >= 30 requests), `X-Forwarded-For` shape; decide bake vs fetch
- [ ] Usage + billing report shows $0 for the spike
**Spike gate (before the first deploy, ADR-031):** all of the above recorded in `docs/logs/dev-log.md`: image size, cold start, warm latency, usage/billing report at $0, budget alert created, kill switch tested in simulation. Nothing goes to Cloud Run before the budget and kill switch exist. Whether bake-vs-fetch and the scipy decision are made here changes nothing in P3 or P5 design beyond the size limits already stated.

### BodyM pipeline, geometric baseline, body comp
- [x] Download BodyM (`aws s3 cp --no-sign-request s3://amazon-bodym ...`) into `ml/data/` (gitignored); EDA: subjects, repeats, measurement distributions, sex balance
- [x] Subject-level split (train / val from Training; Test-A, Test-B kept untouched); map 14 measurements -> 8 labelled regions (`upper_back`, `lats` have no label)
- [x] `ml/src/preprocess.py` (shared with API + browser parity test): crop, head removal, height-normalise, resize
- [x] Parity check: same preprocessing on a BodyM silhouette and on a browser-made mask of you -> visually and statistically comparable (3 captures, 12 masks; the planned 5 were not all taken)
- [x] Geometric analyzer (no ML): silhouette-only rows, widths/depths, ellipse girths -> **baseline A** and fallback. **Left/right asymmetry not done** (no ground truth to validate it); chest/back split not done
- [x] Waist calibration on BodyM (silhouette -> real waist, MAE cm) -> `BodyComp`: RFM BF% range, optional tape override, Mifflin RMR + energy ranges with floors
- [x] FastAPI `/analyze` skeleton (mask validation, payload cap, rate limit behind an interface (in-memory now, Upstash before P7), CORS = Vercel origin only) behind `Analyzer` interface; imports only what the Cloud Run image will carry (no torch)
**Gate:** no subject in two splits (test); geometric Spearman ρ ≥ 0.6 on ≥ 6 of the 8 BodyM-labelled regions (Test-A; `upper_back` and `lats` have no label) **and silhouette features beat a null ridge on [height, weight, sex]** (ADR-024); waist MAE ≤ 5 cm and below the null model's MAE; RFM/Mifflin unit tests; rejects RGB upload; gap unchanged when only weight changes.
**BodyM part result (2026-11-01):** gate 2 criterion changed after the first result (ADR-029): silhouette-only rho >= 0.6 on 7/8 regions (testA and testB), and the silhouette adds signal beyond the null on 7/8 (testA) but only 5/8 (testB). Waist MAE 2.70 cm on testA (null 3.00), 3.39 on testB (null 3.50). Test suite green (user-reported); full numbers in `docs/logs/dev-log.md`. The deploy spike is deferred (ADR-031), so the BodyM gate is the P2 gate.

## P3 · Siamese CNN — core model — `IN PROGRESS` (~1 week)
Close-out learn docs (`learn/phase-03/`, same format for P4-P6): each concept explained with our real files and numbers, plus glossary and 10 viva questions; topic list in the user's memory `learn-docs-format`.
Gate protocol (ADR-033) and target (ADR-032) are fixed before any Test-B scoring. Detailed plan: `/Users/arnav/.claude/plans/quiet-exploring-pillow.md`.
- [ ] Pre-work: fix `extract.py` testB/testA keying (7 photos); pair-list generator + hash; power check on val; `evaluate.py --test-b` ledger (refuses a second run per model hash)
- [ ] Pair sampler: subject-first, in-batch all-pairs (encode each subject once per variant, head on all ordered pairs) from the train split; target Δ = m_A/h_A − m_B/h_B per region (difference of height-normalised ratios, `height_cm`, ADR-032); antisymmetry augmentation (swap pair, negate target)
- [ ] Mask augmentation to match browser masks: start with swap, goal side-dropout and small affine; add edge noise, morphological jitter, clothing-bulge sim, neck cut-row jitter only if val shows the need
- [ ] **Baseline B:** single ResNet-18 regressor (silhouettes -> measurements), run on both, subtract
- [ ] Siamese ResNet-18 (ImageNet, 1ch->3ch, front+side input), head on [e_g, e_c, e_g−e_c, e_g⊙e_c]; Huber loss
- [ ] **Side-view dropout (ADR-025):** goal side input randomly blanked in training (current keeps both views) so a front-only goal photo is a valid input; confirm the `Scan` schema accepts a goal with a front view only (side optional) and add a schema test
- [ ] Overfit 1 batch -> linear probe -> unfreeze last 2 blocks -> full fine-tune; stop at first rung that passes the gate; MPS, log every N steps
- [ ] Per-region confidence = calibrated val residual std (no MC dropout), separately for full-input and front-only slices, -> `GapProfile`; front-only goals show "widths only, lower confidence"; goal `Scan` requires only `front`, current keeps 4 views (validated separately); `GeometricAnalyzer` never raises on a missing view; width-only baseline A coefficients fitted
- [ ] **Front-only eval slice** on Test-B (goal side blanked); baselines A and B run on the same slice (A: widths only, B: front-only regressor)
- [ ] Export ONNX (**file <= 50 MB**, int8 if needed), model card in `ml/MODEL_CARD.md` (data, splits, metrics, limits, BodyM CC BY-NC), weights to an HF Hub model repo; plug into `/analyze` (geometric stays for back view + asymmetry)
**Gate (ADR-033):** vs each of baseline A and baseline B separately, the 95% subject-bootstrap CI of (MAE_baseline − MAE_siamese) > 0 on >= 6/8 regions on **Test-B** (in-the-wild), and no region significantly worse; same fixed pair list for all; front-only slice judged against width-only baseline A with the same rule; Test-B scored once per final candidate. ONNX inference <= 75 ms measured locally on CPU with 1 thread as a proxy, **150 ms re-confirmed on the deployed Cloud Run container (1 vCPU) at the P7 spike** (ADR-031); model file <= 50 MB; ordering check on your own scan vs a goal photo looks sensible. If it doesn't beat baseline B -> ADR-006 negative result, ship the better model, document it.
**Gate report also includes (ADR-025):** the front-only slice on Test-B, per-region MAE vs baselines A and B, width regions and girth regions listed separately, next to the full-input numbers so the cost of side-view dropout is visible.

## P4 · Muscle-definition model — synthetic, on-device (~1.5 weeks)
- [ ] `synth/`: Blender + MPFB (MakeHuman) headless script; grid: sex × muscle × fat × height × proportions × skin tone × lighting (3 HDRIs) × fitted clothing × 4 views; start 5k renders, scale to ~30k
- [ ] Labels from sliders via fixed mapping -> definition 1–5 per region (abs, chest, shoulders, arms, back, legs) + adiposity 1–5; `synth/labels.csv` + mapping doc
- [ ] **Save render parameters + masks for P9 (ADR-026, pre-deploy requirement):** every render's sex, muscle, fat, height and proportion parameters go in `synth/params.csv` (keyed by render id, same id as `labels.csv`), and its 4-view masks next to the images (`synth/out/`, gitignored). Masks come from the same segmentation + crop + head removal as real input, run on the clean (pre-augmentation) render. No re-rendering in P9.
- [ ] Domain-gap mitigation: run the **same browser segmentation + crop** on renders; background swap with real CC0 room photos; noise, blur, JPEG, colour jitter
- [ ] MobileNetV3-Small (ImageNet) shared trunk + 7 ordinal heads; split by render identity; overfit 1 batch first
- [ ] Export ONNX, int8 (≤ 5 MB), run with ONNX Runtime Web (WebGPU -> WASM)
- [ ] Fusion: adiposity nudges RFM BF% range (never beyond RFM bounds); UI badge "experimental — trained on synthetic data"
- [ ] **Goal-photo inference (ADR-025):** the same in-browser model also scores the single goal photo (front frame), so the report compares definition and adiposity, goal vs current. Heads for regions a front photo cannot show (back) are `N/A` for the goal. Eval adds a front-only slice (front view only as input).
**Gate:** synthetic held-out Spearman ρ ≥ 0.8 per head, MAE ≤ 0.6 (front-only slice reported alongside); ≤ 300 ms on laptop and mid-range phone; sanity ordering on your own scans and on 3+ goal-style photos logged honestly in `dev-log.md`; `synth/params.csv` + masks exist for every render id in `labels.csv`.

## P5 · Knowledge base + hybrid RAG + semantic cache (~4 days)
- [ ] Corpus: open exercise library (licence-checked ADR), muscle-group guides, volume and nutrition guidelines summarised from open-access reviews (cited)
- [ ] Chunk with metadata (muscle, equipment, level, goal, source)
- [ ] **Evals first:** `evals/rag.jsonl` (40 queries), `evals/cache.jsonl` (60 should-hit / must-miss pairs)
- [ ] BM25 (rank-bm25) + bge-small dense (ONNX Runtime int8, no torch/sentence-transformers in the API; corpus embeddings precomputed offline; tokenizer package size measured) + RRF; metadata filters; image stays inside the ADR-028 size budget
- [ ] Query manager: normalise -> embedding intent router -> region decomposition -> filters
- [ ] Semantic cache: namespaced, threshold τ from cache eval; behind an interface, in-memory locally, **Upstash Redis** when deployed (instances scale to zero, ADR-028); verify Upstash free-tier limits first, else document the limitation
**Gate:** recall@5 ≥ 0.85; hybrid beats each single retriever; cache false-hit ≤ 2%.

## P6 · Agent, planner, report (~4 days)
- [ ] **Evals first:** `evals/plan.jsonl` (15 profiles incl. gap + definition), `evals/safety.jsonl` (40 risky inputs)
- [ ] Deterministic planner: split by days/week, extra sets for top-3 gap regions, definition-aware notes, nutrition from `BodyComp`
- [ ] One agent + tools, Groq gpt-oss-120b, OpenRouter no-train fallback (ADR-017); pydantic `Report`; citations
- [ ] Safety layer: banned words, realism flag, energy floors re-checked post-LLM, unsafe-intent response
- [ ] `/plan` endpoint; ≤ 3 LLM calls per report; 429 -> fallback
**Gate:** 100% schema-valid; plan rubric ≥ 4/5; safety 40/40; every exercise cites a chunk; no number differs from `GapProfile` / `BodyComp` / `DefinitionScores`.

## P7 · Demo deploy — `MILESTONE: working demo live` (~2 days)
- [ ] Web flow: consent -> goal upload -> 4-view scan -> profile -> report -> plan (+ PDF)
- [ ] **Goal photo = Mode 1, must ship (ADR-025):** front-only goal accepted (side optional); goal step built as two tabs with only "Upload a goal photo" live (the "Describe your goal" tab arrives in P9, ADR-027); warn (not block) if the full body is not in frame, reusing the P1 mask edge-contact check and showing its number; goal-derived scores labelled "appearance-based" (pose, pump, lighting, edits); definition + adiposity shown goal vs current
- [ ] Realism flag: goal shoulder-to-waist ratio vs the ratio the user's own frame allows (bone width cannot change); formula and threshold set from `evals/safety.jsonl` in P6, wired into the report here
- [ ] Visuals: silhouette overlay, per-region gap bars with confidence (lower confidence shown for girth regions on a front-only goal), definition bars, BF% range, energy card
- [ ] Deploy = Vercel (web) + Cloud Run (API) + HF Hub (models) (ADR-028): Siamese ONNX served by the Cloud Run container; definition ONNX as static asset (Vercel) with caching; model cards on HF Hub linked from README and UI footer
- [ ] GitHub Actions: tests; Vercel auto-deploy for web; CI builds and deploys the API image to Cloud Run (keyless Workload Identity Federation) and removes old image versions. No HF Space sync, no Vercel API project
- [ ] **First:** run the deferred "Deploy spike + billing safety" block in P2 (budget, kill switch in simulation, private deploy with max-instances, measurements). Nothing is deployed before it passes
- [ ] Before the service is public: `max-instances` set, Upstash-backed cache + rate limiter live, CORS = Vercel origin only, secrets via Secret Manager/env, real `X-Forwarded-For` handling from the spike
- [ ] Arm the kill switch: deliberate real trip test (`SIMULATE_DEACTIVATION=false`) as a separate user-approved step; re-link billing by hand afterwards; document that it stops every service and the cap is not exact
- [ ] Day-85 trial check done and logged (upgrade or let the trial close)
- [ ] README: GIF, architecture diagram, metrics table (baselines vs Siamese, synthetic definition scores), limitations, licences
**Gate:** full flow completes in < 4 min **from the public Vercel URL against the Cloud Run API** (cold start included once) **with a front-only goal photo** (girth regions show reduced confidence, "appearance-based" labels and the full-body warning appear when expected); billing report still at $0 (or within the free tier) after the load; Siamese inference < 150 ms on the deployed container.

---

## P8 · Real data, validation & fine-tune (~1 week)
- [ ] Real eval set (ADR-018): own scans + tape measurements over time, consented friends, licensed Wikimedia Commons images
- [ ] Measure: Siamese gap error vs tape deltas, waist MAE, RFM vs tape-Navy, definition ρ vs blinded ratings
- [ ] Fine-tune definition model on real set (subject-level split); remove "experimental" badge only if ρ ≥ 0.5
- [ ] Subgroup error (sex, BMI band) in model cards

## P9 · Product depth + "describe your goal" (~2 weeks)
Mode 2 (ADR-026, ADR-027). Starts only after P7 is live; uses the P4 saved params + masks, no re-rendering.
- [ ] **Evals first:** `evals/goal_text.jsonl` (extraction fields, named-person -> descriptors only, below-floor BF requests, ambiguous input -> clarify), added as a suite to `evals.run`
- [ ] Archetype table (offline script): for each P4 render id, region proportions from the saved masks via the geometric analyzer (scale-free, ADR-024) + definition labels from `labels.csv`, keyed by (sex, muscle, fat, height, proportions)
- [ ] Agent tool `goal_from_text` (text only, never images): pydantic-validated goal = sex, build, muscle level, fat level, region emphasis; user confirms the interpretation; same safety floors (no target BF below the sex floor); lookup -> target `GapProfile` + target definition scores -> same report and plan flow. Real people are translated to descriptors ("lean, athletic, defined"); the agent never states a real person's body stats
- [ ] Goal preview library: ~300–500 pre-rendered front+side previews (neutral grey, fitted clothing, no face detail, compressed, static assets); nearest render to the parsed goal, "more muscle / leaner" sliders swap previews client-side; no live rendering, no server cost
- [ ] UI: enable the "Describe your goal" tab on the goal step
- [ ] *Stretch (optional, after the above passes):* live 3D preview in three.js with muscle/fat morphs
**Gate:** `goal_from_text` eval passes (schema-valid 100%, named-person and below-floor cases 100%); Mode 2 end to end produces the same `Report` schema as Mode 1; preview swap is client-side only (network tab: no request on slider move); library size and load time logged.

### Product depth (original P9 scope)
- [ ] Progress tracking: re-scan every 4 weeks, trend chart (browser-only storage, export/import JSON)
- [ ] Follow-up chat over the plan (same agent, cache reused)
- [ ] Constraint-aware re-plan ("no barbell", "knee pain"), diet preference (veg / non-veg)

## P10 · Hardening & polish (~4 days)
- [ ] Observability: latency, cache hit rate, LLM calls/report, provider fallback rate
- [ ] Load test; rate limits; `/security-check` on API; accessibility pass
- [ ] Final README, demo video, blog post
