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
| P7.1 | Multi-user: database for user inputs and plans, proper framework (**not planned yet**) | TBD |
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
- [ ] **Front-only eval slice** on Test-B (goal side blanked); baselines A and B run on the same slice (A: widths only, B: the same single regressor with the goal's side view blanked, ADR-033)
- [ ] Export ONNX (**file <= 50 MB**, int8 if needed), model card in `ml/MODEL_CARD.md` (data, splits, metrics, limits, BodyM CC BY-NC), weights to an HF Hub model repo; plug into `/analyze` (geometric stays as the fallback; its back-view numbers are retired, ADR-037)
- [ ] **Lats + sex handling, strictly in this order (ADR-036, ADR-037); each step starts only when the previous is done and logged:**
  1. [x] ADR-037 written (files only)
  2. [x] Audit: grep that no other ranked region uses a sex term (user ran it 2026-11-02: none; sex appears only in body composition, null-ridge references, splits, tests and the schema; see dev-log)
  3. [x] ADR-036: ANSUR II fit run 2026-11-02: gate PASS (min dR2 .048, CI [.041, .055]), R2_min .408 -> **verdict `unranked`** (shown as "back width (estimate)", not in the priority ranking); numbers in dev-log
  4. [-] Dropped (ADR-037 Amendment 3, user 2026-11-02): no pooled ridge; `lats` needs user_sex and goal_sex set and equal
  5. [x] Schema: `UserProfile.sex`, `GoalProfile.sex` (default `not_specified`), regenerate TS types, `test_schemas.py` (pytest 149 passed 2026-11-02; the TS types were NOT truly regenerated: `npm run types` reads `schema.json`, which only `python -m physique_schemas.export_schema` rewrites. Fixed together with 6a)
  6. [x] (6a done: sex unset -> no body-fat %/calories, `body_fat` optional, 152 tests passed; 6b-1 done: both analyzers return `upper_back` and `lats` as `measured: false`, 150 passed; 6b-2 done 2026-11-02, 162 passed: `app/analyzer/lats.py` applied in the route, `RegionGap.reason`, `AnalyzeResponse.warnings`, schema and TS types regenerated; mode check done: only `p.goal` sets the mode, body fat only feeds the deficit opt-out; `rmr()` now raises on `not_specified` instead of using the female term) Calculation: `lats` is `measured: false` with a reason unless both sexes are set and equal (report-level warning for a different-sex goal); no pooled route; unset user sex -> no body-fat %, no calories, no `lats` (ADR-037 Amendment 4; `BodyComp.body_fat` optional, "set your sex" note; fix the `BF_FLOOR[p.sex]` / `LEAN_RFM[p.sex]` KeyError and the Mifflin non-male-is-female term); check that nothing derives cut/recomp/bulk from body-fat % (the mode stays user-chosen); null baseline, if shown, same condition
  7. [x] (168 passed 2026-11-02: `test_lats.py`, route and bodycomp tests) Tests: missing sex no error; `lats` blocked with a reason for differing or unset sexes; warning for a different-sex goal; B byte-identical with and without sex fields; unset sex gives `body_fat = None`, no energy targets and the note, while gaps still compute, and the mode is untouched
  8. [x] (done 2026-11-02: privacy doc + consent text, model card, README, dev-log; the web app has no sex field, region labels or "not measured" handling yet, so the UI wording moved to the P7 visuals task, no UI files invented) Docs: model card limitation, privacy doc + consent text, dev-log; label "back width (estimate)" in UI, README and report (API key stays `lats`); the sex field reads "needed for body-fat %, calories, lats and the V-taper ceiling" and is never defaulted. (Report note and agent label wired in P6.)
- [ ] **Frame V-taper, M1 (ADR-038)**, each task one commit:
  1. [x] ADR-038 written (files only; the three "mine" items await the user's veto)
  2. [x] (173 passed, TS regenerated once, 2026-11-02; `reliability` is `None` until step 3 writes the block) `GapProfile.v_taper` schema field + calculation in `SiameseSubtractAnalyzer` (index = shoulders / waist from the per-person outputs, delta = goal - current, `None` on the geometric fallback, `measured: false` if shoulders or waist is not measured); unit test with fixed ratios; the all-regions check unchanged; regenerate TS types (2 h). Batch the schema change with step 6's `lats` reason / warning fields if step 6 starts first, to regenerate once
  3. [x] (done 2026-11-02: reliability .64 full / .64 front-only, script commit 62b7348c on a clean tree, 176 passed; two commits, script first, then the JSON) Reliability of the index from val residuals (full and front-only slices, `s_prior` from val pairs), written as a `v_taper` key into `gap_regressor.json` without re-exporting the ONNX, with the script's commit hash (and a clean-tree flag), the checkpoint `model_hash`, the ONNX file SHA-256 and a block version; test that the block's `model_hash` equals the top-level one and that typical values do not clip to 0; no Test-B look (2 h). Commit the script before running it so the hash is real
- [ ] **Reachable ceiling, M2 (after step 6; needs the user's sex, `measured: false` without it, ADR-037 Amendment 4).** Starts with its own ADR (BF-floor range, inversion, one-sided flag): **ADR-040 written 2026-11-02 (files only)**; the items below follow it:
  0. [x] (done 2026-11-02, script commit 22a1ec34 on a clean tree; male floor extrapolated, male 13% not (4 subjects below), female both ends extrapolated; `ceiling_extrapolation.json` next to `lats_coef.json`; numbers in dev-log) Extrapolation check (ADR-040): inverted waist/height at both ends vs the ANSUR II minimum waist/stature per sex, plus the count of subjects below each; one run, stored with the script's commit hash; `extrapolated` flag per sex and end. Run before 2, as the ceiling carries the result
  1. [x] (done 2026-11-02, pending the user's test run: the constant, `SUPPORT_MIN` and the ordering test came early with ADR-041; `waist_at_bf(bf, height_cm, sex)` and its round-trip test, also checked against the script's copy, added after) `bodycomp.py`: new constant `BF_ATHLETE_UPPER` (male 13, female 20) beside `BF_FLOOR`; `waist_at_bf = 20 * height / (64 + 12 * female - BF)` at both ends; tests: `BF_FLOOR < BF_ATHLETE_UPPER` per sex, and waist -> RFM -> inverse round-trips (2 h)
  2. [x] (done 2026-11-02, 204 passed; `Ceiling` + `ZoneLabel` on `VTaper.ceiling`, `analyzer/ceiling.py`, applied in the route after `with_lats`; `Ceiling.zone` is in the schema and stays `None` until step 3, so the TS types are regenerated once) ceiling = the user's B-predicted shoulder ratio / waist_at_bf ratio, as a range (floor end and athlete-upper end), plus the ADR-041 / ADR-042 `label_case`, `support` and exact `text` for zone 1 (high end) and zone 2 (floor end) from the stored step 0 JSON; `measured: false` with "set your sex" if sex is missing; a `ceiling` part on `VTaper`, applied in the route like `with_lats`; one TS regeneration (1 h)
  3. [x] (done 2026-11-02, pending the user's test run: `Ceiling.zone` + `zone_text`, `zone_of` in `analyzer/ceiling.py`, `ZONE3_TEXT` in `bodycomp.py`; the schema gained `zone_text`, so a second TS regeneration) (zone-1 and zone-2 wording fixed beforehand by ADR-041 / ADR-042, before any step 3 code or output: `SUPPORT_MIN` 30, three label cases from the ANSUR support at the high end / floor end; do not tune it here; zone 3 is worded with the "cannot verify" caveat when the floor end is extrapolated) One-sided realism flag, three zones (ADR-040; supersedes this item's earlier "above the ceiling's upper end"): goal index <= ceiling at the athlete-upper end = reachable on frame alone; between = near the floor only, best case, not sustained; above the ceiling at the floor = needs more than frame. No free threshold. This pulls the P7 realism-flag item forward; its P6 safety-eval threshold stays for any margin added later (2 h)
  4. [x] (done 2026-11-02, script commit c1cf6f91 on a clean tree, `evals/unique_share.json`: female .148 (n 98), male .171 (n 153), min .148 < .2 -> **DROP**, the index gap stays display-only and is not a P6 planner input) **Decision point:** apply the ADR-038 rule once: per sex, same-sex val ordered pairs, ground-truth ratios, `unique_share = min over sexes of 1 - R^2(index gap ~ waist gap)`, threshold 0.2 fixed. Yes -> the index gap becomes a planner input (P6); no -> drop it (the likely outcome, ADR-038)
- [ ] **Back regions (ADR-039). User order: 7 -> 1 -> 3 -> 4; 5 and 6 are wait-gated, decided from results, not scheduled.** `upper_back` size is `measured: false` always, fixed reason "Not measurable from a front photo; a back silhouette adds no size information."
  - **7. Planner baseline-volume invariant:** contract recorded (ADR-039); code and test land in P6 because they need the P5 corpus (cited baselines) and the P6 planner, which do not exist, and no baseline may be invented. Tasks sit in P5 and P6 below
  - **1. = step 6b-2** (`lats` as "back width (estimate)", reason shown, needs both sexes set and equal): nothing in it depends on 7, so it proceeds now
  - **3.** back definition from the user's own back view: after P4 (schema prerequisite in P4)
  - **4.** optional goal back photo: P7, after 3 works
  - **5. WAIT** lat flare: gate in P4. **6. WAIT** shoulder height difference: gate in P8
**Gate (ADR-033):** vs each of baseline A and baseline B separately, the 95% subject-bootstrap CI of (MAE_baseline − MAE_siamese) > 0 on >= 6/8 regions on **Test-B** (in-the-wild), and no region significantly worse; same fixed pair list for all; front-only slice judged against width-only baseline A with the same rule; Test-B scored once per final candidate. ONNX latency rule (ADR-035: single-encode p50 <= 75 ms and two-encode p95 <= 150 ms, 1 thread, local proxy), **150 ms re-confirmed on the deployed Cloud Run container (1 vCPU) at the P7 spike** (ADR-031); model file <= 50 MB; ordering check on your own scan vs a goal photo looks sensible. If it doesn't beat baseline B -> ADR-006 negative result, ship the better model, document it.
**Val result (2026-11-02, not the gate):** the learned pair head failed against subtraction, 1/8 at `full` (ADR-034); final candidate is baseline B at `full`, 8/8 vs A on both slices (ADR-035). The Test-B gate is now that model vs A, on both slices.
**Test-B result (2026-11-02):** baseline B `full` seed 0 passes the ADR-033 rule vs A on both slices, 8/8 each, none worse; numbers in `docs/logs/dev-log.md`. Still open: ONNX <= 75 ms, file <= 50 MB, own-scan ordering.
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
- [ ] **Schema for per-head N/A (ADR-039 #3; prerequisite of the back definition):** today `DefinitionScores.back` is a required `Score` and only the whole `Report.definition` can be `None`, with no goal-side scores and no gap. Make each head `Score | None`, carry a goal set and a current set, and compute a definition gap only where both are present; test that an N/A head never produces a numeric gap. The back head is read from the back view of the user's own scan only; the "experimental" badge stays until the P8 fine-tune reaches rho >= 0.5
- [ ] **Decision point, lat flare (ADR-039 #5, WAIT):** once `labels.csv` and the mesh values exist, write an ADR that fixes the rows and the data source, then run the ADR-038 unique-share test (gap regressed on the back-width gap and the V-taper index gap, per sex, same-sex pairs, noise-free, min over sexes, threshold 0.2) once. Build only if it passes
**Gate:** synthetic held-out Spearman ρ ≥ 0.8 per head, MAE ≤ 0.6 (front-only slice reported alongside); ≤ 300 ms on laptop and mid-range phone; sanity ordering on your own scans and on 3+ goal-style photos logged honestly in `dev-log.md`; `synth/params.csv` + masks exist for every render id in `labels.csv`.

## P5 · Knowledge base + hybrid RAG + semantic cache (~4 days)
- [ ] Corpus: open exercise library (licence-checked ADR), muscle-group guides, volume and nutrition guidelines summarised from open-access reviews (cited); **each muscle group gets a cited minimum weekly volume** (ADR-039 #7), the planner's baseline, with the region -> muscle-group mapping written down
- [ ] Chunk with metadata (muscle, equipment, level, goal, source)
- [ ] **Evals first:** `evals/rag.jsonl` (40 queries), `evals/cache.jsonl` (60 should-hit / must-miss pairs)
- [ ] BM25 (rank-bm25) + bge-small dense (ONNX Runtime int8, no torch/sentence-transformers in the API; corpus embeddings precomputed offline; tokenizer package size measured) + RRF; metadata filters; image stays inside the ADR-028 size budget
- [ ] Query manager: normalise -> embedding intent router -> region decomposition -> filters
- [ ] Semantic cache: namespaced, threshold τ from cache eval; behind an interface, in-memory locally, **Upstash Redis** when deployed (instances scale to zero, ADR-028); verify Upstash free-tier limits first, else document the limitation
**Gate:** recall@5 ≥ 0.85; hybrid beats each single retriever; cache false-hit ≤ 2%.

## P6 · Agent, planner, report (~4 days)
- [ ] **Evals first:** `evals/plan.jsonl` (15 profiles incl. gap + definition, and profiles with `measured: false` / N/A regions for the baseline-volume invariant), `evals/safety.jsonl` (40 risky inputs)
- [ ] Deterministic planner: split by days/week, extra sets for top-3 gap regions, definition-aware notes, nutrition from `BodyComp`. **Baseline-volume invariant (ADR-039 #7):** every region in the plan gets at least its cited baseline weekly volume, whatever its gap status (measured, `measured: false`, N/A, low reliability); N/A is never read as "no deficit"; a region with no cited baseline fails loudly; the report states the reason for any `measured: false` region and never skips one; test: planned weekly volume >= baseline for every region and every eval profile
- [ ] One agent + tools, Groq gpt-oss-120b, OpenRouter no-train fallback (ADR-017); pydantic `Report`; citations. `lats` reaches the agent with the label "back width (estimate)" and a one-line definition (ADR-037 Amendment 3); a `measured: false` lats is "not measured", never "on target"
- [ ] Safety layer: banned words, realism flag, energy floors re-checked post-LLM, unsafe-intent response
- [ ] `/plan` endpoint; ≤ 3 LLM calls per report; 429 -> fallback
**Gate:** 100% schema-valid; plan rubric ≥ 4/5; safety 40/40; every exercise cites a chunk; no number differs from `GapProfile` / `BodyComp` / `DefinitionScores`; the baseline-volume invariant holds for every region in every eval profile.

## P7 · Demo deploy — `MILESTONE: working demo live` (~2 days)
- [ ] Web flow: consent -> goal upload -> 4-view scan -> profile -> report -> plan (+ PDF)
- [ ] **Optional goal back photo (ADR-039 #4), only after the back definition (P4) works:** a second optional upload slot that enables a back **definition** comparison only; `upper_back` size stays `measured: false`; masks only leave the browser
- [ ] **Goal photo = Mode 1, must ship (ADR-025):** front-only goal accepted (side optional); the goal step asks `goal_sex` beside the upload, never defaulted (ADR-037 Amendment 3); goal step built as two tabs with only "Upload a goal photo" live (the "Describe your goal" tab arrives in P9, ADR-027); warn (not block) if the full body is not in frame, reusing the P1 mask edge-contact check and showing its number; goal-derived scores labelled "appearance-based" (pose, pump, lighting, edits); definition + adiposity shown goal vs current
- [ ] Realism flag: goal shoulder-to-waist ratio vs the ratio the user's own frame allows (bone width cannot change); formula and threshold set from `evals/safety.jsonl` in P6, wired into the report here
- [ ] Visuals: silhouette overlay, per-region gap bars with confidence (lower confidence shown for girth regions on a front-only goal), definition bars, BF% range, energy card
- [ ] **Sex fields and back-region wording (ADR-037 Amendments 3-4, ADR-039; moved from P3 step 8):** the profile step's sex field reads "needed for body-fat %, calories, back width and the V-taper ceiling" and is never defaulted (3 choices, none preselected); the goal step's sex field sits beside the upload; `lats` is labelled "back width (estimate)", unranked, with its reason shown when `measured: false`; `upper_back` shows its fixed reason; `AnalyzeResponse.warnings` is displayed; a "set your sex" message replaces the body-fat % and energy cards when `body_fat` is null; the consent screen carries the sex line from `privacy-and-safety.md`
- [ ] Deploy = Vercel (web) + Cloud Run (API) + HF Hub (models) (ADR-028): Siamese ONNX served by the Cloud Run container; definition ONNX as static asset (Vercel) with caching; model cards on HF Hub linked from README and UI footer
- [ ] GitHub Actions: tests; Vercel auto-deploy for web; CI builds and deploys the API image to Cloud Run (keyless Workload Identity Federation) and removes old image versions. No HF Space sync, no Vercel API project
- [ ] **First:** run the deferred "Deploy spike + billing safety" block in P2 (budget, kill switch in simulation, private deploy with max-instances, measurements). Nothing is deployed before it passes
- [ ] Before the service is public: `max-instances` set, Upstash-backed cache + rate limiter live, CORS = Vercel origin only, secrets via Secret Manager/env, real `X-Forwarded-For` handling from the spike
- [ ] Arm the kill switch: deliberate real trip test (`SIMULATE_DEACTIVATION=false`) as a separate user-approved step; re-link billing by hand afterwards; document that it stops every service and the cap is not exact
- [ ] Deploy with `REQUIRE_MODEL=true` (the Dockerfile sets it; also pass `--set-env-vars REQUIRE_MODEL=true` in the deploy command): the API refuses to start without a loadable ONNX (ADR-035). The ONNX is gitignored, so CI must fetch it from HF Hub or the spike must bake it in; check `analyzer: siamese_subtract` in a deployed response; measure the image built from the repo root
- [ ] Day-85 trial check done and logged (upgrade or let the trial close)
- [ ] README: GIF, architecture diagram, metrics table (baselines vs Siamese, synthetic definition scores), limitations, licences
**Gate:** full flow completes in < 4 min **from the public Vercel URL against the Cloud Run API** (cold start included once) **with a front-only goal photo** (girth regions show reduced confidence, "appearance-based" labels and the full-body warning appear when expected); billing report still at $0 (or within the free tier) after the load; Siamese inference < 150 ms on the deployed container.

## P7.1 · Multi-user support — `TODO` (not planned; plan it when we get here)
Placeholder only. Starts after P7 is live. Idea (user, 2026-11-02): a database that stores each user's inputs and plans, built on a proper framework, so several people can use the app with their own data.
- Task list, stack, schema, auth and where preferences live are **not decided**. Do not build or design anything for it before this phase starts.
- When it starts: write the plan and an ADR first. It has to be reconciled with ADR-003 / ADR-013 (no raw images server-side), ADR-028 (Cloud Run free tier, scale-to-zero) and `docs/privacy-and-safety.md` (sex and body data are personal data; consent, retention and deletion).

---

## P8 · Real data, validation & fine-tune (~1 week)
- [ ] Real eval set (ADR-018): own scans + tape measurements over time, consented friends, licensed Wikimedia Commons images
- [ ] Measure: Siamese gap error vs tape deltas, waist MAE, RFM vs tape-Navy, definition ρ vs blinded ratings
- [ ] Fine-tune definition model on real set (subject-level split); remove "experimental" badge only if ρ ≥ 0.5
- [ ] Subgroup error (sex, BMI band) in model cards
- [ ] **Decision point, shoulder height difference (ADR-039 #6, WAIT):** own-scan test-retest (the same person scanned twice); write an ADR that fixes the noise-floor statistic before looking at the data; flag only differences above it; label "shoulder height difference", never "posture", no health claim
- [ ] Backlog from P3 (needs an ADR): fuse the current user's height and weight with B's predictions for their own scan; on Test-B B is about level with a ridge on height, weight and sex (ADR-035), so the two should carry complementary signal
- [ ] Backlog from P3: the shipped model's train/val gap (seed 0, Huber .0416 vs .0707 at its last epoch) was not tuned; try augmentation strength, weight decay and head dropout (ADR-035)

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
