# Phase plan
Status values: `TODO` · `IN PROGRESS` · `DONE`. Exactly one phase is `IN PROGRESS`.
Each phase ends with the end-of-phase protocol in `CLAUDE.md` (logs, ADRs, mistakes, tag). `/sensei` -> `learn/phase-NN/` only for the major phases: P3, P4, P5, P6.

**Track A — demo (P0–P7):** capture -> data + baseline -> **Siamese CNN** -> **muscle definition** -> RAG -> agent -> deploy.
**Track B — best (P8–P10):** real data, product depth, hardening. Do not start B until A is live.
Data rule (ADR-020): demo models train on **BodyM + synthetic only**. Real photos come in P8.

| Phase | Outcome | ~Time |
|---|---|---|
| P0 | Repo, schemas, empty deploys live | 1 d |
| P1 | Private 4-view capture in browser | 4 d |
| P2 | BodyM pipeline + geometric baseline + body comp | 4 d |
| **P3** | **Siamese CNN (core model)** | 1 wk |
| **P4** | **Muscle-definition model (synthetic, on-device)** | 1.5 wk |
| P5 | Hybrid RAG + query manager + semantic cache | 4 d |
| P6 | Agent + planner + safety | 4 d |
| **P7** | **Demo deploy** | 2 d |
| P8 | Real data + fine-tune + validation | 1 wk |
| P9 | Product depth | 1 wk |
| P10 | Hardening & polish | 4 d |

---

## P0 · Foundation — `DONE` (~1 day)
- [x] Monorepo scaffold per README folder tree; `uv` for Python, Vite+React+TS for web
- [x] `.gitignore` (ml/data, synth/out, *.onnx, .env, validation-set/), pre-commit (ruff; prettier hook dropped, see mistakes.md)
- [x] Pydantic schemas in `packages/schemas`: `Scan`, `GapProfile`, `BodyComp`, `DefinitionScores`, `UserProfile`, `Report` (+ generated TS types)
- [x] GitHub repo, push; Vercel project linked (web only; API and models stay local, ADR-022)
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
**Result (2026-11-01):** ~8/10 captures valid, accepted by the user; best turn 7.6 s; network-tab check not run (do it before P7). See `docs/logs/dev-log.md`.

## P2 · BodyM data pipeline + geometric baseline + body comp — `IN PROGRESS` (~4 days)
- [ ] Download BodyM (`aws s3 cp --no-sign-request s3://amazon-bodym ...`) into `ml/data/` (gitignored); EDA: subjects, repeats, measurement distributions, sex balance
- [ ] Subject-level split (train / val from Training; Test-A, Test-B kept untouched); map 14 measurements -> 10 regions
- [ ] `ml/src/preprocess.py` (shared with API + browser parity test): crop, head removal, height-normalise, resize
- [ ] Parity check: same preprocessing on a BodyM silhouette and on a browser-made mask of you -> visually and statistically comparable
- [ ] Geometric analyzer (no ML): widths/depths at keypoint rows, chest/back split, ellipse girths, asymmetry -> **baseline A** and fallback
- [ ] Waist calibration on BodyM (silhouette -> real waist, MAE cm) -> `BodyComp`: RFM BF% range, optional tape override, Mifflin RMR + energy ranges with floors
- [ ] FastAPI `/analyze` skeleton (mask validation, payload cap, rate limit, CORS) behind `Analyzer` interface
**Gate:** no subject in two splits (test); geometric Spearman ρ ≥ 0.6 on ≥ 6/10 regions (Test-A); waist MAE ≤ 5 cm; RFM/Mifflin unit tests; rejects RGB upload.

## P3 · Siamese CNN — core model (~1 week)
- [ ] Pair sampler: random subject pairs (A, B) per epoch from train split; target Δ = (meas(A) − meas(B)) / height for each region; antisymmetry augmentation (swap pair, negate target)
- [ ] Mask augmentation to match browser masks: edge noise, morphological jitter, small rotation/scale, clothing-bulge sim
- [ ] **Baseline B:** single ResNet-18 regressor (silhouettes -> measurements), run on both, subtract
- [ ] Siamese ResNet-18 (ImageNet, 1ch->3ch, front+side input), head on [e_g, e_c, e_g−e_c, e_g⊙e_c]; Huber loss
- [ ] Overfit 1 batch -> linear probe -> unfreeze last 2 blocks -> full fine-tune; stop at first rung that passes the gate; MPS, log every N steps
- [ ] Per-region uncertainty (MC dropout) -> confidence in `GapProfile`
- [ ] Export ONNX, model card in `ml/MODEL_CARD.md` (data, splits, metrics, limits, BodyM CC BY-NC); plug into `/analyze` (geometric stays for back view + asymmetry)
**Gate:** per-region MAE of Δ beats baseline A and baseline B on **Test-B** (in-the-wild); ONNX CPU inference < 150 ms; ordering check on your own scan vs a goal photo looks sensible. If it doesn't beat baseline B -> ADR-006 negative result, ship the better model, document it.

## P4 · Muscle-definition model — synthetic, on-device (~1.5 weeks)
- [ ] `synth/`: Blender + MPFB (MakeHuman) headless script; grid: sex × muscle × fat × height × proportions × skin tone × lighting (3 HDRIs) × fitted clothing × 4 views; start 5k renders, scale to ~30k
- [ ] Labels from sliders via fixed mapping -> definition 1–5 per region (abs, chest, shoulders, arms, back, legs) + adiposity 1–5; `synth/labels.csv` + mapping doc
- [ ] Domain-gap mitigation: run the **same browser segmentation + crop** on renders; background swap with real CC0 room photos; noise, blur, JPEG, colour jitter
- [ ] MobileNetV3-Small (ImageNet) shared trunk + 7 ordinal heads; split by render identity; overfit 1 batch first
- [ ] Export ONNX, int8 (≤ 5 MB), run with ONNX Runtime Web (WebGPU -> WASM)
- [ ] Fusion: adiposity nudges RFM BF% range (never beyond RFM bounds); UI badge "experimental — trained on synthetic data"
**Gate:** synthetic held-out Spearman ρ ≥ 0.8 per head, MAE ≤ 0.6; ≤ 300 ms on laptop and mid-range phone; sanity ordering on your own scans logged honestly in `dev-log.md`.

## P5 · Knowledge base + hybrid RAG + semantic cache (~4 days)
- [ ] Corpus: open exercise library (licence-checked ADR), muscle-group guides, volume and nutrition guidelines summarised from open-access reviews (cited)
- [ ] Chunk with metadata (muscle, equipment, level, goal, source)
- [ ] **Evals first:** `evals/rag.jsonl` (40 queries), `evals/cache.jsonl` (60 should-hit / must-miss pairs)
- [ ] BM25 + bge-small dense + RRF; metadata filters
- [ ] Query manager: normalise -> embedding intent router -> region decomposition -> filters
- [ ] Semantic cache: namespaced, threshold τ from cache eval
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
- [ ] Visuals: silhouette overlay, per-region gap bars with confidence, definition bars, BF% range, energy card
- [ ] Siamese ONNX served by the local API; definition ONNX as static asset (Vercel) with caching
- [ ] GitHub Action: test; Vercel auto-deploy. Public API hosting is open (ADR-022): decide at P7 start
- [ ] README: GIF, architecture diagram, metrics table (baselines vs Siamese, synthetic definition scores), limitations, licences
**Gate:** full flow completes in < 4 min on laptop against the local API. The "stranger from a public URL" version is blocked until API hosting is decided (ADR-022).

---

## P8 · Real data, validation & fine-tune (~1 week)
- [ ] Real eval set (ADR-018): own scans + tape measurements over time, consented friends, licensed Wikimedia Commons images
- [ ] Measure: Siamese gap error vs tape deltas, waist MAE, RFM vs tape-Navy, definition ρ vs blinded ratings
- [ ] Fine-tune definition model on real set (subject-level split); remove "experimental" badge only if ρ ≥ 0.5
- [ ] Subgroup error (sex, BMI band) in model cards

## P9 · Product depth (~1 week)
- [ ] Progress tracking: re-scan every 4 weeks, trend chart (browser-only storage, export/import JSON)
- [ ] Follow-up chat over the plan (same agent, cache reused)
- [ ] Constraint-aware re-plan ("no barbell", "knee pain"), diet preference (veg / non-veg)

## P10 · Hardening & polish (~4 days)
- [ ] Observability: latency, cache hit rate, LLM calls/report, provider fallback rate
- [ ] Load test; rate limits; `/security-check` on API; accessibility pass
- [ ] Final README, demo video, blog post
