# Architecture

## 1. End-to-end pipeline

```
 BROWSER (Vercel)                                   API (FastAPI, Cloud Run, ADR-028)
 ┌───────────────────────────────┐                  ┌──────────────────────────────────────────┐
 │ Goal photo (front-only OK)    │                  │ /analyze                                  │
 │ Current scan (camera, timer)  │                  │  validate: binary mask, size, keypoints   │
 │        │                      │                  │        │                                  │
 │ MediaPipe Pose Landmarker     │  silhouette PNG  │  Analyzer.compare(goal, current)          │
 │ Definition model (on-device)  │  (1-bit) +       │   v1: geometric widths at landmark rows   │
 │        │                      │  33 keypoints +  │   v2: Siamese CNN (ONNX) -> region deltas │
 │ quality checks (pose, loose   │  height/sex/     │        │                                  │
 │ clothing, full body in frame) │  equipment/days  │  GapProfile {region: delta, confidence}   │
 │        │                      │ ───────────────► │        │                                  │
 │ head removed, mask binarised  │                  │ /plan  Agent (1 agent + tools)            │
 │ raw frame discarded           │                  │   query manager -> semantic cache ->      │
 │                               │ ◄─────────────── │   hybrid RAG -> deterministic planner ->  │
 │ Report + plan UI              │   Report JSON    │   LLM writer -> safety check -> Report    │
 └───────────────────────────────┘                  └──────────────────────────────────────────┘
                                                     Model weights: HF Hub model repos; ONNX baked into
                                                     the image or fetched at start (decided in the spike)
```
Goal input has two modes (ADR-025..027): **photo** (above; the goal's mask comes from the same Pose Landmarker, side view optional, definition scores computed on-device on the goal frame) and, from P9, **typed description** -> `goal_from_text` agent tool -> archetype table (section 3b). Only masks, keypoints, scores, numbers and (Mode 2) goal text leave the browser.

## 2. Components

| Component | v1 (demo) | v2 (final) |
|---|---|---|
| Capture | MediaPipe Pose Landmarker (WASM, on-device, self-hosted model; its mask is the silhouette), user-paced guided turn -> 4 head-less 384x512 8-bit {0,255} PNG silhouettes + remapped keypoints (ADR-011, ADR-023); code in `apps/web/src/capture/`, RGB frames kept in memory only | + region mode, better frame selection |
| Goal input | **Mode 1 (P7, priority):** goal photo, front-only accepted; girth regions get lower confidence without a side view; scores labelled "appearance-based"; full-body-in-frame warning; realism flag = goal shoulder-to-waist ratio vs the user's own frame (ADR-025). **Mode 2 (P9):** "describe your goal" -> archetype table + static preview library (ADR-026, ADR-027) | + live three.js preview (stretch) |
| Analyzer | Siamese CNN (ResNet-18 shared encoder, front+side, BodyM, goal side-view dropout), ONNX on CPU + geometric analyzer for back view, chest/back split, asymmetry and fallback | Siamese fine-tuned on real deltas (P8) |
| Gap output | 10 regions: shoulders, chest, upper back, lats (V-taper), waist, hips, upper-arm, forearm, thigh, calf + left/right asymmetry | same + per-region uncertainty |
| Body comp | RFM BF% range from silhouette waist + height + sex; Mifflin RMR -> energy ranges with floors; on-device MobileNetV3-Small definition/adiposity scores (synthetic-trained, "experimental"), fused into BF% range | same model fine-tuned on real data, uncertainty, badge removed if ρ ≥ 0.5 |
| Retrieval | Hybrid: BM25 + bge-small dense (ONNX Runtime int8, no torch; corpus embeddings precomputed offline), RRF fusion, metadata filters | + cross-encoder rerank if eval says so |
| Cache + rate limit | exact + semantic, namespaced, behind an interface: in-memory locally, **Upstash Redis (free)** when deployed because Cloud Run instances scale to zero and are separate (ADR-028) | + TTL tuning |
| Agent | 1 LLM agent, 4 tools, pydantic output | + follow-up Q&A over the plan |
| LLM | Groq `gpt-oss-120b` (free), OpenRouter free no-train endpoint as fallback; text only (ADR-017) | smallest model that passes plan eval |

## 3. Siamese model (v2)

```
silhouette_goal ─┐                         ┌─ e_g ─┐
                 ├─ shared ResNet-18 ──────┤       ├─ [e_g, e_c, e_g−e_c, e_g⊙e_c] ─ MLP ─ Δ ∈ R^8
silhouette_cur  ─┘  (ImageNet init,        └─ e_c ─┘
                     1ch -> 3ch, 256×192)
```
- Target (ADR-032): Δ = m_goal/h_goal − m_cur/h_cur (difference of height-normalised ratios, `height_cm`), for 8 regions mapped from BodyM's 14 measurements. A proportion gap: the UI converts to cm as Δ × the user's own height ("at your height"). The net never sees height.
- Loss: Huber on Δ. Antisymmetry augmentation: swap pair, negate target.
- Baseline it must beat: single-image regressor run twice and subtracted. If it doesn't, we report that honestly (ADR-006).
- Front-only silhouettes give widths, not girths -> front+side input in v2.
- **Goal side-view dropout (ADR-025):** in training the goal's side input is randomly blanked (the current scan keeps both views), so a front-only goal photo is a valid input. At inference a missing goal side view lowers confidence on girth regions (arms, thighs, calves, chest depth). Evaluated on a front-only slice of BodyM Test-B against baselines A and B.

## 3b. Mode 2: "describe your goal" (P9)

```
 goal text ─ goal_from_text (LLM, text only) ─ pydantic Goal {sex, build, muscle, fat, region emphasis}
                                                   │  user confirms interpretation
                                                   ▼
                                     safety floors (no target BF below sex floor)
                                                   ▼
        archetype table  ◄── built offline from P4 renders: saved params + 4-view masks
        (region proportions via geometric analyzer + definition labels)
                                                   ▼
        target GapProfile + target definition scores ─► same report + plan flow as Mode 1
        nearest pre-rendered front+side preview ─► client-side "more muscle / leaner" sliders
```
- Named people are turned into descriptors; no real person's body stats are stated.
- Gap quality is geometric-analyzer level (baseline A), not Siamese; confidence reflects that.
- Previews are static assets (~300–500 neutral-grey, fitted-clothing renders, no face detail); no live rendering.

## 4. Query management + RAG + cache (/plan)

1. **Normalise** query (lowercase, unit normalisation, strip PII).
2. **Intent route** (embedding nearest-centroid over labelled examples): `plan | exercise_lookup | explain | off_topic | unsafe`. `unsafe` -> safety response, no retrieval.
3. **Decompose**: one sub-query per priority region from GapProfile (e.g. "lateral delt hypertrophy dumbbell beginner").
4. **Filters**: muscle group, equipment, level.
5. **Semantic cache lookup** per sub-query. Key namespace = `(intent, equipment, level, corpus_hash)`; similarity threshold τ tuned on `evals/cache.jsonl`. Cache **retrieval results and generic sections only**, never a whole personalised report.
6. **Hybrid retrieve**: BM25 top-20 + dense top-20 -> Reciprocal Rank Fusion -> top-k.
7. **Planner (deterministic)**: priority regions get extra weekly sets, split chosen from days/week; volumes from corpus guidelines.
8. **Writer (LLM)**: fills exercise choices, cues, rationale with citations to retrieved chunks; pydantic `Report` schema.
9. **Safety check**: realism flag, no calorie/weight-loss targets below safe floors, no shaming language, "consult a professional" where needed.

## 5. Trust boundaries
- Browser -> API: only 1-bit masks (server rejects 3-channel/greyscale-with-texture images), keypoints, numeric profile. Max payload 512 KB. Rate limited.
- Goal photo: same rule as the current scan; the RGB never leaves the browser, only its mask, keypoints and definition scores do. Mode 2 (P9) sends the user's goal text to the LLM only, after PII normalisation, never logged beyond the intent label.
- Retrieved text -> LLM: treated as data; tool calls whitelisted; no tool can fetch URLs.
- CORS: the production Vercel origin only (preview-deploy origins are not allowed unless added on purpose).
- Client IP: on Cloud Run it arrives via `X-Forwarded-For`; per-IP rate limits are not trusted until the spike's `/whoami` shows the real header shape. Cloud Run's platform logs record the client IP (stated in `privacy-and-safety.md`).

## 6. Deployment
Decision record: ADR-028 (supersedes ADR-022 / ADR-009).

- **Web:** Vercel (React + Vite + TS), preview deploys per PR. The definition ONNX is a static asset of the web app.
- **API:** FastAPI in a Docker container on **Cloud Run**, region `asia-south1` if available (recorded in `dev-log.md`), 1 vCPU / 2 GiB, request-based billing, `min-instances=0`, `max-instances` 1–2 (set before the service is public), small concurrency. Free tier per month: 2M requests, 180k vCPU-s, 360k GiB-s; at this size the compute quotas are both exhausted at 180,000 billed instance-seconds. The UI keeps its "waking server" state for cold starts (measured in the spike).
- **Image:** `python:3.12-slim`, no torch; onnxruntime, numpy, Pillow (+ scipy.ndimage until the spike decides), rank-bm25, bge-small int8 via ONNX Runtime. Design target <= ~250 MB stored so two versions fit Artifact Registry's 0.5 GiB free allowance; old versions removed after each deploy.
- **Models:** HF Hub model repos (public, free) hold weights and model cards, linked from README and UI footer. Siamese ONNX is baked into the image or fetched at container start (spike decides). Model card source in `ml/MODEL_CARD.md`.
- **State:** semantic cache and rate limiter in Upstash Redis when deployed (in-memory locally). In-process state resets on scale-to-zero and is per instance.
- **Secrets:** Secret Manager / Cloud Run env vars, never in repo, image or logs.
- **CI/CD:** GitHub Actions tests; Vercel auto-deploys web; CI builds and deploys the API image to Cloud Run in P7 (keyless Workload Identity Federation). No HF Space sync, no Vercel API project.
- **Billing safety:** Free Trial billing account ($300 / 90 days, closes instead of charging) -> project-scoped budget (~INR 100, alerts) -> Pub/Sub -> Cloud Run function that detaches billing (`infra/kill_switch/`, simulation flag on first). It stops every service in the project, and cost reporting lags, so the cap is not exact. Day-85 upgrade decision is logged in `dev-log.md`.

```
 Vercel (web) ──HTTPS, CORS: Vercel origin only──► Cloud Run (API, 0..2 instances) ──► Upstash Redis (cache, limits)
                                                         │                       └──► Groq / OpenRouter (text only)
 HF Hub (models, cards) ── baked into image or fetched at start
 Budget ─► Pub/Sub ─► kill-switch function ─► detach billing from the project
```

## 7. P2 as built: data pipeline, geometric analyzer, `/analyze` (ADR-029, ADR-030)

```
 BodyM (full-body 720x960 masks)                           browser mask (384x512, head-less)
        │ from_bodym: largest component ► bbox ► +5% margin          │ from_browser: validate shape,
        │ ► letterbox 384x512 ► cut head at the neck row             │ largest component (no re-crop)
        └───────────────────────────┬───────────────────────────────┘
                                    ▼
              canonical 384x512 bool mask (ml/src/preprocess.py)
              body height in px = 2 * (feet_row - 256)         (centred letterbox)
                                    ▼
   rows.py   armpit / crotch / waist / hip rows from run lengths (no keypoints, no centre column)
                                    ▼
   features.py  scale-free: widths, ellipse girths (front width + side depth), arm thickness; / body height
                    │                                         │
         geometric.py: delta = slope x (goal - current)       bodycomp.py: waist model (silhouette + height
         per region, confidence from validated rho            + weight + sex, CURRENT scan only) -> RFM range,
                    │                                         Mifflin RMR, energy ranges with floors
                    └───────────────┬─────────────────────────┘
                                    ▼
        POST /analyze  {goal, current, profile} -> {gap: GapProfile, body_comp: BodyComp}
```

- **`/analyze` guards** (`app/security.py`, `app/analyzer/masks.py`): body size cap checked before parsing (413, also without `Content-Length`); `application/json` only (415); mask must be a PNG of exactly 384x512, mode L, values only {0,255}, not all white, not empty (422, size read from the header before any pixels); per-client sliding-window rate limit behind a `RateLimiter` interface (429 + `Retry-After`; Upstash replaces the in-memory one before P7); validation errors never echo the input; logs hold request id, latency and region deltas only.
- **Two scales** (ADR-024): the gap uses only scale-free silhouette features, so the goal photo needs no weight. Weight enters only the current person's waist estimate. A test pins this: the gap is identical when only the weight changes.
- **Data files** (all derived from BodyM, CC BY-NC 4.0): `analyzer/gap_coef.json` (per-region slope, validated rho), `analyzer/waist_coef.json` (waist model, residual sd). Regenerated by `ml/src/fit_gap.py` and `ml/src/calibrate_waist.py`; features come from `ml/src/extract.py`.
- **For the deployed image:** the API imports `ml/src/preprocess.py` and `scipy.ndimage`; no torch, pandas or scikit-learn (those are the `ml` dependency group only).
