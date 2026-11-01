# Architecture

## 1. End-to-end pipeline

```
 BROWSER (Vercel)                                   LOCAL API (FastAPI, laptop)    
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
                                                     Model weights: ml/ on the laptop (gitignored)
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
| Retrieval | Hybrid: BM25 + bge-small dense, RRF fusion, metadata filters | + cross-encoder rerank if eval says so |
| Cache | exact + semantic, namespaced | + TTL tuning, persisted (Upstash free) |
| Agent | 1 LLM agent, 4 tools, pydantic output | + follow-up Q&A over the plan |
| LLM | Groq `gpt-oss-120b` (free), OpenRouter free no-train endpoint as fallback; text only (ADR-017) | smallest model that passes plan eval |

## 3. Siamese model (v2)

```
silhouette_goal ─┐                         ┌─ e_g ─┐
                 ├─ shared ResNet-18 ──────┤       ├─ [e_g, e_c, e_g−e_c, e_g⊙e_c] ─ MLP ─ Δ ∈ R^8
silhouette_cur  ─┘  (ImageNet init,        └─ e_c ─┘
                     1ch -> 3ch, 256×192)
```
- Target: Δ = (goal measurements − current measurements) / height, for 8 regions mapped from BodyM's 14 measurements.
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
- CORS: Vercel domain only.

## 6. Deployment
- Web: Vercel (React + Vite + TS). Preview deploys per PR.
- API: runs locally (`uvicorn`, ADR-022). Public hosting is undecided until P7; `services/api/Dockerfile` is kept for whichever free host is chosen. The UI keeps its "waking server" state for a future sleeping host.
- Model: weights stay on the laptop (`ml/`, gitignored). Model card in `ml/MODEL_CARD.md` (data, metrics, limitations, BodyM CC BY-NC licence).
