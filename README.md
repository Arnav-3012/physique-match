# PhysiqueGap
Upload a goal physique, scan your own body (privately, on-device), see which regions differ most, and get a cited, personalised training plan.

> Status: P2 in progress (Cloud Run deploy spike first) · see `docs/phase-plan.md`

## How it works (one line each)
1. **Capture** — MediaPipe runs in your browser; only a black-and-white silhouette + joint points are sent. No photo ever leaves your device.
2. **Analyze** — silhouettes compared region by region (v1 geometry, v2 Siamese CNN fine-tuned on BodyM).
3. **Plan** — one AI agent uses hybrid RAG (BM25 + dense) with query routing and a semantic cache to write a report and a weekly plan, every exercise cited.

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Web | React + Vite + TypeScript, Tailwind | fast, Vercel-native |
| On-device vision | MediaPipe tasks-vision (Pose Landmarker, Image Segmenter) | runs in browser -> privacy |
| API | FastAPI + pydantic in a Docker container on **Google Cloud Run** (free tier, scale-to-zero, ADR-028) | free, typed, model-adjacent |
| Training | PyTorch (MPS on M4 Pro), timm ResNet-18 | pretrained, small, fast |
| Inference | ONNX Runtime (CPU, 1 vCPU container); no torch in the image | small image, fast on free hardware |
| Models | Hugging Face Hub model repos (weights + model cards) | free storage, model cards |
| Data | BodyM (Amazon, CC BY-NC 4.0) | silhouettes + 14 real measurements |
| Retrieval | rank-bm25 + bge-small-en-v1.5 (ONNX Runtime int8), RRF, numpy | small corpus, no vector DB needed |
| Cache + rate limit | Upstash Redis free when deployed (instances scale to zero); in-memory locally | state survives scale-to-zero |
| Body comp | RFM (BF% range), Mifflin-St Jeor (energy), on-device MobileNetV3-Small (definition) | validated formulas + private CNN |
| LLM | Groq gpt-oss-120b (free), OpenRouter free no-train fallback, env-swappable | free, no lock-in, text only |
| CI/CD | GitHub Actions (tests, build + deploy API image to Cloud Run); Vercel auto-deploy for web | |
| Cost guard | Free Trial billing account + ~INR 100 budget -> Pub/Sub -> kill-switch function that detaches billing | spend cannot run away (cap is not exact, ADR-028) |

## Folder structure
```
physique-gap/
├── CLAUDE.md                     # Claude Code operating rules
├── README.md
├── docs/
│   ├── architecture.md           # pipeline, components, trust boundaries
│   ├── phase-plan.md             # phases, tasks, exit gates, status
│   ├── decisions.md              # ADR log
│   ├── mistakes.md               # bugs & wrong turns + prevention rules
│   ├── privacy-and-safety.md     # hard requirements
│   ├── project-overview.md       # one-page plan: data, models, why
│   ├── research/                 # L3 research reports
│   └── logs/dev-log.md           # per-session progress + metrics
├── learn/                        # /sensei output per phase
│   ├── README.md
│   └── phase-NN/{concepts,model-architecture,phase-overview}.md
├── apps/web/                     # React app (Vercel)
│   ├── public/models/            # definition.onnx (int8)
│   └── src/{capture,definition,report,lib}/
├── services/api/                 # FastAPI (Cloud Run)
│   ├── Dockerfile
│   └── app/{main.py,routes/,analyzer/,rag/,agent/,safety/}
├── packages/schemas/             # pydantic models + generated TS types
├── ml/
│   ├── configs/                  # yaml per experiment
│   ├── src/{preprocess,dataset,model,train,evaluate,export_onnx}.py
│   ├── notebooks/                # EDA only
│   └── data/                     # gitignored
├── synth/                        # Blender + MakeHuman render scripts, label mapping (out/ gitignored)
├── spikes/cloudrun/              # deploy spike: dummy ONNX + PNG endpoint (ADR-028)
├── infra/kill_switch/            # budget -> Pub/Sub -> detach-billing function (ADR-028)
├── kb/                           # RAG corpus (raw/, chunks.jsonl, sources.md)
├── evals/                        # rag / cache / plan / safety suites + runner
├── tests/
└── .github/workflows/            # ci.yml
```

## Model results
MAE of the per-region proportion gap (measurement / height), on every ordered pair of subjects. Multiply by your height for an approximate cm figure. Data: BodyM, split by subject. Shipped model: a weight-sharing ResNet-18 that predicts each person's proportions, with a fixed subtraction head (baseline B). Honest ablation: the learned pair head did not earn its place.

### Val ablation (val, used for selection)
251 subjects, 62,750 ordered pairs, full input, `full` rung, seed 0. A = geometric analyzer. B = shipped model. Differences are 95% subject-bootstrap CIs; positive means the second model named is better.

| Region | A | B | Pair head | B better than A by | Pair head better than B by |
|---|---|---|---|---|---|
| shoulders | .0094 | .0062 | .0062 | .0032 (.0023, .0041) | .0000 (-.0004, .0004) |
| chest | .0363 | .0193 | .0191 | .0170 (.0138, .0202) | .0002 (-.0014, .0017) |
| waist | .0276 | .0217 | .0199 | .0060 (.0026, .0092) | .0018 (-.0000, .0035) |
| hips | .0298 | .0165 | .0169 | .0132 (.0097, .0175) | -.0004 (-.0017, .0008) |
| upper_arm | .0137 | .0083 | .0079 | .0054 (.0036, .0074) | .0004 (-.0002, .0010) |
| forearm | .0144 | .0058 | .0058 | .0086 (.0071, .0103) | .0000 (-.0004, .0003) |
| thigh | .0240 | .0118 | .0110 | .0121 (.0094, .0148) | .0008 (-.0001, .0017) |
| calf | .0157 | .0081 | .0071 | .0076 (.0060, .0093) | .0010 (.0004, .0016) |

- The pair head has the lower overall val MAE (standardised, both input slices): .2709, one seed, against B's .2791 (shipped seed 0; seeds 1 and 2 reached .2785 and .2754).
- On thigh and calf the pair head is below all three B seeds (thigh .0110 vs .0118-.0123, calf .0071 vs .0078-.0081).
- It still failed the rule written before this run: CI above 0 on at least 6 of 8 regions against the simpler model. It reached 1 of 8 (calf). A lower mean that isn't significant per region doesn't override that (ADR-034, ADR-035).
- Null ridge (height, weight, sex), val, table order: .0070 .0267 .0349 .0196 .0131 .0086 .0155 .0121.
- Val also selected the epoch and the model, so these numbers are optimistic.

### Test-B result (final candidate vs A only)
In-the-wild BodyM, 351 subjects, 122,850 ordered pairs, scored once, on the shipped model only. Differences are 95% subject-bootstrap CIs; positive means B is better. Front-only means the goal has no side view, compared against A's width-only features.

| Region | A | B | Null ridge | B better than A by | Width-only A | B, front-only | B better by |
|---|---|---|---|---|---|---|---|
| shoulders | .0107 | .0083 | .0070 | .0023 (.0016, .0030) | .0107 | .0083 | .0023 (.0016, .0029) |
| chest | .0462 | .0294 | .0294 | .0168 (.0136, .0200) | .0508 | .0311 | .0197 (.0163, .0230) |
| waist | .0397 | .0308 | .0310 | .0089 (.0058, .0121) | .0388 | .0315 | .0072 (.0044, .0101) |
| hips | .0434 | .0256 | .0224 | .0177 (.0139, .0218) | .0456 | .0262 | .0194 (.0149, .0238) |
| upper_arm | .0168 | .0109 | .0117 | .0058 (.0039, .0078) | .0168 | .0114 | .0054 (.0034, .0074) |
| forearm | .0166 | .0077 | .0076 | .0088 (.0074, .0103) | .0166 | .0080 | .0086 (.0071, .0100) |
| thigh | .0312 | .0159 | .0166 | .0153 (.0116, .0192) | .0312 | .0164 | .0149 (.0113, .0186) |
| calf | .0190 | .0111 | .0129 | .0079 (.0062, .0094) | .0190 | .0110 | .0079 (.0063, .0095) |

- B beats A on all 8 regions on both inputs, with no region worse.
- **B is about as accurate as a ridge on height, weight and sex** (the "Null ridge" column, point estimates, no CI): worse on shoulders and hips, slightly better on upper_arm, thigh and calf, tied on chest, waist and forearm. The ridge needs weight, which a goal photo doesn't have, so it's a reference, not a baseline. B's value is that it needs no height or weight for the goal photo.
- Error rose 31-55% from val to Test-B, so expect browser masks to be harder than val suggests. No real-world validation yet (P8).

## Limitations (stated up front)
- **Back width (estimate).** A photo can't see your back from the front, so this is inferred from chest, shoulder and waist ratios with a fit on ANSUR II (US Army body measurements), not measured, and it is not a measure of lat muscle. It explains under half of the real variation (R² .41 to .47), so it is shown but never enters the priority ranking. It needs your sex and the goal's sex, set by you and equal; otherwise it says why it's missing. `upper_back` size is never shown.
- **Sex is optional and never guessed.** Leave it unset and you lose body-fat %, calorie targets and back width; the gaps and the training plan still work, and cut / recomp / bulk is always your choice.
- Silhouettes measure shape/size, not muscle vs. fat. Body-fat % is a formula-based range (RFM), refined by an optional on-device definition model trained on synthetic data.
- Goal photos are often edited, posed or enhanced; realism flag tempers targets.
- BodyM is CC BY-NC: this is a non-commercial research/portfolio project.
- Not medical advice.
