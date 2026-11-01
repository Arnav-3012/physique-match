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

## Limitations (stated up front)
- Silhouettes measure shape/size, not muscle vs. fat. Body-fat % is a formula-based range (RFM), refined by an optional on-device definition model trained on synthetic data.
- Goal photos are often edited, posed or enhanced; realism flag tempers targets.
- BodyM is CC BY-NC: this is a non-commercial research/portfolio project.
- Not medical advice.
