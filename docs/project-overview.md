# PhysiqueGap — project overview
One page: what we build, with which data and models, why, and in what order. Details: `architecture.md`, `decisions.md`, `phase-plan.md`, `research/body-composition-research.md`.

## What it does
1. User uploads a **goal physique** photo.
2. User does a **4-view turn scan** (front, left, back, right) in fitted clothes. Everything visual runs on their phone.
3. App reports, per region: **size gap** (shoulders, chest, upper back, lats, waist, hips, arms, forearms, thighs, calves + left/right asymmetry), **body-fat % range**, optional **muscle-definition scores**.
4. An AI agent with hybrid RAG writes a **report + weekly training plan + calorie/protein targets** (cut / recomp / bulk), each exercise cited.

## Privacy model (one line)
Photos never leave the device. The server gets 1-bit silhouettes, keypoints and numbers. Free LLMs get only numbers and text.

## Data
| Dataset | What | Used for | Why this one | Licence |
|---|---|---|---|---|
| **BodyM** (Amazon) | 8,978 front/side silhouettes, 2,505 subjects, height, weight, 14 tape-style measurements | Geometric calibration (P2), Siamese CNN (P3), waist -> RFM | Only large public set pairing silhouettes with real measurements; silhouettes match our privacy design | CC BY-NC 4.0 (non-commercial) |
| **MakeHuman synthetic renders** (we generate) | ~30k renders × 4 views, labels from muscle/fat sliders | Muscle-definition model (P4) | No public labelled real dataset exists; MakeHuman assets are CC0 | CC0 assets |
| **Real set** (after demo) | Own scans + tape measurements, consented friends, licensed Wikimedia Commons images | Evaluation + fine-tune (P8) | Only honest way to measure real-world accuracy | Consent / per-image licence |
| **Exercise/knowledge corpus** | Open exercise DB + summarised open-access training/nutrition reviews | RAG (P5) | Small, citable, licence-checked | Per source (ADR at P5) |
| ~~Reddit body-fat images~~ | Scraped photos, perceived BF labels | **Not used** | No consent, ToS risk, perceived labels, men only | — |

## Models
| Model | Where it runs | Job | Why |
|---|---|---|---|
| **MediaPipe Pose Landmarker + Image Segmenter** | Browser (WASM) | Keypoints + silhouette | Free, fast on phones, on-device = privacy |
| **Geometric analyzer** (no ML) | API | Region widths/depths, chest/back split, asymmetry | Works on day 1; baseline everything must beat |
| **RFM formula** | API | Body-fat % range | Validated vs DXA (R² 0.69–0.75), needs only waist + height + sex |
| **Mifflin-St Jeor** | API | Resting energy -> targets | Most accurate common RMR equation |
| **Siamese ResNet-18** (ImageNet, fine-tuned) | API, ONNX CPU | Goal-vs-current per-region measurement gap | Binary silhouettes are low-information, 11M params is enough; fast on free CPU |
| **MobileNetV3-Small** (ImageNet, fine-tuned) | Browser, ONNX Runtime Web, int8 | Muscle definition 1–5 per region + adiposity | Phone-sized (≤ 5 MB), runs on the RGB frame without uploading it |
| **bge-small-en-v1.5** | API (local) | Dense retrieval + semantic cache + intent router | Small, strong, no API cost |
| **BM25** | API | Keyword retrieval (exercise names) | Hybrid beats dense-only on exact names |
| **gpt-oss-120b on Groq** (free) | Groq API | Agent: report + plan writing | Free, fast, ~1,000 req/day; OpenRouter no-train free model as fallback |

## How body fat and definition are decided
```
silhouettes ──► waist girth ──► RFM ──► BF% range ──┐
                                                    ├─► fused BF% range + definition bars ─► report
RGB (on device) ─► MobileNetV3 ─► definition 1–5 ───┘   (definition only narrows the range)
user height/weight/age/sex/activity ─► Mifflin RMR ─► energy targets (with safety floors)
```
- No LLM ever guesses body fat. LLMs explain numbers; they don't produce them.

## Phases (Track A = demo, Track B = best)
| Phase | Outcome | ~Time |
|---|---|---|
| P0 | Repo, schemas, empty deploys live | 1 d |
| P1 | Private 4-view capture in browser (webcam or phone) | 4 d |
| P2 | BodyM pipeline + geometric baseline + RFM BF% + energy targets | 4 d |
| **P3** | **Siamese CNN (ResNet-18) on BodyM — core model** | 1 wk |
| **P4** | **Muscle-definition model — synthetic MakeHuman, MobileNetV3-Small, in-browser** | 1.5 wk |
| P5 | Hybrid RAG + query manager + semantic cache (evals first) | 4 d |
| P6 | Agent + planner + safety + `/plan` | 4 d |
| **P7** | **Demo: Vercel UI + local API (public hosting TBD, ADR-022)** | 2 d |
| P8 | Real data (own measurements, Wikimedia Commons), fine-tune + validation | 1 wk |
| P9 | Progress tracking, follow-up chat, constraint re-plans | 1 wk |
| P10 | Hardening, security, docs, demo video | 4 d |

## Biggest risks
1. **Synthetic -> real gap** for definition (in the demo from P4). Mitigation: same browser preprocessing on renders, augmentation, "experimental" badge until real fine-tune in P8 reaches ρ ≥ 0.5.
2. **RFM weaker for lean people** — exactly the goal-physique crowd. Mitigation: definition fusion, show ranges.
3. **Free LLM quotas** (Groq ~1,000/day, OpenRouter 50/day without credits). Mitigation: ≤ 3 calls/report, cache, fallback.
4. **BodyM is non-commercial.** Fine for portfolio/research; a product would need its own data.
