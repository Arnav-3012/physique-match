# P4 overview, from zero: the muscle-definition model

**Status:** demo scope finished on 2026-11-02. The model works on the data it was trained on. **It has not been shown to work on real photos, and the two real scans we tried looked wrong.** The model ships as an experimental, opt-in estimate that stays hidden in the demo build for now. This file tells that story honestly. The other two files explain the details (`concepts.md`, `model-architecture.md`).

**Who this is for.** You do not need to have read anything else. Every term is explained the first time it appears, and the glossary and viva questions are at the end.

---

## 1. The problem, as a plain picture

The app compares your body with a goal body. The size part (shoulders, waist, and so on) comes from outlines (P3). But two people can have the same outline and look completely different: one soft, one lean with visible muscle. An outline cannot tell them apart. To say anything about **leanness and muscularity** we have to look at the real photo, the colours and shading, not only the outline.

So P4 builds a small image model that looks at a front photo and says two things:

1. **Leanness & muscularity (estimate):** how lean and muscle-defined the torso looks (the model mostly reads the abdomen).
2. **Adiposity:** how much softness there is around the waist (a *waist-girth proxy*, not a body-fat percentage).

Each answer is a **band**: low, mid or high.

**The catch that shaped the whole phase:** to teach a model "this torso is lean and defined", you need thousands of photos that someone has labelled with how lean and defined they are. Nobody has published such a set (we searched, section 9). So we **made our own training photos by rendering fake people on a computer** and gave each one a label from the settings we used to build it. A model that learns on fake people can be perfect on fake people and still wrong on real ones. That gap is the main lesson of P4.

---

## 2. What we built, in order

| Step | What | Where | Result |
|---|---|---|---|
| 1 | Spike: can we render fake people fast enough, and what moves the body? | `synth/spike.py` | Cycles on Apple GPU, about 0.42 s per view; per-region muscle targets exist |
| 2 | Decide how a "definition" label is defined | ADR-049, `synth/latents.py` | a hidden number `d`, hidden by fat and capped by muscle |
| 3 | Add visible muscle lines to the smooth body | `synth/cues.py`, `synth/bump.py` | abs readable, chest and shoulders not |
| 4 | Measure what the weight and muscle sliders really do, and fix the labels | `synth/probe.py`, `synth/fit.py`, ADR-053 | adiposity = waist-girth proxy; clamps frozen |
| 5 | Plan the identities and freeze the split | `synth/plan.py`, `synth/coverage.py`, ADR-045 | 3,996 identities; 2,808 train / 396 val / 792 test |
| 6 | Render everything | `synth/render.py` | 2.25 h, 8,042 front and 8,042 back images |
| 7 | Labels from the final settings | `synth/labels.py` | `synth/labels.csv`, SHA-256 `556adaac...c9f` |
| 8 | Train the 2-head model | `ml/src/definition/train.py` | val definition rho .909, adiposity .895 |
| 9 | Train the 8-head model | `train8.py` | val overall definition rho .921 |
| 10 | Score each once on the test set | `evaluate.py`, `evaluate8.py` | both pass the gate; 8-head is the better definition model |
| 11 | Hybrid choice and export | `export_fp32.py` | two fp32 graphs, 7.46 MB |
| 12 | Check it runs in a browser | `ort_web_parity.mjs`, `bench.html` | parity pass; 14.3 ms p50 |
| 13 | Put it in the app | `apps/web/src/definition/` | opt-in, front frame only, privacy rules |
| 14 | Try it on a real photo | the app, `browser_path.diag.ts` | looked wrong; not a browser bug |
| 15 | Look for real data | research | none with the labels we need |

---

## 3. The data flow, plainly

**Offline (once):**
```
settings (sex, muscle, weight, height, skin, light) ──► Blender + MPFB builds a fake person
      │                                                      │ + procedural muscle lines (bump map)
      ▼                                                      ▼
 hidden numbers: d_eff (definition), adip (softness)   front + back images, transparent background
      │ cut into 5 classes by quantiles                      │ pasted on random backgrounds, blurred, noised
      ▼                                                      ▼
  labels.csv  ───────────────────────────────►  MobileNetV3-Small + ordinal heads  ──► two ONNX files
```

**In the browser (each opt-in scan):**
```
camera frame ──► person mask (MediaPipe) ──► crop around the person + 5% margin
   ──► everything outside the mask = grey 128 ──► resize to 192 x 256 ──► numbers 0..1
   ──► two small neural nets run in the browser (ONNX Runtime Web, WASM)
   ──► two bands (low / mid / high) ──► the frame is deleted from memory
```
Only the two bands may leave the device (ADR-013). The photo never does.

---

## 4. Why synthetic data, and why this was risky (ADR-018, ADR-020, ADR-047)

- **Why:** the project rule is "no scraped photos" (ADR-018) and "demo models train on synthetic + BodyM only" (ADR-020). BodyM has silhouettes only, so it cannot teach definition. Real photos come in P8.
- **Risk we accepted:** labels are a **function of the sliders**, so a high score on fake held-out people shows the network can *invert our own formula*. It does not show it can read real bodies (ADR-049 item 6). We wrote this down before training, and it came true.

## 5. The decisions that mattered, with the reason for each

| Decision | Why | Alternative rejected |
|---|---|---|
| Size the render by cost, cap 3,996 identities, 3 h hard stop (ADR-047 Amendment 1) | the downstream cost (train time) matters more than render time | render as many as possible |
| Definition is its own hidden number `d`, not just the muscle slider (ADR-049) | the muscle slider alone barely changes how the mesh looks | label = muscle slider |
| Visible lines come from a **procedural bump map** at Bump Distance 0.008 | the base mesh is smooth, so without lines there is nothing to learn | sculpt muscle by hand (too slow; TBD after demo) |
| One shader-fix round only (ADR-049 Amendment 1) | stops endless tuning | keep tuning until it looks good |
| Labels from a fitted function of the sliders, known before rendering (ADR-053) | per-mesh measurements are noisy; frozen cutpoints avoid label-peeking | measure each mesh after rendering |
| Clamps frozen: muscle 0 to 0.75, weight 0.25 to 0.75 | slider ends give distorted bodies | invented clamps (the owner refused) |
| Two heads for the demo: overall definition and adiposity (ADR-053 Amendment 4) | the six regional labels came from one shared latent, so they are nearly the same | 7 heads |
| Hybrid: definition from the 8-head model, adiposity from the 2-head model (Amendment 6) | each passed its own checks; the 8-head adiposity head failed the "not worse" rule | ship one model |
| fp32, not int8; combined cap raised 5 MB to 8 MB (owner) | every int8 variant failed parity | keep int8 and the 5 MB cap |
| Opt-in, off by default, dev-only for now (ADR-051, owner) | the first real scans looked wrong | show the score to everyone |
| In P7, show it only as goal against current, in plain words (owner) | one number about your own body, unvalidated, can mislead | a stand-alone score |

## 6. The numbers (all from `docs/logs/dev-log.md`)

**Render:** 3,996 identities, 2.25 h (8,091 s), 222 whole 18-identity rounds, 100% on `Apple M4 Pro (GPU - 16 cores)` METAL, `wall_s` median 1.99 s, max 3.88. Images: 8,042 front and 8,042 back = 3,996 x 2 sets (bump and flat) + 50 lighting-slice renders. Split: train 2,808, val 396, test 792; each of the 9 sex x muscle x fat cells per sex has 156 / 22 / 44; every class holds 20.0% of train.

**Training (val, not the gate):**

| Model | Stage | Result |
|---|---|---|
| overfit one batch | 32 identities, 200 steps | exact 1.000 / 1.000 from step 60 (first run failed, see section 8) |
| 2-head probe (frozen trunk) | 12 epochs | best val mean MAE .7399; adiposity rho only .484 |
| 2-head fine-tune | 21 epochs, best at 14 | mean MAE **.3384**; definition rho .909 MAE .331; adiposity rho .895 MAE .346 |
| 8-head probe | 30 epochs | best mean MAE .5245 |
| 8-head fine-tune | 30 epochs | best mean MAE **.3139**; overall rho .921 MAE .278 |

**Gate (test, scored once, rho >= 0.8 and MAE <= 0.6 per head):**

| | 2-head | 8-head |
|---|---|---|
| definition rho / MAE | .909 / .359 | .923 / .302 |
| adiposity rho / MAE | .901 / .337 | .903 / .349 (front-only samples: .901 / .354) |
| lighting slice (held-out light `top`, 50) | def .906 / .340, adip .818 / **.600** | def .909 / .320, adip .779 / .660 |

Both pass the gate. The incremental gate (a region must beat the overall score by 0.05 rho) **failed for every region** (best: legs, +0.019).

**Shipped hybrid:** definition graph 3.73 MB (8-head, checkpoint `f0a1314d...`), adiposity graph 3.73 MB (2-head, checkpoint `c1142f75...`), combined 7.46 MB. Browser: both load in 149 ms; combined p50 14.3 ms, p95 14.9 ms against a 300 ms limit (laptop, Chrome 156, WASM, one effective thread; phone untested). WASM against Python: 0 class mismatches.

**Learning curve (val, one seed per point):**

| Training identities | 2-head mean MAE | 8-head mean MAE |
|---|---|---|
| 705 (25%) | .5922 | .4632 |
| 1,402 (50%) | .3144 | .3658 |
| 2,808 (100%) | .3384 | .3139 |

The 2-head model is flat from 50% to 100%: more fake data would not help it. The 8-head model keeps improving, but it also sees two views, so it has twice the images.

## 7. The real-photo result (the honest part)

- **Scan 1** (bare torso): definition **high**, adiposity **high**. The owner said this did not match their body.
- **Is it a bug?** We tested the browser path on 396 val renders: definition rho .926 / MAE .278 (Python .928 / .265), adiposity .875 / .409 (Python .895 / .346). **PASS**, so crop, resize, scaling and output mapping are fine **on renders**.
- **Scan 2** (dev preview): the model input was clean (whole body, centred, torso skin share 1.00). Logits gave definition class 4 (high) and adiposity class 2 (low). Same person, different adiposity.
- **Conclusion:** not a browser bug. The leading explanation is the **synthetic-to-real gap** (not proven). Also suspect: test-retest instability and knee-length patterned shorts (training bodies wear nothing).
- **Consequence:** the "sanity ordering on your own scans" part of the P4 gate is **not passed**. The model ships as an experimental opt-in.

## 8. What went wrong (and the fix)

| What | Fix | Logged |
|---|---|---|
| Overfit check **failed** at 0.438 / 0.344 although the loss was tiny | the check used BatchNorm's running averages, which lag after 200 steps; measure in batch-statistics mode with dropout off | `mistakes.md` |
| Weight slider mostly softens low-muscle bodies; "bulky" (muscular and soft) cannot be produced | logged as a limit (ADR-044 Amendment 1, ADR-053 Amendment 2); tag renamed `very_muscular` | ADR, model card |
| Six regional labels share one latent, so they carry no extra information | UI shows one overall band, no per-region definition | ADR-053 Amendment 6 |
| 8-head adiposity head missed the "not worse" rule by .007 MAE | hybrid: adiposity from the 2-head model | dev-log |
| **Every int8 variant failed parity** (agreement .15 to .37 against .95 required) | owner raised the cap to 8 MB and shipped fp32; the cause is **not established** (a mistake in my setup is not ruled out) | dev-log |
| Failed int8 files were written into the served folder | deleted; never served | `mistakes.md` |
| First bench attempt failed on a Vite `?import` suffix | absolute `wasmPaths` | dev-log |
| A user scanned with clothes under the bare-torso option and nothing warned | torso check from skin colour (starting thresholds, unvalidated) | dev-log |
| I claimed "frames released after processing" before it was true | `rgbStore` now releases them; a test guards it | dev-log |

## 9. The dataset search (Level 2 research, mid depth)

**No open dataset with leanness or definition labels exists that we could find.** One 2025 paper says no public one exists for photo-based body fat.

| Source | What it has | Fit |
|---|---|---|
| Shape Up! Adults (416 adults), ShapedNet (1,273 adults) | photos and DXA scans, on request | not confirmed to share 2D photos; no definition labels |
| SHAPY | HBW: 35 people, validation 10 people / 781 images with scan-derived measurements; Model Agency: 95K image URLs, 4.4K models with height, chest, waist and hip, crowd words such as "short" and "big" | non-commercial research licence, registration, no mention of muscularity, no consent statement |
| Unsplash, Pexels | free stock photos | licences are silent on ML training and on model releases; Pixabay bans scraping and data mining "for machine learning purposes" |
| UniData free sample on Hugging Face | 21 people, front, side and selfie | CC BY-NC-ND 4.0; labels and adult status not confirmed |
| Reddit progress-photo communities | public posts | **not used:** age cannot be verified, no consent from the people shown, platform terms |
| CAESAR | licensed 3D scans (from memory, not verified) | not photos; no definition labels |

**What we decided:** no scraping of bare-torso photos, whatever the search term, because the owner can lift ADR-018 for the project but cannot give consent for the people in the photos. The model stays as it is for the demo (owner). The ordering check on consented adults, the flat model and real labelled data are post-demo work.

## 10. What was cut, and where it is kept

Per-head N/A schema, goal-photo inference (a P7 prerequisite), fusion with the body-fat range, lat-flare decision, vein and striation detail, real-photo fine-tune (P8), bump-scale ablation, the flat model and real-photo A/B, per-region display, int8 and its diagnosis, the 7-head model. All are marked **TBD after demo** in `docs/phase-plan.md`, listed in `dev-log.md` as skipped, not passed.

---

## 11. Five interview answers

**1. Why synthetic data?** There is no labelled real dataset, and the project rules forbade scraped photos. Rendering gave us thousands of labelled bodies. The cost is a domain gap, which we documented in ADR-049 before training and later saw on a real scan.

**2. How is a "definition" label defined?** From a hidden number `d`, reduced by fat (`hide`) and capped by muscle, then cut into five classes by quantiles of the training set. So a class is relative to the mix of fake bodies, not an absolute amount of muscle.

**3. Why ordinal heads instead of 5-way softmax?** Classes are ordered. A 5 predicted for a 1 should cost more than a 5 for a 4. CORAL uses four yes/no questions, "is the class above k?", and counts the yeses.

**4. You passed the gate. Why is the model not shown to users?** The gate measures fake held-out bodies. Labels come from our own formula, so passing shows the network learned the formula. The real-photo check was not passed, so the output is experimental and opt-in.

**5. What would you do next?** Rank 6 or more consented adults myself first, then compare with the model's order (Spearman). If poor, retrain the flat model and re-check. For real training, request DXA-paired datasets or collect consented volunteers (P8), and have raters label definition.

---

## 12. Glossary

See `concepts.md` (end of file) for the full glossary and the 10 viva questions.
