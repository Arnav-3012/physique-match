# PhysiqueGap: whole-project handbook (data, features, targets, models, training)

Status of this doc: written 2026-11-02 from the repo (`docs/architecture.md`, `docs/decisions.md`, `docs/phase-plan.md`, `ml/MODEL_CARD.md`, `ml/src/*.py`, `services/api/app/analyzer/features.py`). Numbers marked **(computed)** are my arithmetic, not logged results. Numbers marked **(general knowledge)** are standard facts about a public architecture, not read from this repo. Everything else is quoted from the repo.

**What is real today:** P0–P3 are done (the Siamese CNN is trained, tested, exported). P4 (definition model) is planned, rendered nothing yet, **trained nothing yet**. P5 (RAG) is in progress. P6 (agent) and P7 (deploy) have not started. Where a section describes something not yet built, it says so.

---

## 0. The whole project in 60 seconds

**Intuition.** You give the app a photo of the body you want and a scan of the body you have. It compares the two outlines region by region ("your chest is X smaller than the goal, relative to your height"), then a small AI agent turns those gaps into a report and a weekly workout plan using a cited knowledge base.

**Privacy rule that shapes everything (ADR-003, ADR-013):** the photo never leaves the browser. Only a binary silhouette (head removed), 33 body keypoints, and a few typed numbers go to the server.

```
BROWSER                                              SERVER (FastAPI on Cloud Run)
 goal photo + 4-view turn scan                        /analyze
  -> MediaPipe Pose Landmarker (33 keypoints + mask)    -> validate mask (384x512, values {0,255})
  -> head cut, binarise, 384x512 silhouette PNG         -> SiameseSubtract ResNet-18 (ONNX) on goal, on current
  -> (opt-in) MobileNetV3-Small "definition" model         -> gap = goal ratios - current ratios (8 regions)
     runs on the RGB frame, ONLY its 1-5 scores leave      -> + lats estimate, V-taper, body-fat range, calories
                                                         /plan (P6, not built)
                                                          -> hybrid RAG (BM25 + bge-small, RRF) -> planner -> LLM writer -> safety check
```

Pipeline of models, in order of use:

| # | Model | Where it runs | Input | Output | Trained on | Status |
|---|---|---|---|---|---|---|
| 1 | MediaPipe Pose Landmarker (lite) | browser (WASM) | RGB frame | 33 keypoints + person mask | Google's weights, not trained by us | done (P1) |
| 2 | Geometric analyzer (no neural net) | API | silhouette | per-region widths/girths -> gap | fitted slopes on BodyM | done (P2); baseline A + fallback |
| 3 | **SiameseSubtract (ResNet-18)** | API, ONNX CPU | 3×256×192 silhouette tensor | 8 ratios per person | BodyM | **done (P3), the core model** |
| 4 | ANSUR ridge for `lats` | API | 4 numbers | back width / stature | ANSUR II | done (P3) |
| 5 | Waist ridge + RFM + Mifflin | API | silhouette waist, height, weight, sex | waist cm, body-fat % range, calories | BodyM + published formulas | done (P2) |
| 6 | **Definition model (MobileNetV3-Small, 7 heads)** | browser (ONNX Runtime Web) | RGB frame | 7 scores, 1–5 | synthetic renders only | **planned (P4), not trained** |
| 7 | bge-small-en-v1.5 (int8 ONNX) | API | text | dense embedding | public model, not trained by us | in progress (P5) |
| 8 | LLM `gpt-oss-120b` on Groq | external | text only | report text | not trained by us | P6, not built |

---

## 1. Datasets

### 1.1 BodyM (the training data for everything size-related)
- **What:** Amazon's body-measurement dataset (arXiv 2210.05667). Per photo: a front and a left-side **binary silhouette**, plus per subject: `height`, `weight`, and **14 tape-style measurements** (lengths of vertex paths on SMPL meshes in A-pose, so they are mesh measurements, not a person with a tape).
- **Size:** 2,505 subjects. Subjects appear **several times** (Test-A has a median of 18 photos per subject), so every split is **by `subject_id`** or it leaks.
- **Licence:** CC BY-NC 4.0, non-commercial. Every coefficient file and model weight derived from it inherits that. This is stated in `ml/MODEL_CARD.md`.
- **Splits (subjects):**

| Split | Subjects | Used (has an unclipped side mask) | Use |
|---|---|---|---|
| Train | 1,715 | 1,391 | fitting |
| Val | 303 | 251 | epoch selection, reliability, all tuning |
| Test-A | 87 | – | geometric analyzer report |
| Test-B ("in the wild") | 399 | 351 | scored **once** per final model (ledger refuses a second run) |

- **Known defects:** 26% of side masks have the feet cut off by the photo edge ("clipped") and are excluded from every side-view fit and evaluation. Male share: 60% train, 53% Test-A, 39% Test-B. No age or ethnicity in the files we used. People wear tight clothes in an A-pose, so loose clothing is out of distribution.
- **Preprocessing (shared by training and API, `ml/src/preprocess.py`):** largest connected component -> bbox + 5% margin -> letterbox into **384×512 (W×H)** -> cut the head at the neck row -> for the model, 2×2 area-average down to **192×256** and stack as `(front, side, zeros)` -> `float32 (3, 256, 192)` in [0, 1]. Browser masks arrive already in this frame.

### 1.2 ANSUR II (only for the `lats` / back-width estimate)
- U.S. Army anthropometric survey: **4,082 men + 1,986 women = 6,068** people, millimetres, one CSV per sex, `subjectid` unique per person.
- Used because BodyM has no back-width label. Target = `interscyei` (distance between the rear armpit folds) divided by stature.
- Licence: no licence text found, only "cleared for unlimited public release". We do not redistribute the CSVs; only a few fitted coefficients ship. Cite Gordon et al. 2015 (NATICK/TR-15/007).
- Limits: US Army population (fitter, younger than civilians); no silhouettes, so the model that feeds the ridge has never seen these people.

### 1.3 Synthetic renders (only for the definition model, P4, not rendered yet)
- Made with **Blender 5.2.3 + MPFB 2.0.18** (CC0 base mesh), Cycles on Metal, **front and back views**, ~384 px, transparent figure composited on backgrounds later, 3 light rigs (`key_left`, `key_right` train; `top` reserved for the held-out-lighting slice).
- **Plan:** 3,996 identities (18 sex-cells × 222 rounds) -> ~8k images (the flat set for the A/B doubles the renders). Per sex a **3×3 grid of muscle (low/mid/high) × fat (low/mid/high)**, round-robin so any whole-round prefix is balanced; each cell uses its own scrambled Sobol sequence over height, proportions, skin tone, `d`, jitters.
- **Split (ADR-045):** inside each cell, blocks of 10 identities = 7 train / 1 val / 2 test, chosen by SHA-256 of the identity id. `N_MIN = 600` with ≥ 100 held out, otherwise stop and report.
- **Bodies are bare** (no clothing assets were found), and the cues are **procedural bump maps**, not anatomy. Abs is the clearest cue; chest, shoulders, back and legs cues are weak or schematic. This is the main honesty caveat of the definition model.
- No scraped photos (ADR-018), no real photos in training (ADR-020). BodyM is not used for this model.

### 1.4 Knowledge-base sources for RAG (P5, demo scope)
- `yuhonas/free-exercise-db` (Unlicense per its README; licence file still to be read) for exercises, one chunk per exercise.
- Guideline documents written in our own words from open-access papers: Pelland et al. (minimum effective weekly sets, preprint v2), Remmert et al. (per-session cap, preprint), Morton 2018, Iraki 2019, Helms 2014, ISSN 2017 (protein/energy). Every number has a `kb/sources.md` row; a number that cannot be opened and checked is left out.
- Eval set: `evals/rag.jsonl`, ~25 planner-form queries (cache eval and the 40-query set are TBD after demo).

### 1.5 Formulas, not datasets (but part of the data story)
- **RFM** body-fat: `64 − 20·(height/waist) + 12·sex` (sex = 1 female, 0 male), validated by its authors on NHANES vs DXA.
- **Mifflin-St Jeor** resting metabolic rate, then energy ranges with safety floors.

---

## 2. Features and targets: how many, which, and for which model

There are **four different "feature sets"** in this project. Mixing them up is the commonest confusion, so here they are separately.

### 2.1 The CNN (the shipped model): no hand-made features
- **Input:** one tensor per person, shape **(3, 256, 192)**: channel 0 = front silhouette, channel 1 = side silhouette (all zeros if absent), channel 2 = zeros. So the CNN's "features" are pixels: 3 × 256 × 192 = **147,456 values** **(computed)**. It never sees height, weight or sex.
- **Target (per person):** **8 numbers** = `measurement / height_cm` for the regions below. The **gap** is defined as `goal_ratio − current_ratio`, so it is the difference of two per-person predictions (ADR-032, ADR-035).

| # | Region | BodyM column it is trained on | View needed |
|---|---|---|---|
| 1 | shoulders | `shoulder_breadth` | front |
| 2 | chest | `chest` | front + side |
| 3 | waist | `waist` | front + side |
| 4 | hips | `hip` | front + side |
| 5 | upper_arm | `bicep` | front |
| 6 | forearm | `forearm` | front |
| 7 | thigh | `thigh` | front |
| 8 | calf | `calf` | front |

BodyM has 14 measurements; **8 are used** (`ml/src/data/regions.py`). The other 6 are not mapped to a region.
- Why ratios to height: the canonical silhouette removes absolute scale, and a goal photo has no height. "Proportion gap" is then converted to cm in the UI as `gap × your own height`.
- The reported app output has **10 regions**: the 8 above plus `upper_back` (always "not measured", fixed reason) and `lats` ("back width (estimate)", ANSUR ridge, unranked). **V-taper** (shoulders ÷ waist, about .40) is a separate display-only field, not an 11th region.

### 2.2 The geometric analyzer (baseline A / fallback): hand-made features
Source: `services/api/app/analyzer/features.py`. All values are lengths divided by the view's own body height in pixels, so there is no weight and no cm.

| Group | Features | Count |
|---|---|---|
| Front widths at landmark rows | `shoulder_w`, `chest_w`, `waist_w`, `hip_w`, `thigh_w`, `calf_w`, `upper_arm_w`, `forearm_w` | 8 |
| Side depths at the same body fractions | `chest_d`, `waist_d`, `hip_d`, `thigh_d`, `calf_d` | 5 |
| Ellipse girths from width + depth (Ramanujan) | `chest_g`, `waist_g`, `hip_g`, `thigh_g`, `calf_g` | 5 |
| Back-view (retired, now "not measured") | `upper_back_w`, `lats_w` | 2 |
| Quality flags | `arms_separated`, `crotch_found` | 2 |
| **Total keys per person** | | **22** **(computed)** |

Only **one feature per region** is actually used for the gap (`REGION_FEATURE`): for example waist uses `waist_g`, thigh uses `thigh_w`. Landmark rows (armpit, crotch, waist, hip) are found from run lengths in the mask itself, not from keypoints. Gap = fitted slope × (goal feature − current feature) per region.

### 2.3 The small ridge models
| Model | Inputs (count) | Target |
|---|---|---|
| Waist model (`waist_coef.json`) | silhouette waist girth, height, weight, sex (**4**) | waist circumference in cm (used only for the current person's RFM) |
| `lats` ridge (`lats_coef.json`) | chest/stature, shoulders/stature, waist/stature, `is_male` (**4**), taken from the CNN's outputs | `interscyei`/stature (back width) |
| Null ridge reference | height, weight, sex (**3**) | the same 8 ratios; a reference the CNN is compared to, not shipped |

### 2.4 The definition model (planned)
- **Input:** one RGB frame in the browser, about 224–256 px (the exact size is not fixed by an ADR yet).
- **Targets: 7 ordinal heads, each a class 1–5:** 6 definition heads (abs, chest, shoulders, arms, back, legs) + 1 adiposity head. Labels are a function of the render sliders (ADR-049): `d_eff_r = min(d · hide(f), cap(m_r))` binned by 20/40/60/80% quantile edges; adiposity is the same binning of fat `f` alone.
- Which heads are scored at runtime depends on the clothing scope the user picks (arms/shoulders/legs, full body, or clothed scan = all N/A).

---

## 3. Model 1 in detail: SiameseSubtract (the CNN we ship)

### 3.1 Intuition
Imagine two judges who each look at one silhouette and write down "chest = 0.52 of this person's height". The gap is just the difference between their two notes. "Siamese" means both judges are the **same brain** (shared weights), so a mistake in taste cancels when you subtract. We also tried a smarter judge who looks at both silhouettes together (the "pair head"); it did not help, because the target already splits into a per-person difference (ADR-035).

### 3.2 Architecture (`ml/src/model.py`)
```
(3,256,192) --(x-mean)/std (ImageNet stats, inside the graph)-->
   ResNet-18 trunk (conv1 7x7/2 -> BN -> ReLU -> maxpool -> layer1..layer4 -> global avg pool, fc removed)
   -> 512-d embedding e
   -> Linear(512, 256) -> ReLU -> Linear(256, 8)        # the head
   -> value(e) = y_mean + head(e) * y_std               # de-standardise, 8 ratios
gap(goal, current) = value(e_goal) - value(e_current)    # exactly antisymmetric
```
- **Activation functions:** ReLU everywhere inside ResNet-18 and in the head's hidden layer; the output layer is **linear** (a regression, no sigmoid/softmax). BatchNorm after every convolution.
- **Why ResNet-18:** low-information binary shapes do not need a big model; it trains in minutes on an M4 Pro (MPS), runs on a free CPU (ADR-004). DINOv2/ViT and EfficientNet were rejected.
- **Parameters (computed from the standard torchvision ResNet-18 count, which is general knowledge):** the trunk without `fc` is about 11.18 M; the head is 512·256+256 + 256·8+8 = 133,384; total about **11.3 M**. The exported ONNX is **45.2 MB (fp32)**, within the 50 MB limit, no quantisation.
- **Pretrained** on ImageNet (RGB). We feed silhouettes in channels 0–1 and zeros in channel 2, and fine-tune all layers.
- **Label statistics** (`y_mean`, `y_std`) come from the train split and live inside the module, so the ONNX carries them.
- **Pair-head ablation (not shipped):** head on `[e_g, e_c, e_g−e_c, e_g⊙e_c]` (4×512 = 2048 inputs), output scaled by `delta_std = y_std·√2`.

### 3.3 Loss, optimiser, schedule (`ml/src/train.py`, config values from the model card)
| Item | Value |
|---|---|
| Loss | Huber on **standardised** per-person ratios (`F.huber_loss`, PyTorch default δ = 1) |
| Side-view handling | `loss = (1−p)·error(full views) + p·error(front only)`, **p = 0.5** (side-view dropout, so a front-only goal photo is a valid input, ADR-025) |
| Optimiser | **AdamW**, head lr **0.001**, encoder lr **0.0001**, weight decay **0.0001** |
| Schedule | Cosine annealing over `steps_per_epoch × epochs` |
| Batch | **32 subjects** per batch (not photos: one canonical photo per subject per draw) |
| Epochs / stopping | up to 25 in the shipped run, early stopping **patience 8**, best epoch **17 of 25** |
| Augmentation | small affine only, on the device: rotate ±3°, isotropic zoom ±3%, shift ±2% (no stretching, because stretching changes the proportions being predicted) |
| Fine-tuning ladder (ADR-004) | `probe` (frozen trunk) -> `last2` (layer3+layer4) -> `full`; stop at the first rung that passes the gate. Val MAE: probe .4974, last2 .2916, **full** .2709–.2791 depending on the model |
| Checks before the full run | overfit one fixed batch (expect loss ≈ 0), then probe, then unfreeze |
| Device / logging | MPS on the M4 Pro; log every 20 steps, one line per epoch |
| Seeds | seed 0 shipped; seeds 1 and 2 only report spread. Val MAE across the three: .2791, .2785, .2754 |
| Selection | on **val only**: mean standardised MAE, full and front-only slices weighted equally; Test-B never read in training |
| Overfitting, stated honestly | at the last epoch Huber was .0416 train vs .0707 val; tuning it is in the P8 backlog |

### 3.4 Results (Test-B, 351 subjects, 122,850 ordered pairs, scored once)
MAE of the gap, as a fraction of height. "A" = the geometric analyzer.

| Region | CNN | A | Ridge on height, weight, sex |
|---|---|---|---|
| shoulders | .0083 | .0107 | .0070 |
| chest | .0294 | .0462 | .0294 |
| waist | .0308 | .0397 | .0310 |
| hips | .0256 | .0434 | .0224 |
| upper arm | .0109 | .0168 | .0117 |
| forearm | .0077 | .0166 | .0076 |
| thigh | .0159 | .0312 | .0166 |
| calf | .0111 | .0190 | .0129 |

- **Gate (ADR-033):** the 95% subject-bootstrap CI of (MAE_A − MAE_CNN) > 0 on ≥ 6/8 regions with none significantly worse. **Result: 8/8 vs A on the full-input slice and 8/8 vs width-only A on the front-only slice, none worse.**
- **Plain finding:** the CNN is about as accurate as a ridge on height, weight and sex (worse on shoulders and hips, slightly better on arm, thigh, calf). Its value is that it needs **no height or weight for the goal photo**. At 175 cm: about 5.4 cm error on waist, 5.1 on chest, 4.5 on hips.
- **Generalisation:** MAE rose 31–55% from val to Test-B, so browser masks should be expected to be at least as hard.
- **Speed (local proxy, 1 CPU thread):** one encode p50 15.6 ms, two encodes p95 31.6 ms; ONNX vs torch max difference 5.96e-07. Not yet measured on the deployed Cloud Run container.
- **Region reliability** shown in the UI: `clip(1 − √(σ_goal² + σ_current²)/σ_prior, 0, 1)`, fixed per region and per slice, from val residuals (examples, full goal: waist .75, hips .76, shoulders .50).

---

## 4. The other models and formulas

### 4.1 Geometric analyzer (baseline A)
No neural network. Run-length analysis on the mask finds armpit, crotch, waist and hip rows; widths and depths at those rows give the 22 features above; one fitted slope per region turns a feature difference into a gap. Per-region confidence = `clip((ρ_val − 0.3)/0.6, 0, 1)`. Silhouette-only Spearman ρ on Test-B: shoulders .80, chest .83, waist .91, hips .80, upper arm .88, thigh .70, calf .73, forearm .28 (so forearm is near-zero confidence). It stays as the fallback only if the ONNX is missing (and `REQUIRE_MODEL=true` makes that a startup failure in the deployed image).

### 4.2 Back width ("lats") and V-taper
- **Lats:** ridge regression (additive sex term, alpha from a log grid, trained on noisy copies of the inputs to mimic the CNN's errors) from `[chest, shoulders, waist]/stature + is_male` to `interscyei/stature`. Result: gate **passed** (min ΔR² over chest-alone .048, CI [.041, .055]) but `R2_min` .408 < .5, so the verdict is **`unranked`** ("shown, never in the priority ranking", label "back width (estimate)", not lat muscle). Needs user sex and goal sex both set and equal.
- **V-taper:** `shoulders_ratio / waist_ratio` from the CNN's per-person outputs (about .40, not the gym 1.618 ideal). Display-only: the ADR-038 unique-share test gave female .148 / male .171 < .2, so it is **not** a planner input.
- **Reachable ceiling:** RFM inverted at `BF_FLOOR` (male 8, female 18) and `BF_ATHLETE_UPPER` (male 13, female 20) gives the V-taper the user's frame allows; three zones; women's ceiling is outside ANSUR's range, so zone wording carries "cannot verify".

### 4.3 Body composition
- Waist ridge (silhouette waist + height + weight + sex -> cm). MAE: Test-A 2.70 cm (null 3.00), Test-B 3.39 cm (null 3.50), residual sd 4.05 cm.
- RFM -> body-fat **range** (never a single number); Mifflin-St Jeor RMR; energy bands with floors (deficit ≤ 20% of TDEE and never below RMR, surplus ≤ 10%). Unset sex -> no body-fat % and no calories (protein per-kg range still shows). Under 18 -> no targets.

### 4.4 Definition model (P4 plan, nothing trained)
- **Architecture:** MobileNetV3-Small pretrained on ImageNet, one shared trunk, **7 ordinal heads** (5 classes each). MobileNetV3 uses hard-swish and ReLU activations and depthwise-separable convolutions **(general knowledge)**; the head type and the ordinal loss are **not yet fixed in any ADR** (ADR-046 is reserved for the eval protocol and fusion), so I do not state a loss here.
- **Training plan:** overfit one batch -> linear probe -> fine-tune; split by render identity; learning curve at 25/50/100% of the rendered set; ONNX int8 ≤ 5 MB; run in the browser with ONNX Runtime Web (WebGPU -> WASM).
- **Gate:** synthetic held-out Spearman ρ ≥ 0.8 per head, MAE ≤ 0.6, ≤ 300 ms per frame, plus a held-out-lighting slice. This gate is **nearly self-fulfilling** (labels are a function of the sliders), so passing shows the net can invert the mapping, not that it reads real bodies. The UI badge stays "experimental, trained on synthetic data" until the P8 real-data check reaches ρ ≥ 0.5.
- Its adiposity score informs only; it never feeds the body-fat range.

### 4.5 Retrieval (P5)
- **BM25** (`rank-bm25`) top-20 + **bge-small-en-v1.5 int8 ONNX** dense top-20, fused with **Reciprocal Rank Fusion, k = 60** (not tuned), then metadata filters (muscle, equipment, level). bge-small produces 384-d embeddings **(general knowledge)**; corpus embeddings are precomputed offline and the corpus and query must use the same int8 file.
- Gate: recall@5 ≥ 0.85 and the hybrid strictly better than BM25-only and dense-only on the demo query set. Cache gate is deferred.

### 4.6 Agent and planner (P6, not built)
One agent with 4 tools (`retrieve`, `get_gap_profile`, `compute_volume`, `check_safety`), direct OpenAI-compatible SDK, pydantic structured output, no LangChain. The planner is deterministic (split by days/week, extra sets for the top-3 gap regions); the LLM only writes the exercise choices, cues and rationale with citations. Baseline-volume invariant: every region gets at least its cited minimum weekly sets (4 fractional sets/week, primary = 1, secondary = 0.5), whatever its gap status. LLM: Groq `openai/gpt-oss-120b` (fallback OpenRouter no-train), text only, ≤ 3 calls per report.

---

## 5. Deployment and constraints (short)
Web on Vercel; API on Google Cloud Run free tier (1 vCPU / 2 GiB, `min-instances=0`, `max-instances` 1–2); image `python:3.12-slim` with **no torch**; models on Google Cloud (baked into the image or a private bucket, decided in the P7 spike). Cache and rate-limit state in Upstash. Billing safety (budget, kill switch tested in simulation, `max-instances`) comes before the first deploy. **Nothing is deployed yet.**

---

## 6. Glossary
- **Silhouette / mask:** binary image, body = 255, background = 0.
- **Siamese network:** two inputs through the same weights.
- **Backbone / encoder:** the feature-extracting part of a CNN (here ResNet-18 minus its classifier).
- **Embedding:** the 512-d vector the encoder outputs for one silhouette.
- **Head:** the small layers on top of the embedding that produce the final numbers.
- **ReLU:** `max(0, x)`. **BatchNorm:** normalises activations per batch; frozen BatchNorm must keep ImageNet statistics.
- **Huber loss:** squared error for small errors, absolute error for large ones, so outliers do not dominate.
- **AdamW:** Adam with decoupled weight decay. **Cosine schedule:** learning rate decays along a half cosine.
- **Linear probe:** train only the head on frozen features. **Fine-tuning:** also update the trunk.
- **Side-view dropout:** randomly blank the side input during training so a front-only goal is in-distribution.
- **Antisymmetry:** swapping goal and current negates the gap; subtraction guarantees it exactly.
- **Ordinal head:** classes with an order (1 < 2 < … < 5); mistakes far from the truth should cost more.
- **Spearman ρ:** rank correlation. **MAE:** mean absolute error.
- **Subject-level split:** all photos of a person stay in one split.
- **Bootstrap CI:** resample subjects with replacement to get a confidence interval.
- **Null model / ridge reference:** a simple model on height, weight, sex the CNN must be compared to.
- **RFM:** Relative Fat Mass formula. **RMR/TDEE:** resting / total daily energy.
- **BM25:** keyword scoring. **Dense retrieval:** embedding similarity. **RRF:** combine two rankings by `Σ 1/(k + rank)`.
- **ONNX:** portable model file; **int8:** 8-bit quantised weights.
- **Sobol sequence:** low-discrepancy sampling so any prefix covers the space evenly.
- **Bump map:** a texture that fakes surface relief in a render (our "definition" cue).

---

## 7. Ten viva questions
1. Why is the CNN's target `measurement/height` and not centimetres, and what does the UI do with it?
2. Why do we split BodyM by `subject_id`, and what would have happened if we split by photo?
3. The pair head had a lower val MAE (.2709 vs .2791) but was rejected. Why is that correct, not stubborn?
4. Why is the subtraction head exactly antisymmetric, and why does that matter for a goal-versus-current tool?
5. What is side-view dropout, and which real-world input does it make valid?
6. The CNN is about as good as a ridge on height, weight and sex. Why is it still worth shipping?
7. Why can the `lats` estimate not be called a lat-muscle measurement, and why is it unranked even though its gate passed?
8. What makes the definition model's synthetic gate "nearly self-fulfilling", and what evidence would actually count?
9. Why RRF with k = 60 rather than a weighted sum of BM25 and dense scores?
10. Walk through exactly what leaves the browser in a scan, and which ADR guarantees each part.

---

## 8. Where to look next (files)
- Architecture and flow: `docs/architecture.md` · decisions: `docs/decisions.md` (ADR-004, 024, 029, 032–035, 036–042, 044–052) · phase status: `docs/phase-plan.md`
- Model card with all tables: `ml/MODEL_CARD.md`
- CNN code: `ml/src/model.py`, `ml/src/train.py`, `ml/src/augment.py`, `ml/src/preprocess.py`, `ml/src/data/regions.py`
- Geometric features: `services/api/app/analyzer/features.py`
