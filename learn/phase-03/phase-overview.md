# P3 overview — Siamese CNN (core model)

## What was built
- **Protocol before models** (ADR-032, 033): target `Δ = m_A/h_A − m_B/h_B`; pre-registered gate; fixed Test-B pair list (hash 9abfd0c74c1f, 351 subjects, 122,850 ordered pairs) committed before any scoring; a ledger that refuses a second Test-B look per model hash.
- **Data**: BodyM masks cached at 192x256, subject-first sampling (`ml/src/data/dataset.py`, `pairs.py`).
- **Models**: ResNet-18 pair-head Siamese and baseline B (per-person regressor + subtraction), one transfer-learning ladder (`ml/src/model.py`, `train.py`).
- **Export + integration**: single-person ONNX (45.2 MB), calibration JSON (`gap_regressor.json`), `SiameseSubtractAnalyzer` in `/analyze` with the geometric analyzer as fallback, `REQUIRE_MODEL` for deploy.
- **Add-ons on top of the gap** (ADR-036..042): `lats` as "back width (estimate)", sex fields, V-taper index, reachable ceiling with a three-zone flag, `BodyComp.body_fat` optional.

## Data flow
```
browser: 4 masks + keypoints (no RGB)
   -> API: validate masks -> preprocess (crop, head removal, downscale 192x256)
   -> ONNX person encoder on goal   -> ratios_g (8 regions)
   -> ONNX person encoder on current-> ratios_c
   -> gap = ratios_g - ratios_c  (x user's height = cm)
   -> + reliability per region (calibration JSON), lats (ridge), V-taper, ceiling
   -> GapProfile  (geometric analyzer if the model cannot load and REQUIRE_MODEL is off)
```

## Decisions and why
| ADR | Decision | Why |
|---|---|---|
| 032 | Target = difference of height-normalised ratios | no height shortcut; matches baseline A |
| 033 | Gate: 95% subject-bootstrap CI > 0 on ≥ 6/8 vs each baseline, none worse; Test-B once | stops tuning on the test set |
| 034 | Pair head must beat B on val, else negative result | pre-registered before the `full` runs |
| 035 | Ship baseline B (`SiameseSubtract`), pair head as ablation | it won; exactly antisymmetric; simpler |
| 025 | Side-view dropout in training | front-only goal photo is a valid input |
| 036/037 | Lats from ANSUR II, `unranked` | R² .408 < .5; no BodyM label |
| 038 | V-taper index display-only | unique share .148 < .2 |
| 040-042 | Three-zone ceiling, "cannot verify" wording | female ceiling outside ANSUR range |

## Metrics versus the gate
| Gate item | Result |
|---|---|
| Test-B vs A, full input (8 regions) | 8/8, none worse (e.g. waist .0308 vs .0397, chest .0294 vs .0462) |
| Test-B vs width-only A, front-only | 8/8, none worse; front-only costs ≤ .0017 MAE (chest) |
| Thinnest margins | shoulders lower bound .0016, waist .0044 (front-only) |
| ONNX file ≤ 50 MB | 45.2 MB |
| Latency (1 thread, local) | one encode p50 15.6 ms (limit 75); two encodes p95 31.6 ms (limit 150); re-confirm on Cloud Run at P7 |
| Parity ONNX vs torch | 5.96e-07 (tolerance 2e-4) |
| Own-scan ordering | sensible (torso and sleeves up, shoulders/calf flat) with one caveat: torso noise between two captures of one person is 5-9 cm |
| Tests | 204 passed at the last log entry |

Val ablation (val_mae, standardised): probe .4974, Siamese last2 .2916, Siamese full .2709; B last2 .2927, B full .2791 (seeds .2791/.2785/.2754).

**Honest limits.** B's MAE rose 31-55% from val to Test-B. A ridge on height, weight and sex (which a goal photo does not have) ties or beats B on shoulders, hips, chest, waist, forearm. Reliability numbers are optimistic (calibrated on val). The two disclosed weaknesses are in `ml/MODEL_CARD.md`.

## What went wrong (and was fixed)
- `extract.py` assigned 7 testB photos to testA (key by subject, not (split, subject)).
- Compared train Huber with val MAE in one log line.
- `npm run types` does not regenerate `schema.json`, so TS types were stale for one step.
- `rmr()` treated any non-male as female.
- A dirty-tree script run recorded a commit that did not contain the script; discarded and re-run on a clean tree.
- Forearm win (-57%) may be general arm size, not the forearm.

## Viva: 10 questions
1. **Why is the target a difference of height-normalised ratios?** It removes height as a shortcut and makes the gap comparable across people. The UI multiplies by the user's own height to show cm.
2. **Why did the pair head lose to subtract-and-compare?** The target is `y_g − y_c`, which per-person regression plus subtraction fits exactly; the head only adds the chance to be inconsistent (antisymmetry error .0005-.0035). Val: 1/8 regions significantly better (calf).
3. **What is the effective sample size?** About 1,391 training subjects (usable, unclipped side masks), not the 992 pairs per batch.
4. **Why split by subject?** BodyM has repeated photos; a random split leaks identity.
5. **Why bootstrap subjects and weight pairs by `m_i·m_j`?** Pairs share subjects and are not independent; this is the right resampling unit.
6. **What stops you from tuning on Test-B?** The pair list hash is committed first, `evaluate.py` aborts on a changed list, and `testb_ledger.json` refuses a second look for a model hash.
7. **Why is frozen BatchNorm kept in eval mode?** Otherwise running stats drift away from ImageNet's while the weights are "frozen".
8. **What does overfit-one-batch prove?** Labels, loss, shapes and gradients are wired right (loss .000033 by step 300); it says nothing about generalisation.
9. **Why mask a front-only goal, and what did it cost?** A user's goal photo is usually front-only (ADR-025). Side-view dropout at p = .5 in training; Test-B cost ≤ .0017 MAE and no region got worse than width-only A.
10. **Where could this fail in the wild, and how do you say so?** Clothing, pose, and framing (5-9 cm torso noise across two captures of one person), no weight or height input, reliability calibrated on val and optimistic, back regions not validated; each is in the model card and shown as "estimate"/"not measured" rather than hidden.

## Interview in one breath
"Silhouette-only, privacy-first proportion-gap model: a shared ResNet-18 predicts each person's height-normalised measurements, the gap is their difference. A learned pair head was tried and lost on val under a rule fixed beforehand. It beats the geometric baseline on 8/8 regions on a held-out in-the-wild set, with bootstrap CIs, at 45 MB and ~16 ms per scan on one CPU thread."
