# Model card: PhysiqueGap body-proportion model

**Name in code:** `SiameseSubtract` (training), `SiameseSubtractAnalyzer` (API). Trained as "baseline B".
**Checkpoint:** `baseline_b_full`, seed 0, SHA-256 `213383d7c86f33c8040910c7c165f38589eecbdc8b6da4330641a2ee86f26b0b`.
**Decision record:** ADR-032 (target), ADR-033 (gate), ADR-034 and ADR-035 (selection).

## What it does
Takes a body silhouette (front view, plus a side view if there is one) and predicts 8 body proportions, each a measurement divided by height: shoulders, chest, waist, hips, upper arm, forearm, thigh, calf. The API runs it once on a goal silhouette and once on the user's own, then subtracts. The result is a **proportion gap per region**.

A goal needs a front view only. A current scan needs a front view and one side view (left, or right mirrored). A region that can't be measured is returned as not measured, with a reason, not as a zero. `upper_back` is always one (no size number from any view, ADR-039). `lats` is added by the API from this model's chest, shoulders and waist, as "back width (estimate)", and only when the user's sex and the goal's sex are both set and equal (see Limits).

It estimates outline-based proportions only. It can't see muscle versus fat, and it knows nothing about health, fitness or how a body should look.

**Intended use:** a non-commercial demo that suggests which regions differ most from a goal. **Out of scope:** medical or body-composition assessment, tailoring-grade measurement, judging anyone's body, and any commercial use (see Licence).

## Architecture
- ResNet-18 (ImageNet weights, all layers fine-tuned), then a two-layer head (256 hidden units) to 8 outputs.
- Input: float, shape (3, 256, 192): front, side, zeros. A missing side view is an all-zero channel. Normalisation and label scale are inside the ONNX graph.
- Gap = prediction for the goal − prediction for the current scan, so it's exactly antisymmetric.
- A learned pair head was tested as an ablation and not shipped (see Results).

## Data
[BodyM](https://registry.opendata.aws/bodym/) (Amazon): silhouettes and tape measurements. Split by subject, no subject in two splits.

| Split | Subjects in split | Used (has an unclipped side mask) | Use |
|---|---|---|---|
| Train | 1,715 | 1,391 | Fitting |
| Val | 303 | 251 | Epoch and model selection, reliability |
| Test-B (in the wild) | 399 | 351 | Scored once, on this model only |

Photos with a clipped side mask (26%) were excluded throughout, so subjects with only clipped side masks are outside every number here. No synthetic data was used for this model.

## Training
Huber loss on standardised ratios, AdamW (head lr 0.001, encoder lr 0.0001, weight decay 0.0001), cosine schedule, 32 subjects per batch, early stopping (patience 8, best epoch 17 of 25), small affine augmentation, 50% side-view blanking. No head dropout or tuning of the train/val gap: at the last epoch of this seed, Huber was .0416 on train against .0707 on val, so the model does overfit. That's in the P8 backlog.

Seeds: seed 0 was fixed as the shipped model before seeds 1 and 2 ran. Val MAE across the three: .2791, .2785, .2754; seed 0 is the highest of the three. Spread per region never exceeded the CI margin against the geometric baseline.

## Results
MAE of the per-region gap over every ordered pair of subjects, unit = fraction of height. **Test-B, 351 subjects, 122,850 ordered pairs, scored once.** "Better by" is a 95% subject-bootstrap CI against the geometric analyzer (A).

| Region | This model | A | Better by | Height, weight, sex ridge | Front-only goal | Width-only A | Better by |
|---|---|---|---|---|---|---|---|
| shoulders | .0083 | .0107 | .0023 (.0016, .0030) | .0070 | .0083 | .0107 | .0023 (.0016, .0029) |
| chest | .0294 | .0462 | .0168 (.0136, .0200) | .0294 | .0311 | .0508 | .0197 (.0163, .0230) |
| waist | .0308 | .0397 | .0089 (.0058, .0121) | .0310 | .0315 | .0388 | .0072 (.0044, .0101) |
| hips | .0256 | .0434 | .0177 (.0139, .0218) | .0224 | .0262 | .0456 | .0194 (.0149, .0238) |
| upper arm | .0109 | .0168 | .0058 (.0039, .0078) | .0117 | .0114 | .0168 | .0054 (.0034, .0074) |
| forearm | .0077 | .0166 | .0088 (.0074, .0103) | .0076 | .0080 | .0166 | .0086 (.0071, .0100) |
| thigh | .0159 | .0312 | .0153 (.0116, .0192) | .0166 | .0164 | .0312 | .0149 (.0113, .0186) |
| calf | .0111 | .0190 | .0079 (.0062, .0094) | .0129 | .0110 | .0190 | .0079 (.0063, .0095) |

At 175 cm tall, that's about 5.4 cm on waist, 5.1 cm on chest and 4.5 cm on hips.

**The ridge finding, plainly.** This model is about as accurate as a ridge regression on height, weight and sex (point estimates, no CI): worse on shoulders and hips, slightly better on upper arm, thigh and calf, tied on chest, waist and forearm. The ridge can't be used on a goal photo, because a goal photo has no weight. That's this model's value: it needs no height or weight for the goal.

**Pair-head ablation (val only, one seed).** A learned pair head reached CI > 0 against this model on 1 of 8 regions (calf) and was rejected by the rule written before that run. Its overall val MAE was lower (.2709 against .2791), which is recorded and doesn't override the rule. It wasn't scored on Test-B.

## Region reliability
The number shown per region in the UI. It's **fixed per region and per slice, the same for every scan**, so it's not a per-scan uncertainty.

`reliability = clip(1 − sqrt(σ_goal² + σ_current²) / σ_prior, 0, 1)`, from val residuals.

| Region | Full goal | Front-only goal |
|---|---|---|
| shoulders | .50 | .49 |
| chest | .73 | .71 |
| waist | .75 | .74 |
| hips | .76 | .75 |
| upper arm | .68 | .67 |
| forearm | .61 | .61 |
| thigh | .69 | .68 |
| calf | .61 | .61 |

These are optimistic. Val also selected the epoch, and Test-B error ran 31–55% above val. They also assume independent errors for the two people.

## Limits
- **No real-world validation.** Browser masks were compared with BodyM on only 3 captures (12 masks). Error rose 31–55% from val to in-the-wild Test-B, so expect browser masks to be at least that hard. Tape-measured validation comes in P8.
- **Goal photos** are often posed, lit, pumped or edited, and this model takes the outline at face value. Scores from a goal photo are appearance-based.
- **Forearm** may reflect general body size more than the forearm itself: its error matches the ridge's, and the silhouette's own forearm feature is weak.
- **No subgroup results yet** by sex or BMI (P8). BodyM doesn't represent all bodies, and this model hasn't been checked across them.
- **Back width (API key `lats`) is an estimate, not a measurement, and not a measure of lat muscle.** It is a ridge fitted on ANSUR II (4,082 men, 1,986 women; target = the tape distance between the rear armpit folds, over stature) from chest, shoulder and waist ratios plus an additive sex term, applied to this model's outputs. BodyM has no back-width label, so it can't be fitted or checked there. Out-of-fold R² with inputs as noisy as this model's: .47 (men), .41 (women); the extra signal over chest alone is dR² .048 (CI .041 to .055), so it passed its gate, but R² under .5 made it **unranked** (shown, never in the priority ranking). Even with perfect inputs the women's R² is .45. Reliability is about .27 for goal and current. Two disclosed weaknesses: BodyM's shoulder breadth runs about 10% below ANSUR's closest column (biacromial), so this model's shoulder gaps look about 0.9x too small to the ridge, and the ridge's shoulder coefficient is 17% lower for being fitted on noisy inputs. Both are a scale mismatch the fit can't see.
- **Sex fields.** `lats` needs the user's sex and the goal's sex, both typed by the user (never inferred, never defaulted), and equal; then the sex term cancels in the gap. If they differ or either is unset, it is not measured and the reason is shown; a different-sex goal also gets a warning that absolute size gaps may not be reachable. This model itself takes no sex input, and the other seven regions are identical with or without it (tested). Unset sex also removes body-fat % and calorie targets, which need it.
- **`upper_back`** has no size output at all: "Not measurable from a front photo; a back silhouette adds no size information." The back-view fits of the geometric analyzer were retired.
- **Clothing:** trained on BodyM's silhouettes. Loose clothing changes the outline and isn't corrected for.

## Performance (local proxy, M4 Pro, ONNX Runtime CPU, 1 thread)
45.2 MB (fp32, no quantisation). One encode: p50 15.6 ms, p95 16.0 ms. Two encodes: p50 31.3 ms, p95 31.6 ms. ONNX vs torch: max difference 5.96e-07 over 32 val subjects. **Not yet measured on the deployed 1 vCPU container** (P7).

## Licence
Derived from BodyM, **CC BY-NC 4.0: non-commercial use only**, with attribution to the BodyM authors. ResNet-18 ImageNet weights come from torchvision.

---

# Model card: geometric analyzer + waist calibration (P2)

Status: baseline A and the body-composition waist model. The shipped neural model is the card above.

## What it is
- **Geometric analyzer** (`services/api/app/analyzer/`): no neural network. Measures widths, depths and arm thickness on a binary silhouette at rows found from the outline, divides by body height, and converts a feature difference (goal minus current) to a gap with one fitted slope per region.
- **Waist model** (`analyzer/waist_coef.json`): a ridge regression from the silhouette waist girth, height, weight and sex to waist circumference in cm. Used only for the current person, to feed the Relative Fat Mass range.

## Data
- **BodyM** (Amazon, 2,505 subjects, front and left-side silhouettes, height, weight, 14 tape measurements). Licence **CC BY-NC 4.0: research and portfolio use, not commercial.** The coefficient files are derived from it and inherit that licence.
- Splits are by `subject_id` (subjects appear several times): train 1,715 · val 303 · Test-A 87 (median 18 photos per subject) · Test-B 399 (the in-the-wild set; one subject shared with Test-A was dropped from Test-B). 26% of side masks have the feet cut off by the photo edge and are excluded from every side-view fit and evaluation.

- Row windows and feature choices were tuned on train and val only. The gate criterion was changed once after the first result (ADR-029); Test-A and Test-B are reported together.

## Metrics (subject level for the gap, photo level for waist)
| Region | silhouette-only rho, Test-A | Test-B |
|---|---|---|
| shoulders | 0.83 | 0.80 |
| chest | 0.85 | 0.83 |
| waist | 0.91 | 0.91 |
| hips | 0.83 | 0.80 |
| upper arm | 0.87 | 0.88 |
| thigh | 0.78 | 0.70 |
| calf | 0.79 | 0.73 |
| forearm | 0.30 | 0.28 |

Target is measurement / height. A ridge on height, weight and sex alone reaches rho 0.80 to 0.93 on every region; the silhouette-only features do not beat it (they do not use weight), but adding them to it improves rho significantly on 7 of 8 regions on Test-A and 5 of 8 on Test-B.

Waist MAE (cm), silhouette model vs height-weight-sex null: val 3.05 vs 4.32 · Test-A 2.70 vs 3.00 · Test-B 3.39 vs 3.50. 90th percentile error 5.7 cm (A) and 7.3 cm (B). Residual sd 4.05 cm.

## Intended use
Comparing a goal silhouette with the user's own scans, and a rough waist estimate with a stated range. Output is always a range or a per-region confidence; never a single body-fat number.

## Limits
- BodyM subjects wear tight clothing in an A-pose; real users wear shorts and loose tops. On three browser captures, shorts hid the crotch row (row finder falls back to a prior, confidence drops) and loose tops enlarged the silhouette area by 15 to 25%.
- **Forearm is not predicted** (rho about 0.3) and gets near-zero confidence. `upper_back` and `lats` have no BodyM labels; the back-view values this analyzer once gave them (fixed confidence 0.3) are retired and both are now not measured (ADR-037 Amendment 2). Left/right asymmetry is not implemented.
- The silhouette adds little beyond height, weight and sex on chest, hips and thigh in the wild (Test-B).
- A cropped body (feet or head out of frame) breaks the height normalisation behind every feature.
- The files we used carry no age or ethnicity. Male share: 60% (train), 53% (Test-A), 39% (Test-B). Subgroup error by sex and BMI band is planned for P8.
- Parity was checked on 3 captures from one person.
