# Model card: geometric analyzer + waist calibration (P2)

Status: baseline A and the body-composition waist model. The Siamese CNN (P3) will be added to this card.

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
- **Forearm is not predicted** (rho about 0.3) and gets near-zero confidence. `upper_back` and `lats` have no BodyM labels and are fixed at confidence 0.3. Left/right asymmetry is not implemented.
- The silhouette adds little beyond height, weight and sex on chest, hips and thigh in the wild (Test-B).
- A cropped body (feet or head out of frame) breaks the height normalisation behind every feature.
- The files we used carry no age or ethnicity. Male share: 60% (train), 53% (Test-A), 39% (Test-B). Subgroup error by sex and BMI band is planned for P8.
- Parity was checked on 3 captures from one person.
