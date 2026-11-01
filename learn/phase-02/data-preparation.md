# P2 · Everything we did to the data to make it suitable

BodyM arrives as raw silhouettes and tape measurements. This is the full list of what was done to it, in order, why, and where the code is. Numbers are from the real data and the logs.

## 0. The short version

| # | Step | Why | Code |
|---|---|---|---|
| 1 | Download and check the layout | never write a loader from memory | `aws s3 cp` (87.2 MiB, 17,965 files) |
| 2 | Join photos, metadata, measurements | one row per photo pair | [bodym.py](../../ml/src/data/bodym.py) |
| 3 | EDA: repeats, sex, shift, overlap, "what can weight alone do" | find traps before modelling | [eda.py](../../ml/src/eda.py) |
| 4 | Split by subject; keep both test sets untouched | no leakage | [split.py](../../ml/src/data/split.py) |
| 5 | Map 14 measurements to 8 labelled regions; target = measurement ÷ height | scale-free labels | [regions.py](../../ml/src/data/regions.py) |
| 6 | Clean each mask: binarise, keep the largest blob | specks widen the bounding box | [preprocess.py](../../ml/src/preprocess.py) |
| 7 | Crop and letterbox into the browser's 384x512 frame | train and inference see the same thing | `from_bodym` |
| 8 | Cut the head at the neck row | the browser sends head-less masks | `neck_row` |
| 9 | Recover body height from the feet row | no keypoints in BodyM | `body_px` |
| 10 | Drop side masks with cut-off feet | height reference is wrong | `clipped` |
| 11 | Find landmark rows from the outline | one code path for both sources | [rows.py](../../services/api/app/analyzer/rows.py) |
| 12 | Turn rows into scale-free features | goal photo has no weight or height | [features.py](../../services/api/app/analyzer/features.py) |
| 13 | Handle missing values with NaN, never a guess | a made-up number looks real | `features.extract` |
| 14 | Aggregate photos to subjects for the gap; keep photos for waist | one scan is one photo | [eval_geometric.py](../../ml/src/eval_geometric.py), [calibrate_waist.py](../../ml/src/calibrate_waist.py) |
| 15 | Fit on train, tune on val, report on test; always beside a null model | honest numbers | same files |
| 16 | Check that browser masks match BodyM masks | the model will see browser masks | [parity.py](../../ml/src/parity.py) |

## 1. Source and layout

BodyM (Amazon, registry entry `amazon-bodym`): `train/`, `testA/`, `testB/`, each with `mask/` (front), `mask_left/` (left side), and three CSVs: `subject_to_photo_map.csv`, `hwg_metadata.csv` (sex, height, weight), `measurements.csv` (14 tape measurements in cm). Verified on disk, not assumed: masks are `L`-mode PNGs, 720×960, values {0,255}. **Licence CC BY-NC 4.0**: fine for research and portfolio, not commercial; stated in [MODEL_CARD.md](../../ml/MODEL_CARD.md).

| Split | Subjects | Photo pairs | Photos per subject (median / max) | Male share |
|---|---|---|---|---|
| train | 2,018 | 6,134 | 2 / 27 | 60% |
| testA | 87 | 1,684 | 18 / 45 | 53% |
| testB | 400 | 1,160 | 3 / 11 | 39% |

## 2. The join

`load_table` merges photo map → metadata → measurements with `validate="many_to_one"` (many photos, one subject), checks that no photo is lost by the join and that every mask file exists, and renames hyphenated columns to underscores. The measurements and the metadata are **per subject**; the masks are **per photo**.

## 3. What the EDA found, and what we did about it

| Finding | Action |
|---|---|
| Subjects repeat (up to 27 photos) | split by `subject_id`, never by photo |
| 1 subject is in both Test-A and Test-B (train overlaps neither) | dropped from Test-B |
| Test-A has 87 subjects but 1,684 photos | compute rho per subject (average photos); 87 independent points, not 1,684 |
| Test-B is shorter (167 vs 171 cm), more female (61%), wider waist spread | it is the in-the-wild set; reported separately |
| tape `height` differs from metadata `height_cm` by -22.9 to +14.3 cm | use the metadata height (it is also what a user types) |
| weight alone predicts measurement÷height with rho 0.39 to 0.67 (train) | build a null model, since the gate cannot be a bare rho (later: .80 to .93 with height and sex) |
| 26% of side masks touch the bottom edge (feet cut off) | exclude them from side-view work (§8). The rate is the same for men and women (23%/24%) and for height (170.6 vs 171.1 cm), so dropping them does not bias the data |

## 4. Splitting without leaking

- Train/val: `make_split` takes the 2,018 train subjects, stratifies by sex, seeds a permutation (seed 0) and holds out 15% as **val** → **train 1,715 / val 303 subjects**.
- **Test-A (87)** and **Test-B (399 after the dedupe)** are never used to fit or to choose anything. After ADR-029 changed the gate criterion, all feature tuning moved to train (row windows, `sweep_windows.py`) and val; the test sets were read once, together.
- The split is a JSON of subject ids (`ml/data/splits.json`, gitignored with the rest of `ml/data/`). Tests: no subject in two splits (synthetic and the real file), determinism, sex stratification.

## 5. Labels

BodyM has 14 measurements; 8 map to regions: shoulders ← shoulder-breadth, chest ← chest, waist ← waist, hips ← hip, upper_arm ← bicep, forearm ← forearm, thigh ← thigh, calf ← calf. **`upper_back` and `lats` have no measurement**, so they are geometric-only and outside every gate. **Target = measurement ÷ height_cm**, so people of different size are comparable and the label has the same units as the scale-free features.

## 6. Cleaning each mask

1. Threshold at 127 → boolean.
2. **Keep the largest connected blob** (`scipy.ndimage.label`). The raw side mask we looked at had a stray speck; one stray pixel far from the body would stretch the bounding box and shrink the person in the frame. Test: a speck in the corner changes nothing.
3. Reject empty masks (raise) and, at the API, all-white ones.

## 7. The canonical frame (training and inference see the same thing)

The browser sends masks already cropped, letterboxed into 384×512 and head-less ([silhouette.ts](../../apps/web/src/capture/silhouette.ts)). BodyM is 720×960 with a head. `from_bodym` reproduces the browser's steps exactly:

1. bounding box of the (cleaned) body;
2. pad **5% of the box height** on all four sides;
3. `scale = min(384/width, 512/height)`; resize with bilinear and threshold at 0.5; centre in 384×512;
4. **cut the head**: find the narrowest row between 8% and 25% of body height below the crown (the neck, on a 5-row smoothed width profile) and zero everything above it.

Browser masks go through `from_browser` instead: shape check, largest blob, nothing else. **Re-cropping them would be a bug**: their bounding box starts at the neck cut, so re-fitting it would enlarge the body. (My first plan said "re-detect the bbox"; it was wrong, and the test "browser mask is a fixed point" pins the fix.)

`NECK_CUT_OFFSET = 0.0` came from the parity check: with 0.02, BodyM's cut sat lower than the browser's (browser side cut at 0.12–0.16 of body height vs BodyM 0.176). After the change, head-cut |z| ≤ 1.9 on three captures; the phone captures still cut higher, which P3 covers with cut-row jitter augmentation.

## 8. Body height without keypoints, and the clipped-feet problem

Every feature is divided by body height in pixels. The letterbox is centred, so `feet_row_end = 256 + body_px/2` and **`body_px = 2·(feet_row_end − 256)`**. That works on head-less masks (the feet are still there). It fails when the feet are cut off by the photo edge. In BodyM, **26% of side masks** touch the bottom edge (3 of 300 front masks do). The cut-off amount is unknown (the side/front height ratio varies 0.59 to 1.17 within one photo pair, so the front view cannot repair it). So `clipped(mask)` = "bottom row has any white", and those side masks are excluded from every side-view fit and evaluation. Effect: 12 to 29% of subjects lack a chest/waist/hip girth, which is why thigh and calf use front width only. The browser's own "body touches frame edge" check already prevents the same thing at capture time.

## 9. Finding landmark rows from the outline

BodyM has no keypoints, so rows come from run lengths ([rows.py](../../services/api/app/analyzer/rows.py)): a *run* is a horizontal stretch of white pixels in a row (at least 3 pixels).
- **Armpit:** first row, searching 18% to 45% of body height, with 3 or more runs (arm, torso, arm) held 5 rows.
- **Crotch:** first row in 45% to 65% where the widest run falls below 65% of its maximum since the armpit, held 8 rows (the torso splits into legs).
- **Waist:** narrowest width between armpit and crotch; **hip:** widest below the waist; **chest** and **shoulder** at fixed offsets from the armpit.
- If a rule finds nothing, a height-proportion prior is used and a flag (`arms_separated`, `crotch_found`) lowers confidence.

On 400 train front masks: arms separated 99.8%, crotch found 98%. On three browser captures with shorts: crotch found in 2 of 6 views (the hem is not a crotch; see ADR-030).

## 10. Features: what is measured, where, and in what unit

All values below are divided by the view's own `body_px` (so the unit is "body heights").

| Region | Feature | Where | Why this one |
|---|---|---|---|
| shoulders | span of the whole row | armpit − 0.07 of body height | swept on train: rho .636 vs .617 at .045 |
| chest, waist, hips | ellipse girth from front width + side depth | chest / waist / hip row | depth adds information (val rho: waist .92 girth vs .91 width; hips .83 vs .77) |
| upper arm | arm thickness (area ÷ path length) | 0.00 to 0.07 below the armpit | train rho .83 vs bicep |
| forearm | arm thickness | 0.12 to 0.19 below the armpit | honest and weak (rho .3): the upper-arm window scores .78 vs forearm but is just arm size, so rejected |
| thigh, calf | widest front leg width in a range | 2–25% / 50–80% of crotch→feet | single rows were fragile; side depth hurt (hands, clipped feet) |

**Side-view rows** are taken at the same body fraction as the front rows, and a depth is **skipped (NaN)** when it jumps more than 15% against the median of the 6%-of-body-height window around it (a hand or forearm in front of the thigh). **Missing is NaN, never a fill**: a model must not be handed an invented waist.

## 11. Aggregation: subject level vs photo level

- **Gap evaluation:** features averaged per subject (NaN-aware), so each person counts once; Test-A has 87 points.
- **Waist calibration:** photo level, because one scan at run time is one photo, so the error should include photo-to-photo noise (Test-A 2.70 cm MAE).

## 12. Fitting hygiene

- Everything is fit on **train**; **val** picks features and windows; **Test-A/Test-B** only report.
- Ridge models are `StandardScaler + Ridge(alpha=1)` in one pipeline so the scaling is learned on train only; for deployment the scaler is folded into raw-unit coefficients (`waist_coef.json`) so the API needs no scikit-learn.
- Every silhouette result sits beside a **null model** (ridge on height, weight, sex). It reaches rho .80 to .93, which is why a bare "rho ≥ 0.6" would have meant nothing.
- Intervals: 1,000 resamples of subjects.

## 13. Making sure browser masks match

[parity.py](../../ml/src/parity.py) pushes browser masks and 300 BodyM train masks through the same preprocessing and compares: cut position, mask area, width ratio, row-width profile correlation, row-finder flags. On 3 captures (12 masks): profile r .96 to .99 front/back (.83 to .92 side), only 2 of 12 masks with a statistic beyond |z| 2 (area: loose tee), `arms_separated` 6/6. What it exposed that BodyM could not: **shorts hide the crotch**, and **loose clothing inflates area by 15 to 25%**. Not fixed here: P3's mask augmentation (clothing bulge, edge noise) is where that is addressed.

## 14. What we did not do (and why)

| Not done | Reason |
|---|---|
| Fill or impute missing features | a guess looks like a measurement |
| Scrape or add real photos | ADR-018: only BodyM and synthetic data for the demo |
| Remove outlier subjects | the metadata/tape height errors are handled by using one source, not by deleting people |
| Augmentation | P3 (pairs, swap-and-negate, edge noise, clothing bulge, cut-row jitter) |
| Asymmetry and chest/back split | no ground truth to check them against |
| Anything with weight inside the gap | goal photos have no weight (ADR-024) |

## 15. Lineage

`ml/data/bodym/` (raw, gitignored) → `ml/data/splits.json` → `ml/data/features.csv` (one row per photo pair: ids, sex, height, weight, 14 measurements, 8+ features, `set`, `side_clipped`) → `services/api/app/analyzer/gap_coef.json` and `waist_coef.json` (small, committed, CC BY-NC derived). Re-run order: `extract` → `eval_geometric` → `fit_gap` → `calibrate_waist`, all from the repo root with `PYTHONPATH=services/api`.
