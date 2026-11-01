# P2 · BodyM pipeline, geometric baseline, body composition: overview

**One line:** turn the 2,505-subject BodyM silhouette dataset into a clean, leak-free training and evaluation set; measure the body from a silhouette alone with plain geometry (no neural network); check that a browser-made mask looks like a BodyM mask; and expose it all behind `POST /analyze`.

> **Status when written:** the BodyM, geometric-analyzer, body-composition and `/analyze` parts are done. The **Cloud Run deploy spike (ADR-028) is not**: P2 stays open until its gate is logged in [dev-log.md](../../docs/logs/dev-log.md). Add a "Deploy spike results" section here when it is.

Why it matters: P3's Siamese CNN has to beat this baseline (ADR-006), and P7's `/analyze` is this route with the model swapped in behind the same `Analyzer` interface.

## 1. Data flow

```
 BodyM on disk (87 MiB)                                         bodym.py: load_table
   <split>/{mask, mask_left}/<photo_id>.png  +  3 CSVs   ──►   one row per photo pair (+ sex, height, weight, 14 tape measurements)
        │                                                              │
        │                                                  split.py: subject-level split ► ml/data/splits.json
        ▼                                                              │
 preprocess.from_bodym  (largest blob ► crop ► letterbox 384x512 ► cut head)   ◄── same frame as the browser
        │
        ▼
 rows.py   armpit, crotch, waist, hip rows from run lengths
        │
        ▼
 features.py   widths / depths / girths / arm thickness, all ÷ body height          extract.py ► features.csv
        │
        ├─► eval_geometric.py   rho per region vs a height-weight-sex null   (gate 2)
        ├─► fit_gap.py          slope per region ► gap_coef.json
        └─► calibrate_waist.py  silhouette + height + weight + sex ► waist cm ► waist_coef.json  (gate 3)

 At run time (no pandas, sklearn or torch):
   browser Scan ──► POST /analyze ──► security.py (size cap, rate limit) ──► masks.py (strict PNG check)
        ├─► GeometricAnalyzer.compare(goal, current) ──► GapProfile        (weight never enters)
        └─► bodycomp.body_comp(profile, current waist) ──► RFM range + Mifflin energy ranges with floors
```

## 2. What each file does

### Data and analysis (`ml/src/`, needs the `ml` dependency group)
| File | Role |
|---|---|
| [data/bodym.py](../../ml/src/data/bodym.py) | `load_table(split)`, `load_mask`, `clipped` (feet cut off) |
| [data/split.py](../../ml/src/data/split.py) | `make_split`: train/val by subject, sex-stratified; testA/testB fixed; shared subject dropped from testB |
| [data/regions.py](../../ml/src/data/regions.py) | the 8 labelled regions and their BodyM column |
| [eda.py](../../ml/src/eda.py) | sizes, repeats, sex, overlap, per-split stats, "how far can weight alone go" |
| [preprocess.py](../../ml/src/preprocess.py) | the canonical 384x512 frame, shared with the API |
| [tune_rows.py](../../ml/src/tune_rows.py), [show_measurements.py](../../ml/src/show_measurements.py) | overlays of rows and measurement dots, for looking |
| [parity.py](../../ml/src/parity.py) | browser masks vs BodyM through the same preprocessing |
| [sweep_windows.py](../../ml/src/sweep_windows.py) | picks arm and shoulder row windows on train |
| [extract.py](../../ml/src/extract.py) | features for every photo pair ► `ml/data/features.csv` |
| [eval_geometric.py](../../ml/src/eval_geometric.py) | rho per region, null model, bootstrap, gate |
| [fit_gap.py](../../ml/src/fit_gap.py), [calibrate_waist.py](../../ml/src/calibrate_waist.py) | write the two coefficient files |

### Served code (`services/api/app/`, runs in the container)
| File | Role |
|---|---|
| [analyzer/masks.py](../../services/api/app/analyzer/masks.py) | `decode_mask`: the server half of ADR-003 |
| [analyzer/rows.py](../../services/api/app/analyzer/rows.py) | `find_rows`: landmark rows from the silhouette alone |
| [analyzer/features.py](../../services/api/app/analyzer/features.py) | `extract`: scale-free features per region |
| [analyzer/geometric.py](../../services/api/app/analyzer/geometric.py) | `GeometricAnalyzer.compare` ► `GapProfile` |
| [analyzer/base.py](../../services/api/app/analyzer/base.py) | the fixed `Analyzer` protocol |
| [bodycomp.py](../../services/api/app/bodycomp.py) | RFM range, Mifflin, energy ranges with floors, opt-outs |
| [security.py](../../services/api/app/security.py), [routes/analyze.py](../../services/api/app/routes/analyze.py) | the route and its guards |

Tests: `test_split`, `test_preprocess`, `test_rows`, `test_features`, `test_analyzer`, `test_bodycomp`, `test_analyze_route` (69 passed before the route tests; the route tests are reported green by the user).

## 3. The maths worth knowing

**Body height from the feet row.** The letterbox is centred, so the body is symmetric about row 256: `feet_row_end = 256 + body_px/2`, hence `body_px = 2·(feet_row_end − 256)`. This works on a head-less browser mask (the head is gone but the feet are not), so no keypoints are needed.

**Scale-free features.** Every length is divided by the view's own `body_px`, so a person 5 cm taller or a photo taken closer gives the same numbers. The gap uses only these (ADR-024); weight never touches it.

**Girth from two views.** Front gives full width `w`, side gives depth `d`; model the cross-section as an ellipse with `a = w/2`, `b = d/2` and use Ramanujan's perimeter `π·[3(a+b) − √((3a+b)(a+3b))]`. A circle (`a=b=r`) gives `2πr`; this is tested.

**Arm thickness that ignores the A-pose angle.** An arm band of thickness `t` at angle `θ` from vertical has horizontal width `t/cos θ` and path length `1/cos θ` per row, so `thickness = (mean horizontal width) / (mean path step)`. Angle cancels.

**Spearman rho** is rank correlation: only the ordering matters, so any monotone calibration (the slope fitted later) cannot change it. That is why the raw feature can be scored without fitting anything.

**Null model.** Ridge on `[height, weight, sex]` predicting `measurement/height`. It is the honest baseline: if weight alone gets rho 0.9, a silhouette claiming 0.9 has shown nothing. Added value = rho(feature + height/weight/sex) − rho(null), with a 95% interval from resampling subjects 1,000 times.

**Gap.** `delta_region = slope_region · (feature_goal − feature_current)`, in body heights; `slope` is a train-set OLS of `measurement/height` on the feature (`gap_coef.json`).

**Confidence.** `clip((rho_val − 0.3)/0.6, 0, 1) × 0.7^(row-finder fallbacks)`. rho 0.3 is noise and 0.9 is full confidence; each time a row had to come from a prior (arms merged, crotch not found) confidence drops 30%.

**Relative Fat Mass and Mifflin.** `RFM = 64 − 20·height/waist + 12·(female)`. Waist is uncertain by the residual sd (4.05 cm), so RFM is evaluated at waist ± 1 sd and widened by ±20% (RFM's own error vs DXA: about 90% of estimates within 20%, [research doc](../../docs/research/body-composition-research.md) §2.3). `RMR = 10·kg + 6.25·cm − 5·age + 5 (male) or −161 (female)`; TDEE = RMR × activity; energy ranges are fractions of TDEE and floored at RMR, capped at a 20% deficit, 1% of body weight per week and a 10% surplus.

## 4. Decisions (ADR-029, ADR-030 in [decisions.md](../../docs/decisions.md))
- Rows from the silhouette only, one code path for BodyM and browser (BodyM has no keypoints).
- Two scales never mixed: gap = scale-free silhouette; weight only for the current person's waist.
- Gate 2 had to change (see §6); the new criterion is two-part and was fixed before the final numbers.
- Thigh and calf use width, not girth; forearm is honest-and-weak instead of a proxy; clipped side masks are excluded; `NECK_CUT_OFFSET = 0.0` from the parity check.

## 5. Results vs gate

| Gate | Result |
|---|---|
| 1. no subject in two splits | pass (test on the real `splits.json`) |
| 2. silhouette rho >= 0.6 on >= 6 of 8 regions, and adds signal beyond the null | (a) **7/8** on Test-A and Test-B; (b) **7/8** Test-A, **5/8** Test-B. *Original wording failed, 1/8, because it compared a weight-free model with a weight-aware one (ADR-029)* |
| 3. waist MAE <= 5 cm and below the null | **pass**: Test-A 2.70 cm (null 3.00); Test-B 3.39 (3.50); val 3.05 (4.32); p90 5.7 / 7.3 cm |
| 4. RFM / Mifflin tests incl. floors | pass: 11 tests over sex × activity × goal × weight × height × waist |
| 5. `/analyze` rejects RGB; gap unchanged by weight | pass (user-reported green) |
| 6. parity | 12 masks, **3 captures** (planned 5): head-cut |z| <= 1.9, 10/12 masks fully comparable, `arms_separated` 6/6 |

Per-region rho (Test-A / Test-B): shoulders .83/.80 · chest .85/.83 · waist .91/.91 · hips .83/.80 · upper arm .87/.88 · thigh .78/.70 · calf .79/.73 · **forearm .30/.28**. Null rho .80 to .93 on every region.

## 6. What went wrong (details in [mistakes.md](../../docs/mistakes.md))
1. **The gate could not pass.** A weight-free silhouette against a weight-aware null: 1/8. The honest outcome was to say so, change the criterion with the user and log it as an ADR, and tune only on val afterwards.
2. **The crotch was found in 67% of masks.** A prior in the wrong units, a window too short and a centre column that drifts with lean. A width-drop rule fixed it (98%).
3. **26% of side masks have the feet cut off.** Found by looking at a contact sheet, not by counting border pixels in the EDA.
4. **The skip-protruding-arm rule never fired** because its reference window was shorter than the arm. A realistic test caught it.
5. **A proxy that would have passed the forearm gate** (the upper-arm window, rho .78) measures arm size, not forearm; rejected.
6. **Shorts break the crotch row.** The parity check, not BodyM, showed it. The search window now treats a hem as "not found".
7. Process slips: a duplicate ADR number, read-only shell commands against the working agreement, 7 test photos of a shared subject counted in the wrong set.

Pattern: the useful moves were all "draw it and look" (contact sheet, row overlay, measurement dots, parity sheet), and "show the null next to the model".

## 7. Known limits
Forearm is not predicted. Chest, hips and thigh add almost nothing beyond height, weight and sex in the wild (Test-B). Asymmetry and chest/back split are not implemented. `upper_back` and `lats` have no ground truth (confidence 0.3). Parity from one person and 3 captures. BodyM is tight clothing in an A-pose; shorts and loose tops move the numbers. CC BY-NC data: the coefficient files are research and portfolio use only.

## 8. Interview Q&A

1. **Your first gate scored 1 of 8. What did you do?** I saw the null (height, weight, sex) already reached rho .80 to .93, so my silhouette-only model was being compared with something that knew weight. I did not quietly redefine success: I told the user, we changed it to two parts (the silhouette is good enough on its own, and it adds signal beyond the null), logged it as ADR-029, stopped tuning on the test sets and tuned on val only.
2. **Why no neural network here?** There is no label for "muscle development", only tape measurements, and a geometric baseline gives something the Siamese CNN must beat (ADR-006). It also gives a fallback and the back-view features BodyM cannot teach.
3. **How can one code path serve BodyM and browser masks?** Both end in the same 384x512 head-less frame, and rows come from run lengths in that frame. Body height comes from the feet row because the letterbox is centred, so no keypoints are needed.
4. **Why split by subject?** Subjects appear 1 to 27 times; a photo-level split would put the same person in train and test and inflate every number. I also dropped the one subject shared by both test sets.
5. **How do you know the gap does not depend on weight?** It is built from scale-free silhouette features only, and a test posts the same scans with two different weights and asserts the gap is identical while the energy numbers differ.
