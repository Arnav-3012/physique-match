# P3 decisions and additions — what we decided, why, and which concepts sit behind it

Source of truth for each row: `docs/decisions.md` (ADR numbers) and `docs/logs/dev-log.md` (numbers). Concept numbers refer to [concepts.md](concepts.md).

---

## Part 1 — Core model decisions

| # | Decision | Why | Concepts |
|---|---|---|---|
| ADR-002 / 032 | Target = **proportion gap** `m_A/h_A − m_B/h_B` | A goal photo has no height; dividing each person by their own height means the net never sees height, so no shortcut. UI shows cm as `Δ × the user's height` | 4 |
| ADR-004 | ResNet-18, ImageNet, **ladder** probe → last2 → full | Silhouettes are low-information; small model, fast on CPU; stop at the first rung that passes | 9, 10 |
| ADR-006 | Siamese must beat "regress each person, then subtract" (baseline B), else ship the simpler model and say so | Pre-commits to an honest negative result | 3, 17 |
| ADR-025 | **Side-view dropout** (p = .5 on the goal) | Most goal photos are front-only; one model handles both inputs; Test-B cost ≤ .0017 MAE | 12 |
| ADR-033 | **Gate protocol before any Test-B look**: 95% subject-bootstrap CI > 0 on ≥ 6/8 vs each baseline, none worse; fixed pair list with committed hash; ledger refuses a second look | Stops tuning on the test set; subjects are the independent unit | 15, 16, 17 |
| ADR-033 amend. | Baseline B trained with the same side blanking; 3 seeds for the final candidate, spread per region; no ensemble | Like-for-like ablation; an ensemble breaks 50 MB / 75 ms | 9 |
| ADR-034 | Stopping rules written **before** the `full` runs: pair head needs CI > 0 on ≥ 6/8 vs B on val, else negative result | Avoids moving the goalposts after seeing results | 17 |
| ADR-035 | **Ship baseline B** (`SiameseSubtract`), pair head kept as the ablation; single-person ONNX run twice; latency rule single-encode p50 ≤ 75 ms and two-encode p95 ≤ 150 ms; opset 17, legacy exporter | Pair head failed 1/8 at `full`; B is exactly antisymmetric, smaller, simpler | 3, 7, 13 |
| ADR-035 amend. | Goal-prediction cache removed; current scan = front + one side (right mirrored); `RegionGap.measured`; `REQUIRE_MODEL` | Privacy (nothing cached), honest "not measured", no silent fallback in production | 13 |
| Reliability | Per-region **calibrated val residual std** instead of MC dropout; `reliability = clip(1 − √(s_goal²+s_full²)/σ_prior, 0, 1)` | Cheap, deterministic, separate for full and front-only | 14 |

**Result:** Test-B 8/8 vs A on both slices, none worse; 45.2 MB; 15.6 ms/encode (see [phase-overview.md](phase-overview.md)).

---

## Part 2 — Back regions: how `upper_back`, `lats` and sex are handled

### The problem
BodyM has silhouettes (front, side) and 14 tape measurements, but **no back view and no back measurement**. The original P2 geometric analyzer produced `upper_back` and `lats` from a back silhouette, at a fixed confidence 0.3, with nothing to validate it (ADR-024, ADR-030). Looking good in the UI with no ground truth is the thing we wanted to stop.

### Decision tree (ADR-036, 037, 039)
```
upper_back (size)  -> measured:false ALWAYS, fixed reason (ADR-039)
lats               -> "back width (estimate)", via ANSUR II ridge, unranked, only if both sexes set and equal
back definition    -> P4, from the user's own back view only
goal back photo    -> P7, optional, definition only
lat flare          -> WAIT, gated by the ADR-038 unique-share test in P4
shoulder height diff -> WAIT, gated by a test-retest noise floor in P8
```

### `upper_back`
- Stays `measured: false`, always, with the fixed reason: *"Not measurable from a front photo; a back silhouette adds no size information."* (`UPPER_BACK_REASON` in the schema package, used by both analyzers).
- Why: a back silhouette gives the same outline widths as a front one; width there is shoulder/lat width, which is not a separate quantity we can validate. No label exists, so no number is shown.

### `lats` → "back width (estimate)"
- **Idea.** Use a dataset that *does* have both: ANSUR II (US Army, 4,082 men, 1,986 women) has shoulder breadth, chest, waist and back-width measurements. Fit a ridge: back width / stature ≈ chest + shoulders + waist (ratios), then apply it to the chest, shoulder and waist **ratios already predicted by model B**.
- **Why a ridge on B's outputs and not a new CNN?** No back label in BodyM, so a CNN would have nothing to learn from. This gives an estimate with a measurable R².
- **Noise handling.** Training copies get noise at B's own error level (`sigma_full`), so the coefficients are fit to the inputs the API will actually see, not to clean tape measurements (errors-in-variables attenuation).
- **Shoulder column.** Check run before the fit: BodyM `shoulder_breadth` is closer to ANSUR **biacromial** (the default) than bideltoid in both sexes. Known mismatch: BodyM's shoulder breadth is ~10% below ANSUR's, which attenuates the shoulder coefficient (.403 noise-fit vs .488 clean-fit, −17%). Not corrected (the rule was fixed in advance); it is in the model card.
- **Gate and verdict** (fixed before the run). dR² of the full model over chest-only: .048 male / .069 female, minimum .048 ≥ .02, bootstrap CI [.041, .055] → **PASS**. But the noisy-input R²_min is **.408 < .5** → verdict **`unranked`**: shown as "back width (estimate)", **never in the priority ranking**, and the README says why. Even with perfect inputs the female R² is .45, so chest, shoulders and waist explain less than half of back width; this is not B's noise.
- **Coefficients** (raw): chest .1637, shoulders .4026, waist .0472, is_male .0073 (about 1.3 cm at 175 cm vs a typical gap spread of 5 cm). MAE .012 of height (≈ 2 cm). Reliability ≈ .27 for goal and current both at full noise.
- **Sex rule (Amendments 3, 4).** The pooled-ridge route was dropped. `lats` is computed only if `user_sex` and `goal_sex` are both set **and equal**; otherwise `measured: false` with a reason on the region, and a report-level warning (`AnalyzeResponse.warnings`) for a different-sex goal. Order of checks in `app/analyzer/lats.py`: no per-person ratios (geometric fallback) → user sex unset → goal sex unset → sexes differ. The sex term cancels in a same-sex gap, which is why equal sexes are enough.
- **Where it runs.** `with_lats(gap, user_sex, goal_sex)` is applied in the route **after** `Analyzer.compare`, because the analyzer interface `compare(goal, current) -> GapProfile` is fixed and has no sex input.
- **Labels.** API key stays `lats`; the UI, README, report and agent say "back width (estimate)" with a one-line definition. A `measured: false` lats is "not measured", never "on target".

### Sex as a field (Amendments 3-4)
- `Sex` gained `not_specified`; `UserProfile.sex` defaults to it; new `GoalProfile.sex`. The UI field is never defaulted (3 choices, none preselected), reads "needed for body-fat %, calories, back width and the V-taper ceiling".
- Unset user sex → no body-fat %, no calories (`BodyComp.body_fat` optional, "set your sex" note), no `lats`, no V-taper ceiling. **Gaps and plan are unaffected, and cut/recomp/bulk stay user-chosen** (checked: body fat only feeds the deficit opt-out for a lean user, a safety rule).
- `rmr()` raises on `not_specified` instead of silently using the female term (a bug found during the audit; `BF_FLOOR[p.sex]` / `LEAN_RFM[p.sex]` KeyErrors fixed likewise).
- Audit (grep, user-run): no ranked region used a sex term; sex appeared only in body composition, null-ridge references, splits, tests and the schema.

### What comes later for the back (not P3 work)
| Item | Phase | Why not now |
|---|---|---|
| Back **definition** (muscle visibility) from the user's own back view | after P4 | needs the definition model; schema must allow per-head N/A (P4 prerequisite) |
| Optional goal back photo (definition only; size stays unmeasured) | P7 | only after the back definition works |
| Lat flare | P4 decision point | only built if the ADR-038 unique-share test passes on mesh labels |
| Shoulder height difference | P8 | needs a test-retest noise floor first; never labelled "posture" |
| Planner **baseline-volume invariant**: every region gets at least its cited baseline weekly volume, whether measured, `measured: false`, N/A or low reliability; N/A is never "no deficit" | P5 corpus + P6 planner | no baseline may be invented; the contract is recorded in ADR-039 |

---

## Part 3 — Additions on top of the gap

### Frame V-taper (ADR-038)
- **What:** index = shoulders ÷ waist from B's per-person outputs (about .40, no 1.618 score); `delta = goal − current`; its own field `GapProfile.v_taper`, label "frame V-taper (estimate)", not an 11th region.
- **Reliability** from val residuals only (no Test-B look), stored as a `v_taper` block in `gap_regressor.json` without re-exporting the ONNX; block carries the script commit hash, checkpoint `model_hash`, ONNX SHA-256 and a version. Result .64 full and .64 front-only (side view adds nothing to the index; the shoulder and waist errors partly cancel in the ratio).
- **Decision rule fixed first:** `unique_share = 1 − R²(index gap ~ waist gap)` on same-sex val pairs with ground-truth ratios, min over sexes, threshold .2. Result: .148 female, .171 male → **DROP**: the index gap is mostly the waist gap, which the planner already sees. It stays display-only and is not a P6 planner input. The user expected this outcome; it is recorded as legitimate.
- Concepts: ratio-of-predictions, bootstrap/residual calibration, pre-registration, partial R² (unique variance).

### Reachable ceiling (ADR-040, 041, 042)
- **Question:** is the goal's shoulder-to-waist index reachable on this user's frame? Bone width can't change; waist can.
- **Ceiling** = user's B-predicted shoulder ratio ÷ `waist_at_bf` ratio, as a range between a body-fat floor (`BF_FLOOR`: male 8, female 18) and an athlete-upper value (`BF_ATHLETE_UPPER`: male 13, female 20). `waist_at_bf = 20 · height / (64 + 12·female − BF)` is the inverse of RFM, with round-trip tests.
- **Extrapolation check first:** the inverted waist/stature versus the lowest ANSUR value. Male floor extrapolated (8% → .357 < .360), male 13% not (4 subjects below); female both ends extrapolated (9.7% and 6.5% under the minimum). So the whole female ceiling sits outside the data we can verify.
- **Three zones, one-sided, no free threshold:** ≤ athlete-upper ceiling = reachable on frame alone; between = near the floor only, best case, not sustained; above = needs more than frame. Wording fixed before any output (ADR-041/042, `SUPPORT_MIN = 30`); with the stored data women never get a "reachable" claim, and zone 3 carries a "cannot verify" caveat when the floor end is extrapolated.
- Needs the user's sex; `measured: false` with "set your sex" otherwise. This pulled the P7 realism-flag item forward in a one-sided form.

### Process additions
- **Commit the script before running it** so the hash stored in the result JSON is real (V-taper block, extrapolation JSON, unique-share JSON). A first run on a dirty tree recorded a commit that did not contain the script; it was discarded and redone.
- **Pre-registration pattern** used repeatedly: write the rule, threshold and wording before seeing the number (ADR-033, 034, 038, 040-042).
- **Schema regeneration discipline:** `npm run types` does not rewrite `schema.json`; the full order is `export_schema` → `npm run types` → `npm run build`. Batched so types were regenerated as few times as possible (mistake logged).
- **Honest-limits habit:** Test-B MAE ran 31-55% above val; reliability is optimistic; a null ridge that knows weight ties B on several regions; the torso noise floor between two captures is 5-9 cm (one pair). All are in the model card.

---

### Hosting change (ADR-043, 2026-11-02)
HF Hub is dropped. Vercel (web), Cloud Run (API) and the Free Trial stay; the model file is delivered from Google Cloud (baked into the image, or a private Cloud Storage bucket, chosen in the P7 spike). Model cards stay in the repo. Why: one provider and one billing/access surface to guard, no extra account or token. All Google Cloud work is still deferred to deploy time (ADR-031).

---

## Part 4 — Where each thing lives
| Thing | File |
|---|---|
| Model, ladder, freeze | `ml/src/model.py` |
| Training loop | `ml/src/train.py` |
| Augmentation | `ml/src/augment.py` |
| Pair universe, manifest | `ml/src/data/pairs.py`, `evals/testb_pairs.json` |
| Gate, bootstrap, ledger | `ml/src/evaluate.py`, `evals/testb_ledger.json` |
| Export, latency, parity | `ml/src/export_onnx.py` |
| Reliability | `ml/src/calibrate.py`, `gap_regressor.json` |
| Runtime analyzer | `services/api/app/analyzer/siamese_subtract.py` |
| Lats | `ml/src/fit_lats.py`, `app/analyzer/lats.py`, `lats_coef.json` |
| V-taper | `ml/src/v_taper_block.py`, `unique_share.py`, `evals/unique_share.json` |
| Ceiling | `app/analyzer/ceiling.py`, `bodycomp.py`, `ceiling_extrapolation.json` |
| Model card, privacy | `ml/MODEL_CARD.md`, `docs/privacy-and-safety.md` |
