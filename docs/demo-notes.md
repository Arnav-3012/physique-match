# Demo notes (presenter talking points)

Not shown in the app. The app has no per-region definition and no disclaimer about it; the limitation is stated here, in the README, in ADR-053 Amendment 6 and in `ml/MODEL_CARD.md`.

## If asked: "does it measure definition per muscle group?"
No, not yet. Say so plainly:
- The model is trained on synthetic renders only and reports **one overall "leanness & muscularity (estimate)" band** and **one adiposity band** (a waist-girth proxy).
- The model has regional heads (abs, chest, shoulders, arms, back, legs). In held-out evaluation none of them added information beyond the overall score (best gain +0.019 against a pre-registered bar of 0.05), so they are in the model but **not shown**.
- The reason: the synthetic regional labels come from one shared latent, so they are nearly redundant. A high regional score therefore does not show that the model reads individual regions.
- Regional **size** gaps are different: those come from silhouettes (measured, with a reliability tier).
- Planned: independent regional labels, cue fixes and a re-render, real bare-body photos with regional ratings (P8), per-region display once a region passes the bar.

## If asked: "why two models?"
Definition comes from the 8-head model and adiposity from the 2-head model, each from the model that passed its own pre-registered checks. The choice was made after seeing the test results (mild selection; both models and all thresholds were frozen first). Adiposity is about equal between the two; the pre-registered rule says the 8-head model does not beat the 2-head one. All three int8 attempts failed, for a cause not yet established, so both run in fp32 (7.46 MB), and the owner raised the size cap from 5 MB to 8 MB for that.

## If asked: "does it work on real photos?"
Not validated. Held-out scores are on synthetic bodies, where the label is a function of the generator's sliders. The badge "experimental, trained on synthetic data" is there for this reason. Real-photo checks are post-demo (ADR-050, P8).

## Never say
- That the 8-head model passed the adiposity rule (it did not).
- That the model scores or ranks individual regions' definition.
- That the scores are body-fat percentages or medical assessments.
