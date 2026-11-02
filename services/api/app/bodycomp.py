"""Body composition numbers (ADR-012 RFM, ADR-015 Mifflin) with the safety floors from
docs/privacy-and-safety.md. Body fat is always a range. The `BodyComp` schema validates the
energy floors a second time.
"""

import json
from pathlib import Path

import numpy as np

from app.analyzer.features import extract
from app.analyzer.masks import decode_scan, side_view
from physique_schemas.models import BodyComp, BodyFatRange, EnergyTarget, Scan, UserProfile

WAIST = json.loads((Path(__file__).parent / "analyzer" / "waist_coef.json").read_text())
TAPE_SD_CM = 1.0
# RFM vs DXA: ~89-92% of estimates within 20% (docs/research/body-composition-research.md, 2.3).
RFM_REL_ERR = 0.20
LEAN_RFM = {"male": 14.0, "female": 22.0}  # RFM is weakest in lean people (same source)
OLD_AGE = 60  # ...and declines with age
BF_FLOOR = {"male": 8.0, "female": 18.0}  # privacy-and-safety.md: no deficit below these
BF_ATHLETE_UPPER = {"male": 13.0, "female": 20.0}  # ADR-040: top of the ACE athlete bands
# ADR-041, fixed before M2 step 3: ANSUR subjects at or below the high-end waist/stature needed
# for an unqualified "reachable" claim. Do not tune after seeing step 3 output.
SUPPORT_MIN = 30
ZONE1_TEXT = {  # ADR-041 label_case -> exact wording; the agent must use it, never paraphrase
    1: "Reachable on frame alone (estimate)",
    2: "Reachable on frame alone (estimate), thin reference data",
    3: "Outside the range our data covers; cannot verify",
}
ZONE2_TEXT = {  # ADR-042: same rule, applied at the BF_FLOOR end
    1: "Reachable only near the body-fat floor (estimate)",
    2: "Reachable only near the body-fat floor (estimate), thin reference data",
    3: ZONE1_TEXT[3],
}
MIN_BMI = 18.5
# Standard activity multipliers on RMR; not validated here.
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
ENERGY = {  # (low, high) as a fraction of TDEE
    "cut": (0.80, 0.90),
    "recomp": (0.90, 1.00),
    "maintain": (0.95, 1.05),
    "bulk": (1.05, 1.10),
}
MAX_LOSS_PER_WEEK = 0.01  # of body weight (privacy-and-safety.md)
KCAL_PER_KG = 7700
# Widely cited range for lifters (g/kg/day); the P5 corpus replaces this with a cited value.
PROTEIN_G_PER_KG = (1.6, 2.2)

NOTE_ED = (
    "Calorie numbers are left out of this plan. If food or weight feels stressful, "
    "a doctor or counsellor can help."
)
NOTE_LOW = (
    "Your estimated body fat or BMI is already low, so no calorie deficit is suggested. "
    "The plan focuses on strength at maintenance; a health professional can advise on weight."
)
NOTE_NO_SEX = (
    "Body fat and calorie numbers need your sex, because the formulas differ by sex. "
    "Set it in your profile to see them; your gaps and plan are unchanged."
)


def silhouette_waist_g(scan: Scan) -> float:
    """Scale-free silhouette waist girth of one person's scan (NaN when a view is unusable)."""
    masks = decode_scan(scan)
    return extract(masks["front"], side_view(masks))["waist_g"]


def rfm(height_cm: float, waist_cm: float, sex: str) -> float:
    """Relative Fat Mass: 64 - 20 * height / waist + 12 * (female)."""
    return 64 - 20 * height_cm / waist_cm + (12 if sex == "female" else 0)


def label_case(support: int, extrapolated: bool, texts: dict) -> dict:
    """ADR-041/042 wording case. The API image has no ANSUR files, so callers pass the `n_below`
    and `extrapolated` stored by ml/src/ceiling_extrapolation.py (`n_below` is strictly below;
    the rule's `<=` differs only for a subject exactly on the inverted value)."""
    case = 3 if extrapolated else 1 if support >= SUPPORT_MIN else 2
    return {
        "support": support, "extrapolated": extrapolated,
        "label_case": case, "text": texts[case],
    }  # fmt: skip


def zone1_case(support: int, extrapolated: bool) -> dict:
    """Zone 1, from the stored values at BF_ATHLETE_UPPER (ADR-041)."""
    return label_case(support, extrapolated, ZONE1_TEXT)


def zone2_case(support: int, extrapolated: bool) -> dict:
    """Zone 2, from the stored values at BF_FLOOR (ADR-042)."""
    return label_case(support, extrapolated, ZONE2_TEXT)


def rmr(p: UserProfile) -> float:
    """Mifflin-St Jeor resting metabolic rate, kcal/day."""
    sex_term = {"male": 5, "female": -161}[p.sex]  # KeyError on not_specified: never guessed
    return 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age + sex_term


def waist_estimate(p: UserProfile, waist_g: float | None) -> tuple[float, float, str]:
    """(waist cm, sd cm, source). A tape measurement overrides the silhouette. Only the
    current person's scan reaches this model; the goal never does (ADR-024)."""
    if p.tape_waist_cm is not None:
        return p.tape_waist_cm, TAPE_SD_CM, "tape"
    if waist_g is None or np.isnan(waist_g):
        raise ValueError("no waist: the silhouette waist is unavailable and no tape value given")
    x = {
        "waist_cm_proxy": waist_g * p.height_cm,
        "height_cm": p.height_cm,
        "weight_kg": p.weight_kg,
        "male": float(p.sex == "male"),
    }
    cm = WAIST["intercept"] + sum(
        c * x[f] for f, c in zip(WAIST["features"], WAIST["coef"], strict=True)
    )
    return float(cm), WAIST["sd_cm"], "silhouette"


def body_fat(p: UserProfile, waist_cm: float, sd_cm: float, source: str) -> BodyFatRange:
    """RFM at waist -/+ one sd, widened by RFM's own relative error against DXA."""
    low_pt = rfm(p.height_cm, waist_cm - sd_cm, p.sex)
    high_pt = rfm(p.height_cm, waist_cm + sd_cm, p.sex)
    low = np.clip(low_pt - RFM_REL_ERR * abs(low_pt), 3, 59)
    high = np.clip(high_pt + RFM_REL_ERR * abs(high_pt), low + 1, 60)
    mid = rfm(p.height_cm, waist_cm, p.sex)
    weak = mid < LEAN_RFM[p.sex] or p.age >= OLD_AGE
    return BodyFatRange(
        low=round(float(low), 1),
        high=round(float(high), 1),
        method=f"RFM from {'tape' if source == 'tape' else 'silhouette-estimated'} waist",
        confidence="low" if source == "silhouette" or weak else "medium",
    )


def energy_targets(p: UserProfile, rmr_kcal: float, tdee: float, deficit_ok: bool):
    goals = ["maintain"] + ([p.goal] if deficit_ok or p.goal == "bulk" else [])
    max_daily_deficit = MAX_LOSS_PER_WEEK * p.weight_kg * KCAL_PER_KG / 7
    out = []
    for goal in goals:
        lo_f, hi_f = ENERGY[goal]
        low = max(lo_f * tdee, rmr_kcal, tdee - max_daily_deficit)
        high = max(hi_f * tdee, low)
        out.append(
            EnergyTarget(
                goal=goal,
                kcal_low=round(low),
                kcal_high=round(high),
                protein_g_low=round(PROTEIN_G_PER_KG[0] * p.weight_kg),
                protein_g_high=round(PROTEIN_G_PER_KG[1] * p.weight_kg),
            )
        )
    return out


def body_comp(p: UserProfile, waist_g: float | None = None) -> BodyComp:
    """`waist_g` = the current scan's scale-free silhouette waist girth (features.waist_g)."""
    if p.sex == "not_specified":  # RFM, the waist model and Mifflin all have a sex term (ADR-037)
        notes = ([NOTE_ED] if p.ed_history else []) + [NOTE_NO_SEX]
        return BodyComp(body_fat=None, note=" ".join(notes))
    waist, sd, source = waist_estimate(p, waist_g)
    bf = body_fat(p, waist, sd, source)
    if p.ed_history:
        return BodyComp(body_fat=bf, note=NOTE_ED)

    r = rmr(p)
    tdee = r * ACTIVITY[p.activity]
    bmi = p.weight_kg / (p.height_cm / 100) ** 2
    lean = (bf.low + bf.high) / 2 < BF_FLOOR[p.sex] or bmi < MIN_BMI
    deficit_ok = not (lean and p.goal in ("cut", "recomp"))
    return BodyComp(
        body_fat=bf,
        rmr_kcal=round(r),
        tdee_kcal=round(tdee),
        energy=energy_targets(p, round(r), round(tdee), deficit_ok),
        note="" if deficit_ok else NOTE_LOW,
    )
