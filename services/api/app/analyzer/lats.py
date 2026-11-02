"""`lats` = "back width (estimate)": the ANSUR ridge on B's chest, shoulders and waist (ADR-036,
ADR-039). It is not an Analyzer output: it needs the two sexes, which are request fields, so the
route applies it to the GapProfile the analyzer returned, from the per-person ratios already in it.

Both sexes must be set and equal. Then the ridge's sex term is the same for goal and current and
cancels in the gap. Anything else gives `measured: false` with a reason, never a guess.
"""

import json
from pathlib import Path

import numpy as np

from ml.src.calibrate import reliability
from physique_schemas.models import GapProfile, RegionGap, Sex

FIT = json.loads((Path(__file__).parent / "lats_coef.json").read_text())["additive_sex"]
INPUTS = ("chest", "shoulders", "waist")
NEEDS_MODEL = "Back width needs the learned model, which is not loaded."
SET_YOUR_SEX = "Set your sex to estimate back width."
SET_GOAL_SEX = "Say whether the goal physique is male or female to estimate back width."
SEX_DIFFERS = "Back width is only compared between people of the same sex."
DIFFERENT_SEX_WARNING = (
    "The goal physique is the other sex from yours, so absolute size gaps may not be reachable."
)


def _not_measured(reason: str) -> RegionGap:
    return RegionGap(delta=0.0, confidence=0.0, measured=False, reason=reason)


def _predict(ratios: dict[str, float], is_male: float) -> float:
    x = ratios | {"is_male": is_male}
    return FIT["intercept"] + sum(FIT["coef"][k] * x[k] for k in FIT["inputs"])


def lats_gap(gap: GapProfile, user_sex: Sex, goal_sex: Sex) -> RegionGap:
    parts = [gap.regions[r] for r in INPUTS]
    if any(p.goal_ratio is None for p in parts):  # geometric fallback, or an input not measured
        return _not_measured(NEEDS_MODEL)
    if user_sex == "not_specified":
        return _not_measured(SET_YOUR_SEX)
    if goal_sex == "not_specified":
        return _not_measured(SET_GOAL_SEX)
    if user_sex != goal_sex:
        return _not_measured(SEX_DIFFERS)

    is_male = float(user_sex == "male")
    goal = _predict({r: p.goal_ratio for r, p in zip(INPUTS, parts, strict=True)}, is_male)
    current = _predict({r: p.current_ratio for r, p in zip(INPUTS, parts, strict=True)}, is_male)
    rel = FIT["reliability"]
    sigma_goal = rel["resid_std_front" if gap.goal_front_only else "resid_std_full"]
    return RegionGap(
        delta=float(np.clip(goal - current, -1, 1)),
        confidence=float(reliability(sigma_goal, rel["resid_std_full"], rel["sigma_prior"])),
        goal_ratio=goal,
        current_ratio=current,
    )


def with_lats(gap: GapProfile, user_sex: Sex, goal_sex: Sex) -> tuple[GapProfile, list[str]]:
    """The gap with `lats` filled in, and the report-level warnings that go with the sexes."""
    regions = gap.regions | {"lats": lats_gap(gap, user_sex, goal_sex)}
    both_set = "not_specified" not in (user_sex, goal_sex)
    warnings = [DIFFERENT_SEX_WARNING] if both_set and user_sex != goal_sex else []
    return gap.model_copy(update={"regions": regions}), warnings
