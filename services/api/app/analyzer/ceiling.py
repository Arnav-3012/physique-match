"""Reachable frame ceiling (ADR-040): the V-taper index the user's own shoulders allow when the
waist is as lean as RFM says `BF_ATHLETE_UPPER` and `BF_FLOOR` are. Not an Analyzer output, since
it needs the user's sex: the route applies it to the GapProfile, from the ratios already in it.

ceiling = shoulders_ratio / (waist_at_BF / height), and height cancels. The zone wording comes
from the step 0 result stored next to this file (the image has no ANSUR files).
"""

import json
from pathlib import Path

from app.bodycomp import BF_ATHLETE_UPPER, BF_FLOOR, waist_at_bf, zone1_case, zone2_case
from physique_schemas.models import Ceiling, GapProfile, Sex, VTaper

STORED = json.loads((Path(__file__).parent / "ceiling_extrapolation.json").read_text())["sexes"]
SET_YOUR_SEX = "Set your sex to see whether the goal is reachable on your frame."
NEEDS_SHOULDERS = "Your shoulder width was not measured, so no frame ceiling is shown."


def _not_measured(reason: str) -> Ceiling:
    return Ceiling(measured=False, reason=reason)


def _label(case_fn, sex: str, end: str) -> dict:
    stored = STORED[sex]["ends"][end]
    return case_fn(stored["n_below"], stored["extrapolated"])


def ceiling(shoulders_ratio: float | None, user_sex: Sex) -> Ceiling:
    if user_sex == "not_specified":
        return _not_measured(SET_YOUR_SEX)
    if shoulders_ratio is None:
        return _not_measured(NEEDS_SHOULDERS)
    at = {
        end: shoulders_ratio / waist_at_bf(bf[user_sex], 1.0, user_sex)
        for end, bf in (("athlete_upper", BF_ATHLETE_UPPER), ("floor", BF_FLOOR))
    }
    return Ceiling(
        at_athlete_upper=at["athlete_upper"],
        at_floor=at["floor"],
        reachable=_label(zone1_case, user_sex, "athlete_upper"),
        near_floor=_label(zone2_case, user_sex, "floor"),
    )


def with_ceiling(gap: GapProfile, user_sex: Sex) -> GapProfile:
    """The gap with `v_taper.ceiling` filled in; unchanged when there is no V-taper (geometric)."""
    if gap.v_taper is None:
        return gap
    shoulders = gap.regions["shoulders"].current_ratio if gap.v_taper.measured else None
    v_taper: VTaper = gap.v_taper.model_copy(update={"ceiling": ceiling(shoulders, user_sex)})
    return gap.model_copy(update={"v_taper": v_taper})
