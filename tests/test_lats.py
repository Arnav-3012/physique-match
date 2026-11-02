import pytest

from app.analyzer.lats import (
    DIFFERENT_SEX_WARNING,
    FIT,
    NEEDS_MODEL,
    SET_GOAL_SEX,
    SET_YOUR_SEX,
    SEX_DIFFERS,
    with_lats,
)
from ml.src.calibrate import reliability
from physique_schemas.models import REGIONS, GapProfile, RegionGap

RATIOS = {"chest": (0.60, 0.55), "shoulders": (0.24, 0.22), "waist": (0.50, 0.52)}  # goal, current


def profile(front_only=False, ratios=RATIOS) -> GapProfile:
    blank = RegionGap(delta=0.0, confidence=0.0, measured=False)
    regions = dict.fromkeys(REGIONS, blank)
    for region, (g, c) in ratios.items():
        regions[region] = RegionGap(delta=g - c, confidence=0.5, goal_ratio=g, current_ratio=c)
    return GapProfile(analyzer="siamese_subtract", regions=regions, goal_front_only=front_only)


def test_same_sex_gives_the_ridge_gap_and_no_warning():
    out, warnings = with_lats(profile(), "male", "male")
    lats = out.regions["lats"]
    expected = sum(FIT["coef"][r] * (g - c) for r, (g, c) in RATIOS.items())
    assert lats.measured and lats.reason is None and warnings == []
    assert lats.delta == pytest.approx(expected)
    assert lats.delta == pytest.approx(lats.goal_ratio - lats.current_ratio)


def test_the_sex_term_cancels_so_both_sexes_give_the_same_gap_and_reliability():
    male = with_lats(profile(), "male", "male")[0].regions["lats"]
    female = with_lats(profile(), "female", "female")[0].regions["lats"]
    assert male.delta == pytest.approx(female.delta) and male.confidence == female.confidence
    assert male.goal_ratio - female.goal_ratio == pytest.approx(FIT["coef"]["is_male"])


def test_reliability_is_the_stored_one_and_lower_for_a_front_only_goal():
    r = FIT["reliability"]
    full = with_lats(profile(), "male", "male")[0].regions["lats"].confidence
    front = with_lats(profile(front_only=True), "male", "male")[0].regions["lats"].confidence
    assert full == pytest.approx(
        reliability(r["resid_std_full"], r["resid_std_full"], r["sigma_prior"])
    )
    assert 0 < front < full


@pytest.mark.parametrize(
    ("user", "goal", "reason", "n_warnings"),
    [
        ("not_specified", "male", SET_YOUR_SEX, 0),
        ("male", "not_specified", SET_GOAL_SEX, 0),
        ("not_specified", "not_specified", SET_YOUR_SEX, 0),
        ("male", "female", SEX_DIFFERS, 1),
        ("female", "male", SEX_DIFFERS, 1),
    ],
)
def test_lats_is_blocked_with_a_reason_and_other_regions_are_untouched(
    user, goal, reason, n_warnings
):
    gap = profile()
    out, warnings = with_lats(gap, user, goal)
    lats = out.regions["lats"]
    assert not lats.measured and lats.reason == reason and lats.delta == 0 == lats.confidence
    assert len(warnings) == n_warnings and set(warnings) <= {DIFFERENT_SEX_WARNING}
    assert {r: g for r, g in out.regions.items() if r != "lats"} == {
        r: g for r, g in gap.regions.items() if r != "lats"
    }


def test_without_per_person_ratios_the_fallback_says_the_model_is_needed():
    out, _ = with_lats(profile(ratios={}), "male", "male")  # the geometric analyzer's shape
    assert out.regions["lats"].reason == NEEDS_MODEL


def test_a_measured_region_has_no_reason():
    with pytest.raises(ValueError, match="reason"):
        RegionGap(delta=0.1, confidence=0.5, reason="because")
