import pytest

from app.analyzer.ceiling import NEEDS_SHOULDERS, SET_YOUR_SEX, with_ceiling
from app.bodycomp import BF_ATHLETE_UPPER, BF_FLOOR, ZONE3_TEXT
from physique_schemas.models import REGIONS, Ceiling, GapProfile, RegionGap, VTaper


def gap(v_taper, shoulders=0.2131):
    regions = {r: RegionGap(delta=0.0, confidence=0.5) for r in REGIONS}
    regions["shoulders"] = RegionGap(
        delta=0.0, confidence=0.5, goal_ratio=shoulders, current_ratio=shoulders
    )
    return GapProfile(analyzer="siamese_subtract", regions=regions, v_taper=v_taper)


MEASURED = VTaper(goal_index=0.45, current_index=0.40, delta=0.05)


@pytest.mark.parametrize(
    ("sex", "shoulders", "cases"),
    [("male", 0.2131, (2, 3)), ("female", 0.2029, (3, 3))],  # zone 1 case, zone 2 case
)
def test_ceiling_range_and_labels(sex, shoulders, cases):
    c = with_ceiling(gap(MEASURED, shoulders), sex).v_taper.ceiling
    s = 12 if sex == "female" else 0
    # ADR-040: ceiling = shoulders * (64 + 12 s - BF) / 20, height cancels
    assert c.at_athlete_upper == pytest.approx(shoulders * (64 + s - BF_ATHLETE_UPPER[sex]) / 20)
    assert c.at_floor == pytest.approx(shoulders * (64 + s - BF_FLOOR[sex]) / 20)
    assert c.at_floor > c.at_athlete_upper
    assert (c.reachable.label_case, c.near_floor.label_case) == cases
    assert c.measured


def ceiling_for(goal_index, sex="male", shoulders=0.2131):
    v_taper = VTaper(goal_index=goal_index, current_index=0.40, delta=goal_index - 0.40)
    return with_ceiling(gap(v_taper, shoulders), sex).v_taper.ceiling


def test_zones_and_their_boundaries_male():
    c = ceiling_for(0.50)
    low, high = c.at_athlete_upper, c.at_floor
    assert ceiling_for(low).zone == 1  # <= the athlete-upper ceiling
    assert ceiling_for(low + 1e-6).zone == 2
    assert ceiling_for(high).zone == 2  # <= the floor ceiling
    assert ceiling_for(high + 1e-6).zone == 3


def test_flag_is_one_sided():
    assert ceiling_for(0.10).zone == 1  # a small goal is never "too small"


def test_zone_text_is_the_exact_wording_of_that_zone():
    c = ceiling_for(0.50)  # men: zone 1 is case 2 (thin data)
    assert (c.zone, c.zone_text) == (1, c.reachable.text)
    assert "thin reference data" in c.zone_text
    c = ceiling_for(0.57)  # men: zone 2, extrapolated at the floor -> case 3
    assert (c.zone, c.zone_text) == (2, c.near_floor.text)
    assert "reachable" not in c.zone_text.lower()
    c = ceiling_for(0.65)
    assert (c.zone, c.zone_text) == (3, ZONE3_TEXT[True])


def test_female_zones_never_claim_reachable():
    for goal in (0.50, 0.58, 0.65):  # ceilings .568 / .588
        c = ceiling_for(goal, "female", 0.2029)
        assert c.zone == (1 if goal < 0.568 else 2 if goal < 0.588 else 3)
        assert "reachable" not in c.zone_text.lower()


def test_zone_3_caveat_only_when_the_floor_end_is_extrapolated():
    assert "cannot be verified" in ZONE3_TEXT[True]
    assert "cannot be verified" not in ZONE3_TEXT[False]


def test_worked_values_from_adr_040():
    male = with_ceiling(gap(MEASURED, 0.2131), "male").v_taper.ceiling
    assert (male.at_floor, male.at_athlete_upper) == pytest.approx((0.597, 0.543), abs=1e-3)
    female = with_ceiling(gap(MEASURED, 0.2029), "female").v_taper.ceiling
    assert (female.at_floor, female.at_athlete_upper) == pytest.approx((0.588, 0.568), abs=1e-3)


def test_no_sex_is_not_measured():
    c = with_ceiling(gap(MEASURED), "not_specified").v_taper.ceiling
    assert not c.measured and c.reason == SET_YOUR_SEX
    assert c.at_floor is None and c.reachable is None


def test_unmeasured_v_taper_gives_an_unmeasured_ceiling():
    c = with_ceiling(gap(VTaper(measured=False, reason="x")), "male").v_taper.ceiling
    assert not c.measured and c.reason == NEEDS_SHOULDERS


def test_geometric_fallback_has_no_v_taper_and_is_left_alone():
    g = gap(None)
    assert with_ceiling(g, "male") == g


def test_ceiling_validator_rejects_half_filled_parts():
    with pytest.raises(ValueError, match="both ends"):
        Ceiling(at_floor=0.6)
    with pytest.raises(ValueError, match="no numbers"):
        Ceiling(measured=False, reason="x", at_floor=0.6)
