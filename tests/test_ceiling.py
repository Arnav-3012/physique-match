import pytest

from app.analyzer.ceiling import NEEDS_SHOULDERS, SET_YOUR_SEX, with_ceiling
from app.bodycomp import BF_ATHLETE_UPPER, BF_FLOOR
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
    assert c.measured and c.zone is None


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
