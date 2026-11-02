import pandas as pd
import pytest

from app.bodycomp import BF_ATHLETE_UPPER, BF_FLOOR, rfm
from app.bodycomp import waist_at_bf as api_waist_at_bf
from ml.src.ceiling_extrapolation import BF_ENDS, check, waist_at_bf


@pytest.mark.parametrize("sex", ["male", "female"])
def test_floor_end_is_the_planner_floor(sex):
    assert BF_ENDS[sex]["floor"] == BF_FLOOR[sex]


@pytest.mark.parametrize("sex", ["male", "female"])
def test_athlete_upper_end_is_the_planner_constant(sex):
    assert BF_ENDS[sex]["athlete_upper"] == BF_ATHLETE_UPPER[sex]


@pytest.mark.parametrize("sex", ["male", "female"])
def test_inverse_round_trips_rfm(sex):
    for bf in BF_ENDS[sex].values():
        assert rfm(1.0, waist_at_bf(bf, sex == "female"), sex) == pytest.approx(bf)


@pytest.mark.parametrize("sex", ["male", "female"])
@pytest.mark.parametrize("height_cm", [150.0, 180.0])
def test_api_inverse_round_trips_rfm_and_matches_the_script(sex, height_cm):
    for bf in BF_ENDS[sex].values():
        waist = api_waist_at_bf(bf, height_cm, sex)
        assert rfm(height_cm, waist, sex) == pytest.approx(bf)
        assert waist / height_cm == pytest.approx(waist_at_bf(bf, sex == "female"))


def frame(male_ratios, female_ratios):
    rows = [(r, 1.0) for r in male_ratios] + [(r, 0.0) for r in female_ratios]
    return pd.DataFrame(rows, columns=["waist", "is_male"]).assign(stature=1.0)


def test_flag_and_count_below():
    # male floor 8% -> .357, athlete upper 13% -> .392; female 18% -> .345, 20% -> .357
    out = check(frame([0.30, 0.40, 0.50], [0.30, 0.40]))["male"]["ends"]
    assert out["floor"]["n_below"] == 1 and not out["floor"]["extrapolated"]
    assert out["athlete_upper"]["n_below"] == 1
    out = check(frame([0.38, 0.40], [0.36, 0.40]))
    assert out["male"]["ends"]["floor"]["extrapolated"]  # .357 < lowest .38
    assert out["male"]["ends"]["floor"]["n_below"] == 0
    assert not out["male"]["ends"]["athlete_upper"]["extrapolated"]  # .392 > .38
    assert out["female"]["ends"]["floor"]["extrapolated"]  # .345 < lowest .36
    assert out["female"]["ends"]["athlete_upper"]["extrapolated"]  # .357 < .36
