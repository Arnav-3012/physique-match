import json
from pathlib import Path

import numpy as np
import pytest

from app.bodycomp import (
    BF_ATHLETE_UPPER,
    BF_FLOOR,
    SUPPORT_MIN,
    ZONE1_TEXT,
    ZONE2_TEXT,
    zone1_case,
    zone2_case,
)

ANALYZER = Path(__file__).resolve().parents[1] / "services/api/app/analyzer"
STORED = ANALYZER / "ceiling_extrapolation.json"


@pytest.mark.parametrize("sex", ["male", "female"])
def test_floor_is_below_athlete_upper(sex):
    assert BF_FLOOR[sex] < BF_ATHLETE_UPPER[sex]


def stored_case(sex, end, case_fn):
    stored = json.loads(STORED.read_text())["sexes"][sex]["ends"][end]
    return case_fn(stored["n_below"], stored["extrapolated"])


def test_men_at_13_percent_are_thin():
    case = stored_case("male", "athlete_upper", zone1_case)
    assert (case["label_case"], case["support"]) == (2, 4)
    assert "thin reference data" in case["text"]


def test_women_at_20_percent_get_no_reachable_claim():
    case = stored_case("female", "athlete_upper", zone1_case)
    assert (case["label_case"], case["support"]) == (3, 0)
    assert "reachable" not in case["text"].lower()


@pytest.mark.parametrize("sex", ["male", "female"])
def test_zone2_at_the_floor_gets_no_reachable_claim(sex):
    # ADR-042: both sexes' floor ends sit below the ANSUR minimum in the stored step 0 data
    case = stored_case(sex, "floor", zone2_case)
    assert (case["label_case"], case["support"]) == (3, 0)
    assert "reachable" not in case["text"].lower()


def test_zone2_texts_differ_from_zone1_only_in_the_reachable_cases():
    assert ZONE2_TEXT[3] == ZONE1_TEXT[3]
    assert all("floor" in ZONE2_TEXT[c] and "floor" not in ZONE1_TEXT[c] for c in (1, 2))


# ratios 0.36 .. 0.60, 100 evenly spaced: 5 are <= .37, 38 are <= .45
RATIOS = np.linspace(0.36, 0.60, 100)


def support_case(inverted, ratios, case_fn=zone1_case):
    """What the step 0 script stores, computed from raw ratios (the API has no ANSUR files)."""
    return case_fn(int((ratios <= inverted).sum()), bool(inverted < ratios.min()))


def test_inside_range_with_enough_support_is_normal():
    case = support_case(0.45, RATIOS)
    assert (case["label_case"], case["support"]) == (1, 38)
    assert case["text"] == ZONE1_TEXT[1]
    assert support_case(0.45, RATIOS, zone2_case)["text"] == ZONE2_TEXT[1]


def test_inside_range_with_thin_support_and_outside_range():
    assert support_case(0.37, RATIOS)["label_case"] == 2
    below = support_case(0.35, RATIOS)
    assert (below["label_case"], below["support"], below["extrapolated"]) == (3, 0, True)


def test_support_min_boundary_is_inclusive():
    assert zone1_case(SUPPORT_MIN, False)["label_case"] == 1
    assert zone1_case(SUPPORT_MIN - 1, False)["label_case"] == 2
    assert zone1_case(SUPPORT_MIN + 5, True)["label_case"] == 3  # extrapolated wins
