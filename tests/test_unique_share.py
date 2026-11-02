import numpy as np
import pandas as pd
import pytest

from ml.src.unique_share import THRESHOLD, check, keep_index_gap, unique_share

RNG = np.random.default_rng(0)
WAIST = RNG.normal(0.52, 0.07, 200)


def test_threshold_is_the_adr_038_value():
    assert THRESHOLD == 0.2


def test_index_exactly_linear_in_waist_has_no_unique_share():
    shoulders = WAIST * (0.8 - WAIST)  # index = 0.8 - waist
    assert unique_share(shoulders, WAIST) == pytest.approx(0, abs=1e-9)


def test_index_independent_of_waist_is_almost_all_unique():
    shoulders = WAIST * RNG.normal(0.40, 0.03, 200)
    assert unique_share(shoulders, WAIST) > 0.9


@pytest.mark.parametrize(
    ("shares", "keep"),
    [
        ({"male": 0.2, "female": 0.5}, True),  # >= threshold
        ({"male": 0.1999, "female": 0.5}, False),
        ({"male": 0.9, "female": 0.05}, False),  # the minimum over sexes decides
    ],
)
def test_decision_uses_the_minimum_over_sexes(shares, keep):
    assert keep_index_gap(shares) is keep


def test_check_splits_by_sex_and_ignores_case():
    n = 100
    df = pd.DataFrame(
        {
            "sex": ["Male"] * n + ["female"] * n,
            "y_waist": np.r_[WAIST[:n], WAIST[n:]],
            "y_shoulders": np.r_[
                WAIST[:n] * (0.8 - WAIST[:n]), WAIST[n:] * RNG.normal(0.4, 0.03, n)
            ],
        }
    )
    out = check(df)
    assert {s: v["n"] for s, v in out["sexes"].items()} == {"male": n, "female": n}
    assert out["sexes"]["male"]["unique_share"] < 1e-6  # linear: nothing unique
    assert out["unique_share"] == out["sexes"]["male"]["unique_share"] and not out["keep"]
