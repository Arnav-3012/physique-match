import numpy as np
import pandas as pd
import pytest

from ml.src.data.pairs import canonical_subjects, labels, true_deltas
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN
from ml.src.evaluate import floors


def fake_features(n=60):
    """Ratios driven only by sex and weight, so the null ridge can recover them."""
    rng = np.random.default_rng(0)
    male = rng.random(n) > 0.5
    df = pd.DataFrame(
        {
            "subject_id": [f"s{k:03d}" for k in range(n)],
            "photo_id": [f"p{k:03d}" for k in range(n)],
            "side_clipped": False,
            "set": ["train"] * (n - 20) + ["val"] * 20,
            "sex": np.where(male, "male", "female"),
            "height_cm": rng.uniform(150, 195, n),
            "weight_kg": rng.uniform(50, 110, n),
        }
    )
    ratio = 0.3 + 0.05 * male + 0.002 * df.weight_kg
    for column in REGION_COLUMN.values():
        df[column] = ratio * df.height_cm
    return df


def test_mean_floor_is_the_mean_absolute_true_delta():
    df = fake_features()
    val, train = canonical_subjects("val", df), canonical_subjects("train", df)
    deltas = np.abs(true_deltas(labels(val)))
    expected = deltas.sum(axis=(0, 1)) / (len(val) * (len(val) - 1))
    assert floors(val, train).mae_mean.to_numpy() == pytest.approx(expected)


def test_null_floor_recovers_ratios_it_can_see_and_has_one_row_per_region():
    df = fake_features()
    t = floors(canonical_subjects("val", df), canonical_subjects("train", df))
    assert t.region.tolist() == list(LABELLED_REGIONS)
    assert (t.mae_null < 0.1 * t.mae_mean).all()
