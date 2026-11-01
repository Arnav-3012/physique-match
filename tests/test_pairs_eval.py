import json

import numpy as np
import pandas as pd
import pytest

from ml.src.data.pairs import baseline_a, canonical_subjects, true_deltas
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN
from ml.src.evaluate import claim_test_b, compare, verdict, weighted_mae

R = len(LABELLED_REGIONS)


def photos() -> pd.DataFrame:
    rows = [  # subject, photo, side_clipped
        ("s1", "p1", True), ("s1", "p2", False), ("s1", "p3", False),
        ("s2", "p4", False), ("s3", "p5", True), ("s4", "p6", False),
    ]  # fmt: skip
    df = pd.DataFrame(rows, columns=["subject_id", "photo_id", "side_clipped"])
    df["set"] = ["testB"] * 5 + ["train"]
    df["height_cm"] = 170.0
    for col in REGION_COLUMN.values():
        df[col] = 85.0
    return df


def test_canonical_one_unclipped_photo_per_subject():
    one = canonical_subjects("testB", photos())
    assert one.subject_id.tolist() == ["s1", "s2"]  # s3 has only a clipped side, s4 is train
    assert one.photo_id[0] in ("p2", "p3")
    assert canonical_subjects("testB", photos()).equals(one)  # deterministic
    assert one.y_chest.eq(0.5).all()


def test_true_deltas_antisymmetric_zero_diagonal():
    d = true_deltas(np.random.default_rng(0).normal(size=(6, R)))
    assert np.allclose(d, -d.transpose(1, 0, 2))
    assert np.allclose(d[np.arange(6), np.arange(6)], 0)


def test_baseline_a_nan_stays_nan():
    df = pd.DataFrame({"f": [0.1, np.nan, 0.3]})
    coef = {r: {"feature": "f", "slope": 2.0} for r in LABELLED_REGIONS}
    out = baseline_a(df, coef)
    assert out[0, 2, 0] == pytest.approx(-0.4)
    assert np.isnan(out[0, 1]).all() and np.isnan(out[1, 0]).all()


def synthetic(noise_base: float, noise_model: float, n: int = 30):
    rng = np.random.default_rng(1)
    y = rng.normal(0.3, 0.1, size=(n, R))
    truth = true_deltas(y)
    return (
        y,
        truth + rng.normal(0, noise_base, truth.shape),
        truth + rng.normal(0, noise_model, truth.shape),
    )


def test_weighted_mae_with_ones_is_plain_mean():
    err = np.abs(np.random.default_rng(2).normal(size=(8, 8, R)))
    err[np.arange(8), np.arange(8)] = np.nan
    assert np.allclose(weighted_mae(err, np.ones((1, 8)))[0], np.nanmean(err, axis=(0, 1)))


def test_better_model_wins_every_region():
    y, base, model = synthetic(0.05, 0.01)
    t = compare(base, model, y, n_boot=200)
    assert (t.lo > 0).all() and verdict(t)["passed"]


def test_identical_models_do_not_win():
    y, base, _ = synthetic(0.05, 0.05)
    t = compare(base, base, y, n_boot=200)
    assert (t["diff"] == 0).all() and not verdict(t)["passed"]


def test_nan_pairs_are_masked_not_zero():
    y, base, model = synthetic(0.05, 0.01)
    base[:10] = np.nan  # the baseline has no prediction for these goals
    t = compare(base, model, y, n_boot=50)
    assert (t.coverage < 1).all() and (t.mae_base < 0.2).all()  # not dragged to 0 or inflated


def test_verdict_rules():
    def table(lo, hi):
        return pd.DataFrame({"region": list("abcdefgh"), "lo": lo, "hi": hi})

    assert verdict(table([1] * 6 + [-1] * 2, [2] * 6 + [1] * 2))["passed"]
    assert not verdict(table([1] * 5 + [-1] * 3, [2] * 5 + [1] * 3))["passed"]  # only 5 wins
    worse = verdict(table([1] * 7 + [-3], [2] * 7 + [-1]))  # 7 wins but one significantly worse
    assert not worse["passed"] and worse["significantly_worse"] == ["h"]


def test_ledger_refuses_second_test_b_look(tmp_path):
    path = tmp_path / "ledger.json"
    claim_test_b("abc", path)
    claim_test_b("def", path)
    with pytest.raises(SystemExit, match="second look"):
        claim_test_b("abc", path)
    assert set(json.loads(path.read_text())) == {"abc", "def"}
