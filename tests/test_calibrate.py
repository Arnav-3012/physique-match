import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.src.calibrate import calibrate, index_sigmas, reliability
from ml.src.data.regions import LABELLED_REGIONS

N_REGIONS = len(LABELLED_REGIONS)
META = Path(__file__).resolve().parents[1] / "services/api/app/analyzer/gap_regressor.json"
ONNX = META.with_suffix(".onnx")


def standardised(rng, shape):
    """Normals rescaled to mean 0 and sample std exactly 1 per column, so tests need no tolerance
    for sampling noise."""
    z = rng.normal(0, 1, shape)
    return (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)


def fake_split(n=200, noise_full=0.01, noise_front=0.03):
    """Ratios with std 0.05 per region; predictions off by exactly noise_full / noise_front std."""
    rng = np.random.default_rng(0)
    y = 0.3 + 0.05 * standardised(rng, (n, N_REGIONS))
    noise = standardised(rng, y.shape)
    return y + noise_full * noise, y + noise_front * noise, y


def test_sigmas_follow_the_noise_and_the_region_order():
    out = calibrate(*fake_split())
    assert list(out) == list(LABELLED_REGIONS)
    for s in out.values():
        assert s["sigma_full"] == pytest.approx(0.01)
        assert s["sigma_front"] == pytest.approx(0.03)
        assert s["sigma_prior"] == pytest.approx(np.sqrt(2) * 0.05)


def test_sigma_prior_is_the_std_of_the_true_gap_over_ordered_pairs():
    _, _, y = fake_split(n=40)
    brute = np.array(
        [[y[i, k] - y[j, k] for i in range(40) for j in range(40) if i != j]
         for k in range(N_REGIONS)]
    )  # fmt: skip
    out = calibrate(y, y, y)
    got = np.array([out[r]["sigma_prior"] for r in LABELLED_REGIONS])
    assert got == pytest.approx(brute.std(axis=1))
    # the same number as the per-person std scaled by sqrt(2): the pair std is mean-free
    assert got == pytest.approx(np.sqrt(2) * y.std(axis=0, ddof=1))


def test_reliability_is_one_minus_the_error_share_clipped_to_the_unit_interval():
    assert reliability(0.03, 0.04, 0.10) == pytest.approx(0.5)  # sigma_gap = 0.05
    assert reliability(0.0, 0.0, 0.10) == 1.0
    assert reliability(0.10, 0.10, 0.10) == 0.0  # error 1.4x the spread: clipped, not negative
    assert reliability(0.05, 0.05, 0.10) > reliability(0.06, 0.06, 0.10)


def test_typical_regions_do_not_clip_to_zero():
    full, front, y = fake_split()
    out = calibrate(full, front, y)
    for s in out.values():
        for goal in ("sigma_full", "sigma_front"):
            assert reliability(s[goal], s["sigma_full"], s["sigma_prior"]) > 0.5


def test_front_only_goal_is_never_more_reliable_than_a_full_one():
    out = calibrate(*fake_split())
    for s in out.values():
        full = reliability(s["sigma_full"], s["sigma_full"], s["sigma_prior"])
        front = reliability(s["sigma_front"], s["sigma_full"], s["sigma_prior"])
        assert front <= full


def test_index_sigmas_are_those_of_shoulders_over_waist():
    full, front, y = fake_split(n=40)
    s, w = LABELLED_REGIONS.index("shoulders"), LABELLED_REGIONS.index("waist")

    def index(a):
        return a[:, s] / a[:, w]

    gaps = [index(y)[i] - index(y)[j] for i in range(40) for j in range(40) if i != j]
    out = index_sigmas(full, front, y)
    assert list(out) == ["sigma_full", "sigma_front", "sigma_prior"]
    assert out["sigma_full"] == pytest.approx((index(full) - index(y)).std(ddof=1))
    assert out["sigma_front"] == pytest.approx((index(front) - index(y)).std(ddof=1))
    assert out["sigma_prior"] == pytest.approx(np.std(gaps))
    assert out["sigma_full"] < out["sigma_front"]  # fake_split: the front noise is 3x larger


@pytest.mark.skipif(not META.exists(), reason="run ml.src.export_onnx first")
def test_exported_calibration_does_not_clip_typical_regions():
    """Thresholds are mine: no region exactly 0 on either slice, at least 6 of 8 above 0.3."""
    table = json.loads(META.read_text())["calibration"]
    for goal in ("sigma_full", "sigma_front"):
        rel = [reliability(s[goal], s["sigma_full"], s["sigma_prior"]) for s in table.values()]
        assert min(rel) > 0
        assert sum(r > 0.3 for r in rel) >= 6


@pytest.mark.skipif(not META.exists(), reason="run ml.src.export_onnx first")
def test_v_taper_block_is_of_this_model_and_does_not_clip():
    meta = json.loads(META.read_text())
    assert "v_taper" in meta, "export_onnx rewrites the JSON; re-run ml.src.v_taper_block"
    block = meta["v_taper"]
    assert block["model_hash"] == meta["model_hash"]
    for goal in ("sigma_full", "sigma_front"):
        assert 0 < reliability(block[goal], block["sigma_full"], block["sigma_prior"]) < 1


@pytest.mark.skipif(not ONNX.exists(), reason="the ONNX is gitignored; run ml.src.export_onnx")
def test_v_taper_block_records_the_onnx_in_use():
    with ONNX.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    assert json.loads(META.read_text())["v_taper"]["onnx_sha256"] == digest
