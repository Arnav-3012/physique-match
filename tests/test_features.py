import numpy as np
import pytest
from test_rows import figure

from app.analyzer.features import GIRTH_REGIONS, extract, girth
from ml.src.preprocess import from_bodym

H_FIG = 600


def side_figure(spike=False):
    """Profile with known depths (fractions of height): torso .10 deep, legs .06 deep.
    `spike` adds a protruding arm, 50% deeper, 4% of the height tall, over the waist."""
    h = H_FIG
    m = np.zeros((700, 400), bool)
    cx, top = 200, 40

    def band(r0, r1, half):
        m[top + int(r0 * h) : top + int(r1 * h), int(cx - half * h) : int(cx + half * h)] = True

    band(0.0, 0.13, 0.045)  # head
    band(0.13, 0.17, 0.025)  # neck
    band(0.17, 0.52, 0.05)  # torso, depth .10
    if spike:
        band(0.36, 0.40, 0.075)
    band(0.52, 1.0, 0.03)  # legs overlap in profile, depth .06
    return m


@pytest.fixture
def feats():
    return extract(from_bodym(figure()), from_bodym(side_figure()))


def test_arm_thickness_ignores_the_arm_angle(feats):
    # figure arms: 0.03h wide horizontally, sloping 0.3 sideways per row -> true thickness 0.0287h
    for k in ("upper_arm_w", "forearm_w"):
        assert feats[k] == pytest.approx(0.03 * np.cos(np.arctan(0.3)), abs=0.004)


def test_merged_arms_give_nan_thickness():
    f = extract(from_bodym(figure(arms="merged")))
    assert np.isnan(f["upper_arm_w"]) and np.isnan(f["forearm_w"])


def test_girth_of_a_circle():
    assert girth(0.2, 0.2) == pytest.approx(2 * np.pi * 0.1)


def test_front_widths_in_body_heights(feats):
    assert feats["chest_w"] == pytest.approx(0.18, abs=0.012)
    assert feats["hip_w"] == pytest.approx(0.20, abs=0.012)
    assert feats["waist_w"] == pytest.approx(0.14, abs=0.012)
    assert feats["thigh_w"] == pytest.approx(0.08, abs=0.012)


def test_side_depths_and_girth(feats):
    assert feats["chest_d"] == pytest.approx(0.10, abs=0.012)
    assert feats["thigh_d"] == pytest.approx(0.06, abs=0.012)
    assert feats["hip_g"] == pytest.approx(girth(feats["hip_w"], feats["hip_d"]))


def test_features_are_scale_free():
    small = extract(from_bodym(figure()), from_bodym(side_figure()))
    big_front = np.kron(figure(), np.ones((2, 2), bool))  # same person, 2x the pixels
    big_side = np.kron(side_figure(), np.ones((2, 2), bool))
    big = extract(from_bodym(big_front), from_bodym(big_side))
    for k in ("chest_g", "waist_g", "hip_g", "thigh_g", "shoulder_w"):
        assert big[k] == pytest.approx(small[k], abs=0.01)


def test_protruding_arm_rows_are_skipped():
    f = extract(from_bodym(figure()), from_bodym(side_figure(spike=True)))
    assert np.isnan(f["waist_d"]) and np.isnan(f["waist_g"])
    assert not np.isnan(f["chest_d"])  # rows away from the spike stay valid


def test_missing_views_give_nan_not_a_guess():
    f = extract(from_bodym(figure()))
    assert all(np.isnan(f[f"{n}_g"]) for n in GIRTH_REGIONS)
    assert np.isnan(f["upper_back_w"]) and not np.isnan(f["chest_w"])
