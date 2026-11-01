import numpy as np
import pytest

from app.analyzer.rows import ARMPIT_PRIOR, CROTCH_PRIOR, find_rows
from ml.src.preprocess import body_px, from_bodym

H_FIG = 600


def figure(arms="apose", legs_gap=True):
    """Full-body mask with known rows (fractions of height from the crown):
    neck .13-.17, torso .17-.52 (waist band .38-.42 narrower, hip band .44-.52 widest),
    legs from .52. A-pose arms are attached to the torso down to .30, then slope outward
    leaving a gap; "merged" arms overlap the torso all the way down.
    """
    h = H_FIG
    m = np.zeros((700, 400), bool)
    cx, top = 200, 40

    def band(r0, r1, lo, hi):  # symmetric columns cx -/+ lo*h .. hi*h
        rows = slice(top + int(r0 * h), top + int(r1 * h))
        for sign in (-1, 1):
            a, b = sorted((cx + sign * lo * h, cx + sign * hi * h))
            m[rows, int(a) : int(b)] = True

    band(0.0, 0.13, 0, 0.045)  # head
    band(0.13, 0.17, 0, 0.025)  # neck
    band(0.17, 0.38, 0, 0.09)  # torso
    band(0.38, 0.42, 0, 0.07)  # waist
    band(0.42, 0.44, 0, 0.09)
    band(0.44, 0.52, 0, 0.10)  # hip
    if arms == "apose":
        band(0.17, 0.30, 0.09, 0.12)  # attached
        for r in np.arange(0.30, 0.50, 1 / h):  # sloping away, 1 px per row
            inner = 0.10 + 0.06 * (r - 0.30) / 0.20
            band(r, r + 1 / h, inner, inner + 0.03)
    elif arms == "merged":
        band(0.17, 0.50, 0.04, 0.13)  # overlaps the torso: never a gap
    band(0.52, 1.0, *((0.01, 0.09) if legs_gap else (0, 0.09)))  # legs
    return m


@pytest.fixture
def rows():
    return find_rows(from_bodym(figure()))


def u(rows, row):  # row -> fraction of body height below the crown
    return (row - rows.crown) / (rows.feet - rows.crown)


def test_known_rows_are_found(rows):
    assert rows.arms_separated and rows.crotch_found and rows.found
    assert u(rows, rows.armpit) == pytest.approx(0.30, abs=0.02)
    assert 0.37 <= u(rows, rows.waist) <= 0.43  # inside the narrow band
    assert 0.43 <= u(rows, rows.hip) <= 0.52  # inside the wide band
    assert u(rows, rows.crotch) == pytest.approx(0.52, abs=0.02)


def test_rows_are_ordered(rows):
    order = [rows.cut, rows.shoulder, rows.armpit, rows.chest, rows.waist, rows.hip]
    assert order == sorted(order) and rows.hip < rows.crotch < rows.feet


def test_crown_matches_body_height():
    mask = from_bodym(figure())
    r = find_rows(mask)
    assert r.feet - r.crown == pytest.approx(body_px(mask), abs=2)


def test_merged_arms_use_the_prior_and_flag_it():
    r = find_rows(from_bodym(figure(arms="merged")))
    assert not r.arms_separated and not r.found
    assert u(r, r.armpit) == pytest.approx(ARMPIT_PRIOR, abs=0.01)


def test_touching_legs_use_the_prior_and_flag_it():
    r = find_rows(from_bodym(figure(legs_gap=False)))
    assert not r.crotch_found
    assert u(r, r.crotch) == pytest.approx(CROTCH_PRIOR, abs=0.01)


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        find_rows(np.zeros((512, 384), bool))
