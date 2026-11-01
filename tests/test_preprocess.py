import numpy as np
import pytest

from ml.src.preprocess import MARGIN, H, W, body_px, from_bodym, from_browser


def figure(h=600, canvas=(700, 400), top=40, left=150, arm_half=0.0):
    """Blocky full-body mask: head, narrow neck, torso (+ optional arms), two legs."""
    m = np.zeros(canvas, bool)
    cx = left + 0.1 * h

    def band(r0, r1, half, dx=0.0):
        rows = slice(top + int(r0 * h), top + int(r1 * h))
        cols = slice(int(cx + dx - half * h), int(cx + dx + half * h))
        m[rows, cols] = True

    band(0.0, 0.13, 0.045)  # head
    band(0.13, 0.17, 0.025)  # neck
    band(0.17, 0.52, max(0.09, arm_half))  # torso, wider if arms are out
    for dx in (-0.05, 0.05):
        band(0.52, 1.0, 0.025, dx)  # legs
    return m


def first_row(mask):
    return np.flatnonzero(mask.any(axis=1))[0]


def test_output_is_canonical_and_head_is_gone():
    out = from_bodym(figure())
    assert out.shape == (H, W) and out.dtype == bool
    top = (H - body_px(out)) / 2
    assert top + 0.12 * body_px(out) < first_row(out) < top + 0.20 * body_px(out)  # cut in the neck


def test_body_height_is_recovered_from_the_feet_row():
    out = from_bodym(figure())
    assert body_px(out) == pytest.approx(H / (1 + 2 * MARGIN), abs=2)  # height-limited


def test_width_limited_body_is_smaller_and_still_recovered():
    wide = figure(h=300, canvas=(400, 700), left=20, arm_half=0.0)
    wide[60:200, 20:640] = True  # arms-out span wider than the frame ratio
    out = from_bodym(wide)
    expected = 300 * min(W / (620 + 2 * MARGIN * 300), H / (300 * (1 + 2 * MARGIN)))
    assert body_px(out) < H / (1 + 2 * MARGIN) - 20
    assert body_px(out) == pytest.approx(expected, abs=3)


def test_invariant_to_position_and_scale():
    a = from_bodym(figure(h=600, top=40, left=150))
    b = from_bodym(figure(h=1000, canvas=(1200, 700), top=90, left=300))
    iou = (a & b).sum() / (a | b).sum()
    assert iou > 0.95


def test_stray_pixels_are_ignored():
    clean = figure()
    speck = clean.copy()
    speck[3:8, 3:8] = True  # far corner: would widen the bbox if kept
    assert (from_bodym(clean) == from_bodym(speck)).all()


def test_browser_mask_is_a_fixed_point():
    out = from_bodym(figure())
    assert (from_browser(out) == out).all()
    assert (from_browser(from_browser(out)) == out).all()


def test_browser_rejects_wrong_shape_and_empty():
    with pytest.raises(ValueError):
        from_browser(np.zeros((100, 100), bool))
    with pytest.raises(ValueError):
        from_browser(np.zeros((H, W), bool))
