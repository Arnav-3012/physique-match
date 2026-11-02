import base64
import io

import numpy as np
import pytest
from PIL import Image
from test_features import side_figure
from test_rows import figure

from app.analyzer.geometric import GeometricAnalyzer
from app.analyzer.masks import decode_mask
from ml.src.preprocess import H, W, from_bodym
from physique_schemas.models import REGIONS, Keypoint, Scan, ViewData

KEYPOINTS = [Keypoint(x=0.5, y=0.5, z=0.0, visibility=1.0)] * 33


def png_b64(array: np.ndarray, fmt: str = "PNG") -> str:
    """Mode follows the array: (H, W) -> L, (H, W, 3) -> RGB, (H, W, 4) -> RGBA."""
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def mask_b64(mask: np.ndarray) -> str:
    return png_b64(mask.astype(np.uint8) * 255)


def scan(front: np.ndarray, side: np.ndarray, back: np.ndarray | None = None) -> Scan:
    views = {"front": front, "left": side} | ({"back": back} if back is not None else {})
    return Scan(views={k: ViewData(mask_png_b64=mask_b64(m), keypoints=KEYPOINTS)
                       for k, m in views.items()})  # fmt: skip


@pytest.fixture
def base():
    return from_bodym(figure()), from_bodym(side_figure())


@pytest.fixture
def wide(base):  # same height, twice the front width: a bigger person on the width features
    return from_bodym(np.kron(figure(), np.ones((1, 2), bool))), base[1]


def test_identical_scans_have_no_gap(base):
    gap = GeometricAnalyzer().compare(scan(*base), scan(*base))
    assert gap.analyzer == "geometric" and set(gap.regions) == set(REGIONS)
    assert all(r.delta == 0 for r in gap.regions.values())


def test_gap_is_antisymmetric_and_points_the_right_way(base, wide):
    a, b = GeometricAnalyzer(), (scan(*wide), scan(*base))
    ab, ba = a.compare(*b), a.compare(*b[::-1])
    assert ab.regions["shoulders"].delta > 0 > ba.regions["shoulders"].delta  # goal is wider
    for region in REGIONS:
        assert ab.regions[region].delta == pytest.approx(-ba.regions[region].delta)


def test_confidence_follows_validated_rho(base, wide):
    gap = GeometricAnalyzer().compare(scan(*wide), scan(*base))
    assert gap.regions["waist"].confidence > 0.9  # rho_val .92, rows found on the figure
    assert gap.regions["forearm"].confidence < 0.1  # rho_val .32
    assert gap.regions["waist"].measured


def test_back_regions_are_retired_with_or_without_a_back_view(base, wide):
    """ADR-037: no back-view size number; not measured, not just zero confidence."""
    for goal, current in ((scan(*wide), scan(*base)), (scan(*wide, wide[0]), scan(*base, base[0]))):
        gap = GeometricAnalyzer().compare(goal, current)
        for region in ("upper_back", "lats"):
            r = gap.regions[region]
            assert r.measured is False and r.delta == 0 == r.confidence


def test_front_only_goal_uses_width_fits_and_never_raises(base, wide):
    goal = Scan(views={"front": ViewData(mask_png_b64=mask_b64(wide[0]), keypoints=KEYPOINTS)})
    gap = GeometricAnalyzer().compare(goal, scan(*base))
    assert gap.goal_front_only is True
    assert gap.regions["waist"].confidence > 0 and gap.regions["waist"].delta > 0
    both = GeometricAnalyzer().compare(scan(*wide), scan(*base))
    assert gap.regions["shoulders"].delta == pytest.approx(both.regions["shoulders"].delta)
    assert both.goal_front_only is False


def test_decode_accepts_a_good_mask(base):
    assert decode_mask(mask_b64(base[0])).shape == (H, W)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda: "not base64!!", id="not-base64"),
        pytest.param(lambda: base64.b64encode(b"hello").decode(), id="not-an-image"),
        pytest.param(lambda: png_b64(np.zeros((H, W, 3), np.uint8)), id="rgb"),
        pytest.param(lambda: png_b64(np.zeros((H, W, 4), np.uint8)), id="rgba"),
        pytest.param(lambda: png_b64(np.zeros((H, W), np.uint8) + 128), id="grey-values"),
        pytest.param(lambda: png_b64(np.zeros((100, 100), np.uint8)), id="wrong-size"),
        pytest.param(lambda: png_b64(np.full((H, W), 255, np.uint8)), id="all-white"),
        pytest.param(lambda: png_b64(np.zeros((H, W), np.uint8)), id="empty"),
        pytest.param(lambda: png_b64(np.zeros((H, W, 3), np.uint8), "JPEG"), id="jpeg"),
    ],
)
def test_decode_rejects_anything_that_is_not_a_binary_mask(bad):
    with pytest.raises(ValueError):
        decode_mask(bad())


def test_grey_mask_with_texture_is_rejected(base):
    textured = base[0].astype(np.uint8) * 255
    textured[300, 100:110] = 77  # one grey run is enough
    with pytest.raises(ValueError, match="only the values 0 and 255"):
        decode_mask(png_b64(textured))
