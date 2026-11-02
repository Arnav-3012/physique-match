import numpy as np
import pytest
from PIL import Image
from test_analyzer import KEYPOINTS, mask_b64, png_b64, scan
from test_features import side_figure
from test_rows import figure

from app.analyzer.siamese_subtract import (
    ONNX_PATH,
    V_TAPER_NOT_MEASURED,
    SiameseSubtractAnalyzer,
    v_taper,
)
from ml.src.calibrate import reliability
from ml.src.data.regions import LABELLED_REGIONS
from ml.src.preprocess import H, W, from_bodym
from ml.src.sanity_ordering import load_scan
from physique_schemas.models import REGIONS, RegionGap, Scan, ViewData, VTaper

SIGMAS = {"sigma_full": 0.01, "sigma_front": 0.03, "sigma_prior": 0.1}
META = {
    "output": {"regions": list(LABELLED_REGIONS)},
    "calibration": {r: SIGMAS for r in LABELLED_REGIONS},
}
META_VT = META | {"v_taper": SIGMAS}  # the block's sigmas are named like a calibration entry


class StubSession:
    """Proportions grow with the filled area of the front view; the side view shifts them a little,
    so a test can tell whether it was used. Counts encodes."""

    def __init__(self):
        self.runs = 0

    def run(self, _, feed):
        self.runs += 1
        x = feed["x"]
        value = x[:, 0].mean(axis=(1, 2)) + 0.01 * x[:, 1].mean(axis=(1, 2))
        return [np.repeat(value[:, None], len(LABELLED_REGIONS), axis=1)]


@pytest.fixture
def analyzer():
    return SiameseSubtractAnalyzer(session=StubSession(), meta=META)


@pytest.fixture
def base():
    return from_bodym(figure()), from_bodym(side_figure())


@pytest.fixture
def wide(base):
    return from_bodym(np.kron(figure(), np.ones((1, 2), bool))), base[1]


def view(mask_or_b64) -> ViewData:
    b64 = mask_or_b64 if isinstance(mask_or_b64, str) else mask_b64(mask_or_b64)
    return ViewData(mask_png_b64=b64, keypoints=KEYPOINTS)


def front_only(front) -> Scan:
    return Scan(views={"front": view(front)})


def with_right(front, side) -> Scan:
    """A scan whose only side view is the right profile, the mirror image of the left one."""
    return Scan(views={"front": view(front), "right": view(np.fliplr(side))})


def test_identical_scans_have_no_gap_and_every_region_is_present(analyzer, base):
    gap = analyzer.compare(scan(*base), scan(*base))
    assert gap.analyzer == "siamese_subtract" and set(gap.regions) == set(REGIONS)
    assert all(r.delta == 0 for r in gap.regions.values())
    assert gap.goal_front_only is False


def test_gap_is_goal_minus_current_and_carries_both_proportions(analyzer, base, wide):
    gap = analyzer.compare(scan(*wide), scan(*base))
    r = gap.regions["chest"]
    assert r.measured and r.delta > 0
    assert r.delta == pytest.approx(r.goal_ratio - r.current_ratio)


def test_gap_is_exactly_antisymmetric(analyzer, base, wide):
    ab = analyzer.compare(scan(*wide), scan(*base))
    ba = analyzer.compare(scan(*base), scan(*wide))
    for region in LABELLED_REGIONS:
        assert ab.regions[region].delta == -ba.regions[region].delta


def test_reliability_is_fixed_per_region_and_slice_not_per_scan(analyzer, base, wide):
    a = analyzer.compare(scan(*wide), scan(*base)).regions["chest"].confidence
    b = analyzer.compare(scan(*base), scan(*wide)).regions["chest"].confidence
    assert a == b == pytest.approx(reliability(0.01, 0.01, 0.1))


def test_front_only_goal_is_flagged_and_less_reliable(analyzer, base):
    full = analyzer.compare(scan(*base), scan(*base))
    front = analyzer.compare(front_only(base[0]), scan(*base))
    assert front.goal_front_only is True
    assert front.regions["waist"].measured
    assert front.regions["waist"].confidence == pytest.approx(reliability(0.03, 0.01, 0.1))
    assert front.regions["waist"].confidence < full.regions["waist"].confidence


def test_a_right_side_view_gives_the_same_gap_as_the_left_one(analyzer, base, wide):
    left = analyzer.compare(scan(*wide), scan(*base))
    right = analyzer.compare(with_right(*wide), with_right(*base))
    assert right == left
    assert right.goal_front_only is False


def test_back_regions_are_not_measured_here_whatever_the_views(analyzer, base, wide):
    """upper_back stays unmeasured (ADR-037 Amendment 2); lats is filled in later."""
    without = analyzer.compare(scan(*wide), scan(*base))
    with_back = analyzer.compare(scan(*wide, wide[0]), scan(*base, base[0]))
    assert with_back == without
    for region in ("upper_back", "lats"):
        r = without.regions[region]
        assert r.measured is False and r.delta == 0 == r.confidence and r.goal_ratio is None
    assert all(without.regions[r].measured for r in LABELLED_REGIONS)


def test_an_unmeasured_region_cannot_carry_numbers():
    RegionGap(delta=0.0, confidence=0.0, measured=False)
    with pytest.raises(ValueError, match="no numbers"):
        RegionGap(delta=0.2, confidence=0.0, measured=False)
    with pytest.raises(ValueError, match="no numbers"):
        RegionGap(delta=0.0, confidence=0.0, goal_ratio=0.3, measured=False)


def test_every_request_encodes_the_goal_and_the_current_scan(analyzer, base, wide):
    goal = scan(*wide)
    analyzer.compare(goal, scan(*base))
    assert analyzer.session.runs == 2
    analyzer.compare(goal, scan(*base))  # same goal again: nothing is remembered
    assert analyzer.session.runs == 4
    assert not hasattr(analyzer, "_goal_cache")


def test_a_bad_goal_is_rejected(analyzer, base):
    rgb = png_b64(np.full((H, W, 3), 120, np.uint8))
    with pytest.raises(ValueError, match="1-channel"):
        analyzer.compare(Scan(views={"front": view(rgb)}), scan(*base))


def test_every_view_is_validated_even_when_the_model_does_not_use_it(analyzer, base):
    rgb = png_b64(np.full((H, W, 3), 120, np.uint8))
    unused = Scan(views={**scan(*base).views, "right": view(rgb)})  # left present, right unused
    with pytest.raises(ValueError, match="1-channel"):
        analyzer.compare(scan(*base), unused)


def test_input_is_the_canonical_frame_with_a_blank_side_when_absent(analyzer, base):
    seen = []
    run = analyzer.session.run
    analyzer.session.run = lambda names, feed: seen.append(feed["x"]) or run(names, feed)
    analyzer.compare(front_only(base[0]), scan(*base))
    goal_x, current_x = seen
    assert goal_x.shape == (1, 3, H // 2, W // 2) and goal_x.dtype == np.float32
    assert not goal_x[0, 1].any() and current_x[0, 1].any()


def test_sanity_script_loads_only_the_views_on_disk(tmp_path, base):
    Image.fromarray(base[0].astype(np.uint8) * 255).save(tmp_path / "front.png")
    assert list(load_scan(tmp_path).views) == ["front"]
    assert list(load_scan(tmp_path, ("front", "left")).views) == ["front"]


class FixedSession:
    """Returns the given per-person outputs in call order; the goal is encoded first."""

    def __init__(self, *outputs):
        self.outputs = iter(outputs)

    def run(self, _, feed):
        return [next(self.outputs)]


def person(shoulders: float, waist: float) -> np.ndarray:
    row = dict.fromkeys(LABELLED_REGIONS, 0.3) | {"shoulders": shoulders, "waist": waist}
    return np.array([[row[r] for r in LABELLED_REGIONS]])


def fixed(goal, current, meta=META_VT) -> SiameseSubtractAnalyzer:
    return SiameseSubtractAnalyzer(session=FixedSession(person(*goal), person(*current)), meta=meta)


def test_v_taper_is_shoulders_over_waist_and_the_delta_is_goal_minus_current(base):
    vt = fixed((0.24, 0.48), (0.22, 0.52)).compare(scan(*base), scan(*base)).v_taper
    assert vt.measured and vt.reason is None
    assert vt.goal_index == pytest.approx(0.5)
    assert vt.current_index == pytest.approx(0.22 / 0.52)
    assert vt.delta == pytest.approx(0.5 - 0.22 / 0.52)


def test_v_taper_delta_is_exactly_antisymmetric(base):
    ab = fixed((0.24, 0.48), (0.22, 0.52)).compare(scan(*base), scan(*base)).v_taper
    ba = fixed((0.22, 0.52), (0.24, 0.48)).compare(scan(*base), scan(*base)).v_taper
    assert ab.delta == -ba.delta != 0


def test_v_taper_reliability_follows_the_slice_and_is_none_without_the_block(base):
    full = fixed((0.24, 0.48), (0.22, 0.52)).compare(scan(*base), scan(*base)).v_taper
    front = fixed((0.24, 0.48), (0.22, 0.52)).compare(front_only(base[0]), scan(*base)).v_taper
    assert full.reliability == pytest.approx(reliability(0.01, 0.01, 0.1))
    assert front.reliability == pytest.approx(reliability(0.03, 0.01, 0.1))
    assert front.reliability < full.reliability
    bare = fixed((0.24, 0.48), (0.22, 0.52), meta=META).compare(scan(*base), scan(*base)).v_taper
    assert bare.measured and bare.reliability is None


def test_v_taper_is_not_measured_when_shoulders_or_waist_is_not():
    def region(g, c):
        return RegionGap(delta=g - c, confidence=0.5, goal_ratio=g, current_ratio=c)

    blank = RegionGap(delta=0.0, confidence=0.0, measured=False)
    ok = region(0.24, 0.22), region(0.5, 0.5)
    for shoulders, waist in [(blank, ok[1]), (ok[0], blank), (ok[0], region(0.0, 0.5))]:
        out = v_taper({"shoulders": shoulders, "waist": waist}, False, SIGMAS)
        assert out == VTaper(measured=False, reason=V_TAPER_NOT_MEASURED)


def test_v_taper_schema_keeps_numbers_and_measured_in_step():
    VTaper(goal_index=0.4, current_index=0.38, delta=0.02)
    VTaper(measured=False, reason="because")
    with pytest.raises(ValueError, match="needs both indices"):
        VTaper(goal_index=0.4, delta=0.02)
    with pytest.raises(ValueError, match="carries no numbers"):
        VTaper(measured=False, delta=0.02)
    with pytest.raises(ValueError, match="carries no numbers"):
        VTaper(measured=False, reliability=0.0)
    with pytest.raises(ValueError, match="reason"):
        VTaper(goal_index=0.4, current_index=0.38, delta=0.02, reason="because")


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="run ml.src.export_onnx first")
def test_real_model_runs_and_is_antisymmetric(base, wide):
    analyzer = SiameseSubtractAnalyzer()
    ab = analyzer.compare(scan(*wide), scan(*base))
    ba = analyzer.compare(scan(*base), scan(*wide))
    for region in LABELLED_REGIONS:
        assert ab.regions[region].delta == -ba.regions[region].delta
        assert 0 < ab.regions[region].confidence < 1
    assert ab.v_taper.measured and ab.v_taper.delta == -ba.v_taper.delta
    assert 0 < ab.v_taper.reliability < 1  # the block in gap_regressor.json is read
