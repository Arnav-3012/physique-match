import numpy as np
import pytest

from synth import cues

ALL = {r: 1.0 for r in ("abs", "chest", "shoulders", "arms", "back", "legs")}


def test_zero_amplitude_is_flat():
    for view in ("front", "back"):
        m = cues.height_map(view, {r: 0.0 for r in ALL}, seed=1)
        assert m.shape == (cues.H, cues.W) and m.dtype == np.float32
        assert np.all(m == 1.0)


def test_values_in_unit_range_and_cues_exist():
    m = cues.height_map("front", ALL, seed=1)
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.min() < 0.7  # grooves are visible at full amplitude


def test_deterministic_and_seed_dependent():
    a = cues.height_map("front", ALL, seed=3)
    assert np.array_equal(a, cues.height_map("front", ALL, seed=3))
    assert not np.array_equal(a, cues.height_map("front", ALL, seed=4))


def test_stronger_amplitude_cuts_deeper():
    weak = cues.height_map("front", {r: 0.3 for r in ALL}, seed=2).min()
    strong = cues.height_map("front", ALL, seed=2).min()
    assert strong < weak < 1.0


@pytest.mark.parametrize(("region", "view"), [("abs", "back"), ("chest", "back"), ("back", "front")])
def test_region_only_appears_in_its_view(region, view):
    amp = {r: 0.0 for r in ALL} | {region: 1.0}
    assert np.all(cues.height_map(view, amp, seed=5) == 1.0)


def test_left_and_right_share_the_pattern_with_a_side_gain():
    m = cues.height_map("front", {r: 0.2 for r in ALL}, seed=6)  # low amplitude: no clipping
    left, right = m[:, : cues.W // 2][:, ::-1], m[:, cues.W // 2 :]
    assert np.corrcoef((1 - left).ravel(), (1 - right).ravel())[0, 1] > 0.999
    assert not np.array_equal(left, right)  # the side gains differ a little


def test_map_edges_are_flat_so_edge_extension_adds_no_cue():
    m = cues.height_map("back", ALL, seed=7)
    assert np.all(m[:, :8] == 1.0) and np.all(m[:, -8:] == 1.0)
    assert np.all(m[:8] == 1.0) and np.all(m[-8:] == 1.0)
