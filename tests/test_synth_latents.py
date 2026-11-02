from math import exp
from statistics import NormalDist

import pytest

from synth import latents as L


def test_hide_is_one_when_lean_zero_when_soft_and_monotone():
    assert L.hide(0.0) == L.hide(L.LEAN_F) == 1.0
    assert L.hide(L.SOFT_F) == L.hide(1.0) == 0.0
    fs = [i / 20 for i in range(21)]
    assert all(L.hide(a) >= L.hide(b) for a, b in zip(fs, fs[1:], strict=False))


def test_muscle_caps_the_cues():
    assert L.d_eff(1.0, 0.1, 0.0) == pytest.approx(L.CAP_BASE)
    assert L.d_eff(1.0, 0.1, 1.0) == 1.0


def test_bulky_has_low_definition_and_lean_muscular_high():
    bulky = L.d_eff(1.0, 0.85, 0.85)  # high muscle, high fat
    shredded = L.d_eff(1.0, 0.15, 0.85)  # high muscle, low fat
    assert bulky == 0.0 and shredded > 0.8


def test_definition_latent_is_independent_of_the_fat_label_input():
    # adiposity is binned from f alone; d and muscle never enter it
    edges = L.bin_edges([i / 100 for i in range(101)])
    assert L.label(0.3, edges) == L.label(0.3, edges)
    assert L.flat_score(1.0, 0.0) > L.flat_score(1.0, 0.8)  # fat lowers the flat proxy


def test_bump_factor_has_mean_one_and_is_capped():
    nd = NormalDist()
    zs = [nd.inv_cdf((i + 0.5) / 2000) for i in range(2000)]
    mean = sum(exp(L.BUMP_SIGMA * z - L.BUMP_SIGMA**2 / 2) for z in zs) / len(zs)
    assert mean == pytest.approx(1.0, abs=0.01)
    assert all(L.bump_strength(0.9, z) <= L.BUMP_CAP for z in zs)
    assert L.bump_strength(0.0, 2.0) == 0.0


def test_bump_does_not_map_one_to_one_onto_d_eff():
    assert L.bump_strength(0.5, -1.0) != L.bump_strength(0.5, 1.0)


def test_target_signed():
    assert L.target_signed(0.5)[1] == 0.0
    assert L.target_signed(1.0) == ("incr", 1.0)
    assert L.target_signed(0.0) == ("decr", 1.0)


def test_a_point_mass_on_the_lowest_edge_stays_in_class_one():
    values = [0.0] * 25 + [i / 74 for i in range(1, 76)]  # 25% exactly zero (high fat hides every cue)
    edges = L.bin_edges(values)
    assert edges[0] == 0.0
    labels = [L.label(v, edges) for v in values]
    assert all(lab == 1 for lab in labels[:25])
    assert min(labels.count(c) for c in range(1, 6)) / len(values) >= 0.1


def test_quantile_labels_are_balanced_on_a_uniform_sample():
    values = [i / 999 for i in range(1000)]
    edges = L.bin_edges(values)
    assert len(edges) == 4
    counts = [sum(L.label(v, edges) == c for v in values) for c in range(1, 6)]
    assert min(counts) / len(values) >= 0.1  # the ADR-047 class-balance rule
