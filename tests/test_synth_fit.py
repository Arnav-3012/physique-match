"""The fitted adiposity lookup and its pre-registered checks (ADR-053), on a toy table: no Blender, no files."""

import copy
import math

import pytest

from synth import fit, fit_labels

SEXES = ("female", "male")
CLAMPS = {s: {"muscle": [0.0, 0.75], "weight": [0.25, 0.75]} for s in SEXES}


def toy_girth(weight_gain=10.0):
    return {s: [[70 + weight_gain * j / 4 - 3 * i / 4 for j in range(5)] for i in range(5)] for s in SEXES}


def toy_fit(girth=None) -> fit.Fit:
    data = fit.build(girth or toy_girth(), copy.deepcopy(CLAMPS))  # tests mutate it
    data["sha256"] = fit.coefficient_hash(data)
    return fit.Fit(data)


def toy_probe(thigh_by_muscle=lambda i: 38 + i, chest_by_muscle=lambda i: 22 + i):
    rows = [{"thigh_w_cm": thigh_by_muscle(i), "chest_d_cm": chest_by_muscle(i)} for i in range(5) for _ in range(5)]
    return {s: rows for s in SEXES}


def test_girth_of_a_circle_is_pi_times_the_diameter():
    assert fit.girth_cm(30, 30) == pytest.approx(math.pi * 30)
    assert fit.girth_cm(40, 20) < fit.girth_cm(30, 30) * 1.1  # an ellipse is shorter than the bounding circle


def test_the_lookup_is_exact_on_grid_points_and_linear_between():
    f = toy_fit()
    assert f.girth("male", 0.5, 0.5) == pytest.approx(70 + 10 * 2 / 4 - 3 * 2 / 4)
    mid = (f.girth("male", 0.5, 0.5) + f.girth("male", 0.5, 0.75)) / 2
    assert f.girth("male", 0.5, 0.625) == pytest.approx(mid)


def test_adiposity_is_zero_at_the_lowest_weight_and_one_at_the_reference_corner():
    f = toy_fit()
    for sex in SEXES:
        assert f.adiposity(sex, 0.3, 0.25) == 0.0
        top = max(f.adiposity(sex, m, 0.75) for m in fit.lattice([0.0, 0.75]))
        assert top == pytest.approx(1.0)


def test_adiposity_rises_with_weight_at_fixed_muscle():
    f = toy_fit()
    ws = fit.lattice([0.25, 0.75])
    for m in (0.0, 0.4, 0.75):
        vals = [f.adiposity("female", m, w) for w in ws]
        assert vals == sorted(vals)


def test_the_coefficient_hash_covers_the_tables_and_clamps_but_not_the_check_report():
    a = toy_fit()
    b = toy_fit()
    b.data["checks"] = {"anything": True}
    assert fit.coefficient_hash(a.data) == fit.coefficient_hash(b.data)
    assert fit.coefficient_hash(a.data) != fit.coefficient_hash(toy_fit(toy_girth(12.0)).data)


def test_the_checks_pass_on_a_monotone_table():
    report, failures = fit_labels.run_checks(toy_probe(), toy_fit().data)
    assert failures == []
    assert all(report["male"].values())


def test_a_girth_that_falls_with_weight_fails_the_monotone_check():
    girth = toy_girth()
    girth["male"][2][3] = girth["male"][2][2] - 1.0  # weight 0.75 narrower than 0.5 at muscle 0.5
    _, failures = fit_labels.run_checks(toy_probe(), toy_fit(girth).data)
    assert any("male" in x and "C1" in x for x in failures)


def test_a_measured_muscle_effect_that_dips_is_recorded_but_does_not_fail_the_cap():
    u_shaped = lambda i: 38 + (2 - i) ** 2  # noqa: E731 - soft mass at low muscle, as the probe found for the male
    report, failures = fit_labels.run_checks(toy_probe(thigh_by_muscle=u_shaped), toy_fit().data)
    assert failures == []  # the cap uses the muscle position, which is monotone by construction
    assert report["male"]["cap_calibrated_to_measured_muscle_effect_would_be_monotone"] is False
    assert report["male"]["C4_cap_monotone_up_in_muscle"] is True


def test_clamps_outside_zero_one_are_rejected():
    data = toy_fit().data
    data["clamps"]["male"]["weight"] = [0.25, 1.2]
    _, failures = fit_labels.run_checks(toy_probe(), data)
    assert any("clamps" in x for x in failures)
