"""The planned sequence, split, lighting slice and label balance (ADR-045, ADR-053), from the sliders alone: no
Blender. The structural tests use a toy fit so they do not depend on a generated file; the coverage test uses the
real synth/plan/fit.json and is skipped until `python -m synth.fit_labels` has written it."""

from collections import Counter

import pytest

from synth import coverage, latents, plan
from synth import fit as fitmod


def toy_fit() -> fitmod.Fit:
    girth = {s: [[70 + 10 * j / 4 - 3 * i / 4 for j in range(5)] for i in range(5)] for s in plan.SEXES}
    clamps = {s: {"muscle": [0.0, 0.75], "weight": [0.25, 0.75]} for s in plan.SEXES}
    data = fitmod.build(girth, clamps)
    data["sha256"] = fitmod.coefficient_hash(data)
    return fitmod.Fit(data)


@pytest.fixture(scope="module")
def rows():
    return plan.build_plan(fit=toy_fit())


def test_the_plan_is_deterministic():
    assert plan.build_plan(72, fit=toy_fit()) == plan.build_plan(72, fit=toy_fit())


def test_cells_are_round_robin_and_every_whole_round_prefix_is_balanced(rows):
    assert len(rows) == plan.N_PLANNED
    assert all(r["cell"] == r["index"] % plan.N_CELLS for r in rows)
    for k in (1, 7, 34, 100):
        counts = Counter(r["cell"] for r in rows[: k * plan.N_CELLS])
        assert set(counts.values()) == {k}
    for sex in plan.SEXES:  # any prefix is near-balanced by sex
        assert abs(sum(r["sex"] == sex for r in rows[:601]) - 300) <= 1


def test_positions_sit_inside_their_cell_third(rows):
    for r in rows:
        tol = 1e-6  # the plan is rounded to 6 decimals
        assert r["muscle_level"] / 3 - tol <= r["m"] <= (r["muscle_level"] + 1) / 3 + tol
        assert r["fat_level"] / 3 - tol <= r["f"] <= (r["fat_level"] + 1) / 3 + tol
        assert 0 <= r["d"] <= 1


def test_slider_values_are_the_positions_mapped_through_the_clamps(rows):
    clamps = toy_fit().data["clamps"]
    for r in rows[:300]:
        (m_lo, m_hi), (w_lo, w_hi) = clamps[r["sex"]]["muscle"], clamps[r["sex"]]["weight"]
        assert r["muscle_slider"] == pytest.approx(m_lo + r["m"] * (m_hi - m_lo), abs=2e-6)
        assert r["weight_slider"] == pytest.approx(w_lo + r["f"] * (w_hi - w_lo), abs=2e-6)
        assert m_lo - 1e-6 <= r["muscle_slider"] <= m_hi + 1e-6
        assert w_lo - 1e-6 <= r["weight_slider"] <= w_hi + 1e-6


def test_adiposity_rises_with_the_weight_position_inside_a_cell_and_across_cells(rows):
    for sex in plan.SEXES:
        same = [r for r in rows if r["sex"] == sex and r["muscle_level"] == 1]
        low = [r["adip"] for r in same if r["fat_level"] == 0]
        high = [r["adip"] for r in same if r["fat_level"] == 2]
        assert max(low) < min(high) + 0.35 and sum(high) / len(high) > sum(low) / len(low) + 0.4
    assert all(0 <= r["adip"] <= 1 for r in rows)


def test_every_identity_is_in_exactly_one_split_and_ids_are_unique(rows):
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_the_split_is_stratified_by_cell_in_blocks_of_ten(rows):
    for c in range(plan.N_CELLS):
        cell = [r for r in rows if r["cell"] == c]
        for b in range(0, len(cell) - len(plan.BLOCK) + 1, len(plan.BLOCK)):
            block = Counter(r["split"] for r in cell[b : b + len(plan.BLOCK)])
            assert block == Counter(plan.BLOCK)


def test_the_n_min_prefix_holds_out_at_least_100_and_every_cell_has_some(rows):
    prefix = rows[: 34 * plan.N_CELLS]  # first whole round >= N_MIN = 600
    test = [r for r in prefix if r["split"] == "test"]
    assert len(test) >= 100
    assert {r["cell"] for r in test} == set(range(plan.N_CELLS))


def test_rig_roles_the_slice_rig_is_never_a_training_rig(rows):
    assert plan.SLICE_RIG not in plan.TRAIN_RIGS
    assert {r["rig"] for r in rows} == set(plan.TRAIN_RIGS)
    for c in range(plan.N_CELLS):  # the two training rigs alternate inside each cell
        assert {r["rig"] for r in rows if r["cell"] == c} == set(plan.TRAIN_RIGS)


def test_the_lighting_slice_is_the_first_50_held_out_identities_in_order(rows):
    in_slice = [r for r in rows if r["slice_top"]]
    assert len(in_slice) == plan.SLICE_N
    assert all(r["split"] == "test" for r in in_slice)
    first_tests = [r for r in rows if r["split"] == "test"][: plan.SLICE_N]
    assert [r["id"] for r in in_slice] == [r["id"] for r in first_tests]


def test_label_inputs_follow_the_latents(rows):
    for r in rows[:200]:
        assert r["d_eff_abs"] == pytest.approx(latents.d_eff(r["d"], r["adip"], r["m_abs"]), abs=5e-6)
        if r["adip"] >= latents.SOFT_F + 1e-5:
            assert all(r[f"d_eff_{x}"] == 0 for x in latents.REGIONS)  # high adiposity hides every cue
        assert r["m_abs"] == r["m"]  # abs has no muscle target


def test_labels_are_a_pure_function_of_the_plan_columns_and_the_edges(rows):
    edges = coverage.compute_edges(rows)
    again = [{k: v for k, v in r.items() if k not in ("f", "muscle_level", "fat_level")} for r in rows]
    a = [coverage.head_values(r) for r in rows]
    b = [coverage.head_values(r) for r in again]
    assert a == b
    assert [latents.label(a[0][h], edges[h]) for h in edges] == [latents.label(b[0][h], edges[h]) for h in edges]


def test_coverage_edges_use_the_training_partition_only(rows):
    edges = coverage.compute_edges(rows)
    train = [r["adip"] for r in rows if r["split"] == "train"]
    assert edges["adiposity"] == latents.bin_edges(train)


@pytest.mark.skipif(not fitmod.FIT_PATH.exists(), reason="run `python -m synth.fit_labels` first (ADR-053)")
def test_coverage_every_class_of_every_head_is_at_least_10_percent_on_train_with_the_real_fit():
    real = plan.build_plan()
    _, failures = coverage.check(real, coverage.compute_edges(real))
    assert failures == []
