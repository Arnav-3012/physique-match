"""Exemplar selection for the class sheet (ADR-053 Amendment 3): pure Python on a toy fit."""

from synth import coverage, exemplars, latents, plan
from synth import fit as fitmod


def _rows():
    girth = {s: [[70 + 10 * j / 4 - 3 * i / 4 for j in range(5)] for i in range(5)] for s in plan.SEXES}
    clamps = {s: {"muscle": [0.0, 0.75], "weight": [0.25, 0.75]} for s in plan.SEXES}
    data = fitmod.build(girth, clamps)
    data["sha256"] = fitmod.coefficient_hash(data)
    return plan.build_plan(fit=fitmod.Fit(data))


def test_one_training_exemplar_per_class_per_head_per_sex_with_the_right_label():
    rows = _rows()
    edges = coverage.compute_edges(rows)
    selection = exemplars.select(rows, edges)
    by_id = {r["id"]: r for r in rows}
    for sex, heads in selection.items():
        assert set(heads) == set(exemplars.HEADS)
        for head, classes in heads.items():
            assert set(classes) == set(range(1, 6))
            for c, e in classes.items():
                assert e is not None
                row = by_id[e["id"]]
                assert row["sex"] == sex and row["split"] == "train"
                assert latents.label(coverage.head_values(row)[head], edges[head]) == c


def test_the_id_list_is_unique_and_sorted():
    rows = _rows()
    ids = exemplars.ids_of(exemplars.select(rows, coverage.compute_edges(rows)))
    assert ids == sorted(set(ids))
    assert len(ids) <= 2 * len(exemplars.HEADS) * 5
