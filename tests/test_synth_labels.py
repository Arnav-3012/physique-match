"""The demo label table (ADR-053 Amendment 4): two trained heads plus untrained per-region columns."""

from synth import coverage, labels, latents, plan
from synth import fit as fitmod


def _rows():
    girth = {s: [[70 + 10 * j / 4 - 3 * i / 4 for j in range(5)] for i in range(5)] for s in plan.SEXES}
    clamps = {s: {"muscle": [0.0, 0.75], "weight": [0.25, 0.75]} for s in plan.SEXES}
    data = fitmod.build(girth, clamps)
    data["sha256"] = fitmod.coefficient_hash(data)
    return plan.build_plan(fit=fitmod.Fit(data))


def test_the_trained_heads_reuse_the_frozen_abs_flat_and_adiposity_edges():
    rows = _rows()
    edges = coverage.compute_edges(rows)
    header, table = labels.label_table(rows, edges)
    assert header[:5] == ["id", "split", "definition", "definition_flat", "adiposity"]
    col = {name: i for i, name in enumerate(header)}
    for r, out in zip(rows, table, strict=True):
        v = coverage.head_values(r)
        assert out[col["definition"]] == latents.label(v["def_abs"], edges["def_abs"])
        assert out[col["definition_flat"]] == latents.label(v["flat_abs"], edges["flat_abs"])
        assert out[col["adiposity"]] == latents.label(v["adiposity"], edges["adiposity"])


def test_per_region_labels_are_kept_but_marked_untrained():
    rows = _rows()
    header, _ = labels.label_table(rows, coverage.compute_edges(rows))
    regions = [h for h in header if h.startswith("untrained_")]
    assert len(regions) == 10  # chest, shoulders, arms, back, legs, each for the bump set and the flat set
    assert "untrained_def_chest" in regions and "untrained_def_abs" not in regions


def test_labels_depend_only_on_the_row_values_and_the_edges():
    rows = _rows()
    edges = coverage.compute_edges(rows)
    stripped = [{k: v for k, v in r.items() if k.startswith(("d_eff_", "flat_", "adip", "id", "split"))} for r in rows]
    assert labels.label_table(rows, edges) == labels.label_table(stripped, edges)
