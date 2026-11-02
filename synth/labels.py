"""labels.csv from params.csv and edges.json alone (ADR-053): regenerable without re-rendering, never from the mesh.

    uv run python -m synth.labels synth/out/render/params.csv
    uv run python -m synth.labels synth/plan/identities.csv --out synth/plan/labels_planned.csv

params.csv holds the plan row of every rendered identity (slider values, fitted adiposity proxy, d_eff, flat
proxy, split, the fit's hash), and edges.json holds the class edges fixed before the render, so the labels are a
pure function of the two files. The per-mesh measurements are for validation only (synth/validate_fit.py).

Demo heads (ADR-053 Amendment 4, the ADR-047 2-head fallback): `definition` (overall = the abs head's value, the
muscle-capped d_eff on the macro muscle position, abs-dominated), `definition_flat` (the flat set's overall value)
and `adiposity` (the waist-girth proxy). They reuse the frozen edges of `def_abs`, `flat_abs` and `adiposity`
(the values are identical, so no new edges exist). The six per-region labels stay as `untrained_*` columns.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from synth import coverage, latents

OUT_PATH = Path(__file__).parent / "labels.csv"
TRAINED = {
    "definition": "def_abs",
    "definition_flat": "flat_abs",
    "adiposity": "adiposity",
}  # head -> edges key


def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def label_table(rows: list[dict], edges: dict[str, tuple[float, ...]]) -> tuple[list[str], list[list]]:
    """(header, rows): the trained heads first, then every other edges key as an `untrained_` column."""
    untrained = [h for h in edges if h not in TRAINED.values()]
    header = ["id", "split", *TRAINED, *(f"untrained_{h}" for h in untrained)]
    source = [*TRAINED.values(), *untrained]
    table = []
    for r in rows:
        v = coverage.head_values(r)
        table.append([r["id"], r["split"], *(latents.label(v[s], edges[s]) for s in source)])
    return header, table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    ap.add_argument("--edges", type=Path, default=coverage.EDGES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    edges = {h: tuple(e) for h, e in json.loads(args.edges.read_text()).items()}
    with args.params.open(newline="") as fh:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    header, table = label_table(rows, edges)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(table)
    fits = {r["fit_sha256"] for r in rows}
    print(f"{len(rows)} identities -> {args.out}  sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    print(f"fit hashes in the input: {sorted(fits)}")


if __name__ == "__main__":
    main()
