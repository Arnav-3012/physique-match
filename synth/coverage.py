"""Label edges and coverage, from the planned sliders alone, BEFORE any render (ADR-049 item 3, ADR-045).

    uv run python -m synth.coverage            # check and report only
    uv run python -m synth.coverage --write    # also write synth/plan/edges.json (once)

Edges are the 20/40/60/80% quantiles of each head's value over the planned sequence's TRAINING partition, once.
`--write` refuses to overwrite an existing edges.json (they are never recomputed) and writes nothing if a check
fails: a mapping that cannot reach the class balance is fixed before the big render, never by dropping classes.
Paste the printed report into docs/logs/dev-log.md.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter

from synth import latents, plan

MIN_CLASS_SHARE = 0.10  # each class 1-5 >= 10% of the training partition, per head (ADR-047 Amendment 1, D)
MIN_CELL_SHARE = 0.07  # each sex-cell >= 7% of its sex's N (item B)
MIN_HELD_OUT = 100  # at N_MIN (ADR-047 item 8)
EDGES_PATH = plan.PLAN_PATH.with_name("edges.json")


def head_values(row: dict) -> dict[str, float]:
    """Every head's value for one identity: definition (bump set), flat proxy (flat set), adiposity."""
    return (
        {f"def_{r}": row[f"d_eff_{r}"] for r in latents.REGIONS}
        | {f"flat_{r}": row[f"flat_{r}"] for r in latents.REGIONS}
        | {"adiposity": row["adip"]}
    )


def compute_edges(rows: list[dict]) -> dict[str, tuple[float, ...]]:
    train = [head_values(r) for r in rows if r["split"] == "train"]
    return {h: latents.bin_edges([v[h] for v in train]) for h in train[0]}


def class_shares(values: list[float], edges: tuple[float, ...]) -> list[float]:
    counts = Counter(latents.label(v, edges) for v in values)
    return [counts[c] / len(values) for c in range(1, 6)]


def check(rows: list[dict], edges: dict[str, tuple[float, ...]]) -> tuple[list[str], list[str]]:
    """(report lines, failures)."""
    lines, failures = [], []
    by_split = {s: [head_values(r) for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    lines.append(f"identities {len(rows)}: " + ", ".join(f"{s} {len(v)}" for s, v in by_split.items()))
    lines.append("class shares per head on train (min >= 10%); zero = share of exact 0:")
    for head, e in edges.items():
        shares = class_shares([v[head] for v in by_split["train"]], e)
        zero = sum(v[head] == 0.0 for v in by_split["train"]) / len(by_split["train"])
        shape = " ".join(f"{s:.3f}" for s in shares)
        lines.append(f"  {head:<12} edges {[round(x, 4) for x in e]}  shares {shape}  zero {zero:.3f}")
        if min(shares) < MIN_CLASS_SHARE:
            failures.append(f"{head}: smallest class share {min(shares):.3f} < {MIN_CLASS_SHARE}")
        if len(set(e)) < len(e):
            failures.append(f"{head}: duplicate edges {e} (a point mass wider than one class)")

    for sex in plan.SEXES:
        sex_rows = [r for r in rows if r["sex"] == sex]
        counts = Counter((r["muscle_level"], r["fat_level"]) for r in sex_rows)
        worst = min(counts.values()) / len(sex_rows)
        lines.append(f"{sex}: {len(sex_rows)} identities, smallest cell share {worst:.3f}")
        if worst < MIN_CELL_SHARE:
            failures.append(f"{sex}: smallest cell share {worst:.3f} < {MIN_CELL_SHARE}")

    prefix = rows[: -(-plan.N_MIN // plan.N_CELLS) * plan.N_CELLS]  # the first whole round at or above N_MIN
    held = sum(r["split"] == "test" for r in prefix)
    lines.append(f"prefix of {len(prefix)} (first whole round >= N_MIN): held out {held}")
    if held < MIN_HELD_OUT:
        failures.append(f"the N_MIN prefix holds out {held} < {MIN_HELD_OUT}")
    per_cell = Counter(r["cell"] for r in prefix if r["split"] == "test")
    lo, hi = min(per_cell.values()), max(per_cell.values())
    lines.append(f"  held out per cell there: min {lo}, max {hi}")
    slice_n = sum(r["slice_top"] for r in rows)
    lines.append(f"lighting slice: {slice_n} held-out identities under {plan.SLICE_RIG}")
    lines.append(f"rig roles: train {plan.TRAIN_RIGS}, slice {plan.SLICE_RIG}")
    return lines, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing edges.json (needs a new ADR)")
    args = ap.parse_args()
    rows = plan.build_plan()
    edges = compute_edges(rows)
    lines, failures = check(rows, edges)
    print("\n".join(lines))
    if failures:
        print("\nFAILED (fix the mapping before the render, never drop classes):\n  " + "\n  ".join(failures))
        return 1
    print("\nall coverage checks passed")
    if args.write:
        if EDGES_PATH.exists() and not args.force:
            print(f"{EDGES_PATH} exists: edges are never recomputed; nothing written")
            return 1
        EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
        EDGES_PATH.write_text(json.dumps({h: list(e) for h, e in edges.items()}, indent=1) + "\n")
        print(f"{EDGES_PATH} sha256 {hashlib.sha256(EDGES_PATH.read_bytes()).hexdigest()}")
        print("labels: `uv run python -m synth.labels synth/plan/identities.csv --out synth/plan/labels_planned.csv`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
