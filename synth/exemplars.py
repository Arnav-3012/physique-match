"""One exemplar identity per class per head per sex, for the class-exemplar sheet (ADR-053 Amendment 3).

    uv run python -m synth.exemplars          # needs synth/plan/edges.json (coverage --write)

Class shares are 20% by construction (quantile cutpoints), which says nothing about whether adjacent classes LOOK
different. The sheet is how the user checks that before the full render. For each sex and each bump-set head the
exemplar of a class is the training identity of that class whose value is closest to the class's median value.
Writes synth/plan/exemplars.json and exemplar_ids.txt (the ids, for `render.py --ids`).
"""

import json
import statistics

from synth import coverage, latents, plan

OUT_JSON = plan.PLAN_PATH.with_name("exemplars.json")
OUT_IDS = plan.PLAN_PATH.with_name("exemplar_ids.txt")
HEADS = [f"def_{r}" for r in latents.REGIONS] + ["adiposity"]  # the bump set's heads; the flat heads are not shown


def select(rows: list[dict], edges: dict[str, tuple[float, ...]]) -> dict:
    out: dict = {}
    for sex in plan.SEXES:
        mine = [(r, coverage.head_values(r)) for r in rows if r["sex"] == sex and r["split"] == "train"]
        out[sex] = {}
        for head in HEADS:
            out[sex][head] = {}
            for c in range(1, 6):
                cands = [(r, v[head]) for r, v in mine if latents.label(v[head], edges[head]) == c]
                if not cands:
                    out[sex][head][c] = None
                    continue
                mid = statistics.median(v for _, v in cands)
                r, v = min(cands, key=lambda rv: (abs(rv[1] - mid), rv[0]["index"]))
                out[sex][head][c] = {"id": r["id"], "value": round(v, 4), "muscle_slider": r["muscle_slider"],
                                     "weight_slider": r["weight_slider"], "adip": r["adip"], "d": r["d"]}  # fmt: skip
    return out


def ids_of(selection: dict) -> list[str]:
    return sorted({e["id"] for sex in selection.values() for head in sex.values() for e in head.values() if e})


def main() -> None:
    edges = {h: tuple(e) for h, e in json.loads(coverage.EDGES_PATH.read_text()).items()}
    selection = select(plan.build_plan(), edges)
    OUT_JSON.write_text(json.dumps(selection, indent=1) + "\n")
    ids = ids_of(selection)
    OUT_IDS.write_text("\n".join(ids) + "\n")
    missing = [(s, h, c) for s, hs in selection.items() for h, cs in hs.items() for c, e in cs.items() if e is None]
    print(f"{len(ids)} exemplar identities -> {OUT_IDS}; selection -> {OUT_JSON}; empty classes: {missing or 'none'}")


if __name__ == "__main__":
    main()
