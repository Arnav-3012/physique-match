"""Extrapolation check for the reachable frame ceiling (ADR-040, phase-plan M2 step 0).

RFM is inverted at the two pre-registered body-fat ends:
`waist/height = 20 / (64 + 12 * female - BF)`. For each sex and end, that waist/height is compared
with the lowest waistcircumference / stature in ANSUR II of the same sex: below it, the ceiling
carries `extrapolated: true`. Also stored, as
information only: how many subjects of that sex sit below each inverted value. Descriptive, one run;
the ANSUR waist is at the navel and the people are US Army personnel (ADR-036, limitation 3).
Commit this file first, so the recorded commit contains it and `tree_clean` is true. Run:
  uv run python -m ml.src.ceiling_extrapolation
"""

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from ml.src.data.ansur import ROOT, load

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "services/api/app/analyzer/ceiling_extrapolation.json"
VERSION = 1
# ADR-040 body-fat ends in %: `floor` is bodycomp.BF_FLOOR, `athlete_upper` is BF_ATHLETE_UPPER.
# ml must not import app, so tests/test_ceiling_extrapolation.py ties the floor to bodycomp.
BF_ENDS = {
    "male": {"floor": 8.0, "athlete_upper": 13.0},
    "female": {"floor": 18.0, "athlete_upper": 20.0},
}


def waist_at_bf(bf: float, female: bool) -> float:
    """RFM inverted: waist / height at body fat `bf` (height cancels)."""
    return 20 / (64 + 12 * female - bf)


def check(df: pd.DataFrame) -> dict[str, dict]:
    """Per sex: n, lowest waist/stature, and per end the inverted value, flag and count below."""
    out = {}
    for sex, ends in BF_ENDS.items():
        ratio = (df.waist / df.stature)[df.is_male == float(sex == "male")].to_numpy()
        out[sex] = {"n": len(ratio), "min_waist_stature": float(ratio.min()), "ends": {
            end: {
                "bf": bf,
                "waist_stature": (w := waist_at_bf(bf, sex == "female")),
                "extrapolated": bool(w < ratio.min()),
                "n_below": int((ratio < w).sum()),
            }
            for end, bf in ends.items()
        }}  # fmt: skip
    return out


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def main() -> None:
    result = check(load())
    block = {
        "version": VERSION,
        "script_commit": git("rev-parse", "HEAD"),
        "tree_clean": not git("status", "--porcelain"),
        "data": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(ROOT.glob("*.csv"))},
    } | {"sexes": result}  # fmt: skip
    OUT.write_text(json.dumps(block, indent=1))

    print(f"ceiling extrapolation v{VERSION}: commit {block['script_commit'][:12]}, "
          f"tree {'clean' if block['tree_clean'] else 'DIRTY'}")  # fmt: skip
    for sex, r in result.items():
        print(f"{sex}: n {r['n']}, lowest ANSUR waist/stature {r['min_waist_stature']:.3f}")
        for end, e in r["ends"].items():
            print(f"  {end:13} BF {e['bf']:4.1f}  inverted {e['waist_stature']:.3f}  "
                  f"below: {e['n_below']:4d}  extrapolated: {e['extrapolated']}")  # fmt: skip
    print("->", OUT)


if __name__ == "__main__":
    main()
