"""ADR-038 decision point: does the V-taper index gap carry signal beyond the waist gap?

Per sex, over the val same-sex ordered pairs, regress the true index gap on the true waist gap
(OLS with intercept) and take unique_share = 1 - R^2. Both gaps come from the ground-truth BodyM
ratios (measurement / height_cm), no model output. unique_share = min over sexes; keep the index gap
as a planner input (P6) only if it is >= THRESHOLD, which was fixed before this was computed.
Computed once. Commit this file first, so the recorded commit contains it and `tree_clean` is true.
Run:  uv run python -m ml.src.unique_share
Reads val only, never Test-B. Writes evals/unique_share.json.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.data.pairs import OUT, canonical_subjects

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "evals" / "unique_share.json"
VERSION = 1
THRESHOLD = 0.2  # ADR-038, not changed after the run


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def unique_share(shoulders: np.ndarray, waist: np.ndarray) -> float:
    """1 - R^2 of the index gap on the waist gap over all ordered pairs i != j of one sex.
    With one regressor and an intercept, R^2 is the squared correlation."""
    index = shoulders / waist
    i, j = np.nonzero(~np.eye(len(waist), dtype=bool))
    r = np.corrcoef(waist[i] - waist[j], index[i] - index[j])[0, 1]
    return float(1 - r**2)


def keep_index_gap(shares: dict[str, float]) -> bool:
    return min(shares.values()) >= THRESHOLD


def check(df: pd.DataFrame) -> dict:
    """df: canonical val subjects with `sex`, `y_shoulders`, `y_waist`."""
    by_sex = {}
    for sex, one in df.groupby(df.sex.str.lower()):
        by_sex[sex] = {
            "n": len(one),
            "unique_share": unique_share(one.y_shoulders.to_numpy(), one.y_waist.to_numpy()),
        }
    shares = {sex: v["unique_share"] for sex, v in by_sex.items()}
    return {"sexes": by_sex, "unique_share": min(shares.values()), "keep": keep_index_gap(shares)}


def main() -> None:
    val = canonical_subjects("val")
    if set(val.sex.str.lower()) != {"male", "female"}:
        raise SystemExit(f"unexpected sex values in val: {sorted(set(val.sex))}")
    result = check(val)
    with OUT.open("rb") as f:
        features_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    record = {
        "version": VERSION,
        "script_commit": git("rev-parse", "HEAD"),
        "tree_clean": not git("status", "--porcelain"),
        "features_csv_sha256": features_sha256,
        "n_val_subjects": len(val),
        "threshold": THRESHOLD,
    } | result
    RESULT.write_text(json.dumps(record, indent=1) + "\n")

    tree = "clean" if record["tree_clean"] else "DIRTY"
    print(f"unique share v{VERSION}: commit {record['script_commit'][:12]}, tree {tree}")
    for sex, v in result["sexes"].items():
        print(f"{sex:7s} n {v['n']:4d}  unique_share {v['unique_share']:.3f}")
    verdict = "KEEP as a planner input" if result["keep"] else "DROP (display-only)"
    print(f"min {result['unique_share']:.3f} vs threshold {THRESHOLD}: {verdict}")


if __name__ == "__main__":
    main()
