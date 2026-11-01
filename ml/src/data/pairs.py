"""Evaluation universe (ADR-032, ADR-033): one canonical photo per subject, per-subject labels and
the committed Test-B manifest.
Run: uv run python -m ml.src.data.pairs     (writes evals/testb_pairs.json; commit it)

A model is scored on every ordered pair (goal i, current j), i != j, of a split's canonical
subjects, so there is no pair sampling and the list is fully set by the subject list. A subject's
photo is a seeded pick among its photos with an unclipped side mask, so the full-input and
front-only slices use the same photo. delta[i, j] = y_i - y_j, y = measurement / height_cm.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.data.bodym import ROOT
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN
from ml.src.data.split import SPLITS_JSON

OUT = ROOT.parent / "features.csv"  # written by ml.src.extract
SEED = 0
MANIFEST = Path(__file__).resolve().parents[3] / "evals" / "testb_pairs.json"


def canonical_subjects(name: str, features: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per subject of split `name`, sorted by subject_id, with y_<region> labels."""
    df = pd.read_csv(OUT) if features is None else features
    df = df[(df.set == name) & ~df.side_clipped]
    one = df.sample(frac=1, random_state=SEED).drop_duplicates("subject_id")
    one = one.sort_values("subject_id").reset_index(drop=True)
    for region in LABELLED_REGIONS:
        one[f"y_{region}"] = one[REGION_COLUMN[region]] / one.height_cm
    return one


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[[f"y_{r}" for r in LABELLED_REGIONS]].to_numpy()


def true_deltas(y: np.ndarray) -> np.ndarray:
    """(n, n, regions); [i, j] = goal i minus current j."""
    return y[:, None, :] - y[None, :, :]


def baseline_a(df: pd.DataFrame, coef: dict) -> np.ndarray:
    """Geometric gap for every ordered pair, same formula as GeometricAnalyzer; NaN = no feature."""
    out = np.empty((len(df), len(df), len(LABELLED_REGIONS)))
    for k, region in enumerate(LABELLED_REGIONS):
        f = df[coef[region]["feature"]].to_numpy()
        out[:, :, k] = np.clip(coef[region]["slope"] * (f[:, None] - f[None, :]), -1, 1)
    return out


def pair_hash(df: pd.DataFrame) -> str:
    rows = "\n".join(f"{s},{p}" for s, p in zip(df.subject_id, df.photo_id, strict=True))
    return hashlib.sha256(rows.encode()).hexdigest()


def main() -> None:
    df = canonical_subjects("testB")
    in_split = len(json.loads(SPLITS_JSON.read_text())["testB"])
    n = len(df)
    MANIFEST.write_text(
        json.dumps(
            {
                "split": "testB",
                "seed": SEED,
                "hash": pair_hash(df),
                "subjects_in_split": in_split,
                "subjects": n,
                "ordered_pairs": n * (n - 1),
                "photos": dict(
                    zip(df.subject_id.astype(str), df.photo_id.astype(str), strict=True)
                ),
            },
            indent=1,
        )  # fmt: skip
    )
    print(f"testB: {n} of {in_split} subjects have an unclipped side photo")
    print(f"{n * (n - 1)} ordered pairs")
    print("hash", pair_hash(df), "->", MANIFEST)


if __name__ == "__main__":
    main()
