"""Subject-level splits. train -> train/val by `subject_id` (never by photo); testA/testB are
fixed by BodyM and only read for final numbers. Run: uv run python -m ml.src.data.split"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.data.bodym import ROOT, SPLITS, load_table

SEED = 0
VAL_FRAC = 0.15
SPLITS_JSON = ROOT.parent / "splits.json"


def make_split(
    tables: dict[str, pd.DataFrame], seed: int = SEED, val_frac: float = VAL_FRAC
) -> dict[str, list[str]]:
    """Subject-id lists for train / val / testA / testB, pairwise disjoint.

    val is stratified by sex. A subject present in both test sets is dropped from testB.
    """
    rng = np.random.default_rng(seed)
    subjects = tables["train"].drop_duplicates("subject_id").sort_values("subject_id")
    val: list[str] = []
    for _, ids in subjects.groupby("sex").subject_id:
        ids = ids.to_numpy()
        val += list(rng.permutation(ids)[: round(len(ids) * val_frac)])
    out = {
        "train": sorted(set(subjects.subject_id) - set(val)),
        "val": sorted(val),
        "testA": sorted(set(tables["testA"].subject_id)),
    }
    out["testB"] = sorted(set(tables["testB"].subject_id) - set(out["testA"]))
    assert_disjoint(out)
    return out


def assert_disjoint(split: dict[str, list[str]]) -> None:
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = set(split[a]) & set(split[b])
            if shared:
                raise ValueError(f"{len(shared)} subjects in both {a} and {b}")


def load_split_table(name: str, path: Path = SPLITS_JSON) -> pd.DataFrame:
    """Photo-pair rows for one named split ('train', 'val', 'testA', 'testB')."""
    ids = set(json.loads(path.read_text())[name])
    df = load_table("train" if name in ("train", "val") else name)
    return df[df.subject_id.isin(ids)].reset_index(drop=True)


def main() -> None:
    split = make_split({s: load_table(s) for s in SPLITS})
    SPLITS_JSON.write_text(json.dumps(split, indent=1))
    print({k: len(v) for k, v in split.items()}, "->", SPLITS_JSON)


if __name__ == "__main__":
    main()
