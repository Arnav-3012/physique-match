"""Silhouette features for every BodyM photo pair -> ml/data/features.csv (gitignored).
Run: PYTHONPATH=services/api uv run python -m ml.src.extract    (a few minutes, all cores)

Clipped side masks (feet cut by the photo edge) get NaN depths: they are never used for girths.
Rows carry their subject's set (train / val / testA / testB) from ml/data/splits.json.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from app.analyzer.features import extract
from ml.src.data.bodym import ROOT, SPLITS, clipped, load_mask, load_table
from ml.src.data.split import SPLITS_JSON
from ml.src.preprocess import from_bodym

OUT = ROOT.parent / "features.csv"


def one(paths) -> dict:
    front_path, side_path = paths
    side_raw = load_mask(side_path)
    side = None if clipped(side_raw) else from_bodym(side_raw)
    return extract(from_bodym(load_mask(front_path)), side) | {"side_clipped": side is None}


def main() -> None:
    owner = {s: name for name, ids in json.loads(SPLITS_JSON.read_text()).items() for s in ids}
    df = pd.concat([load_table(s) for s in SPLITS], ignore_index=True)
    df["set"] = df.subject_id.map(owner)
    # A row stays only if its source split holds its subject: the shared subject's testB photos
    # (owner = testA) are dropped here instead of being counted in testA.
    from_train = df.split.eq("train") & df.set.isin(["train", "val"])
    df = df[from_train | df.split.eq(df.set)].reset_index(drop=True)
    with ProcessPoolExecutor() as pool:
        feats = list(pool.map(one, zip(df.front_path, df.side_path, strict=True), chunksize=32))
    out = pd.concat([df.drop(columns=["front_path", "side_path"]), pd.DataFrame(feats)], axis=1)
    out.to_csv(OUT, index=False)
    print(f"{len(out)} photo pairs -> {OUT}")
    print(out.groupby("set").agg(photos=("photo_id", "size"), subjects=("subject_id", "nunique")))
    print("side_clipped rate:", out.side_clipped.mean().round(3))


if __name__ == "__main__":
    main()
