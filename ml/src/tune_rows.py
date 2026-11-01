"""Row-finder health on the BodyM TRAIN split only (never testA/testB; ADR-024).
Run: PYTHONPATH=services/api uv run python -m ml.src.tune_rows

Prints flag rates and the spread of each row (fraction of body height below the crown), and
saves an overlay sheet of 12 front masks to ml/data/preview/rows.png.
"""

import numpy as np
import pandas as pd
from PIL import Image

from app.analyzer.rows import find_rows
from ml.src.data.bodym import ROOT, load_mask
from ml.src.data.split import load_split_table
from ml.src.preprocess import from_bodym

N = 400
SEED = 0
ROW_NAMES = ("shoulder", "armpit", "chest", "waist", "hip", "crotch")
COLOURS = {"armpit": (255, 60, 60), "chest": (255, 160, 0), "waist": (60, 220, 60),
           "hip": (60, 120, 255), "crotch": (255, 230, 0)}  # fmt: skip


def overlay(mask: np.ndarray, rows) -> np.ndarray:
    img = np.repeat(mask[..., None] * 200, 3, axis=2).astype(np.uint8)
    for name, colour in COLOURS.items():
        img[getattr(rows, name)] = colour
    return img


def main() -> None:
    df = load_split_table("train").drop_duplicates("subject_id").sample(N, random_state=SEED)
    records, tiles = [], []
    for _, r in df.iterrows():
        mask = from_bodym(load_mask(r.front_path))
        rows = find_rows(mask)
        span = rows.feet - rows.crown
        records.append(
            {"arms_separated": rows.arms_separated, "crotch_found": rows.crotch_found}
            | {n: (getattr(rows, n) - rows.crown) / span for n in ROW_NAMES}
        )
        if len(tiles) < 12:
            tiles.append(overlay(mask, rows))
    out = pd.DataFrame(records)
    print(f"front masks: {N}, train only")
    print(out[["arms_separated", "crotch_found"]].mean().round(3).to_dict())
    print(out.drop(columns=["arms_separated", "crotch_found"]).describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3))  # fmt: skip
    sheet = np.vstack([np.hstack(tiles[i : i + 4]) for i in range(0, 12, 4)])
    path = ROOT.parent / "preview" / "rows.png"
    path.parent.mkdir(exist_ok=True)
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(path)
    print("overlay ->", path)


if __name__ == "__main__":
    main()
