"""BodyM EDA. Run: uv run python -m ml.src.eda   (numbers go into docs/logs/dev-log.md)"""

import numpy as np
import pandas as pd
from PIL import Image

from ml.src.data.bodym import MEASUREMENTS, SPLITS, load_table

REGION_COLS = ["shoulder_breadth", "chest", "waist", "hip", "bicep", "forearm", "thigh", "calf"]
SEED = 0


def main() -> None:
    pd.set_option("display.width", 160, "display.precision", 2)
    tables = {s: load_table(s) for s in SPLITS}

    print("== sizes, repeats, sex ==")
    for s, df in tables.items():
        per = df.groupby("subject_id").size()
        sex = df.drop_duplicates("subject_id").sex.value_counts(normalize=True).round(2).to_dict()
        print(
            f"{s}: {df.subject_id.nunique()} subjects, {len(df)} photo pairs, photos/subject "
            f"min {per.min()} median {per.median():.0f} max {per.max()}, sex {sex}"
        )

    print("\n== subject overlap between splits (must be 0) ==")
    ids = {s: set(df.subject_id) for s, df in tables.items()}
    for a, b in [("train", "testA"), ("train", "testB"), ("testA", "testB")]:
        print(f"{a} & {b}: {len(ids[a] & ids[b])}")

    subj = {s: df.drop_duplicates("subject_id") for s, df in tables.items()}
    print("\n== subject-level stats (mean / std), by split ==")
    cols = ["height_cm", "weight_kg", *MEASUREMENTS]
    print(pd.concat({s: d[cols].agg(["mean", "std"]).T.stack() for s, d in subj.items()}, axis=1))

    tr = subj["train"]
    print("\n== train: tape `height` vs metadata height_cm (cm) ==")
    print((tr.height - tr.height_cm).describe().round(2).to_dict())

    print("\n== train: Spearman(weight, measurement/height) = what a null model can reach ==")
    rho = {c: tr.weight_kg.corr(tr[c] / tr.height_cm, method="spearman") for c in REGION_COLS}
    print({c: round(v, 2) for c, v in rho.items()})

    print("\n== mask sizes / values (200 random photos per split) ==")
    rng = np.random.default_rng(SEED)
    for s, df in tables.items():
        sample = df.iloc[rng.choice(len(df), 200, replace=False)]
        sizes = {Image.open(p).size for p in (*sample.front_path, *sample.side_path)}
        print(f"{s}: sizes {sizes}")


if __name__ == "__main__":
    main()
