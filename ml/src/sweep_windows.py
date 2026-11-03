"""Pick arm and shoulder row windows on the TRAIN split (ADR-024); val stays untouched.
Run: PYTHONPATH=services/api uv run python -m ml.src.sweep_windows

Spearman of the feature vs measurement/height, one photo per subject (600 subjects), for
  arm thickness over a 0.07-body-height window starting `lo` below the armpit
  shoulder span `k` body heights above the armpit
"""

import numpy as np
import pandas as pd

from app.analyzer.features import _at, _span, arm_thickness
from app.analyzer.rows import find_rows
from ml.src.data.bodym import load_mask
from ml.src.data.split import load_split_table
from ml.src.eval_geometric import rho
from ml.src.preprocess import body_px, from_bodym

N = 600
SEED = 0
WINDOW = 0.07
LOS = np.arange(0.0, 0.241, 0.02)
KS = np.arange(0.0, 0.101, 0.01)


def main() -> None:
    df = load_split_table("train").drop_duplicates("subject_id").sample(N, random_state=SEED)
    masks = [from_bodym(load_mask(p)) for p in df.front_path]
    rows = [find_rows(m) for m in masks]
    bpx = [body_px(m) for m in masks]
    cols = ("bicep", "forearm", "shoulder_breadth")
    targets = {c: (df[c] / df.height_cm).to_numpy() for c in cols}

    def score(values, target):
        v = np.array(values, float)
        ok = ~np.isnan(v)
        return rho(v[ok], target[ok]), 1 - ok.mean()

    arms = []
    for lo in LOS:
        t = [
            arm_thickness(m, r.armpit + round(lo * b), r.armpit + round((lo + WINDOW) * b)) / b
            for m, r, b in zip(masks, rows, bpx, strict=True)
        ]
        (rb, nan), (rf, _) = score(t, targets["bicep"]), score(t, targets["forearm"])
        arms.append({"lo": round(lo, 2), "rho_bicep": rb, "rho_forearm": rf, "nan": nan})
    shoulders = []
    for k in KS:
        s = [_at(m, r.armpit - round(k * b), _span) / b for m, r, b in zip(masks, rows, bpx, strict=True)]
        rho_s = score(s, targets["shoulder_breadth"])[0]
        shoulders.append({"k": round(k, 2), "rho_shoulder": rho_s})
    pd.set_option("display.precision", 3)
    print(f"train, {N} subjects, window {WINDOW} body heights\n")
    print(pd.DataFrame(arms).to_string(index=False))
    print("\nshoulder span k body heights above the armpit:")
    print(pd.DataFrame(shoulders).to_string(index=False))


if __name__ == "__main__":
    main()
