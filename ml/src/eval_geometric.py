"""Gate 2 (ADR-024, two-part criterion): per labelled region, on subjects (photos averaged).
Run: PYTHONPATH=services/api uv run python -m ml.src.eval_geometric [--final]

Target = measurement / height_cm (scale-free). Columns:
  sil   Spearman of the raw scale-free silhouette feature (rank-based, no fit)
  null  ridge on [height, weight, sex], fit on train
  comb  ridge on [feature, height, weight, sex], fit on train
  d     sil - null            (the original gate; kept for the record)
  c     comb - null, with a 95% subject bootstrap interval (does the silhouette add signal?)
Default prints val only: all tuning happens there. --final adds testA / testB and the gate.
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analyzer.features import REGION_FEATURE
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN
from ml.src.extract import OUT

SEED = 0
BOOT = 1000
GATE_RHO = 0.6
GATE_REGIONS = 6
NULL_COLS = ["height_cm", "weight_kg", "male"]
GIRTH_PREFIX = {"chest": "chest", "waist": "waist", "hips": "hip", "thigh": "thigh", "calf": "calf"}


def load_subjects() -> pd.DataFrame:
    df = pd.read_csv(OUT)
    first = df.groupby("subject_id")[["sex", "set"]].first()
    num = df.drop(columns=["split", "photo_id", "sex", "set"]).groupby("subject_id").mean()
    subj = num.join(first)
    subj["male"] = (subj.sex == "male").astype(float)
    return subj


def rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


def ridge(X: pd.DataFrame, y: pd.Series):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def evaluate(region: str, subj: pd.DataFrame, sets, rng: np.random.Generator) -> list[dict]:
    feat = REGION_FEATURE[region]
    d = subj.dropna(subset=[feat])
    d = d.assign(y=d[REGION_COLUMN[region]] / d.height_cm)
    train = d[d.set == "train"]
    null = ridge(train[NULL_COLS], train.y)
    comb = ridge(train[[feat, *NULL_COLS]], train.y)
    out = []
    for name in sets:
        t = d[d.set == name]
        sil, y = t[feat].to_numpy(), t.y.to_numpy()
        p_null, p_comb = null.predict(t[NULL_COLS]), comb.predict(t[[feat, *NULL_COLS]])
        idx = rng.integers(0, len(t), (BOOT, len(t)))
        c_boot = [rho(p_comb[i], y[i]) - rho(p_null[i], y[i]) for i in idx]
        out.append(
            {
                "region": region, "set": name, "n": len(t),
                "sil": rho(sil, y), "null": rho(p_null, y), "comb": rho(p_comb, y),
                "d": rho(sil, y) - rho(p_null, y), "c": rho(p_comb, y) - rho(p_null, y),
                "c_lo": np.percentile(c_boot, 2.5), "c_hi": np.percentile(c_boot, 97.5),
            }
        )  # fmt: skip
    return out


def components(subj: pd.DataFrame) -> pd.DataFrame:
    """Width vs depth vs girth alone, val only: which half of a weak girth is broken."""
    val, rows = subj[subj.set == "val"], []
    for region, p in GIRTH_PREFIX.items():
        d = val.dropna(subset=[f"{p}_d"])
        y = d[REGION_COLUMN[region]] / d.height_cm
        rows.append(
            {"region": region, "n": len(d)} | {k: rho(d[f"{p}_{k}"], y) for k in ("w", "d", "g")}
        )
    return pd.DataFrame(rows)


def main() -> None:
    final = "--final" in sys.argv
    sets = ("val", "testA", "testB") if final else ("val",)
    subj = load_subjects()
    rng = np.random.default_rng(SEED)
    res = pd.DataFrame([r for reg in LABELLED_REGIONS for r in evaluate(reg, subj, sets, rng)])
    pd.set_option("display.width", 200, "display.precision", 3)
    for name in sets:
        print(f"\n== {name} ==")
        print(res[res.set == name].drop(columns="set").to_string(index=False))
    print("\n== val: width / depth / girth alone (rho) ==")
    print(components(subj).to_string(index=False))

    for name in sets[1:]:  # --final: testA is the gate set, testB is the tighter in-the-wild check
        t = res[res.set == name]
        a = t[t.sil >= GATE_RHO]
        b = t[t.c_lo > 0]
        print(f"\nGATE {name} (a) silhouette-only rho >= {GATE_RHO}: {len(a)}/{len(t)}"
              f" (need {GATE_REGIONS}): {', '.join(a.region)}")  # fmt: skip
        print(f"GATE {name} (b) comb beats null, interval excludes 0: {len(b)}/{len(t)}"
              f" (need {GATE_REGIONS}): {', '.join(b.region)}")  # fmt: skip
    print("\nfeature NaN rate (subject level):")
    feats = [REGION_FEATURE[r] for r in LABELLED_REGIONS]
    print(subj[feats].isna().groupby(subj.set).mean().round(3).to_string())


if __name__ == "__main__":
    main()
