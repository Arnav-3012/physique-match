"""Gate 3: silhouette -> real waist (cm). Photo level, because one scan is one photo.
Run: PYTHONPATH=services/api uv run python -m ml.src.calibrate_waist

  sil   ridge on [silhouette waist girth in cm, height, weight, sex]   (current person only)
  null  ridge on [height, weight, sex]
Fit on train; val for tuning; testA / testB reported. Writes waist_coef.json (raw-unit
coefficients) with the val residual sd, which sets the body-fat range width.
Derived from BodyM (CC BY-NC 4.0, ADR-002).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.extract import OUT

COEF_PATH = Path(__file__).resolve().parents[2] / "services/api/app/analyzer/waist_coef.json"
SIL = ["waist_cm_proxy", "height_cm", "weight_kg", "male"]
NULL = SIL[1:]
GATE_MAE_CM = 5.0


def fit(X: pd.DataFrame, y: pd.Series):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def raw_coefficients(model, columns: list[str]) -> dict:
    """Fold the scaler into the ridge: waist = intercept + sum(coef_i * x_i) in raw units."""
    scaler, ridge = model[0], model[1]
    coef = ridge.coef_ / scaler.scale_
    return {
        "features": columns,
        "coef": coef.tolist(),
        "intercept": float(ridge.intercept_ - (coef * scaler.mean_).sum()),
    }


def main() -> None:
    df = pd.read_csv(OUT).dropna(subset=["waist_g"])
    df["male"] = (df.sex == "male").astype(float)
    df["waist_cm_proxy"] = df.waist_g * df.height_cm  # scale-free girth x height
    train = df[df.set == "train"]
    sil, null = fit(train[SIL], train.waist), fit(train[NULL], train.waist)

    rows = []
    for name in ("val", "testA", "testB"):
        t = df[df.set == name]
        e_sil = np.abs(sil.predict(t[SIL]) - t.waist)
        e_null = np.abs(null.predict(t[NULL]) - t.waist)
        rows.append({"set": name, "photos": len(t), "mae_sil": e_sil.mean(),
                     "mae_null": e_null.mean(), "p90_sil": e_sil.quantile(0.9)})  # fmt: skip
    res = pd.DataFrame(rows)
    pd.set_option("display.precision", 2)
    print(res.to_string(index=False))

    val = df[df.set == "val"]
    sd = float((sil.predict(val[SIL]) - val.waist).std(ddof=1))
    COEF_PATH.write_text(json.dumps(raw_coefficients(sil, SIL) | {"sd_cm": sd}, indent=1))
    print(f"\nval residual sd {sd:.2f} cm -> {COEF_PATH}")
    a = res[res.set == "testA"].iloc[0]
    ok = a.mae_sil <= GATE_MAE_CM and a.mae_sil < a.mae_null
    print(f"GATE 3 (testA): mae {a.mae_sil:.2f} <= {GATE_MAE_CM} and < null {a.mae_null:.2f}: "
          f"{'PASS' if ok else 'FAIL'}")  # fmt: skip


if __name__ == "__main__":
    main()
