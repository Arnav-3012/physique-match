"""Per-region linear map: scale-free silhouette feature -> measurement / height (ADR-024).
Fit on TRAIN subjects only; rho_val (val subjects) feeds the per-region confidence.
Run: PYTHONPATH=services/api uv run python -m ml.src.fit_gap
Writes services/api/app/analyzer/gap_coef.json. Derived from BodyM (CC BY-NC 4.0, ADR-002).
"""

import json
from pathlib import Path

import numpy as np

from app.analyzer.features import REGION_FEATURE
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN
from ml.src.eval_geometric import load_subjects, rho

OUT = Path(__file__).resolve().parents[2] / "services/api/app/analyzer/gap_coef.json"


def main() -> None:
    subj = load_subjects()
    coef = {}
    for region in LABELLED_REGIONS:
        feat = REGION_FEATURE[region]
        d = subj.dropna(subset=[feat])
        y = d[REGION_COLUMN[region]] / d.height_cm
        train, val = d.set == "train", d.set == "val"
        slope, intercept = np.polyfit(d[feat][train], y[train], 1)
        coef[region] = {
            "feature": feat,
            "slope": float(slope),
            "intercept": float(intercept),
            "rho_val": rho(d[feat][val], y[val]),
        }
        print(f"{region:10s} {feat:12s} slope {slope:6.3f}  rho_val {coef[region]['rho_val']:.3f}")
    OUT.write_text(json.dumps(coef, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
