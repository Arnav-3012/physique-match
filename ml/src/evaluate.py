"""P3 evaluation under the pre-registered protocol (ADR-033). Run:
  uv run python -m ml.src.evaluate --split val
  uv run python -m ml.src.evaluate --split testB --siamese s.npz --baseline-b b.npz

A prediction file (.npz) holds: subjects (ids in canonical_subjects order), full (n, n, 8) and
front (n, n, 8) gap predictions for every ordered pair (goal i, current j), and model_hash.
Gate: per baseline, the 95% subject-bootstrap CI of MAE_baseline - MAE_model is > 0 on >= 6 of 8
regions and < 0 on none. Bootstrap = resample subjects, weight pair (i, j) by mult(i) * mult(j).
Baseline NaN pairs are masked on both sides and reported as coverage, never read as delta = 0.
`--split testB` checks the committed manifest and refuses a model hash it has scored before.
Every table also prints two reference floors, fit on train and outside the gate: mae_mean (predict
the train mean, so delta = 0) and mae_null (ridge on height, weight, sex).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.data.pairs import (
    MANIFEST,
    baseline_a,
    canonical_subjects,
    labels,
    pair_hash,
    true_deltas,
)
from ml.src.data.regions import LABELLED_REGIONS, SIDE_REGIONS

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "evals" / "testb_ledger.json"
COEF_DIR = REPO / "services/api/app/analyzer"
SEED = 0
BOOT = 1000
GATE_REGIONS = 6
MIN_REL_EFFECT = 0.10  # power check: the CI must be able to resolve a 10% relative MAE gain


def abs_errors(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|pred - truth| per ordered pair and region; the diagonal (a subject with itself) is NaN."""
    err = np.abs(pred - true_deltas(y))
    i = np.arange(len(y))
    err[i, i] = np.nan
    return err


def region_mae(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.nanmean(abs_errors(pred, y), axis=(0, 1))


def _null_inputs(df: pd.DataFrame) -> pd.DataFrame:
    return df[["height_cm", "weight_kg"]].assign(male=df.sex.eq("male").astype(float))


def null_ridge(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """P2's null model (ADR-024) as a gap for every ordered pair of df, fit on train subjects.
    It knows weight and a goal photo has none, so it is a reference, not a baseline (ADR-029).
    Same ridge as eval_geometric.py, which this module cannot import: it needs PYTHONPATH."""
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(_null_inputs(train), labels(train))
    return true_deltas(model.predict(_null_inputs(df)))


def floors(df: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Per-region MAE of delta for the two reference floors."""
    y = labels(df)
    return pd.DataFrame(
        {
            "region": LABELLED_REGIONS,
            "mae_mean": region_mae(np.zeros((len(y), len(y), y.shape[1])), y),
            "mae_null": region_mae(null_ridge(df, train), y),
        }
    )


def weighted_mae(err: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(B, regions) mean of err over valid pairs, pair (i, j) weighted by w[b, i] * w[b, j]."""
    ok = ~np.isnan(err)
    num = np.einsum("bi,ijr,bj->br", w, np.where(ok, err, 0.0), w, optimize=True)
    den = np.einsum("bi,ijr,bj->br", w, ok.astype(float), w, optimize=True)
    return num / den


def compare(base: np.ndarray, model: np.ndarray, y: np.ndarray, n_boot: int = BOOT) -> pd.DataFrame:
    """Per region: MAE of both, their difference (base - model) with a 95% CI, coverage."""
    e_base, e_model = abs_errors(base, y), abs_errors(model, y)
    off = ~np.eye(len(y), dtype=bool)
    skip = np.isnan(e_base) | np.isnan(e_model)  # pairs either side cannot score
    e_base, e_model = np.where(skip, np.nan, e_base), np.where(skip, np.nan, e_model)
    n = len(y)
    counts = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=n_boot)
    w = np.vstack([np.ones(n), counts]).astype(float)  # row 0 = the point estimate
    b, m = weighted_mae(e_base, w), weighted_mae(e_model, w)
    lo, hi = np.nanpercentile((b - m)[1:], [2.5, 97.5], axis=0)
    t = pd.DataFrame(
        {
            "region": LABELLED_REGIONS,
            "group": ["side" if r in SIDE_REGIONS else "width" for r in LABELLED_REGIONS],
            "mae_base": b[0], "mae_model": m[0], "diff": b[0] - m[0], "lo": lo, "hi": hi,
            "coverage": (~skip[off]).mean(axis=0),
        }
    )  # fmt: skip
    t["underpowered"] = (t.hi - t.lo) / 2 > MIN_REL_EFFECT * t.mae_base
    return t


def verdict(t: pd.DataFrame, need: int = GATE_REGIONS) -> dict:
    wins = int((t.lo > 0).sum())
    worse = t.region[t.hi < 0].tolist()
    return {"wins": wins, "need": need, "significantly_worse": worse,
            "passed": wins >= need and not worse}  # fmt: skip


def claim_test_b(model_hash: str, path: Path = LEDGER) -> None:
    """Record a Test-B look for this model, or refuse if there already was one."""
    ledger = json.loads(path.read_text()) if path.exists() else {}
    if model_hash in ledger:
        raise SystemExit(
            f"model {model_hash[:12]} was scored on Test-B on {ledger[model_hash]}; "
            "a second look needs an ADR (ADR-033)"
        )
    ledger[model_hash] = time.strftime("%Y-%m-%d %H:%M")
    path.write_text(json.dumps(ledger, indent=1))


def load_preds(path: str, subjects: pd.Series) -> dict:
    z = np.load(path, allow_pickle=False)
    if z["subjects"].astype(str).tolist() != subjects.astype(str).tolist():
        raise SystemExit(f"{path}: subjects differ from the canonical list for this split")
    shape = (len(subjects),) * 2 + (len(LABELLED_REGIONS),)
    if z["full"].shape != shape or z["front"].shape != shape:
        raise SystemExit(f"{path}: expected full/front of shape {shape}")
    return {"full": z["full"], "front": z["front"], "hash": str(z["model_hash"])}


def show(title: str, t: pd.DataFrame, refs: pd.DataFrame) -> None:
    v = verdict(t)
    print(f"\n== {title} ==")
    print(t.merge(refs, on="region").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wins {v['wins']}/{len(t)} (need {v['need']}); significantly worse: "
          f"{', '.join(v['significantly_worse']) or 'none'}; "
          f"{'PASS' if v['passed'] else 'FAIL'}")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "testB"), default="val")
    ap.add_argument("--siamese")
    ap.add_argument("--baseline-b")
    args = ap.parse_args()

    df = canonical_subjects(args.split)
    y = labels(df)
    if args.split == "testB":
        if json.loads(MANIFEST.read_text())["hash"] != pair_hash(df):
            raise SystemExit("testB subject/photo list differs from the committed manifest")
    n = len(df)
    print(f"{args.split}: {n} subjects, {n * (n - 1)} ordered pairs, hash {pair_hash(df)[:12]}")

    coef = json.loads((COEF_DIR / "gap_coef.json").read_text())
    coef_w = json.loads((COEF_DIR / "gap_coef_widths.json").read_text())
    a_full, a_width = baseline_a(df, coef), baseline_a(df, coef_w)
    preds = {k: load_preds(p, df.subject_id) for k, p in
             (("siamese", args.siamese), ("b", args.baseline_b)) if p}  # fmt: skip
    if args.split == "testB":
        for p in preds.values():
            claim_test_b(p["hash"])

    zero = np.zeros_like(a_full)
    refs = floors(df, canonical_subjects("train"))
    show(
        "baseline A (2-view) vs predicting 0: MAE_base = zero, MAE_model = A",
        compare(zero, a_full, y),
        refs,
    )
    show(
        "width-only A vs predicting 0 (front-only slice reference)",
        compare(zero, a_width, y),
        refs,
    )
    if "siamese" in preds:
        s = preds["siamese"]
        show("Siamese (full input) vs baseline A", compare(a_full, s["full"], y), refs)
        show("Siamese (front-only goal) vs width-only A  [front-only gate]",
             compare(a_width, s["front"], y), refs)  # fmt: skip
        if "b" in preds:
            show("Siamese (full input) vs baseline B",
                 compare(preds["b"]["full"], s["full"], y), refs)  # fmt: skip
            show("Siamese (front-only goal) vs baseline B  [info]",
                 compare(preds["b"]["front"], s["front"], y), refs)  # fmt: skip


if __name__ == "__main__":
    main()
