"""Metrics, the gate and the per-group reporting rule for the definition model (ADR-045, ADR-053 Amendment 4).

    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m ml.src.definition.evaluate --run definition_finetune

Gate, unchanged and not loosened: Spearman >= 0.8 and MAE <= 0.6 (in classes) on the TEST partition, per head.
A fail is reported as a fail. The test split is scored once per final candidate: a ledger in the run folder refuses
a second scoring without --again. The lighting slice (the first 50 test identities rendered under `top`) is
reported beside it. The functions above `main` need only numpy and scipy.
"""

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from ml.src.definition.items import HEADS, Item

GATE_RHO, GATE_MAE = 0.8, 0.6
MIN_GROUP = 30  # a group is reported only with at least this many test identities (ADR-045 item 5)


def head_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """pred, true: classes for one head. Spearman is NaN when either side is constant."""
    rho = float(spearmanr(pred, true).statistic) if len(set(pred)) > 1 and len(set(true)) > 1 else float("nan")
    return {"n": int(len(true)), "spearman": rho, "mae": float(np.abs(pred - true).mean()),
            "exact": float((pred == true).mean())}  # fmt: skip


def gate(metrics: dict[str, float]) -> bool:
    return bool(metrics["spearman"] >= GATE_RHO and metrics["mae"] <= GATE_MAE)  # NaN fails


def groupings(items: list[Item]) -> dict[str, list[int]]:
    """Index lists for every reporting group, finest first: cell, sex x muscle, sex x fat, sex, pooled."""
    out: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        for key in (f"cell {it.cell}", f"{it.sex} x muscle {it.muscle_level}", f"{it.sex} x fat {it.fat_level}",
                    f"sex {it.sex}", "pooled"):  # fmt: skip
            out[key].append(i)
    return out


def group_report(items: list[Item], pred: np.ndarray, true: np.ndarray) -> list[dict]:
    """Per group with >= MIN_GROUP identities, per head. pred, true: (N, 2) classes."""
    rows = []
    for name, idx in groupings(items).items():
        if len(idx) < MIN_GROUP:
            continue
        for h, head in enumerate(HEADS):
            rows.append({"group": name, "head": head, **head_metrics(pred[idx, h], true[idx, h])})
    return rows


def predict(model, dataset, device: str, batch_size: int = 64) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    from ml.src.definition.model import predict_class

    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            out.append(predict_class(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    import torch

    from ml.src.definition.data import DefinitionDataset
    from ml.src.definition.items import load_items
    from ml.src.definition.model import DefinitionNet
    from ml.src.definition.train import RUNS, device_name

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a folder under ml/data/runs with best.pt and config.json")
    ap.add_argument("--again", action="store_true", help="score the test split a second time (log why)")
    args = ap.parse_args()
    run = RUNS / args.run
    cfg = json.loads((run / "config.json").read_text())
    ckpt = run / "best.pt"
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    ledger = run / "test_scored.json"
    if ledger.exists() and not args.again:
        raise SystemExit(f"{ledger} exists: the test split is scored once per final candidate (ADR-045)")

    device = device_name()
    model = DefinitionNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    items = load_items(cfg["labels_csv"], cfg["params_csv"], cfg["set_name"])
    test = [i for i in items if i.split == "test"]
    ds = DefinitionDataset(test, cfg["images_dir"], cfg["set_name"], cfg["height"])
    pred, true = (
        predict(model, ds, device),
        np.array([[i.definition - 1, i.adiposity - 1] for i in test]),
    )
    print(f"test identities {len(test)}  checkpoint sha256 {sha}")
    for row in group_report(test, pred, true):
        if row["group"] in ("pooled", "sex female", "sex male"):
            print(f"{row['group']:<14}{row['head']:<11} n {row['n']:>4}  rho {row['spearman']:.3f}  "
                  f"MAE {row['mae']:.3f}  exact {row['exact']:.3f}  {'PASS' if gate(row) else 'FAIL'}")  # fmt: skip
    verdicts = {h: gate(head_metrics(pred[:, k], true[:, k])) for k, h in enumerate(HEADS)}
    print(f"GATE (pooled, rho >= {GATE_RHO}, MAE <= {GATE_MAE}): {verdicts}")

    sliced = [i for i in test if i.slice_top and cfg["set_name"] == "bump"]
    if sliced:
        sds = DefinitionDataset(sliced, cfg["images_dir"], "bump", cfg["height"], rig="top")
        spred, strue = (
            predict(model, sds, device),
            np.array([[i.definition - 1, i.adiposity - 1] for i in sliced]),
        )
        for k, h in enumerate(HEADS):
            m = head_metrics(spred[:, k], strue[:, k])
            print(f"lighting slice {h:<11} n {m['n']}  rho {m['spearman']:.3f}  MAE {m['mae']:.3f}")
    ledger.write_text(json.dumps({"checkpoint_sha256": sha, "gate": verdicts}))


if __name__ == "__main__":
    main()
