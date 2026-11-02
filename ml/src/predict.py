"""Checkpoint -> the prediction file evaluate.py scores (full and front (n, n, 8), model_hash). Run:
uv run --group train python -m ml.src.predict --ckpt ml/data/runs/<name>/best.pt --split val
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from ml.src.data.dataset import Views, as_input, blank_side
from ml.src.data.regions import LABELLED_REGIONS
from ml.src.model import ARCHS

BATCH = 64


def _embed(model, x: torch.Tensor, device: str) -> torch.Tensor:
    return torch.cat([model.encoder(b.to(device)) for b in x.split(BATCH)])


def _table(model, e_goal: torch.Tensor, e_current: torch.Tensor) -> np.ndarray:
    rows = [model.gap(g.expand(len(e_current), -1), e_current) for g in e_goal]
    return torch.stack(rows).cpu().numpy()


@torch.no_grad()
def predict_pairs(model, views: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    """uint8 (n, 2, H, W) -> gap for every ordered pair (goal i, current j), (n, n, regions):
    both views on both sides, and front-only goal. Current always keeps its side view."""
    model.eval()
    x = as_input(views.float() / 255)
    e_full, e_front = _embed(model, x, device), _embed(model, blank_side(x), device)
    return _table(model, e_full, e_full), _table(model, e_front, e_full)


@torch.no_grad()
def person_values(model, views: torch.Tensor, device: str) -> tuple[np.ndarray, np.ndarray]:
    """SiameseSubtract only. uint8 (n, 2, H, W) -> each person's predicted measurement / height,
    (n, regions): with both views, and with the side view blanked (a front-only goal)."""
    model.eval()
    x = as_input(views.float() / 255)
    full = model.value(_embed(model, x, device))
    front = model.value(_embed(model, blank_side(x), device))
    return full.cpu().numpy(), front.cpu().numpy()


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = ARCHS[ckpt["config"]["arch"]](ckpt["config"]["hidden"], pretrained=False)
    model.load_state_dict(ckpt["state"])
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "testB"), default="val")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Views(args.split)
    model = load_checkpoint(args.ckpt, device)
    full, front = predict_pairs(model, data.pick(np.arange(len(data))), device)
    with args.ckpt.open("rb") as f:
        model_hash = hashlib.file_digest(f, "sha256").hexdigest()

    out = args.ckpt.with_name(f"{args.split}.npz")
    np.savez(out, subjects=np.array(data.subjects), full=full, front=front, model_hash=model_hash)
    print(f"{args.split}: {len(data)} subjects, model {model_hash[:12]} -> {out}")
    # Measured, not enforced: a perfect gap model has f(i, j) = -f(j, i).
    antisym = np.abs(full + full.transpose(1, 0, 2)).mean(axis=(0, 1))
    print("antisymmetry |f(i,j) + f(j,i)|:")
    for region, value in zip(LABELLED_REGIONS, antisym, strict=True):
        print(f"  {region:10s} {value:.4f}")


if __name__ == "__main__":
    main()
