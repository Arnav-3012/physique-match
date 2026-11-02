"""Train the Siamese or baseline B (P3). Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m ml.src.train \
      --config ml/configs/<name>.yaml

A batch is K subjects, each encoded with and without its side view. The best checkpoint is chosen
on val only (mean standardised MAE, full and front-only slices equally) and goes to
ml/data/runs/<config name>/best.pt, next to train.log. Test-B is never read here (ADR-033).

Each epoch line compares train and val in one unit, the standardised Huber of the loss, next to the
constant-prediction floor (delta = 0). Train still carries augmentation and side dropout.
"""

import argparse
import logging
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from ml.src.augment import jitter
from ml.src.data.bodym import ROOT
from ml.src.data.dataset import Views, as_input, blank_side
from ml.src.data.pairs import canonical_subjects
from ml.src.data.regions import LABELLED_REGIONS
from ml.src.evaluate import abs_errors, floors, region_mae
from ml.src.model import ARCHS
from ml.src.predict import predict_pairs

RUNS = ROOT.parent / "runs"


@dataclass
class Config:
    name: str
    arch: str = "siamese"
    rung: str = "last2"
    hidden: int = 256  # 0 = linear head
    subjects_per_batch: int = 32
    epochs: int = 40
    patience: int = 8
    lr_head: float = 0.001
    lr_encoder: float = 0.0001
    weight_decay: float = 0.0001
    side_dropout: float = 0.5
    # Same-subject (i, i) pairs are excluded by default: evaluate.py never scores them (diagonal is
    # NaN), and their target is always 0, so they would only pull the head towards predicting 0.
    self_pairs: bool = False
    augment: bool = True
    overfit: bool = False  # one fixed batch, no val: the check before any full run
    log_every: int = 20
    seed: int = 0


def load_config(path: Path, seed: int | None = None) -> Config:
    """With `seed`, the run gets its own folder: <config name>_seed<seed> (ADR-034)."""
    cfg = Config(name=path.stem, **yaml.safe_load(path.read_text()))
    return cfg if seed is None else replace(cfg, name=f"{cfg.name}_seed{seed}", seed=seed)


def subject_batches(n: int, k: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for start in range(0, n - k + 1, k):
        yield order[start : start + k]


def train_step(model, views: torch.Tensor, y: torch.Tensor, cfg: Config, device: str):
    x = views.to(device).float() / 255
    if cfg.augment:
        x = jitter(x)
    x = as_input(x)
    # One forward pass, so BatchNorm statistics cover both inputs seen at inference.
    e_full, e_front = model.encoder(torch.cat([x, blank_side(x)])).chunk(2)
    return model.loss(e_full, e_front, y.to(device), cfg.side_dropout, cfg.self_pairs)


def standardised_huber(pred: np.ndarray, y: np.ndarray, std: np.ndarray) -> float:
    """The training loss of a gap table, on every ordered pair i != j (NaN diagonal skipped)."""
    e = abs_errors(pred, y) / std
    return float(np.nanmean(np.where(e < 1, 0.5 * e**2, e - 0.5)))


def metrics(full: np.ndarray, front: np.ndarray, y: np.ndarray, std: np.ndarray) -> dict:
    """Both slices weigh equally, as side_dropout = 0.5 does in training."""
    mae_full, mae_front = region_mae(full, y), region_mae(front, y)
    return {
        "mae": float(((mae_full + mae_front) / 2 / std).mean()),
        "huber": (standardised_huber(full, y, std) + standardised_huber(front, y, std)) / 2,
        "mae_full": mae_full,
        "mae_front": mae_front,
    }


def score_val(model, val: Views, device: str) -> dict:
    full, front = predict_pairs(model, val.pick(np.arange(len(val))), device)
    return metrics(full, front, val.y.numpy(), model.delta_std.cpu().numpy())


def floor_val(model, val: Views) -> dict:
    zero = np.zeros((len(val), len(val), len(LABELLED_REGIONS)))
    return metrics(zero, zero, val.y.numpy(), model.delta_std.cpu().numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config, args.seed)

    run_dir = RUNS / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "train.log", mode="w"),
        ],
    )

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train = Views("train")
    val = None if cfg.overfit else Views("val")

    model = ARCHS[cfg.arch](cfg.hidden, pretrained=True)
    model.fit_labels(train.y)
    model.encoder.freeze(cfg.rung)
    model.to(device)
    groups = [{"params": list(model.head.parameters()), "lr": cfg.lr_head}]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": cfg.lr_encoder})
    optim = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = 1 if cfg.overfit else len(train) // cfg.subjects_per_batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps_per_epoch * cfg.epochs)
    logging.info(f"{cfg.name}: {cfg.arch}/{cfg.rung} on {device}, {len(train)} subjects, "
                 f"{steps_per_epoch} steps/epoch")  # fmt: skip
    floor = None if cfg.overfit else floor_val(model, val)

    fixed = np.arange(cfg.subjects_per_batch)
    best, best_epoch, stale, step = float("inf"), -1, 0, 0
    for epoch in range(cfg.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = []
        for subjects in [fixed] if cfg.overfit else subject_batches(
            len(train), cfg.subjects_per_batch, rng
        ):  # fmt: skip
            views = train.pick(subjects, None if cfg.overfit else rng)
            loss = train_step(model, views, train.y[subjects], cfg, device)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            step += 1
            epoch_losses.append(loss.item())
            if step % cfg.log_every == 0:
                logging.info(f"step {step} epoch {epoch} loss {loss.item():.4f}")
        if cfg.overfit:
            continue

        scores = score_val(model, val, device)
        improved = scores["mae"] < best
        logging.info(
            f"epoch {epoch} train_huber {np.mean(epoch_losses):.4f} "
            f"val_huber {scores['huber']:.4f} floor_huber {floor['huber']:.4f} "
            f"val_mae {scores['mae']:.4f} floor_mae {floor['mae']:.4f}"
            f"{' *' if improved else ''} ({time.time() - t0:.0f} s)"
        )
        if improved:
            best, best_epoch, stale = scores["mae"], epoch, 0
            best_maes = scores["mae_full"], scores["mae_front"]
            torch.save(
                {"state": model.state_dict(), "config": asdict(cfg), "epoch": epoch, "val": best},
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= cfg.patience:
                logging.info(f"stopped: no val improvement in {cfg.patience} epochs")
                break

    if cfg.overfit:
        logging.info(f"final loss {loss.item():.6f} (one fixed batch; expect ~0)")
        return
    refs = floors(canonical_subjects("val"), canonical_subjects("train"))
    logging.info(f"best epoch {best_epoch} val_mae {best:.4f}; MAE of delta per region:")
    logging.info(f"  {'region':10s} {'full':>6s} {'front':>6s} {'mean':>6s} {'null':>6s}"
                 "   (mean = predict train mean, null = ridge on height, weight, sex)")  # fmt: skip
    columns = zip(LABELLED_REGIONS, *best_maes, refs.mae_mean, refs.mae_null, strict=True)
    for region, full, front, mean, null in columns:
        logging.info(f"  {region:10s} {full:6.4f} {front:6.4f} {mean:6.4f} {null:6.4f}")


if __name__ == "__main__":
    main()
