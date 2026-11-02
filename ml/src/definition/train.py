"""Train the definition model (P4). Run, in this order (ADR-047 item 10, CLAUDE.md training rules):

  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m ml.src.definition.train \\
      --config ml/configs/definition_overfit.yaml
  ... --config ml/configs/definition_probe.yaml       # frozen ImageNet trunk, heads only
  ... --config ml/configs/definition_finetune.yaml    # everything, starting from the probe's best.pt

Rungs: `overfit` trains one fixed batch with no augmentation and must reach (near) perfect accuracy on both heads
before any full run; `probe` freezes the trunk; `finetune` unfreezes it. The best checkpoint is chosen on val only
(mean MAE of the two heads) and goes to ml/data/runs/<config name>/best.pt next to train.log and config.json. The
test split is never read here (ADR-045); evaluate.py scores it once.

`max_round` limits training to the identities of the first N rendered rounds, for the learning curve at 25 / 50 /
100% of the rendered set (ADR-047 item 9): same val, same seeds.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ml.src.definition.data import DefinitionDataset
from ml.src.definition.evaluate import head_metrics, predict
from ml.src.definition.items import HEADS, load_items
from ml.src.definition.model import DefinitionNet, coral_loss, predict_class

RUNS = Path(__file__).resolve().parents[2] / "data" / "runs"  # ml/data/runs, gitignored
OVERFIT_ACCURACY = 0.99  # the one-batch check passes when both heads are this exact on the batch


@dataclass
class Config:
    name: str
    rung: str = "probe"  # overfit | probe | finetune
    set_name: str = "bump"  # bump (set B, the demo model) or flat (set A)
    images_dir: str = "synth/out/render/images"
    params_csv: str = "synth/out/render/params.csv"
    labels_csv: str = "synth/labels.csv"
    height: int = 256  # model input height; width = 3/4 of it (ADR-050 assumed 224-256)
    batch_size: int = 32
    epochs: int = 30
    patience: int = 6
    lr_head: float = 0.001
    lr_backbone: float = 0.0001
    weight_decay: float = 0.0001
    max_round: int | None = None
    init_from: str | None = None  # a run folder name whose best.pt starts the fine-tune
    pretrained: bool = True
    augment: bool = True
    log_every: int = 20  # steps
    seed: int = 0


def load_config(path: Path, seed: int | None = None) -> Config:
    """With `seed`, the run gets its own folder: <config name>_seed<seed>."""
    cfg = Config(name=path.stem, **yaml.safe_load(path.read_text()))
    return cfg if seed is None else replace(cfg, name=f"{cfg.name}_seed{seed}", seed=seed)


def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def make_loader(items, cfg: Config, train: bool) -> DataLoader:
    ds = DefinitionDataset(items, cfg.images_dir, cfg.set_name, cfg.height, train=train and cfg.augment, seed=cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=0, drop_last=False,
                      generator=torch.Generator().manual_seed(cfg.seed))  # fmt: skip


def val_metrics(model, items, cfg: Config, device: str) -> tuple[dict, float]:
    ds = DefinitionDataset(items, cfg.images_dir, cfg.set_name, cfg.height)
    pred = predict(model, ds, device)
    true = np.array([[i.definition - 1, i.adiposity - 1] for i in items])
    per_head = {h: head_metrics(pred[:, k], true[:, k]) for k, h in enumerate(HEADS)}
    return per_head, float(np.mean([m["mae"] for m in per_head.values()]))


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = device_name()
    out = RUNS / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log", mode="w"),
                        logging.StreamHandler()])  # fmt: skip
    log = logging.getLogger("definition")

    items = load_items(cfg.labels_csv, cfg.params_csv, cfg.set_name)
    train_items = [i for i in items if i.split == "train" and (cfg.max_round is None or i.round < cfg.max_round)]
    val_items = [i for i in items if i.split == "val"]
    log.info("device %s  rung %s  set %s  train %d  val %d  (test not read)", device, cfg.rung, cfg.set_name,
             len(train_items), len(val_items))  # fmt: skip
    if not train_items:
        raise SystemExit("no training identities: has the render written params.csv, and does labels.csv cover it?")

    model = DefinitionNet(pretrained=cfg.pretrained).to(device)
    if cfg.init_from:
        model.load_state_dict(torch.load(RUNS / cfg.init_from / "best.pt", map_location=device))
    if cfg.rung == "probe":
        model.freeze_trunk()
    groups = [{"params": model.heads.parameters(), "lr": cfg.lr_head}]
    if cfg.rung != "probe":
        groups.append({"params": model.trunk.parameters(), "lr": cfg.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    if cfg.rung == "overfit":
        overfit(model, opt, train_items[: cfg.batch_size], cfg, device, log)
        return

    loader = make_loader(train_items, cfg, train=True)
    best, since, step = float("inf"), 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        if cfg.rung == "probe":
            model.trunk.eval()  # a frozen trunk keeps its BatchNorm statistics
        t0, running = time.perf_counter(), []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = coral_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            step += 1
            if step % cfg.log_every == 0:
                log.info("step %d  loss %.4f", step, float(np.mean(running[-cfg.log_every :])))
        per_head, mae = val_metrics(model, val_items, cfg, device)
        shown = "  ".join(f"{h} rho {m['spearman']:.3f} MAE {m['mae']:.3f}" for h, m in per_head.items())
        log.info("epoch %d  train loss %.4f  val %s  mean MAE %.4f  %.0fs", epoch, float(np.mean(running)), shown, mae,
                 time.perf_counter() - t0)  # fmt: skip
        if mae < best:
            best, since = mae, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            since += 1
            if since >= cfg.patience:
                log.info("early stop: no val improvement for %d epochs", cfg.patience)
                break
    log.info("best val mean MAE %.4f -> %s", best, out / "best.pt")


def overfit(model, opt, batch_items, cfg: Config, device: str, log) -> None:
    """One fixed batch, no augmentation, no val: the check before any full run (CLAUDE.md)."""
    ds = DefinitionDataset(batch_items, cfg.images_dir, cfg.set_name, cfg.height)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    y = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    model.train()
    for step in range(1, cfg.epochs + 1):
        loss = coral_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.epochs:
            # batch-statistics BatchNorm, dropout off: BN running stats (momentum 0.01) lag after 200 steps
            model.dropout.eval()
            with torch.no_grad():
                acc = (predict_class(model(x)) == y).float().mean(0)
            model.dropout.train()
            log.info(
                "step %d  loss %.4f  exact definition %.3f adiposity %.3f",
                step,
                loss.item(),
                *acc.tolist(),
            )
    passed = bool((acc >= OVERFIT_ACCURACY).all())
    log.info(
        "OVERFIT CHECK %s (both heads >= %.2f exact on the batch)",
        "PASS" if passed else "FAIL",
        OVERFIT_ACCURACY,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config, args.seed))


if __name__ == "__main__":
    main()
