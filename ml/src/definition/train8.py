"""Train the 8-head definition model (ADR-053 Amendment 5). Same rungs and config as `train.py`; samples are
(identity, view) images and the loss is masked by view. Run only AFTER the render has finished (MPS competes with
the Metal render); the one-batch overfit check may run earlier on CPU:

  DEFINITION_DEVICE=cpu uv run --group train python -m ml.src.definition.train8 \\
      --config ml/configs/definition8_overfit.yaml
  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --group train python -m ml.src.definition.train8 \\
      --config ml/configs/definition8_probe.yaml
  ... --config ml/configs/definition8_finetune.yaml     # everything, starting from the probe's best.pt

The best checkpoint is chosen on val only (mean MAE over the heads with samples). The test split is never read here.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.src.definition.data8 import DefinitionDataset8
from ml.src.definition.evaluate8 import masked_head_metrics, predict_samples, sample_arrays
from ml.src.definition.items8 import load_items8, samples
from ml.src.definition.model import predict_class
from ml.src.definition.model8 import DefinitionNet8, masked_coral_loss
from ml.src.definition.train import OVERFIT_ACCURACY, RUNS, Config, device_name, load_config


def val_metrics(model, val_samples, cfg: Config, device: str) -> tuple[dict, float]:
    ds = DefinitionDataset8(val_samples, cfg.images_dir, cfg.set_name, cfg.height)
    true, mask = sample_arrays(val_samples)
    per_head = masked_head_metrics(predict_samples(model, ds, device), true, mask)
    return per_head, float(np.mean([m["mae"] for m in per_head.values() if m["n"]]))


def run(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = os.environ.get("DEFINITION_DEVICE") or device_name()
    out = RUNS / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "train.log", mode="w"),
                        logging.StreamHandler()])  # fmt: skip
    log = logging.getLogger("definition8")

    items = load_items8(cfg.labels_csv, cfg.params_csv, cfg.set_name)
    train_items = [i for i in items if i.split == "train" and (cfg.max_round is None or i.round < cfg.max_round)]
    val_items = [i for i in items if i.split == "val"]
    train_samples, val_samples = samples(train_items), samples(val_items)
    log.info("device %s  rung %s  set %s  train %d identities (%d images)  val %d  (test not read)", device, cfg.rung,
             cfg.set_name, len(train_items), len(train_samples), len(val_items))  # fmt: skip
    if not train_items:
        raise SystemExit("no training identities: has the render written params.csv, and does labels.csv cover it?")

    model = DefinitionNet8(pretrained=cfg.pretrained).to(device)
    if cfg.init_from:
        model.load_state_dict(torch.load(RUNS / cfg.init_from / "best.pt", map_location=device))
    if cfg.rung == "probe":
        model.freeze_trunk()
    groups = [{"params": model.heads.parameters(), "lr": cfg.lr_head}]
    if cfg.rung != "probe":
        groups.append({"params": model.trunk.parameters(), "lr": cfg.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

    if cfg.rung == "overfit":
        overfit(model, opt, train_samples[: cfg.batch_size], cfg, device, log)
        return

    ds = DefinitionDataset8(train_samples, cfg.images_dir, cfg.set_name, cfg.height, train=cfg.augment, seed=cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(cfg.seed))  # fmt: skip
    best, since, step = float("inf"), 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        if cfg.rung == "probe":
            model.trunk.eval()  # a frozen trunk keeps its BatchNorm statistics
        t0, running = time.perf_counter(), []
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            loss = masked_coral_loss(model(x), y, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            step += 1
            if step % cfg.log_every == 0:
                log.info("step %d  loss %.4f", step, float(np.mean(running[-cfg.log_every :])))
        per_head, mae = val_metrics(model, val_samples, cfg, device)
        shown = "  ".join(f"{h} n {m['n']} rho {m['spearman']:.3f} MAE {m['mae']:.3f}" for h, m in per_head.items())
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


def overfit(model, opt, batch_samples, cfg: Config, device: str, log) -> None:
    """One fixed batch of (identity, view) images, no augmentation, no val: the check before any full run."""
    ds = DefinitionDataset8(batch_samples, cfg.images_dir, cfg.set_name, cfg.height)
    x, y, mask = (torch.stack([ds[i][k] for i in range(len(ds))]).to(device) for k in range(3))
    for m in model.modules():  # torchvision's MobileNetV3 uses BN momentum 0.01: 80 steps would leave the running
        if isinstance(m, torch.nn.BatchNorm2d):  # statistics mostly ImageNet's and eval mode would not match train
            m.momentum = 0.1
    model.train()
    for step in range(1, cfg.epochs + 1):
        loss = masked_coral_loss(model(x), y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0 or step == cfg.epochs:
            model.eval()
            with torch.no_grad():
                hit = ((predict_class(model(x)) == y) & mask).sum(0).float() / mask.sum(0).clamp(min=1)
            model.train()
            log.info(
                "step %d  loss %.4f  exact per head %s",
                step,
                loss.item(),
                [round(v, 3) for v in hit.tolist()],
            )
    used = mask.any(0)
    passed = bool((hit[used] >= OVERFIT_ACCURACY).all())
    log.info("OVERFIT CHECK %s (every used head >= %.2f exact on its valid samples)", "PASS" if passed else "FAIL",
             OVERFIT_ACCURACY)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    run(load_config(args.config, args.seed))


if __name__ == "__main__":
    main()
