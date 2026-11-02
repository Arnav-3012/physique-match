"""The definition model's metrics, items and (when torch is installed) model and dataset. The torch tests are
skipped in the default environment; run them with `uv run --group train pytest tests/test_definition_model.py`."""

import csv

import numpy as np
import pytest

from ml.src.definition import evaluate
from ml.src.definition.items import Item, load_items


def _item(i, sex="female", cell=0, muscle=0, fat=0, definition=3, adiposity=2):
    return Item(f"id{i:05d}", "test", sex, cell, muscle, fat, 0, "key_left", False, definition, adiposity)


# --- metrics (numpy and scipy only) -------------------------------------------
def test_head_metrics_perfect_and_off_by_one():
    true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    perfect = evaluate.head_metrics(true, true)
    assert perfect["spearman"] == pytest.approx(1.0) and perfect["mae"] == 0 and perfect["exact"] == 1
    off = evaluate.head_metrics(np.clip(true + 1, 0, 4), true)
    assert off["mae"] == pytest.approx(0.8) and off["spearman"] > 0.9


def test_the_gate_needs_both_numbers_and_a_constant_prediction_fails():
    assert evaluate.gate({"spearman": 0.8, "mae": 0.6})
    assert not evaluate.gate({"spearman": 0.79, "mae": 0.1})
    assert not evaluate.gate({"spearman": 0.95, "mae": 0.61})
    constant = evaluate.head_metrics(np.zeros(10, int), np.arange(10) % 5)
    assert np.isnan(constant["spearman"]) and not evaluate.gate(constant)


def test_a_group_is_reported_only_with_at_least_30_identities():
    items = [_item(i, sex="female", cell=0) for i in range(35)] + [_item(100 + i, sex="male", cell=1) for i in range(5)]
    pred = true = np.array([[i % 5, i % 5] for i in range(40)])
    groups = {r["group"] for r in evaluate.group_report(items, pred, true)}
    assert "pooled" in groups and "sex female" in groups and "cell 0" in groups
    assert "sex male" not in groups and "cell 1" not in groups  # 5 identities: not reported


# --- items --------------------------------------------------------------------
def test_load_items_joins_params_with_the_labels_of_the_chosen_set(tmp_path):
    with (tmp_path / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split", "definition", "definition_flat", "adiposity"])
        w.writerow(["id00000", "train", 4, 2, 1])
    with (tmp_path / "params.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split", "sex", "cell", "muscle_level", "fat_level", "round", "rig", "slice_top"])
        w.writerow(["id00000", "train", "male", 3, 1, 2, 0, "key_right", 1])
    bump = load_items(tmp_path / "labels.csv", tmp_path / "params.csv", "bump")[0]
    flat = load_items(tmp_path / "labels.csv", tmp_path / "params.csv", "flat")[0]
    assert (bump.definition, flat.definition, bump.adiposity) == (4, 2, 1)
    assert bump.rig == "key_right" and bump.slice_top and bump.sex == "male"


# --- model and dataset (torch) --------------------------------------------------
def test_the_model_has_two_ordinal_heads_and_the_loss_falls_on_one_batch():
    torch = pytest.importorskip("torch")
    from ml.src.definition.model import DefinitionNet, coral_loss, cumulative_targets, predict_class

    torch.manual_seed(0)
    model = DefinitionNet(pretrained=False)
    x = torch.randn(8, 3, 128, 96)
    logits = model(x)
    assert logits.shape == (8, 2, 4)
    y = torch.tensor([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0], [2, 2], [1, 1], [3, 3]])
    assert cumulative_targets(y)[0, 1].tolist() == [1, 1, 1, 1]  # class 4 is above every threshold
    assert cumulative_targets(y)[0, 0].tolist() == [0, 0, 0, 0]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = coral_loss(model(x), y).item()
    for _ in range(25):
        loss = coral_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
    model.eval()
    assert predict_class(model(x)).shape == (8, 2)


def test_freezing_the_trunk_leaves_only_the_heads_trainable():
    pytest.importorskip("torch")
    from ml.src.definition.model import DefinitionNet

    model = DefinitionNet(pretrained=False)
    model.freeze_trunk()
    assert not any(p.requires_grad for p in model.trunk.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())


def test_the_dataset_composites_the_transparent_render_and_returns_the_labels(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from ml.src.definition.data import DefinitionDataset

    rgba = Image.new("RGBA", (288, 384), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (100, 300), (200, 150, 120, 255)), (90, 40))
    for name in ("id00000_bump_key_left_front.png", "id00000_bump_top_front.png"):
        rgba.save(tmp_path / name)
    item = _item(0, definition=4, adiposity=2)
    ds = DefinitionDataset([item], tmp_path, "bump", height=256)
    x, y = ds[0]
    assert tuple(x.shape) == (3, 256, 192) and y.tolist() == [3, 1]
    assert ds.path(item).name == "id00000_bump_key_left_front.png"
    assert DefinitionDataset([item], tmp_path, "bump", rig="top").path(item).name == "id00000_bump_top_front.png"
    assert (ds[0][0] == x).all()  # evaluation is deterministic: a fixed grey background, no augmentation
    train = DefinitionDataset([item], tmp_path, "bump", train=True, seed=1)
    assert tuple(train[0][0].shape) == (3, 256, 192) and not (train[0][0] == x).all()
