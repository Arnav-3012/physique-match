"""The 8-head definition path (ADR-053 Amendment 5): view map, items, masked metrics, the incremental gate, the A/B
rule (numpy and scipy only), and the model, loss mask and dataset (torch tests are skipped without torch; run them
with `uv run --group train pytest tests/test_definition_model8.py`). The 2-head path has its own tests."""

import csv

import numpy as np
import pytest

from ml.src.definition import evaluate8, views
from ml.src.definition.items8 import Item8, Sample, load_items8, samples


def _item8(i, labels=(3, 3, 3, 3, 3, 3, 3, 2), split="test"):
    return Item8(f"id{i:05d}", split, "female", 0, 0, 0, 0, "key_left", False, tuple(labels))


# --- the view map ---------------------------------------------------------------
def test_the_view_map_is_the_pre_registered_one():
    assert views.HEADS8 == (
        "definition",
        "abs",
        "chest",
        "shoulders",
        "arms",
        "back",
        "legs",
        "adiposity",
    )
    assert views.valid_heads("front") == (
        "definition",
        "abs",
        "chest",
        "shoulders",
        "arms",
        "legs",
        "adiposity",
    )
    assert views.valid_heads("back") == ("arms", "back", "legs", "adiposity")
    assert views.valid_mask("front") == (True, True, True, True, True, False, True, True)
    assert views.valid_mask("back") == (False, False, False, False, True, True, True, True)


def test_evaluation_uses_the_one_view_map():
    assert evaluate8.valid_mask is views.valid_mask  # the same object, not a copy of the rule


# --- items ----------------------------------------------------------------------
def test_load_items8_reads_the_eight_labels_and_overall_equals_abs(tmp_path):
    cols = ["id", "split", "definition", "definition_flat", "adiposity", "untrained_def_chest",
            "untrained_def_shoulders",
            "untrained_def_arms", "untrained_def_back", "untrained_def_legs"]  # fmt: skip
    with (tmp_path / "labels.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow(["id00000", "train", 4, 2, 1, 5, 3, 2, 1, 3])
    with (tmp_path / "params.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "split", "sex", "cell", "muscle_level", "fat_level", "round", "rig", "slice_top"])
        w.writerow(["id00000", "train", "male", 3, 1, 2, 0, "key_right", 0])
    item = load_items8(tmp_path / "labels.csv", tmp_path / "params.csv")[0]
    # HEADS8 order: definition, abs, chest, shoulders, arms, back, legs, adiposity
    assert item.labels == (4, 4, 5, 3, 2, 1, 3, 1)


def test_both_views_of_an_identity_share_its_split():
    items = [_item8(0, split="train"), _item8(1, split="val")]
    out = samples(items)
    assert [(s.id, s.view) for s in out] == [("id00000", "front"), ("id00000", "back"), ("id00001", "front"),
                                             ("id00001", "back")]  # fmt: skip
    assert all(s.item.split == ("train" if s.id == "id00000" else "val") for s in out)


# --- metrics --------------------------------------------------------------------
def test_each_head_is_scored_over_its_own_valid_samples_with_the_count():
    items = [_item8(i, labels=(i % 5 + 1,) * 8) for i in range(10)]
    ss = samples(items)
    true, mask = evaluate8.sample_arrays(ss)
    m = evaluate8.masked_head_metrics(true.copy(), true, mask)
    assert m["definition"]["n"] == 10 and m["back"]["n"] == 10  # front only / back only: one image per identity
    assert m["arms"]["n"] == 20 and m["adiposity"]["n"] == 20  # both views
    assert m["chest"]["spearman"] == pytest.approx(1.0) and m["chest"]["mae"] == 0


def test_a_head_with_no_valid_sample_has_zero_count_not_a_crash():
    ss = samples([_item8(0)], views=("front",))
    true, mask = evaluate8.sample_arrays(ss)
    m = evaluate8.masked_head_metrics(true.copy(), true, mask)
    assert m["back"]["n"] == 0 and np.isnan(m["back"]["mae"])


# --- the incremental gate -------------------------------------------------------
def _gate_setup(region_good: bool):
    """40 identities; chest label random. Overall head perfect on the chest label, or the chest head is."""
    rng = np.random.default_rng(0)
    chest = rng.integers(0, 5, 40)
    items = [_item8(i, labels=(3, 3, int(c) + 1, 3, 3, 3, 3, 2)) for i, c in enumerate(chest)]
    ss = samples(items)
    pred = np.zeros((len(ss), 8))
    noise = rng.integers(0, 5, 40)
    for n, c in enumerate(chest):
        front = 2 * n
        pred[front, 0] = noise[n] if region_good else c  # overall head
        pred[front, 2] = c if region_good else noise[n]  # chest head
    return ss, pred


def test_a_region_head_that_beats_the_overall_score_is_shown_as_regional():
    ss, pred = _gate_setup(region_good=True)
    metrics = {h: {"n": 40, "spearman": 0.9, "mae": 0.3, "exact": 0.5} for h in views.HEADS8}
    d = evaluate8.regional_decisions(ss, pred, metrics)["chest"]
    assert d["decision"] == "regional" and d["rho_region"] - d["rho_overall"] >= evaluate8.INCREMENTAL_MARGIN


def test_a_region_head_that_does_not_beat_the_overall_score_gets_the_fixed_reason():
    ss, pred = _gate_setup(region_good=False)
    metrics = {h: {"n": 40, "spearman": 0.9, "mae": 0.3, "exact": 0.5} for h in views.HEADS8}
    d = evaluate8.regional_decisions(ss, pred, metrics)["chest"]
    assert d["decision"] == "overall" and d["reason"] == "regional head did not beat the overall score"


def test_the_margin_is_inclusive_and_a_failed_primary_gate_or_nan_never_shows_a_region():
    m = evaluate8.INCREMENTAL_MARGIN
    assert evaluate8.decide(True, 0.80, 0.80 - m - 1e-9)[0] == "regional"
    assert evaluate8.decide(True, 0.80, 0.80 - m + 1e-3)[0] == "overall"
    assert evaluate8.decide(False, 0.99, 0.10) == ("na", evaluate8.REASON_NO_GATE)
    assert evaluate8.decide(True, float("nan"), 0.5)[0] == "overall"


def test_abs_is_never_compared_with_itself():
    ss, pred = _gate_setup(region_good=True)
    metrics = {h: {"n": 40, "spearman": 0.9, "mae": 0.3, "exact": 0.5} for h in views.HEADS8}
    assert evaluate8.regional_decisions(ss, pred, metrics)["abs"]["reason"] == evaluate8.REASON_ABS


def test_the_back_head_is_read_from_the_back_image_and_the_overall_head_from_the_front():
    ss = samples([_item8(0, labels=(3, 3, 3, 3, 3, 5, 3, 2))])
    pred = np.zeros((2, 8))
    pred[0, 0], pred[1, 5] = 1, 4  # overall (front row), back head (back row)
    pred[0, 5], pred[1, 0] = 99, 99  # invalid cells must be ignored
    _, mean, true = evaluate8.identity_level(ss, pred)
    assert mean[0, 0] == 1 and mean[0, 5] == 4 and true[0, 5] == 4
    pred[:, 4] = (2, 4)  # arms are valid on both views: their mean
    assert evaluate8.identity_level(ss, pred)[1][0, 4] == 3


# --- the A/B rule ---------------------------------------------------------------
def test_the_8_head_model_ships_only_if_overall_and_abs_are_not_worse_than_the_2_head_one():
    m2 = {"spearman": 0.90, "mae": 0.40}
    same = {"spearman": 0.895, "mae": 0.405}
    worse = {"spearman": 0.85, "mae": 0.40}
    assert evaluate8.ab_check({"definition": same, "abs": same}, m2)["ship_8_head"]
    out = evaluate8.ab_check({"definition": same, "abs": worse}, m2)
    assert out == {"definition": True, "abs": False, "ship_8_head": False}


# --- torch: model, loss mask, dataset -------------------------------------------
def test_the_8_head_model_outputs_one_logit_row_per_head():
    torch = pytest.importorskip("torch")
    from ml.src.definition.model import predict_class
    from ml.src.definition.model8 import DefinitionNet8

    model = DefinitionNet8(pretrained=False)
    logits = model(torch.randn(2, 3, 128, 96))
    assert logits.shape == (2, 8, 4) and predict_class(logits).shape == (2, 8)
    assert list(model.heads) == list(views.HEADS8)


def test_a_masked_label_has_zero_effect_on_the_loss_and_the_gradient():
    torch = pytest.importorskip("torch")
    from ml.src.definition.model8 import masked_coral_loss

    torch.manual_seed(0)
    logits = torch.randn(4, 8, 4, requires_grad=True)
    y = torch.randint(0, 5, (4, 8))
    mask = torch.zeros(4, 8, dtype=torch.bool)
    mask[:, :3] = True
    loss = masked_coral_loss(logits, y, mask)
    loss.backward()
    grad = logits.grad.clone()
    y2 = y.clone()
    y2[~mask] = (y[~mask] + 2) % 5  # change only the masked labels
    logits.grad = None
    loss2 = masked_coral_loss(logits, y2, mask)
    loss2.backward()
    assert loss.item() == loss2.item() and torch.equal(grad, logits.grad)
    assert (grad[:, 3:] == 0).all() and (grad[:, :3] != 0).any()
    changed = y.clone()
    changed[mask] = (y[mask] + 2) % 5  # changing a valid label does change the loss
    assert masked_coral_loss(logits.detach(), changed, mask).item() != loss.item()


def test_a_fully_masked_batch_gives_zero_loss_not_nan():
    torch = pytest.importorskip("torch")
    from ml.src.definition.model8 import masked_coral_loss

    loss = masked_coral_loss(
        torch.randn(2, 8, 4),
        torch.zeros(2, 8, dtype=torch.long),
        torch.zeros(2, 8, dtype=torch.bool),
    )
    assert loss.item() == 0.0


def test_the_8_head_loss_falls_on_one_batch():
    torch = pytest.importorskip("torch")
    from ml.src.definition.model8 import DefinitionNet8, masked_coral_loss

    torch.manual_seed(0)
    model = DefinitionNet8(pretrained=False)
    x = torch.randn(8, 3, 128, 96)
    y = torch.randint(0, 5, (8, 8))
    mask = torch.tensor([views.valid_mask("front" if i % 2 == 0 else "back") for i in range(8)])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = masked_coral_loss(model(x), y, mask).item()
    for _ in range(25):
        loss = masked_coral_loss(model(x), y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_the_dataset_loads_each_view_and_uses_the_shared_mask(tmp_path):
    pytest.importorskip("torch")
    from PIL import Image

    from ml.src.definition import data8

    assert data8.valid_mask is views.valid_mask
    rgba = Image.new("RGBA", (288, 384), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (100, 300), (200, 150, 120, 255)), (90, 40))
    for view in ("front", "back"):
        rgba.save(tmp_path / f"id00000_bump_key_left_{view}.png")
    item = _item8(0, labels=(5, 5, 4, 3, 2, 1, 3, 2))
    ds = data8.DefinitionDataset8(samples([item]), tmp_path, "bump", height=256)
    (xf, yf, mf), (_, yb, mb) = ds[0], ds[1]
    assert tuple(xf.shape) == (3, 256, 192) and ds.path(Sample(item, "back")).name == "id00000_bump_key_left_back.png"
    assert yf.tolist() == yb.tolist() == [4, 4, 3, 2, 1, 0, 2, 1]  # all 8 real classes on both views
    assert tuple(mf.tolist()) == views.valid_mask("front") and tuple(mb.tolist()) == views.valid_mask("back")
    assert mf.dtype == mb.dtype == data8.torch.bool


def test_the_2_head_path_is_unchanged():
    torch = pytest.importorskip("torch")
    from ml.src.definition.items import HEADS
    from ml.src.definition.model import DefinitionNet

    assert HEADS == ("definition", "adiposity")
    assert DefinitionNet(pretrained=False)(torch.randn(2, 3, 128, 96)).shape == (2, 2, 4)
