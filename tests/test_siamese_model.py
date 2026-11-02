import math

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # train group only; CI and the API image stay torch-free

from ml.src.augment import affine, jitter  # noqa: E402
from ml.src.data.dataset import Views, as_input, blank_side  # noqa: E402
from ml.src.data.regions import LABELLED_REGIONS, REGION_COLUMN  # noqa: E402
from ml.src.model import ARCHS, EMBED, Siamese, SiameseSubtract  # noqa: E402
from ml.src.preprocess import MODEL_H, MODEL_W, H, W, downscale, model_input  # noqa: E402
from ml.src.train import load_config, standardised_huber, subject_batches  # noqa: E402

N_REGIONS = len(LABELLED_REGIONS)


def test_old_checkpoints_and_configs_still_name_the_shipped_model_regressor():
    assert ARCHS["regressor"] is ARCHS["subtract"] is SiameseSubtract


def test_a_seed_gets_its_own_run_folder(tmp_path):
    path = tmp_path / "b_full.yaml"
    path.write_text("arch: regressor\n")
    assert load_config(path).name == "b_full"
    seeded = load_config(path, seed=2)
    assert (seeded.name, seeded.seed, seeded.arch) == ("b_full_seed2", 2, "regressor")


def test_val_huber_is_in_the_same_unit_as_the_training_loss():
    torch.manual_seed(0)
    y = torch.randn(6, N_REGIONS)
    std = torch.full((N_REGIONS,), 0.5)
    pred = (y[:, None] - y[None, :]) + torch.randn(6, 6, N_REGIONS)
    goal, current = (torch.eye(6) == 0).nonzero(as_tuple=True)
    loss = torch.nn.functional.huber_loss(pred[goal, current] / std, (y[goal] - y[current]) / std)
    assert standardised_huber(pred.numpy(), y.numpy(), std.numpy()) == pytest.approx(loss.item())


def tiny_batch(n=6):
    torch.manual_seed(0)
    return torch.rand(n, 3, 64, 48), torch.randn(n, N_REGIONS)


def test_gap_has_one_output_per_labelled_region():
    x, _ = tiny_batch()
    model = Siamese(hidden=16, pretrained=False).eval()
    assert model(x, x.flip(0)).shape == (6, N_REGIONS)


def test_gap_output_is_in_delta_units():
    e = torch.randn(4, EMBED)
    model = Siamese(hidden=16, pretrained=False).eval()
    before = model.gap(e, e.flip(0))
    model.y_std.mul_(3)
    assert torch.allclose(model.gap(e, e.flip(0)), 3 * before)


@pytest.mark.parametrize("cls", [Siamese, SiameseSubtract])
def test_loss_falls_on_a_fixed_batch(cls):
    x, y = tiny_batch()
    model = cls(hidden=16, pretrained=False)
    model.encoder.freeze("full")
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        e_full, e_front = model.encoder(torch.cat([x, blank_side(x)])).chunk(2)
        loss = model.loss(e_full, e_front, y, side_dropout=0.0)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def stub_gap(model, sign=1):
    model.gap = lambda g, c: sign * (g - c)[:, :N_REGIONS]


def perfect_embeddings(y):
    e = torch.zeros(len(y), EMBED)
    e[:, :N_REGIONS] = y
    return e


def test_pair_target_is_goal_minus_current():
    _, y = tiny_batch()
    e = perfect_embeddings(y)
    model = Siamese(hidden=16, pretrained=False)
    stub_gap(model)
    assert model.loss(e, e, y, side_dropout=0.0).item() == pytest.approx(0)
    stub_gap(model, sign=-1)
    assert model.loss(e, e, y, side_dropout=0.0).item() > 0.1


def test_same_subject_pairs_are_excluded_by_default():
    _, y = tiny_batch()
    e = perfect_embeddings(y)
    model = Siamese(hidden=16, pretrained=False)
    sizes = []

    def recording_gap(g, c):
        sizes.append(len(g))
        return (g - c)[:, :N_REGIONS]

    model.gap = recording_gap
    model.loss(e, e, y, side_dropout=0.0)
    model.loss(e, e, y, side_dropout=0.0, self_pairs=True)
    assert sizes == [6 * 5, 6 * 6]


def test_side_dropout_blanks_the_goal_only():
    _, y = tiny_batch()
    e = perfect_embeddings(y)
    model = Siamese(hidden=16, pretrained=False)
    stub_gap(model)
    assert model.loss(e, e + 1, y, side_dropout=0.0).item() == pytest.approx(0)  # front unused
    assert model.loss(e, e + 1, y, side_dropout=1.0).item() > 0.1  # front used, for goal


def test_regressor_gap_is_exactly_antisymmetric_and_ignores_the_mean():
    e = torch.randn(5, EMBED)
    model = SiameseSubtract(hidden=16, pretrained=False).eval()
    gap = model.gap(e, e.flip(0))
    assert torch.allclose(gap, -model.gap(e.flip(0), e))
    model.y_mean.add_(7)
    assert torch.allclose(model.gap(e, e.flip(0)), gap, atol=1e-4)


def test_fit_labels_sets_the_scale_used_for_pairs():
    y = torch.tensor([[1.0] * N_REGIONS, [3.0] * N_REGIONS])
    model = SiameseSubtract(hidden=16, pretrained=False)
    model.fit_labels(y)
    assert model.y_mean[0].item() == pytest.approx(2)
    assert model.delta_std[0].item() == pytest.approx(y[:, 0].std().item() * math.sqrt(2))


def test_freeze_stops_gradients_and_keeps_frozen_batchnorm_in_eval():
    model = Siamese(hidden=16, pretrained=False)
    model.encoder.freeze("last2")
    model.train()
    net = model.encoder.net
    assert not net.layer2[0].conv1.weight.requires_grad
    assert net.layer4[0].conv1.weight.requires_grad
    assert not net.bn1.training
    assert not net.layer2[0].bn1.training
    assert net.layer4[0].bn1.training
    assert model.head.training


def test_probe_freezes_the_whole_encoder():
    model = Siamese(hidden=0, pretrained=False)
    model.encoder.freeze("probe")
    assert not any(p.requires_grad for p in model.encoder.parameters())


def test_training_input_matches_the_api_input():
    rng = np.random.default_rng(0)
    front, side = rng.random((H, W)) > 0.5, rng.random((H, W)) > 0.5
    cached = torch.from_numpy(np.stack([downscale(front), downscale(side)]))[None]
    x = as_input(cached.float() / 255)
    assert np.array_equal(x[0].numpy(), model_input(front, side))
    assert np.array_equal(blank_side(x)[0].numpy(), model_input(front))
    assert x[0].shape == (3, MODEL_H, MODEL_W)


def test_affine_with_no_motion_is_identity():
    x = torch.rand(2, 1, 64, 48)
    none, one = torch.zeros(2), torch.ones(2)
    assert torch.allclose(affine(x, none, one, none, none), x, atol=1e-4)


def test_quarter_turn_keeps_lengths_in_a_non_square_frame():
    x = torch.zeros(1, 1, 64, 48)
    x[0, 0, 30:34, 14:34] = 1  # 20 px wide, 4 px tall, around the centre
    none, one = torch.zeros(1), torch.ones(1)
    turned = affine(x, torch.tensor([math.pi / 2]), one, none, none)[0, 0] > 0.5
    assert turned.any(dim=1).sum().item() == 20
    assert turned.any(dim=0).sum().item() == 4


def test_jitter_keeps_shape_and_roughly_the_area():
    torch.manual_seed(0)
    x = torch.zeros(4, 2, 256, 192)
    x[:, :, 60:200, 60:130] = 1
    out = jitter(x)
    assert out.shape == x.shape
    assert out.sum().item() == pytest.approx(x.sum().item(), rel=0.1)


def test_batches_hold_each_subject_at_most_once_per_epoch():
    seen = np.concatenate(list(subject_batches(10, 3, np.random.default_rng(0))))
    assert len(seen) == 9 and len(set(seen.tolist())) == 9


def fake_split(tmp_path):
    """Subject a has one clipped photo, b has two good ones, c has only clipped ones."""
    rows = [("a", "p1", True), ("a", "p2", False), ("b", "p3", False), ("b", "p4", False),
            ("c", "p5", True)]  # fmt: skip
    df = pd.DataFrame(rows, columns=["subject_id", "photo_id", "side_clipped"])
    df["set"] = "train"
    df["height_cm"] = 100.0
    for column in REGION_COLUMN.values():
        df[column] = 50.0
    good = ["p2", "p3", "p4"]
    views = np.stack([np.full((2, 4, 4), k, np.uint8) for k in range(len(good))])
    np.savez(tmp_path / "train.npz", photo_id=np.array(good), views=views)
    return Views("train", features=df, cache=tmp_path)


def test_views_drops_clipped_photos_and_all_clipped_subjects(tmp_path):
    data = fake_split(tmp_path)
    assert data.subjects == ["a", "b"]
    assert data.y.shape == (2, N_REGIONS)
    assert data.y[0, 0].item() == pytest.approx(0.5)


def test_pick_stays_inside_the_subject(tmp_path):
    data = fake_split(tmp_path)
    rng = np.random.default_rng(0)
    picks_a = {data.pick(np.array([0]), rng)[0, 0, 0, 0].item() for _ in range(20)}
    picks_b = {data.pick(np.array([1]), rng)[0, 0, 0, 0].item() for _ in range(20)}
    assert picks_a == {0}
    assert picks_b == {1, 2}
    assert data.pick(np.array([1]))[0, 0, 0, 0].item() == data.canonical[1]
