"""Pair-head Siamese (the ablation) and SiameseSubtract (shipped, ADR-035). ADR-004, 006, 032.
Shared interface:
  encoder(x) -> embeddings; gap(e_goal, e_current) -> delta per region, in delta units;
  loss(e_full, e_front, y, side_dropout) -> scalar on standardised targets.
Normalisation and label scales live in the module, so an ONNX export carries them.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from ml.src.data.regions import LABELLED_REGIONS

EMBED = 512
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
STEM = ("conv1", "bn1")
RUNGS = {  # ADR-004 ladder: blocks that train
    "probe": (),
    "last2": ("layer3", "layer4"),
    "full": (*STEM, "layer1", "layer2", "layer3", "layer4"),
}


class Encoder(nn.Module):
    """ResNet-18 without its classifier: float (n, 3, H, W) in [0, 1] -> (n, EMBED)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Identity()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.frozen: list[nn.Module] = []

    def freeze(self, rung: str) -> None:
        self.frozen = [m for name, m in self.net.named_children() if name not in RUNGS[rung]]
        for m in self.frozen:
            m.requires_grad_(False)

    def train(self, mode: bool = True) -> "Encoder":
        super().train(mode)
        for m in self.frozen:  # frozen BatchNorm must keep its ImageNet statistics
            m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


class GapNet(nn.Module):
    def __init__(self, head_in: int, hidden: int, pretrained: bool):
        super().__init__()
        out = len(LABELLED_REGIONS)
        self.encoder = Encoder(pretrained)
        self.head = (
            nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(), nn.Linear(hidden, out))
            if hidden
            else nn.Linear(head_in, out)
        )
        self.register_buffer("y_mean", torch.zeros(out))
        self.register_buffer("y_std", torch.ones(out))

    def fit_labels(self, y: torch.Tensor) -> None:
        """Label statistics from the train split; `y` = measurement / height_cm per subject."""
        self.y_mean.copy_(y.mean(dim=0))
        self.y_std.copy_(y.std(dim=0))

    @property
    def delta_std(self) -> torch.Tensor:
        return self.y_std * math.sqrt(2)  # std of y_i - y_j for independent subjects

    def forward(self, goal: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.gap(self.encoder(goal), self.encoder(current))


class Siamese(GapNet):
    def __init__(self, hidden: int = 256, pretrained: bool = True):
        super().__init__(4 * EMBED, hidden, pretrained)

    def gap(self, e_goal: torch.Tensor, e_current: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([e_goal, e_current, e_goal - e_current, e_goal * e_current], dim=-1)
        return self.head(pair) * self.delta_std

    def loss(
        self,
        e_full: torch.Tensor,
        e_front: torch.Tensor,
        y: torch.Tensor,
        side_dropout: float,
        self_pairs: bool = False,
    ) -> torch.Tensor:
        """Ordered pairs of the batch, (i, i) only if `self_pairs`, then with target 0.
        Only the goal loses its side view (ADR-025)."""
        n = len(y)
        keep = torch.ones(n, n, device=y.device)
        if not self_pairs:
            keep = keep - torch.eye(n, device=y.device)
        goal, current = keep.nonzero(as_tuple=True)
        blanked = torch.rand(len(goal), 1, device=y.device) < side_dropout
        e_goal = torch.where(blanked, e_front[goal], e_full[goal])
        std = self.delta_std
        return F.huber_loss(self.gap(e_goal, e_full[current]) / std, (y[goal] - y[current]) / std)


class SiameseSubtract(GapNet):
    """Shipped model (ADR-035, trained as baseline B): a shared encoder predicts each person's
    ratios (measurement / height); the gap is their difference, so it is exactly antisymmetric."""

    def __init__(self, hidden: int = 256, pretrained: bool = True):
        super().__init__(EMBED, hidden, pretrained)

    def value(self, e: torch.Tensor) -> torch.Tensor:
        return self.y_mean + self.head(e) * self.y_std

    def gap(self, e_goal: torch.Tensor, e_current: torch.Tensor) -> torch.Tensor:
        return self.value(e_goal) - self.value(e_current)

    def loss(
        self,
        e_full: torch.Tensor,
        e_front: torch.Tensor,
        y: torch.Tensor,
        side_dropout: float,
        self_pairs: bool = False,  # unused: this loss is per subject, it has no pairs
    ) -> torch.Tensor:
        def error(e: torch.Tensor) -> torch.Tensor:
            return F.huber_loss(self.value(e) / self.y_std, y / self.y_std)

        return (1 - side_dropout) * error(e_full) + side_dropout * error(e_front)


# "regressor" is the name checkpoints and configs stored before the rename (ADR-035).
ARCHS = {"siamese": Siamese, "subtract": SiameseSubtract, "regressor": SiameseSubtract}
