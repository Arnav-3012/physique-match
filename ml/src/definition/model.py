"""MobileNetV3-Small trunk with two ordinal heads (ADR-053 Amendment 4): definition and adiposity.

Each head is a CORAL / cumulative-link head: 4 logits, logit k = P(class > k + 1), trained with binary cross-entropy
on the cumulative targets. The class is 1 + the number of logits above 0. That respects the order (a 5 mistaken
for a 1 costs more than for a 4) and gives the MAE and Spearman the gate uses.
"""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

N_CLASSES = 5
FEATURES = 576  # MobileNetV3-Small width after the last conv block


class DefinitionNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.trunk = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(FEATURES, N_CLASSES - 1) for h in ("definition", "adiposity")})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 2, 4): logits per head, definition first."""
        z = self.dropout(self.trunk(x))
        return torch.stack([self.heads[h](z) for h in ("definition", "adiposity")], dim=1)

    def freeze_trunk(self, frozen: bool = True) -> None:
        for p in self.trunk.parameters():
            p.requires_grad = not frozen


def cumulative_targets(y: torch.Tensor) -> torch.Tensor:
    """y: (B, 2) classes 0-4 -> (B, 2, 4) with 1 where the class is above threshold k."""
    return (y[..., None] > torch.arange(N_CLASSES - 1, device=y.device)).float()


def coral_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, cumulative_targets(y))


def predict_class(logits: torch.Tensor) -> torch.Tensor:
    """(B, 2, 4) -> (B, 2) classes 0-4."""
    return (logits > 0).sum(-1)
