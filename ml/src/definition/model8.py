"""The 8-head model: the same MobileNetV3-Small trunk and CORAL heads as `model.py`, one head per HEADS8 entry
(ADR-053 Amendment 5). The loss ignores the heads a view cannot show through an explicit mask."""

import torch
from torch import nn

from ml.src.definition.model import FEATURES, N_CLASSES, DefinitionNet, cumulative_targets
from ml.src.definition.views import HEADS8


class DefinitionNet8(DefinitionNet):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__(pretrained, dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(FEATURES, N_CLASSES - 1) for h in HEADS8})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 8, 4): logits per head in HEADS8 order."""
        z = self.dropout(self.trunk(x))
        return torch.stack([self.heads[h](z) for h in HEADS8], dim=1)


def masked_coral_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """logits (B, 8, 4), y (B, 8) classes 0-4, mask (B, 8) bool. Masked entries add nothing to the loss or the
    gradient, whatever their label; the mean is over the valid entries only."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, cumulative_targets(y), reduction="none")
    w = mask[..., None].float()
    return (bce * w).sum() / (w.sum() * logits.shape[-1]).clamp(min=1.0)
