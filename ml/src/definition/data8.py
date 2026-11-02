"""Dataset of (identity, view) images for the 8-head model (ADR-053 Amendment 5). The 2-head `data.py` is untouched.

Returns (image, labels (8,) as 0-4, mask (8,) bool). The labels are always real classes; the mask, from the
shared view map, says which of them this view may be trained and scored on. Compositing and augmentation are
inherited from DefinitionDataset.
"""

import torch

from ml.src.definition.data import DefinitionDataset, to_tensor
from ml.src.definition.views import valid_mask


class DefinitionDataset8(DefinitionDataset):
    def path(self, sample):
        return self.dir / f"{sample.id}_{self.set}_{self.rig or sample.rig}_{sample.view}.png"

    def __getitem__(self, i: int):
        s = self.items[i]
        return (
            to_tensor(self.image(s)),
            torch.tensor(s.item.labels) - 1,
            torch.tensor(valid_mask(s.view)),
        )
