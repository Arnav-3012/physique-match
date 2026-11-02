"""Mask augmentation, run on the device. P3 starts small: small affine here, side-view dropout in
the loss, and pair swapping needs no code (all-pairs batches hold both (i, j) and (j, i)). Edge
noise, morphological jitter, clothing bulge and neck cut-row jitter wait until val shows the need.
"""

import math

import torch
import torch.nn.functional as F

ROTATE_DEG = 3.0
ZOOM = 0.03  # isotropic only: stretching one axis would change the proportions being predicted
SHIFT = 0.02  # fraction of the half-frame


def affine(
    views: torch.Tensor,
    angle: torch.Tensor,
    zoom: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
) -> torch.Tensor:
    """Rotate and zoom without shear: (n, 1, H, W) with per-image parameters of shape (n,)."""
    h, w = views.shape[-2:]
    cos, sin = zoom * angle.cos(), zoom * angle.sin()
    # affine_grid works in [-1, 1] on each axis; the h/w factors undo the non-square frame.
    theta = torch.stack(
        [
            torch.stack([cos, -sin * h / w, dx], dim=1),
            torch.stack([sin * w / h, cos, dy], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, views.shape, align_corners=False)
    return F.grid_sample(views, grid, align_corners=False, padding_mode="zeros")


def jitter(views: torch.Tensor) -> torch.Tensor:
    """Float (k, 2, H, W): an independent small affine per view."""
    k, c, h, w = views.shape

    def draw(span: float) -> torch.Tensor:
        return (2 * torch.rand(k * c, device=views.device) - 1) * span

    flat = views.reshape(k * c, 1, h, w)
    moved = affine(flat, draw(math.radians(ROTATE_DEG)), 1 + draw(ZOOM), draw(SHIFT), draw(SHIFT))
    return moved.reshape(k, c, h, w)
