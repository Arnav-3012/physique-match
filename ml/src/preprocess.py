"""Canonical silhouette frame, shared by training (BodyM) and the API (browser masks). ADR-023/024.

Browser masks arrive already cropped, letterboxed and head-less
(apps/web/src/capture/silhouette.ts), so `from_browser` only validates and cleans them.
Re-cropping would rescale the body: the bbox of a head-less mask starts at the cut row.
BodyM masks are full-body, so `from_bodym` repeats the browser's crop and letterbox exactly
and then removes the head.

Pure numpy + Pillow + scipy.ndimage: no torch, so the API imports it.
"""

import numpy as np
from PIL import Image
from scipy import ndimage

# Mirrors apps/web/src/capture/config.ts (maskW, maskH, bboxMargin).
W, H = 384, 512
MARGIN = 0.05  # of bbox height, added on every side before the letterbox

# Starting guesses, tuned in the parity check (docs/phase-plan.md P2). Not facts.
NECK_WINDOW = (0.08, 0.25)  # search the narrowest row this far below the crown (body heights)
NECK_SMOOTH = 5  # rows, moving average over the width profile
# Cut this far below the narrowest neck row. Parity check 2026-11-01 (3 captures): at 0.02 the
# browser cut sat 0.013 (front) / 0.031 (side) body heights higher than BodyM; 0.0 closes most
# of it. Phone captures cut higher still (z ~ -3), left to cut-row jitter augmentation in P3.
NECK_CUT_OFFSET = 0.0
MODEL_H, MODEL_W = H // 2, W // 2  # Siamese input, ResNet-18 at 192x256 (P3)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return labels == 1 + np.argmax(np.bincount(labels.ravel())[1:])


def _rows(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        raise ValueError("empty mask")
    return rows


def body_px(mask: np.ndarray) -> float:
    """Full-body height in px inside the canonical frame (head included, even if cut away).

    The letterbox is centred, so the feet row alone gives it: feet = H/2 + body_px/2.
    """
    return 2.0 * (_rows(mask)[-1] + 1 - H / 2)


def neck_row(mask: np.ndarray) -> int:
    """Narrowest row between the head and the shoulders of a full-body canonical mask."""
    top, height = _rows(mask)[0], body_px(mask)
    widths = np.convolve(mask.sum(axis=1), np.ones(NECK_SMOOTH) / NECK_SMOOTH, mode="same")
    lo, hi = (top + int(f * height) for f in NECK_WINDOW)
    return lo + int(np.argmin(widths[lo:hi]))


def from_browser(mask: np.ndarray) -> np.ndarray:
    """Browser mask (already canonical, head-less) -> clean bool (H, W)."""
    if mask.shape != (H, W):
        raise ValueError(f"expected {(H, W)} mask, got {mask.shape}")
    mask = largest_component(mask.astype(bool))
    _rows(mask)
    return mask


def from_bodym(raw: np.ndarray) -> np.ndarray:
    """Full-body bool mask of any size -> canonical head-less bool (H, W)."""
    raw = largest_component(raw.astype(bool))
    rows, cols = _rows(raw), np.flatnonzero(raw.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pad = (y1 - y0) * MARGIN
    sw, sh = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    scale = min(W / sw, H / sh)
    nw, nh = min(W, round(sw * scale)), min(H, round(sh * scale))

    p = int(np.ceil(pad)) + 1  # zero margin so the crop box never leaves the image
    img = Image.fromarray(np.pad(raw, p).astype(np.uint8) * 255)
    box = (x0 - pad + p, y0 - pad + p, x1 + pad + p, y1 + pad + p)
    crop = np.asarray(img.resize((nw, nh), Image.Resampling.BILINEAR, box=box)) > 127

    mask = np.zeros((H, W), bool)
    oy, ox = (H - nh) // 2, (W - nw) // 2
    mask[oy : oy + nh, ox : ox + nw] = crop
    mask[: neck_row(mask) + round(NECK_CUT_OFFSET * body_px(mask))] = False
    return mask


def downscale(mask: np.ndarray) -> np.ndarray:
    """Canonical bool (H, W) -> uint8 (MODEL_H, MODEL_W), 2x2 area average on 0..255."""
    filled = mask.reshape(MODEL_H, 2, MODEL_W, 2).sum(axis=(1, 3))
    return (filled * 255 // 4).astype(np.uint8)


def model_input(front: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
    """Canonical masks -> float32 (3, MODEL_H, MODEL_W) in [0, 1]: front, side, zeros.
    A missing side view is an all-zero channel, the input side-view dropout trains on (ADR-025)."""
    blank = np.zeros((MODEL_H, MODEL_W), np.uint8)
    views = [downscale(front), blank if side is None else downscale(side), blank]
    return np.stack(views).astype(np.float32) / 255
