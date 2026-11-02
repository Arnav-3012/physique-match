"""Dataset for the definition model: the transparent renders of synth/, composited on a background.

Only the front view is used: the demo has no back head (ADR-053 Amendment 4). Items and labels: items.py.

Domain-gap augmentation (train only, applied to the clean render): small scale and shift, colour jitter, blur,
noise, a JPEG round trip, on a random solid / gradient / noise background. Real CC0 backgrounds are TBD.
"""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from ml.src.definition.items import Item

MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet statistics: the backbone is ImageNet-pretrained
STD = np.array([0.229, 0.224, 0.225], np.float32)
EVAL_BACKGROUND = (128, 128, 128)


def _background(rng: np.random.Generator, size: tuple[int, int]) -> Image.Image:
    w, h = size
    kind = rng.integers(3)
    if kind == 0:  # solid
        return Image.new("RGB", size, tuple(int(v) for v in rng.integers(0, 256, 3)))
    if kind == 1:  # vertical gradient
        top, bottom = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        grad = (top * (1 - t) + bottom * t) * np.ones((1, w, 1))
        return Image.fromarray(grad.astype(np.uint8))
    base = rng.integers(60, 200, 3)  # noise around a colour
    return Image.fromarray(np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8))


def _affine(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Small scale and shift of the RGBA render; the new border is transparent, so the background shows."""
    s = rng.uniform(0.9, 1.1)
    dx, dy = rng.uniform(-0.04, 0.04) * img.width, rng.uniform(-0.04, 0.04) * img.height
    cx, cy = img.width / 2, img.height / 2
    return img.transform(img.size, Image.Transform.AFFINE, (1 / s, 0, cx - cx / s - dx, 0, 1 / s, cy - cy / s - dy),
                         Image.Resampling.BILINEAR)  # fmt: skip


def _photometric(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        arr = np.asarray(img, np.float32) + rng.normal(0, rng.uniform(1, 6), (img.height, img.width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(40, 96)))
        img = Image.open(buf).convert("RGB")
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = (np.asarray(img, np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class DefinitionDataset(Dataset):
    """(image tensor 3 x H x W, labels tensor [definition, adiposity] as 0-4)."""

    def __init__(self, items: list[Item], images_dir: Path, set_name: str = "bump", height: int = 256,
                 train: bool = False, rig: str | None = None, seed: int = 0):  # fmt: skip
        self.items, self.dir, self.set, self.train = items, Path(images_dir), set_name, train
        self.rig = rig  # None = each identity's own training rig; "top" for the lighting slice
        self.size = (height * 3 // 4, height)  # the renders are 3:4
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def path(self, item: Item) -> Path:
        return self.dir / f"{item.id}_{self.set}_{self.rig or item.rig}_front.png"

    def image(self, item: Item) -> Image.Image:
        img = Image.open(self.path(item)).convert("RGBA")
        if self.train:
            img = _affine(img, self.rng)
        background = _background(self.rng, img.size) if self.train else Image.new("RGB", img.size, EVAL_BACKGROUND)
        background.paste(img, (0, 0), img)
        if self.train:
            background = _photometric(background, self.rng)
        return background.resize(self.size, Image.Resampling.BILINEAR)

    def __getitem__(self, i: int):
        item = self.items[i]
        return to_tensor(self.image(item)), torch.tensor([item.definition - 1, item.adiposity - 1])
