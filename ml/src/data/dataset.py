"""Training data: BodyM masks in the canonical frame, cached downscaled, sampled subject-first.
Run once: uv run --group train python -m ml.src.data.dataset     (ml/data/cache/, gitignored)
That builds train and val. Test-B stays unread until the gate: --splits testB, only then.

Only photos with an unclipped side mask are used (ADR-030). The labels and the canonical photo of
each subject come from `canonical_subjects`, so training, val and evaluate.py agree on both.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.src.data.bodym import ROOT, load_mask
from ml.src.data.pairs import OUT, canonical_subjects, labels
from ml.src.preprocess import downscale, from_bodym

CACHE = ROOT.parent / "cache"
VIEW_DIRS = ("mask", "mask_left")  # front, side


def usable_photos(name: str, features: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.read_csv(OUT) if features is None else features
    return df[(df.set == name) & ~df.side_clipped]


def _views(row: tuple[str, str]) -> np.ndarray:
    source_split, photo = row
    d = ROOT / source_split
    return np.stack([downscale(from_bodym(load_mask(d / v / f"{photo}.png"))) for v in VIEW_DIRS])


def build(name: str) -> None:
    df = usable_photos(name)
    with ProcessPoolExecutor() as pool:
        rows = zip(df.split, df.photo_id, strict=True)
        views = np.stack(list(pool.map(_views, rows, chunksize=32)))
    ids = np.array(df.photo_id.astype(str).tolist())
    np.savez_compressed(CACHE / f"{name}.npz", photo_id=ids, views=views)
    print(f"{name}: {views.shape} -> {CACHE / f'{name}.npz'}")


class Views:
    """One split in memory: per-subject labels, and each subject's usable photos."""

    def __init__(self, name: str, features: pd.DataFrame | None = None, cache: Path = CACHE):
        canon = canonical_subjects(name, features)
        z = np.load(cache / f"{name}.npz", allow_pickle=False)
        row_of = {p: k for k, p in enumerate(z["photo_id"].tolist())}
        by_subject = usable_photos(name, features).groupby("subject_id").photo_id
        self.views = torch.from_numpy(z["views"])  # uint8 (photos, 2, H, W)
        self.y = torch.from_numpy(labels(canon)).float()
        self.subjects = canon.subject_id.astype(str).tolist()
        self.canonical = np.array([row_of[str(p)] for p in canon.photo_id])
        self.photos = [
            np.array([row_of[str(p)] for p in by_subject.get_group(s)]) for s in canon.subject_id
        ]

    def __len__(self) -> int:
        return len(self.subjects)

    def pick(self, subjects: np.ndarray, rng: np.random.Generator | None = None) -> torch.Tensor:
        """uint8 (k, 2, H, W), one photo per subject: random with `rng`, else the canonical one."""
        if rng is None:
            return self.views[self.canonical[subjects]]
        return self.views[np.array([rng.choice(self.photos[s]) for s in subjects])]


def as_input(views: torch.Tensor) -> torch.Tensor:
    """Float (k, 2, H, W) in [0, 1] -> (k, 3, H, W): front, side, zeros, as `model_input`."""
    return torch.cat([views, torch.zeros_like(views[:, :1])], dim=1)


def blank_side(x: torch.Tensor) -> torch.Tensor:
    blanked = x.clone()
    blanked[:, 1] = 0
    return blanked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", choices=("train", "val", "testB"),
                    default=["train", "val"])  # fmt: skip
    CACHE.mkdir(exist_ok=True)
    for name in ap.parse_args().splits:
        build(name)


if __name__ == "__main__":
    main()
