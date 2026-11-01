"""BodyM loader. Layout (checked on the downloaded copy, 2026-11-01):

    ml/data/bodym/<split>/{mask,mask_left}/<photo_id>.png   720x960, mode L, values {0, 255}
    ml/data/bodym/<split>/{hwg_metadata,measurements,subject_to_photo_map}.csv

`mask` = front view, `mask_left` = left side view. Several photo pairs per subject, so any
split must be made on `subject_id` (docs/mistakes.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "data" / "bodym"
SPLITS = ("train", "testA", "testB")  # testA = lab-like, testB = in the wild
MEASUREMENTS = (
    "ankle", "arm_length", "bicep", "calf", "chest", "forearm", "height", "hip",
    "leg_length", "shoulder_breadth", "shoulder_to_crotch", "thigh", "waist", "wrist",
)  # fmt: skip


def load_table(split: str, root: Path = ROOT) -> pd.DataFrame:
    """One row per photo pair: ids, sex, height_cm, weight_kg, 14 measurements (cm), mask paths."""
    d = root / split
    photos = pd.read_csv(d / "subject_to_photo_map.csv")
    meta = pd.read_csv(d / "hwg_metadata.csv").rename(columns={"gender": "sex"})
    meas = pd.read_csv(d / "measurements.csv").rename(columns=lambda c: c.replace("-", "_"))
    df = photos.merge(meta, on="subject_id", validate="many_to_one")
    df = df.merge(meas, on="subject_id", validate="many_to_one")
    if len(df) != len(photos):
        raise ValueError(f"{split}: {len(photos) - len(df)} photos lost in the join")
    df["front_path"] = [d / "mask" / f"{p}.png" for p in df.photo_id]
    df["side_path"] = [d / "mask_left" / f"{p}.png" for p in df.photo_id]
    missing = [p for p in (*df.front_path, *df.side_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{split}: {len(missing)} masks missing, first {missing[0]}")
    df.insert(0, "split", split)
    return df


def clipped(mask: np.ndarray) -> bool:
    """Feet cut off by the photo edge. 24% of side masks (train sample, 2026-11-01), random in
    sex/height. Excluded from every side-view fit and eval: the height reference is short."""
    return bool(mask[-1].any())


def load_mask(path: Path) -> np.ndarray:
    """Boolean (H, W) mask."""
    return np.asarray(Image.open(path).convert("L")) > 127
