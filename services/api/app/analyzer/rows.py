"""Landmark rows from the silhouette alone (ADR-024): one code path for BodyM and browser masks.

Front/back views of a canonical, head-less mask (ml/src/preprocess.py). Rows are pixel rows in
the 384x512 frame. Run-length analysis; `width` = the widest run in a row, which is the torso
once the arms have separated, so no centre column is needed (it drifts with lean and pose):
  armpit = first row where the arms separate from the torso (>= 3 runs, held 5 rows)
  crotch = first row where the widest run falls below CROTCH_DROP x its maximum since the
           armpit, held 8 rows (the torso splits into two legs)
  waist  = narrowest width between armpit and crotch
  hip    = widest width between waist and crotch
When a rule finds nothing, a height-proportion prior is used and the flag says so.
"""

from dataclasses import dataclass

import numpy as np

from ml.src.preprocess import H, body_px

# Starting guesses in body heights below the crown; windows are tuned on the BodyM train split
# only (ml/src/tune_rows.py). Not facts.
ARMPIT_SEARCH = (0.18, 0.45)
ARMPIT_PRIOR = 0.28
# BodyM train: 0.583 +/- 0.028. Past 0.65 the "crotch" is a shorts hem (parity check 2026-11-01:
# 0.667-0.694 on shorts), so it counts as not found and the prior is used.
CROTCH_SEARCH = (0.45, 0.65)
CROTCH_PRIOR = 0.56
CROTCH_DROP = 0.65
WAIST_WINDOW = (0.25, 0.85)  # of the armpit -> crotch span
CHEST_OFFSET = 0.20  # chest row = armpit + this * (waist - armpit)
SHOULDER_ABOVE_ARMPIT = 0.07  # body heights; train sweep rho .636 (was .617 at .045)
SMOOTH = 5  # rows
ARM_GAP_HOLD = 5  # rows the arm/torso gap must persist
CROTCH_HOLD = 8  # rows the width drop must persist
MIN_RUN = 3  # px; shorter runs are mask noise


@dataclass(frozen=True)
class Rows:
    crown: int  # head-top row, even though the head is cut away
    cut: int  # first row of the head-less mask
    shoulder: int
    armpit: int
    chest: int
    waist: int
    hip: int
    crotch: int
    feet: int  # one past the last body row
    arms_separated: bool
    crotch_found: bool

    @property
    def found(self) -> bool:
        return self.arms_separated and self.crotch_found


def row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each horizontal run of at least MIN_RUN px in one mask row."""
    edges = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, ends, strict=True) if b - a >= MIN_RUN]


def _smooth(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.ones(SMOOTH) / SMOOTH, mode="same")


def find_rows(mask: np.ndarray) -> Rows:
    """Rows of a front or back canonical mask. Raises ValueError on an empty mask."""
    height = body_px(mask)
    body = np.flatnonzero(mask.any(axis=1))
    cut, feet = int(body[0]), int(body[-1]) + 1
    crown = round((H - height) / 2)

    def at(u: float) -> int:
        return crown + round(u * height)

    runs = [row_runs(r) for r in mask]
    width = np.array([max((b - a for a, b in r), default=0) for r in runs], float)

    three = np.array([len(r) >= 3 for r in runs])
    gap = [r for r in range(*map(at, ARMPIT_SEARCH)) if three[r : r + ARM_GAP_HOLD].all()]
    armpit, arms_separated = (gap[0], True) if gap else (at(ARMPIT_PRIOR), False)

    ref = np.maximum.accumulate(width[armpit:])  # widest torso/hip row so far
    narrow = width[armpit:] < CROTCH_DROP * np.concatenate(([ref[0]], ref[:-1]))
    split = [
        r
        for r in range(max(at(CROTCH_SEARCH[0]), armpit + 1), at(CROTCH_SEARCH[1]))
        if narrow[r - armpit : r - armpit + CROTCH_HOLD].all()
    ]
    crotch, crotch_found = (split[0], True) if split else (at(CROTCH_PRIOR), False)

    smooth = _smooth(width)
    span = max(crotch - armpit, 1)
    lo, hi = (armpit + round(f * span) for f in WAIST_WINDOW)
    waist = lo + int(np.argmin(smooth[lo:hi])) if hi > lo else (lo + hi) // 2
    below = smooth[waist : max(crotch - 2, waist + 1)]
    hip = waist + int(np.argmax(below))

    return Rows(
        crown=crown,
        cut=cut,
        shoulder=armpit - round(SHOULDER_ABOVE_ARMPIT * height),
        armpit=armpit,
        chest=armpit + round(CHEST_OFFSET * (waist - armpit)),
        waist=waist,
        hip=hip,
        crotch=crotch,
        feet=feet,
        arms_separated=arms_separated,
        crotch_found=crotch_found,
    )
