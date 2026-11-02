"""Identities with all 8 labels, and their (identity, view) samples (stdlib only, so tests need no torch).

Labels come from labels.csv (`python -m synth.labels`, a pure function of params.csv and the frozen edges):
`definition` is both the overall and the abs label (m_abs = m), the other five regions are the `untrained_*`
columns written next to the two trained heads, `adiposity` is as before (ADR-053 Amendment 5).
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from ml.src.definition.items import LABEL_COLUMN, Item, load_items
from ml.src.definition.views import HEADS8, VIEWS


@dataclass(frozen=True)
class Item8:
    id: str
    split: str
    sex: str
    cell: int
    muscle_level: int
    fat_level: int
    round: int
    rig: str
    slice_top: bool
    labels: tuple[int, ...]  # classes 1-5 in HEADS8 order


@dataclass(frozen=True)
class Sample:
    """One image: an identity seen from one view. Both views of an identity share its split."""

    item: Item8
    view: str

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def rig(self) -> str:
        return self.item.rig


def label_columns(set_name: str) -> dict[str, str]:
    overall = LABEL_COLUMN[set_name]
    prefix = "def" if set_name == "bump" else "flat"
    regional = {r: f"untrained_{prefix}_{r}" for r in ("chest", "shoulders", "arms", "back", "legs")}
    return {"definition": overall, "abs": overall, "adiposity": "adiposity", **regional}


def _widen(it: Item, labels: tuple[int, ...]) -> Item8:
    return Item8(
        it.id,
        it.split,
        it.sex,
        it.cell,
        it.muscle_level,
        it.fat_level,
        it.round,
        it.rig,
        it.slice_top,
        labels,
    )


def load_items8(labels_csv: Path, params_csv: Path, set_name: str = "bump") -> list[Item8]:
    """The rendered identities with their 8 labels, in sequence order."""
    columns = label_columns(set_name)
    with Path(labels_csv).open(newline="") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}
    return [
        _widen(it, tuple(int(rows[it.id][columns[h]]) for h in HEADS8))
        for it in load_items(labels_csv, params_csv, set_name)
    ]


def samples(items: list[Item8], views: tuple[str, ...] = VIEWS) -> list[Sample]:
    return [Sample(it, v) for it in items for v in views]
