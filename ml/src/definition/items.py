"""The rendered identities with their labels (stdlib only, so metrics and tests need no torch).

Items come from params.csv (what was rendered: split, sex, cell, rig, slice flag) joined with labels.csv (the
classes, from `python -m synth.labels`, a pure function of params.csv and the frozen edges). The split is by
identity and was fixed before the render (ADR-045); nothing here re-splits.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

HEADS = ("definition", "adiposity")
LABEL_COLUMN = {
    "bump": "definition",
    "flat": "definition_flat",
}  # which labels.csv column the set trains on


@dataclass(frozen=True)
class Item:
    id: str
    split: str
    sex: str
    cell: int
    muscle_level: int
    fat_level: int
    round: int
    rig: str
    slice_top: bool
    definition: int  # classes 1-5
    adiposity: int


def load_items(labels_csv: Path, params_csv: Path, set_name: str = "bump") -> list[Item]:
    """The rendered identities with their labels, in sequence order."""
    with Path(labels_csv).open(newline="") as fh:
        labels = {r["id"]: r for r in csv.DictReader(fh)}
    column = LABEL_COLUMN[set_name]
    items = []
    with Path(params_csv).open(newline="") as fh:
        for r in csv.DictReader(fh):
            lab = labels[r["id"]]
            items.append(Item(
                id=r["id"], split=r["split"], sex=r["sex"], cell=int(r["cell"]),
                muscle_level=int(r["muscle_level"]), fat_level=int(r["fat_level"]), round=int(r["round"]),
                rig=r["rig"], slice_top=r["slice_top"] == "1",
                definition=int(lab[column]), adiposity=int(lab["adiposity"]),
            ))  # fmt: skip
    return items
