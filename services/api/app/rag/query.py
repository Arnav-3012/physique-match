"""Query manager, demo scope (ADR-048): normalise, region decomposition, filters.

No free text reaches retrieval in the demo: sub-queries come from the region map and a closed
vocabulary. Equipment strings outside the dataset's vocabulary are dropped, never embedded or
logged. The embedding intent router is TBD after demo.
"""

import json
import re
import unicodedata
from dataclasses import dataclass

from app.rag.corpus import KB_DIR
from app.rag.retrieve import Filters

LEVELS = ("beginner", "intermediate", "advanced")
# user-facing word -> the dataset's `equipment` value. Unknown words are dropped.
EQUIPMENT = {
    "barbell": "barbell",
    "barbells": "barbell",
    "dumbbell": "dumbbell",
    "dumbbells": "dumbbell",
    "machine": "machine",
    "machines": "machine",
    "cable": "cable",
    "cables": "cable",
    "bodyweight": "body only",
    "body weight": "body only",
    "body only": "body only",
    "none": "body only",
    "bands": "bands",
    "band": "bands",
    "resistance bands": "bands",
    "kettlebell": "kettlebells",
    "kettlebells": "kettlebells",
    "e-z curl bar": "e-z curl bar",
    "ez bar": "e-z curl bar",
    "exercise ball": "exercise ball",
    "medicine ball": "medicine ball",
    "foam roll": "foam roll",
}


@dataclass(frozen=True)
class SubQuery:
    text: str
    group: str
    filters: Filters


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def dataset_equipment(items: list[str]) -> tuple[str, ...]:
    return tuple(sorted({EQUIPMENT[e] for e in map(normalise, items) if e in EQUIPMENT}))


def levels_up_to(level: str) -> tuple[str, ...]:
    """OUR rule: a user sees exercises at their level and below (not a cited number)."""
    return LEVELS[: LEVELS.index(level) + 1]


def decompose(region: str, equipment: list[str], level: str) -> list[SubQuery]:
    """One sub-query per muscle group of the region, filtered to the user's setup."""
    region_map = json.loads((KB_DIR / "region_map.json").read_text())["regions"]
    if region not in region_map:
        raise KeyError(f"unknown region {region!r}")
    gear, levels = dataset_equipment(equipment), levels_up_to(level)
    return [
        SubQuery(f"{group} exercise", group, Filters(muscle=(group,), equipment=gear, level=levels))
        for group in region_map[region]["groups"]
    ]
