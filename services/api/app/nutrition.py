"""Cited protein and energy ranges (kb/nutrition.json, ADR-048 Amendment 2) as the API uses them.

The app's own calorie limits override the cited ranges; `check_energy` makes any band that sits
outside a cited range either an explicit, declared override or an error at import.
"""

import json
from pathlib import Path

from physique_schemas.models import ProteinRange


def _find() -> Path:
    """kb/nutrition.json above this file: the repo root locally, the image's app dir deployed."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "kb" / "nutrition.json").exists():
            return parent / "kb" / "nutrition.json"
    raise FileNotFoundError("kb/nutrition.json not found above app/nutrition.py")


NUTRITION = json.loads(_find().read_text())
PROTEIN_G_PER_KG = tuple(NUTRITION["protein"]["g_per_kg_day"])


def protein_range(weight_kg: float) -> ProteinRange:
    """The whole cited range, never a point inside it, in grams of total body weight."""
    p = NUTRITION["protein"]
    low, high = p["g_per_kg_day"]
    also = " ".join(x["text"] for x in p["also_listed"])
    return ProteinRange(
        g_per_kg_low=low, g_per_kg_high=high,
        g_low=round(low * weight_kg), g_high=round(high * weight_kg),
        unit=p["unit"], label=p["label"], also_listed=also, text=p["agent_text"],
    )  # fmt: skip


def check_energy(energy: dict, max_loss: float, nutrition: dict | None = None) -> None:
    """Each band in `energy` ({goal: (low, high)} as fractions of daily needs) must lie inside the
    cited surplus range or match a declared override in nutrition.json; the weekly loss cap must
    not exceed the cited upper rate. Raises ValueError listing every problem."""
    cited = (nutrition or NUTRITION)["energy"]
    overrides, problems = cited["app_overrides"], []
    gain_lo, gain_hi = cited["gain_surplus_pct_above_maintenance"]
    for goal, (lo, hi) in energy.items():
        pct = [round((lo - 1) * 100, 6), round((hi - 1) * 100, 6)]
        if pct[0] >= gain_lo and pct[1] <= gain_hi:
            continue  # inside the cited surplus range
        override = overrides.get(goal)
        if override is None:
            problems.append(f"{goal}: {pct}% is outside the cited range and has no override")
        elif [round(x, 6) for x in override["pct_vs_maintenance"]] != pct:
            problems.append(f"{goal}: code band {pct} differs from the declared override")
        elif override["relation"] == "stricter_than_cited" and pct[1] > gain_hi:
            problems.append(f"{goal}: declared stricter than cited, but {pct[1]}% is above it")
    for goal in overrides.keys() - energy.keys():
        problems.append(f"{goal}: override declared for a goal the code does not have")
    if max_loss * 100 > cited["loss_pct_bodyweight_per_week"][1]:
        problems.append(f"loss cap {max_loss * 100}%/week is above the cited upper rate")
    if problems:
        raise ValueError("; ".join(problems))
