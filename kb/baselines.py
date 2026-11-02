"""Cited baseline weekly volumes and the region -> muscle-group map (ADR-039 item 7, ADR-048).

`validate` is the "fail loudly" rule: a region with no group, a group with no baseline, or a number
with no source listed in `sources.md` is an error, never a default. Stdlib only.
"""

import json
import re
from pathlib import Path

KB = Path(__file__).parent


def source_ids() -> set[str]:
    """Ids in the first column of the `sources.md` tables (`| `id` | ...`)."""
    return set(re.findall(r"^\| `([a-z0-9-]+)` \|", (KB / "sources.md").read_text(), re.M))


def load() -> tuple[dict, dict]:
    return (
        json.loads((KB / "baselines.json").read_text()),
        json.loads((KB / "region_map.json").read_text()),
    )


def validate(baselines: dict, region_map: dict, sources: set[str], regions: tuple[str, ...]):
    problems = []
    mapped = {g for r in region_map["regions"].values() for g in r["groups"]}
    for region in regions:
        if not region_map["regions"].get(region, {}).get("groups"):
            problems.append(f"region {region!r} maps to no muscle group")
    for group in sorted(mapped - baselines["groups"].keys()):
        problems.append(f"muscle group {group!r} has no baseline")
    for group in sorted(baselines["groups"].keys() - mapped):
        problems.append(f"baseline for {group!r} belongs to no region")
    cited = {
        **baselines["groups"],
        "counting": baselines["counting"],
        "session_cap": baselines["session_cap"],
    }
    for name, entry in cited.items():
        if entry["source_id"] not in sources:
            problems.append(f"{name!r} cites {entry['source_id']!r}, which is not in sources.md")
        if not entry.get("locator"):
            problems.append(f"{name!r} has no locator")
    for group, entry in baselines["groups"].items():
        if not entry["min_sets_per_week"] > 0:
            problems.append(f"baseline for {group!r} is not positive")
    if problems:
        raise ValueError("; ".join(problems))
