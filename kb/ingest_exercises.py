"""Exercise chunks from the free-exercise-db dump (Unlicense, kb/raw/LICENSE.md; ADR-048).

One chunk per exercise, id `ex:<dataset id>`. Image paths are dropped. Stdlib only: the API
never imports this, it only reads the chunks that `build_index.py` writes.

    python3 -I kb/ingest_exercises.py     # prints counts as a sanity check
"""

import json
from collections import Counter
from pathlib import Path

RAW = Path(__file__).parent / "raw" / "exercises.json"
SOURCE = "free-exercise-db"
LEVEL = {"beginner": "beginner", "intermediate": "intermediate", "expert": "advanced"}  # our Level
# The planner's default pool (hypertrophy work). Stretching, cardio, plyometrics, strongman and
# olympic lifts stay in the index for explicit filters.
PLANNABLE = ("strength", "powerlifting")


def chunk(e: dict) -> dict:
    primary, secondary = e["primaryMuscles"], e["secondaryMuscles"]
    equipment = e["equipment"] or "none"
    style = " ".join(x for x in (e["mechanic"], e["force"], e["category"]) if x)
    text = (
        f"{e['name']}. Targets {', '.join(primary)}"
        + (f"; also {', '.join(secondary)}" if secondary else "")
        + f". Equipment: {equipment}. Level: {LEVEL[e['level']]}. {style}. "
        + " ".join(e["instructions"])
    )
    return {
        "id": f"ex:{e['id']}",
        "kind": "exercise",
        "name": e["name"],
        "text": text,
        "muscle": primary,
        "secondary": secondary,
        "equipment": equipment,
        "level": LEVEL[e["level"]],
        "category": e["category"],
        "plannable": e["category"] in PLANNABLE,
        "source": SOURCE,
    }


def load(path: Path = RAW) -> list[dict]:
    chunks = [chunk(e) for e in json.loads(path.read_text())]
    dupes = [i for i, n in Counter(c["id"] for c in chunks).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate exercise ids: {dupes[:5]}")
    return chunks


if __name__ == "__main__":
    cs = load()
    print(len(cs), "chunks;", sum(c["plannable"] for c in cs), "plannable")
    print(Counter(m for c in cs for m in c["muscle"]))
