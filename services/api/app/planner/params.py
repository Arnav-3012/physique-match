"""Planner parameters and split layouts (ADR-055).

Every number here is OUR rule, untuned, fixed before any plan was produced. The cited numbers (floor,
counting rule, per-session soft cap) are read from kb/baselines.json and never live here.
"""

import json

from app.rag.corpus import KB_DIR

SETS_PER_EXERCISE = 3
MAX_EXERCISES_PER_DAY = 9
EXTRA_SETS_PER_PRIORITY_GROUP = 2.0
MAX_WEEKLY_SETS_PER_GROUP = 10.0
MAX_SETS_PER_GROUP_PER_SESSION = 8.0  # applied as the lower of this and the cited cap
MAX_SESSIONS_PER_GROUP_PER_WEEK = 4
TOP_K = 10
SHORT_DAYS_NOTE = (  # code-generated, shown when days <= 2 and a shortfall exists (ADR-056)
    "With 2 or fewer days not every region can be covered; the regions below are short of their weekly minimum."
)

PUSH = frozenset({"chest", "shoulders", "triceps"})
PULL = frozenset({"lats", "middle back", "traps", "biceps", "forearms"})
LEGS = frozenset({"quadriceps", "hamstrings", "glutes", "abductors", "adductors", "calves"})
CORE = frozenset({"abdominals"})
ALL = PUSH | PULL | LEGS | CORE
UPPER, LOWER = PUSH | PULL | CORE, LEGS | CORE
LOW_VOLUME = frozenset({"abdominals", "calves", "forearms", "traps"})

# Focus groups per day; an exercise chosen for group G goes only on days whose focus has G.
_PPL = [PUSH, PULL, LOWER]
LAYOUTS: dict[int, list[frozenset[str]]] = {
    1: [ALL],
    2: [ALL, ALL],
    3: [ALL, ALL, ALL],  # the shape of the existing 3-day template
    4: [UPPER, LOWER, UPPER, LOWER],
    5: [PUSH, PULL, LOWER, UPPER, LOWER],
    6: _PPL + _PPL,
    7: _PPL + _PPL + [LOW_VOLUME],
}


def kb_json(name: str) -> dict:
    return json.loads((KB_DIR / name).read_text())


def session_limit(baselines: dict) -> float:
    return min(MAX_SETS_PER_GROUP_PER_SESSION, baselines["session_cap"]["value"])
