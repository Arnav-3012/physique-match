"""Deterministic weekly plan (ADR-055): a pure function of its arguments, no LLM, no retriever.

Fills every muscle group to its cited floor, one group at a time and only from its own ranked
candidate list (no cross-group substitution; secondary credit still counts by the cited rule), then
adds extra sets for the priority regions. A group that cannot reach its floor is returned as a named
shortfall; the floor is never lowered. `check` verifies the result independently before it is returned.
"""

from collections import defaultdict
from dataclasses import dataclass

from app.planner.check import check
from app.planner.params import (
    EXTRA_SETS_PER_PRIORITY_GROUP,
    LAYOUTS,
    MAX_EXERCISES_PER_DAY,
    MAX_SESSIONS_PER_GROUP_PER_WEEK,
    MAX_WEEKLY_SETS_PER_GROUP,
    SETS_PER_EXERCISE,
    SHORT_DAYS_NOTE,
    kb_json,
    session_limit,
)
from physique_schemas.models import UserProfile


@dataclass(frozen=True)
class Slot:
    exercise_id: str  # the retrieved chunk id, also its citation
    name: str
    sets: int


@dataclass(frozen=True)
class Shortfall:
    group: str
    planned: float
    floor: float


@dataclass
class WeeklyPlan:
    days: list[list[Slot]]
    weekly: dict[str, float]  # fractional sets per tracked muscle group
    shortfalls: list[Shortfall]
    note: str = ""  # fixed sentence for short weeks, code-generated (ADR-056)

    @property
    def empty_days(self) -> list[int]:
        return [i for i, day in enumerate(self.days) if not day]


def build(
    profile: UserProfile,
    priority_regions: list[str],
    chunks: list[dict],
    candidates: dict[str, list[str]],
) -> WeeklyPlan:
    baselines = kb_json("baselines.json")
    regions = kb_json("region_map.json")["regions"]
    counting, limit = baselines["counting"], session_limit(baselines)
    floors = {g: float(e["min_sets_per_week"]) for g, e in baselines["groups"].items()}
    boosted = {g for r in priority_regions for g in regions[r]["groups"]}
    by_id = {c["id"]: c for c in chunks}
    for group, ids in candidates.items():
        if any(group not in by_id[i]["muscle"] for i in ids):
            raise ValueError(f"candidate list for {group!r} holds an exercise that does not train it")

    focus = LAYOUTS[profile.days_per_week]
    days: list[list[Slot]] = [[] for _ in focus]
    day_load: list[dict[str, float]] = [defaultdict(float) for _ in focus]
    week: dict[str, float] = defaultdict(float)
    sessions: dict[str, set[int]] = defaultdict(set)
    credits: dict[str, dict[str, float]] = {}

    def credit(cid: str) -> dict[str, float]:
        if cid not in credits:
            out: dict[str, float] = defaultdict(float)
            for muscle in by_id[cid]["muscle"]:
                out[muscle] += SETS_PER_EXERCISE * counting["primary"]
            for muscle in by_id[cid]["secondary"]:
                out[muscle] += SETS_PER_EXERCISE * counting["secondary"]
            credits[cid] = out
        return credits[cid]

    def place(cid: str, group: str, only_day: int | None = None, dry: bool = False) -> bool:
        """Put the block on the best day where every cap holds; `dry` only answers whether it fits."""
        cr = credit(cid)
        tracked = {m: v for m, v in cr.items() if m in floors}
        if any(week[m] + v > MAX_WEEKLY_SETS_PER_GROUP for m, v in tracked.items()):
            return False
        primaries = [m for m in by_id[cid]["muscle"] if m in floors]
        options = []
        for d in range(len(days)) if only_day is None else [only_day]:
            if (
                group in focus[d]
                and len(days[d]) < MAX_EXERCISES_PER_DAY
                and all(s.exercise_id != cid for s in days[d])
                and all(day_load[d][m] + v <= limit for m, v in tracked.items())
                and all(d in sessions[m] or len(sessions[m]) < MAX_SESSIONS_PER_GROUP_PER_WEEK for m in primaries)
            ):
                options.append((day_load[d][group], len(days[d]), d))
        if not options:
            return False
        if dry:
            return True
        d = min(options)[2]
        days[d].append(Slot(cid, by_id[cid]["name"], SETS_PER_EXERCISE))
        for m, v in cr.items():
            day_load[d][m] += v
            week[m] += v
        for m in primaries:
            sessions[m].add(d)
        return True

    def fill(group: str, target: dict[str, float]) -> None:
        while week[group] < target[group]:

            def useful(cid: str) -> float:
                return sum(min(v, max(0.0, target[m] - week[m])) for m, v in credit(cid).items() if m in target)

            ranked = sorted(enumerate(candidates.get(group, [])), key=lambda e: (-useful(e[1]), *e))
            if not any(place(cid, group) for _, cid in ranked):
                return

    extras = {
        g: min(f + (EXTRA_SETS_PER_PRIORITY_GROUP if g in boosted else 0.0), MAX_WEEKLY_SETS_PER_GROUP)
        for g, f in floors.items()
    }

    def uncovered_credit(cid: str) -> float:
        return sum(min(v, max(0.0, floors[m] - week[m])) for m, v in credit(cid).items() if m in floors)

    while True:  # pass 1: floors, the most constrained unmet group first (ADR-056)
        live = {g: [c for c in candidates.get(g, []) if place(c, g, dry=True)] for g in floors if week[g] < floors[g]}
        live = {g: cs for g, cs in live.items() if cs}
        if not live:
            break
        group = min(live, key=lambda g: (len(live[g]), -(floors[g] - week[g]), g))
        best = min(live[group], key=lambda c: (-uncovered_credit(c), candidates[group].index(c), c))
        if not place(best, group):  # the dry run said it fits: an internal invariant
            raise RuntimeError(f"{best} fit {group} in the dry run but not when placed")
    for group in floors:  # pass 2: priority extras, only where every cap has room
        fill(group, extras)

    for d in (i for i, day in enumerate(days) if not day):  # one low-volume filler, caps still apply
        for group in (g for g in floors if g in focus[d]):
            if any(place(cid, group, only_day=d) for cid in candidates.get(group, [])):
                break

    weekly = {g: week[g] for g in floors}
    shortfalls = [Shortfall(g, weekly[g], floors[g]) for g in floors if weekly[g] < floors[g]]
    check(days, shortfalls, by_id)
    note = SHORT_DAYS_NOTE if profile.days_per_week <= 2 and shortfalls else ""
    return WeeklyPlan(days, weekly, shortfalls, note)
