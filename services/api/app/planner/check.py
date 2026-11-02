"""Independent floor and cap check (ADR-055). Recomputes every group's credit from the final plan,
reading kb/baselines.json itself with its own counting code, and raises on an internal invariant
breach: a group under its floor and not named, a named number that disagrees, a cap exceeded."""

from collections import defaultdict

from app.planner.params import (
    MAX_EXERCISES_PER_DAY,
    MAX_SESSIONS_PER_GROUP_PER_WEEK,
    MAX_WEEKLY_SETS_PER_GROUP,
    kb_json,
    session_limit,
)


class PlannerInvariantError(Exception):
    pass


def weekly_credit(days, chunks_by_id: dict[str, dict]) -> dict[str, float]:
    """Weekly fractional sets per muscle, from the plan's exercises alone (the writer's plan is judged
    with this, never with a total the model or the planner reports)."""
    counting = kb_json("baselines.json")["counting"]
    week: dict[str, float] = defaultdict(float)
    for day in days:
        for slot in day:
            chunk = chunks_by_id[slot.exercise_id]
            for muscle in chunk["muscle"]:
                week[muscle] += slot.sets * counting["primary"]
            for muscle in chunk["secondary"]:
                week[muscle] += slot.sets * counting["secondary"]
    return dict(week)


def check(days, shortfalls, chunks_by_id: dict[str, dict]) -> dict[str, float]:
    baselines = kb_json("baselines.json")
    counting, limit = baselines["counting"], session_limit(baselines)
    floors = {g: e["min_sets_per_week"] for g, e in baselines["groups"].items()}
    week: dict[str, float] = defaultdict(float)
    sessions: dict[str, set[int]] = defaultdict(set)
    problems = []
    for d, day in enumerate(days):
        ids = [s.exercise_id for s in day]
        if len(ids) > MAX_EXERCISES_PER_DAY or len(ids) != len(set(ids)):
            problems.append(f"day {d}: {len(ids)} exercises or a repeat")
        load: dict[str, float] = defaultdict(float)
        for slot in day:
            chunk = chunks_by_id[slot.exercise_id]
            for muscle in chunk["muscle"]:
                load[muscle] += slot.sets * counting["primary"]
                sessions[muscle].add(d)
            for muscle in chunk["secondary"]:
                load[muscle] += slot.sets * counting["secondary"]
        for muscle, value in load.items():
            week[muscle] += value
            if muscle in floors and value > limit:
                problems.append(f"day {d}: {muscle} {value} > session limit {limit}")
    named = {s.group: s for s in shortfalls}
    for group, floor in floors.items():
        if week[group] > MAX_WEEKLY_SETS_PER_GROUP:
            problems.append(f"{group}: {week[group]} > weekly limit {MAX_WEEKLY_SETS_PER_GROUP}")
        if len(sessions[group]) > MAX_SESSIONS_PER_GROUP_PER_WEEK:
            problems.append(f"{group}: {len(sessions[group])} sessions")
        short = week[group] < floor
        if short and group not in named:
            problems.append(f"{group}: {week[group]} < floor {floor} and not named")
        if group in named and not (short and named[group].planned == week[group]):
            problems.append(f"{group}: named shortfall disagrees with the recomputed {week[group]}")
        if group in named and named[group].floor != floor:
            problems.append(f"{group}: named floor {named[group].floor} != cited {floor}")
    problems += [f"{g}: named but not a tracked group" for g in named.keys() - floors.keys()]
    if problems:
        raise PlannerInvariantError("; ".join(problems))
    return dict(week)
