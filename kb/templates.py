"""Weekly muscle-group credit of a split template and the floor feasibility check (ADR-048).

A set counts `counting["primary"]` per primary muscle and `counting["secondary"]` per secondary
muscle of its exercise. `check` raises with every group under its floor or over the per-session cap.
"""

import json
from collections import defaultdict

from kb.baselines import KB


def load_exercises() -> dict[str, dict]:
    return {e["id"]: e for e in json.loads((KB / "raw" / "exercises.json").read_text())}


def day_credit(day: dict[str, int], exercises: dict[str, dict], counting: dict) -> dict[str, float]:
    credit: dict[str, float] = defaultdict(float)
    for exercise_id, sets in day.items():
        exercise = exercises[exercise_id]
        for muscle in exercise["primaryMuscles"]:
            credit[muscle] += sets * counting["primary"]
        for muscle in exercise["secondaryMuscles"]:
            credit[muscle] += sets * counting["secondary"]
    return credit


def check(template: dict, exercises: dict[str, dict], baselines: dict) -> dict[str, float]:
    """Weekly credit per group; raises ValueError listing every floor or cap violation."""
    counting, cap = baselines["counting"], baselines["session_cap"]["value"]
    problems, week = [], defaultdict(float)
    for name, day in template["days"].items():
        for group, credit in day_credit(day, exercises, counting).items():
            week[group] += credit
            if group in baselines["groups"] and credit > cap:
                problems.append(f"{name}: {group} {credit} > session cap {cap}")
    for group, entry in baselines["groups"].items():
        if week[group] < entry["min_sets_per_week"]:
            problems.append(f"{group}: {week[group]} < floor {entry['min_sets_per_week']}")
    if problems:
        raise ValueError("; ".join(problems))
    return dict(week)
