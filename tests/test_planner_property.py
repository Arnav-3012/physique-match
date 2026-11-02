"""Property test for the planner fill (ADR-056): with full gear, 3 to 7 days and any level, every muscle
group is at its floor or named in the shortfall list, and `check` passes. A miss is reported, never
tuned away: the fill order is not changed to satisfy this test.
"""

import json
from pathlib import Path

import pytest

from app.planner.check import check
from app.planner.params import kb_json
from app.planner.plan import build
from app.rag import corpus as kb_corpus
from physique_schemas.models import UserProfile

FIXTURES = Path(__file__).parent / "fixtures"
FULL_GEAR = ["barbell", "dumbbell", "cable", "machine", "body only"]
BASE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", goal="recomp",
    equipment=FULL_GEAR,
)  # fmt: skip
FLOORS = {g: e["min_sets_per_week"] for g, e in kb_json("baselines.json")["groups"].items()}
CORPUS = kb_corpus.load()
BY_ID = {c["id"]: c for c in CORPUS.chunks}
PRIORITIES = [[], ["shoulders", "chest", "upper_arm"], ["hips", "thigh", "calf"]]


@pytest.mark.parametrize("mode", ["hybrid", "bm25"])
@pytest.mark.parametrize("level", ["beginner", "intermediate", "advanced"])
@pytest.mark.parametrize("days", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("priority", PRIORITIES, ids=["none", "upper", "lower"])
def test_every_group_is_at_its_floor_or_named(mode, level, days, priority):
    ranks = json.loads((FIXTURES / f"planner_ranks_{mode}.json").read_text())["ranks"]
    profile = UserProfile(**{**BASE, "level": level, "days_per_week": days})
    key = f"{level}|{','.join(sorted(FULL_GEAR))}"
    plan = build(profile, priority, CORPUS.chunks, ranks[key])

    week = check(plan.days, plan.shortfalls, BY_ID)  # raises if a shortfall is silent
    named = {s.group for s in plan.shortfalls}
    assert {g for g in FLOORS if week[g] < FLOORS[g]} == named
    assert not named, f"full gear, {days} days, {level}: floors missed for {sorted(named)}"
    assert plan.empty_days == [] and plan.note == ""
