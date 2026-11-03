"""The user's picked regions (`focus`, ADR-060) and the unchanged default rule (ADR-054)."""

import pytest
from pydantic import ValidationError

from app.planner.priorities import MAX_PRIORITIES, priorities
from physique_schemas.models import REGIONS, GapProfile, PlanRequest, RegionGap, UserProfile

UNMEASURED = {"upper_back", "lats"}


def _region(r: str, deltas: dict[str, float]) -> RegionGap:
    if r in UNMEASURED:  # an unmeasured region carries no numbers (the schema enforces it)
        return RegionGap(delta=0.0, confidence=0.0, measured=False, reason="x")
    return RegionGap(delta=deltas.get(r, 0.0), confidence=0.5)


def _gap(deltas: dict[str, float]) -> GapProfile:
    return GapProfile(analyzer="siamese_subtract", regions={r: _region(r, deltas) for r in REGIONS})


GAP = _gap({"shoulders": 0.04, "chest": 0.03, "thigh": 0.02, "calf": -0.01, "upper_arm": 0.015, "forearm": 0.005})


def test_no_focus_is_the_gap_rule_unchanged():
    assert priorities(GAP) == ["shoulders", "chest", "thigh"]
    assert priorities(GAP, []) == priorities(GAP)


def test_focus_replaces_the_gap_rule_largest_gap_first():
    assert priorities(GAP, ["thigh", "calf"]) == ["thigh", "calf"]
    assert priorities(GAP, ["forearm"]) == ["forearm"]  # a region below MIN_GAP is still prioritised when asked for


def test_focus_is_capped_at_three_regions():
    out = priorities(GAP, ["shoulders", "chest", "thigh", "calf", "upper_arm"])
    assert len(out) == MAX_PRIORITIES and out == ["shoulders", "chest", "thigh"]


def test_unrankable_or_unmeasured_picks_are_ignored_and_the_gap_rule_applies():
    assert priorities(GAP, ["waist", "lats", "upper_back"]) == priorities(GAP)
    assert priorities(GAP, ["waist", "forearm"]) == ["forearm"]  # a rankable pick survives next to an unrankable one


def test_plan_request_accepts_focus_and_rejects_unknown_regions():
    profile = UserProfile(
        height_cm=175, weight_kg=72, age=29, sex="male", activity="light", level="beginner", goal="recomp",
        days_per_week=3, equipment=["body only"],
    )  # fmt: skip
    ok = PlanRequest(profile=profile, gap=GAP, focus=["thigh", "calf"])
    assert ok.focus == ["thigh", "calf"] and PlanRequest(profile=profile, gap=GAP).focus == []
    with pytest.raises(ValidationError):
        PlanRequest(
            profile=profile, gap=GAP, focus=["legs"]
        )  # the UI's "Legs" is mapped to thigh + calf before it is sent
