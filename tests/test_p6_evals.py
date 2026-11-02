"""Shape checks for the frozen P6 eval files, and the false-positive guards for their output patterns.

The priorities in every case are recomputed from the stated rule (evals/p6-README.md) so a wrong
expectation in a case fails here, before any planner exists.
"""

import importlib.util
import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from physique_schemas.models import REGIONS, Equipment, UserProfile

EVALS = Path(__file__).resolve().parent.parent / "evals"
_spec = importlib.util.spec_from_file_location("p6_patterns", EVALS / "patterns.py")
patterns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patterns)

BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip
INPUT_KEYS = {
    "profile", "scope", "definition", "gap", "not_measured", "confidence", "goal_front_only",
    "goal_sex", "equipment", "extra_fields",
}  # fmt: skip
EXPECT_KEYS = {
    "priorities", "days", "energy", "invariant", "body_fat", "must_include", "must_not_contain",
    "supportive_note", "cut_plan", "plan", "definition", "session_cap", "unranked", "realism_note",
    "intermediate_target", "said_once", "bands_shown", "no_regional_definition_claim", "response",
    "note", "shortfall_named_per_region", "unmet_never_marked_covered", "floor_unchanged",
    "kcal_never_below_rmr", "deficit_max_pct_of_tdee", "surplus_max_pct_of_tdee",
    "loss_max_pct_bodyweight_per_week", "no_number_differs_from_body_comp",
    "banned_patterns_absent", "no_growth_target_for_negative_gaps",
}  # fmt: skip
MIN_GAP = 0.010  # priority rule, a placeholder (evals/p6-README.md)
UNRANKED = {"lats", "upper_back", "waist"}


def _read(name: str) -> list[dict]:
    lines = (EVALS / name).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


PLAN, SAFETY, CHECKS = _read("plan.jsonl"), _read("safety.jsonl"), _read("pattern_checks.jsonl")


def priorities(inp: dict) -> list[str]:
    """The README priority rule applied to a case's gap."""
    conf, skipped = inp.get("confidence", {}), inp.get("not_measured", {})
    rows = [
        (-d, -conf.get(r, 0.6), REGIONS.index(r), r)
        for r, d in inp.get("gap", {}).items()
        if r not in UNRANKED and r not in skipped and round(d, 6) >= MIN_GAP
    ]
    return [r for *_, r in sorted(rows)[:3]]


def test_counts_and_unique_ids():
    assert (len(PLAN), len(SAFETY)) == (15, 40)
    for cases in (PLAN, SAFETY, CHECKS):
        ids = [c["id"] for c in cases]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", PLAN + SAFETY, ids=lambda c: c["id"])
def test_case_shape(case):
    inp, expect = case["input"], case["expect"]
    assert set(inp) <= INPUT_KEYS and expect and set(expect) <= EXPECT_KEYS
    assert set(inp.get("profile", {})) <= set(UserProfile.model_fields)
    for key in ("gap", "not_measured", "confidence"):
        assert set(inp.get(key, {})) <= set(REGIONS)
    if "priorities" in expect and "gap" in inp:
        assert expect["priorities"] == priorities(inp)
    if "days" in expect:
        assert expect["days"] == inp.get("profile", {}).get("days_per_week", 3)


@pytest.mark.parametrize(
    "case", [c for c in PLAN + SAFETY if "equipment" not in c["input"]], ids=lambda c: c["id"]
)  # the equipment cases run in test_out_of_list_equipment_is_rejected and tests/test_plan_route.py
def test_profile_validity_matches_expectation(case):
    inp = case["input"]
    rejected = case["expect"].get("response") == "rejected_422" and "profile" in inp
    if rejected:
        with pytest.raises(ValidationError):
            UserProfile(**{**BASE_PROFILE, **inp["profile"]})
    elif "profile" in inp:
        UserProfile(**{**BASE_PROFILE, **inp["profile"]})


@pytest.mark.parametrize("case", [c for c in SAFETY if "equipment" in c["input"]], ids=lambda c: c["id"])
def test_out_of_list_equipment_is_rejected(case):
    # Echo-free 422 bodies are the API handler's job: tests/test_validation_errors.py.
    assert case["expect"]["response"] == "rejected_422"
    with pytest.raises(ValidationError):
        UserProfile(**{**BASE_PROFILE, "equipment": case["input"]["equipment"]})


def test_equipment_checklist_is_the_planner_pool():
    raw = EVALS.parent / "kb" / "raw" / "exercises.json"
    if not raw.exists():
        pytest.skip("kb/raw/exercises.json not present")
    pool = {
        e["equipment"] or "none" for e in json.loads(raw.read_text()) if e["category"] in ("strength", "powerlifting")
    }
    assert set(get_args(Equipment)) == pool


@pytest.mark.parametrize("case", CHECKS, ids=lambda c: c["id"])
def test_pattern_fixtures(case):
    assert patterns.flagged(case["check"], case["text"]) is case["flag"]
