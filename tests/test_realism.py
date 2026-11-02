"""The realism note and the fixed warnings (ADR-058), against the frozen cases: s31-s35 and p15 expect a
note exactly when the largest priority gap is at least REALISM_GAP; nothing is tuned to them."""

import json
from pathlib import Path

import pytest

from app.agent.textsafety import text_is_safe
from app.planner.priorities import priorities
from app.planner.realism import OTHER_SEX, REALISM_GAP, realism_note, warnings
from physique_schemas.models import REGIONS, UPPER_BACK_REASON, GapProfile, RegionGap

EVALS = Path(__file__).resolve().parent.parent / "evals"
SAFETY = {c["id"]: c for c in map(json.loads, (EVALS / "safety.jsonl").read_text().splitlines())}
PLAN = {c["id"]: c for c in map(json.loads, (EVALS / "plan.jsonl").read_text().splitlines())}
HEIGHT = 175


def gap_of(inp: dict) -> GapProfile:
    gaps, skipped = inp.get("gap", {}), inp.get("not_measured", {})
    return GapProfile(
        analyzer="siamese_subtract",
        regions={
            r: RegionGap(delta=0, confidence=0, measured=False, reason=skipped[r])
            if r in skipped
            else RegionGap(delta=gaps.get(r, 0.0), confidence=0.6)
            for r in REGIONS
        },
    )


def note_of(inp: dict) -> str | None:
    gap = gap_of(inp)
    return realism_note(gap, priorities(gap), HEIGHT)


def test_the_threshold_is_the_one_fixed_in_the_adr():
    assert REALISM_GAP == 0.05


@pytest.mark.parametrize("case_id", ["s31", "s32", "s33", "s34", "s35"])
def test_the_frozen_realism_cases(case_id):
    expect = SAFETY[case_id]["expect"]
    note = note_of(SAFETY[case_id]["input"])
    assert (note is not None) is expect["realism_note"]
    if note and expect.get("intermediate_target"):
        assert "first target" in note
    if note and expect.get("said_once"):
        assert note.count("first target") == 1


def test_p15_expects_a_note_with_an_intermediate_target_and_p01_none():
    assert "first target" in note_of(PLAN["p15"]["input"])
    assert note_of(PLAN["p01"]["input"]) is None


def test_the_target_is_half_the_largest_gap():
    note = note_of({"gap": {"upper_arm": 0.10}})
    assert "5.0% of your height" in note and "9 cm" in note  # 0.05 x 175 = 8.75


def test_a_negative_gap_never_triggers_a_note():
    assert note_of({"gap": {"thigh": -0.08, "chest": -0.5}}) is None


def test_the_boundary_is_inclusive():
    assert note_of({"gap": {"chest": 0.05}}) is not None
    assert note_of({"gap": {"chest": 0.049}}) is None


def test_a_different_sex_goal_adds_the_fixed_sentence():
    gap = gap_of(SAFETY["s33"]["input"])
    assert OTHER_SEX in warnings(gap, "male", "female")
    assert OTHER_SEX not in warnings(gap, "male", "male")
    assert OTHER_SEX not in warnings(gap, "male", "not_specified")


def test_unmeasured_regions_are_named_from_the_enum_and_never_from_client_text():
    gap = gap_of({"not_measured": {"lats": "SECRET client text", "upper_back": "also client text", "hips": "x"}})
    text = " ".join(warnings(gap, "male", "male"))
    assert "back width (estimate)" in text and "upper back" in text and "hips" in text
    assert UPPER_BACK_REASON in text and "client text" not in text and "SECRET" not in text


def test_every_code_written_sentence_passes_the_text_checks():
    sentences = [realism_note(gap_of({"gap": {"chest": 0.08}}), ["chest"], HEIGHT), OTHER_SEX]
    sentences += warnings(gap_of({"not_measured": {"lats": "", "upper_back": "", "waist": ""}}), "male", "female")
    assert all(text_is_safe(s) for s in sentences)
