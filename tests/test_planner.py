"""The planner against the frozen P6 plan cases (ADR-055). Constraints, never exact exercise names.

The planner never calls the retriever: it runs on committed rank fixtures (tests/fixtures), so CI
covers it without the bge-small model. All 15 cases run on the hybrid snapshot; p01, p10 and p12 also run
on the BM25 set. One local smoke test uses the live hybrid retriever and skips without the model.
Group credit is recomputed here with kb.templates.day_credit, independently of the planner and of
app.planner.check.
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from app.planner.candidates import ranked_candidates
from app.planner.check import PlannerInvariantError, check
from app.planner.params import (
    LAYOUTS,
    MAX_SESSIONS_PER_GROUP_PER_WEEK,
    MAX_WEEKLY_SETS_PER_GROUP,
    SHORT_DAYS_NOTE,
    TOP_K,
    kb_json,
    session_limit,
)
from app.planner.plan import Slot, build
from app.planner.priorities import priorities
from app.rag import corpus as kb_corpus
from app.rag.embed import MODEL_FILE, Embedder
from app.rag.retrieve import Retriever
from kb.templates import day_credit, load_exercises
from physique_schemas.models import REGIONS, GapProfile, RegionGap, UserProfile

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
MODEL_DIR = ROOT / "ml" / "data" / "models" / "bge-small"  # gitignored: absent in CI
CASES = [json.loads(line) for line in (ROOT / "evals" / "plan.jsonl").read_text().splitlines()]
SENSITIVITY = {"p01", "p10", "p12"}
BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip
BASELINES = kb_json("baselines.json")
GROUPS = list(BASELINES["groups"])
RANKS = {m: json.loads((FIXTURES / f"planner_ranks_{m}.json").read_text()) for m in ("hybrid", "bm25")}
CORPUS = kb_corpus.load()
BY_ID = {c["id"]: c for c in CORPUS.chunks}


def rank_key(profile: UserProfile) -> str:
    return f"{profile.level}|{','.join(sorted(profile.equipment)) or 'body only'}"


def profile_of(case: dict) -> UserProfile:
    return UserProfile(**{**BASE_PROFILE, **case["input"].get("profile", {})})


def gap_of(inp: dict) -> GapProfile:
    gaps, conf, skipped = inp.get("gap", {}), inp.get("confidence", {}), inp.get("not_measured", {})
    return GapProfile(
        analyzer="siamese_subtract",
        regions={
            r: RegionGap(delta=0, confidence=0, measured=False, reason=skipped[r])
            if r in skipped
            else RegionGap(delta=gaps.get(r, 0.0), confidence=conf.get(r, 0.6))
            for r in REGIONS
        },
    )


def assert_constraints(case: dict, plan, profile: UserProfile) -> None:
    expect = case["expect"]
    assert len(plan.days) == expect["days"] and plan.empty_days == []
    for slot in (s for day in plan.days for s in day):  # retrieved, plannable, in the user's gear
        chunk = BY_ID[slot.exercise_id]
        assert chunk["plannable"] and chunk["equipment"] in (profile.equipment or ["body only"])

    exercises, week = load_exercises(), defaultdict(float)
    sessions: dict[str, set[int]] = defaultdict(set)
    for d, day in enumerate(plan.days):
        credit = day_credit({s.exercise_id.removeprefix("ex:"): s.sets for s in day}, exercises, BASELINES["counting"])
        for group, value in credit.items():
            week[group] += value
            if group in BASELINES["groups"]:
                assert value <= session_limit(BASELINES)
        for s in day:
            sessions_of = BY_ID[s.exercise_id]["muscle"]
            for group in sessions_of:
                sessions[group].add(d)
    for group in GROUPS:
        assert week[group] <= MAX_WEEKLY_SETS_PER_GROUP
        assert len(sessions[group]) <= MAX_SESSIONS_PER_GROUP_PER_WEEK
    short = {g for g in GROUPS if week[g] < BASELINES["groups"][g]["min_sets_per_week"]}

    assert short == {s.group for s in plan.shortfalls}  # named, never silent
    for s in plan.shortfalls:
        assert s.planned == week[s.group]
        assert s.floor == BASELINES["groups"][s.group]["min_sets_per_week"]
    if expect["invariant"] == "met":
        assert not short, f"planner finding, floor not lowered: {sorted(short)}"


def run(case: dict, mode: str):
    profile = profile_of(case)
    ranks = RANKS[mode]["ranks"][rank_key(profile)]
    return profile, build(profile, priorities(gap_of(case["input"])), CORPUS.chunks, ranks)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_priorities_match_the_frozen_expectation(case):
    assert priorities(gap_of(case["input"])) == case["expect"]["priorities"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_plan_on_the_hybrid_snapshot(case):
    profile, plan = run(case, "hybrid")
    assert_constraints(case, plan, profile)


@pytest.mark.parametrize("case", [c for c in CASES if c["id"] in SENSITIVITY], ids=lambda c: c["id"])
def test_plan_on_the_bm25_set(case):
    profile, plan = run(case, "bm25")
    assert_constraints(case, plan, profile)


def test_fixtures_match_the_corpus_and_cover_every_case():
    for mode, fixture in RANKS.items():
        assert fixture["corpus_hash"] == CORPUS.manifest["corpus_hash"], f"regenerate {mode}"
        assert {rank_key(profile_of(c)) for c in CASES} <= fixture["ranks"].keys()


def test_every_layout_covers_every_group_and_has_the_right_length():
    for n, layout in LAYOUTS.items():
        assert len(layout) == n and set().union(*layout) == set(GROUPS)


def test_priority_groups_get_the_extra_sets():
    case = next(c for c in CASES if c["id"] == "p01")
    _, plan = run(case, "hybrid")
    region_map = kb_json("region_map.json")["regions"]
    for region in case["expect"]["priorities"]:
        for group in region_map[region]["groups"]:
            assert plan.weekly[group] >= BASELINES["groups"][group]["min_sets_per_week"] + 2


def test_note_is_shown_with_a_shortfall_at_two_days_or_fewer():
    plan = run(next(c for c in CASES if c["id"] == "p12"), "hybrid")[1]  # 1 day
    assert plan.shortfalls and plan.note == SHORT_DAYS_NOTE


@pytest.mark.parametrize("level", ["beginner", "intermediate", "advanced"])
def test_note_is_absent_without_a_shortfall_at_two_days(level):
    profile = UserProfile(**{**BASE_PROFILE, "level": level, "days_per_week": 2})
    plan = build(profile, [], CORPUS.chunks, RANKS["hybrid"]["ranks"][rank_key(profile)])
    if plan.shortfalls:
        pytest.skip(f"2-day {level} case has a shortfall: {sorted(s.group for s in plan.shortfalls)}")
    assert plan.note == ""


def test_note_is_absent_with_a_shortfall_at_three_days_or_more():
    plan = run(next(c for c in CASES if c["id"] == "p10"), "hybrid")[1]  # 3 days
    assert plan.shortfalls and plan.note == ""


def test_plan_is_deterministic():
    case = CASES[0]
    assert run(case, "hybrid")[1] == run(case, "hybrid")[1]


def test_empty_equipment_pool_names_every_group_and_does_not_raise():
    plan = build(UserProfile(**BASE_PROFILE), [], CORPUS.chunks, {})
    assert {s.group for s in plan.shortfalls} == set(GROUPS)
    assert all(s.planned == 0 for s in plan.shortfalls) and len(plan.empty_days) == 3


def test_a_candidate_from_another_groups_list_is_refused():
    chest = next(c["id"] for c in CORPUS.chunks if c["kind"] == "exercise" and c["muscle"] == ["chest"])
    with pytest.raises(ValueError, match="does not train it"):
        build(UserProfile(**BASE_PROFILE), [], CORPUS.chunks, {"calves": [chest]})


def test_check_raises_on_a_silent_shortfall_and_on_a_cap_breach():
    with pytest.raises(PlannerInvariantError, match="not named"):
        check([[], [], []], [], BY_ID)
    chest = next(c for c in CORPUS.chunks if c["kind"] == "exercise" and c["muscle"] == ["chest"])
    with pytest.raises(PlannerInvariantError, match="session limit"):
        check([[Slot(chest["id"], chest["name"], 20)], [], []], [], BY_ID)


@pytest.mark.skipif(not (MODEL_DIR / MODEL_FILE).exists(), reason="bge-small model not present")
def test_live_hybrid_ranks_match_the_snapshot_and_plan_passes():
    case = next(c for c in CASES if c["id"] == "p01")
    profile = profile_of(case)
    retriever = Retriever(CORPUS, Embedder(MODEL_DIR))
    live = ranked_candidates(profile, lambda q, f: retriever.search(q, "hybrid", f, k=TOP_K))
    assert live == RANKS["hybrid"]["ranks"][rank_key(profile)], "snapshot is stale: regenerate"
    assert_constraints(case, build(profile, case["expect"]["priorities"], CORPUS.chunks, live), profile)
