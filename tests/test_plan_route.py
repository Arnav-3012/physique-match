"""POST /plan (ADR-057): validation, echo-free 422s, body composition and safety through the route.

The frozen eval cases are run here for their deterministic parts only (energy, cut_plan, the 422 gates and
the equipment checklist). Report-text expectations (banned language, must_include, supportive_note,
realism) need the LLM writer and stay skipped with a reason.
"""

import json
import logging
from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from app import security
from app.agent.llm import LLMError
from app.agent.textsafety import text_is_safe
from app.main import app
from app.rag.service import load_retrieval
from app.routes import plan as plan_route
from physique_schemas.models import REGIONS, BodyComp, EnergyTarget, Equipment, PlanResponse

SAFETY = [
    json.loads(line)
    for line in (Path(__file__).resolve().parent.parent / "evals" / "safety.jsonl").read_text().splitlines()
]
BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip
GAP = {"analyzer": "siamese_subtract", "regions": {r: {"delta": 0.0, "confidence": 0.6} for r in REGIONS}}
WAIST_G = 0.5  # supplied by this harness: the safety cases carry no waist (reported in the dev-log)
client = TestClient(app)


@pytest.fixture(autouse=True)
def _generous_rate_limit(monkeypatch):
    monkeypatch.setattr(security, "limiter", security.SlidingWindowLimiter(1_000_000))
    monkeypatch.delenv("LLM_WRITER_ENABLED", raising=False)  # the writer is off unless a test turns it on


def payload(profile=None, waist_g=WAIST_G, **extra) -> dict:
    return {"profile": {**BASE_PROFILE, **(profile or {})}, "gap": GAP, "waist_g": waist_g, **extra}


def case(case_id: str) -> dict:
    return next(c for c in SAFETY if c["id"] == case_id)


def ids(prefix_from: int, to: int) -> list[dict]:
    return [c for c in SAFETY if prefix_from <= int(c["id"][1:]) <= to]


def assert_caps(bc: dict, weight_kg: float) -> None:
    rmr, tdee = bc["rmr_kcal"], bc["tdee_kcal"]
    for t in bc["energy"]:
        assert t["kcal_low"] >= max(rmr, 0.8 * tdee) - 0.5  # deficit <= 20% of TDEE, never below RMR
        assert t["kcal_high"] <= 1.1 * tdee + 0.5  # surplus <= 10%
        assert (tdee - t["kcal_low"]) * 7 / 7700 <= 0.01 * weight_kg + 0.01  # loss <= 1% / week


# --- the happy path ---------------------------------------------------------
def test_a_valid_request_returns_a_deterministic_plan():
    r = client.post("/plan", json=payload())
    assert r.status_code == 200, r.text
    out = PlanResponse.model_validate(r.json())
    assert (out.served_by, out.fallback_reason) == ("deterministic", "disabled")
    assert out.retrieval == plan_route.RETRIEVAL.mode and out.report is not None
    assert len(out.days) == 3 and all(out.days) and out.body_comp is not None


# --- validation: generic, echo-free 422 -------------------------------------
@pytest.mark.parametrize("c", ids(21, 30), ids=lambda c: c["id"])
def test_out_of_list_equipment_is_a_422_that_does_not_echo(c):
    r = client.post("/plan", json=payload({"equipment": c["input"]["equipment"]}))
    assert r.status_code == 422
    for bad in (e for e in c["input"]["equipment"] if e not in get_args(Equipment)):
        assert bad not in r.text


def gate_payload(inp: dict) -> dict:
    extra = dict(inp.get("extra_fields", {}))
    extra.update({k: inp[k] for k in ("scope", "definition") if k in inp})
    return payload(inp.get("profile"), **extra)


@pytest.mark.parametrize("c", ids(36, 40), ids=lambda c: c["id"])
def test_the_gates_are_422s(c):
    r = client.post("/plan", json=gate_payload(c["input"]))
    assert r.status_code == 422
    assert not any(s in r.text for s in ("base64", "AAAA", "image"))


def test_bounds_on_waist_and_gap_are_422s_without_the_value():
    for bad in (payload(waist_g=5), payload(waist_g=0.01)):
        r = client.post("/plan", json=bad)
        assert r.status_code == 422 and '"input"' not in r.text
    wide = {**GAP, "regions": {**GAP["regions"], "chest": {"delta": 2.0, "confidence": 0.6}}}
    r = client.post("/plan", json={**payload(), "gap": wide})
    assert r.status_code == 422 and "2.0" not in r.text


# --- body composition and the deterministic safety expectations --------------
@pytest.mark.parametrize("c", ids(1, 15), ids=lambda c: c["id"])
def test_energy_and_cut_expectations(c):
    inp, expect = c["input"], c["expect"]
    r = client.post("/plan", json=payload(inp.get("profile")))
    assert r.status_code == 200, r.text
    bc = r.json()["body_comp"]
    goals = [t["goal"] for t in bc["energy"]]
    weight = {**BASE_PROFILE, **inp.get("profile", {})}["weight_kg"]
    if expect.get("energy") == "none":
        assert bc["energy"] == [] and bc["energy_measured"] is False
    if expect.get("energy") == "within_caps":
        assert bc["energy"]
    assert_caps(bc, weight)
    if "cut_plan" in expect:
        assert ("cut" in goals) is expect["cut_plan"]
    if expect.get("plan") == "maintenance_strength_only":
        assert set(goals) <= {"maintain"}


def test_no_waist_returns_the_plan_without_body_composition():
    r = client.post("/plan", json=payload(waist_g=None))
    assert r.status_code == 200
    out = r.json()
    assert out["body_comp"] is None and out["body_comp_note"] == plan_route.NOTE_NO_WAIST
    assert len(out["days"]) == 3


def test_no_waist_with_an_eating_disorder_history_gets_the_supportive_note_not_the_calorie_note():
    out = client.post("/plan", json=payload({"ed_history": True}, waist_g=None)).json()
    assert out["body_comp"]["energy"] == [] and out["body_comp"]["note"] == plan_route.NOTE_ED
    assert out["body_comp_note"] == ""


@pytest.mark.parametrize("profile", [{"tape_waist_cm": 85.0}, {"sex": "not_specified"}])
def test_a_tape_waist_or_an_unset_sex_needs_no_waist_g(profile):
    out = client.post("/plan", json=payload(profile, waist_g=None)).json()
    assert out["body_comp"] is not None and out["body_comp_note"] == ""


# --- an internal breach is a generic 500 ------------------------------------
def test_a_safety_breach_is_a_generic_500_with_the_detail_only_in_the_log(monkeypatch, caplog):
    leaky = BodyComp.model_construct(
        body_fat=None,
        rmr_kcal=1700,
        tdee_kcal=2500,
        protein=None,
        energy_measured=True,
        note="",
        energy=[EnergyTarget(goal="maintain", kcal_low=2400, kcal_high=2500, protein_g_low=1, protein_g_high=2)],
    )  # calorie numbers for an eating-disorder history: the rules must refuse
    monkeypatch.setattr(plan_route, "body_comp", lambda p, w=None: leaky)
    with caplog.at_level(logging.ERROR, logger="plan"):
        r = client.post("/plan", json=payload({"ed_history": True}))
    assert r.status_code == 500 and r.json() == {"detail": "internal error"}
    assert "calorie numbers given" in caplog.text and "calorie numbers" not in r.text


# --- retrieval startup ------------------------------------------------------
def test_missing_embedder_falls_back_to_bm25_or_fails_when_required(tmp_path):
    assert load_retrieval(tmp_path, require=False).mode == "bm25"
    with pytest.raises(RuntimeError, match="REQUIRE_EMBEDDER"):
        load_retrieval(tmp_path, require=True)


# --- the report text: code-written here, the writer's text is tested in test_agent_writer.py ---
def region_gap(inp: dict) -> dict:
    gaps, skipped = inp.get("gap", {}), inp.get("not_measured", {})
    regions = {
        r: {"delta": 0, "confidence": 0, "measured": False, "reason": skipped[r]}
        if r in skipped
        else {"delta": gaps.get(r, 0.0), "confidence": 0.6}
        for r in REGIONS
    }
    return {"analyzer": "siamese_subtract", "regions": regions}


def report_text(data: dict) -> str:
    rep = data["report"]
    parts = [rep["summary"], rep["realism_note"] or "", rep["disclaimer"], *data["warnings"]]
    for day in rep["plan"]:
        parts += [day["label"], *(e["cue"] for e in day["exercises"])]
    return " ".join(parts)


@pytest.mark.parametrize("c", ids(16, 20) + ids(31, 35), ids=lambda c: c["id"])
def test_report_text_cases(c):
    inp, expect = c["input"], c["expect"]
    body = payload(gap=region_gap(inp), goal_sex=inp.get("goal_sex", "not_specified"))
    r = client.post("/plan", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    text = report_text(data)
    assert text_is_safe(text)  # banned patterns, regional claims, drug words, URLs, markdown, HTML
    if "priorities" in expect:
        assert data["priorities"] == expect["priorities"]
    for s in expect.get("must_not_contain", []):
        assert s.lower() not in text.lower()
    for s in expect.get("must_include", []):
        assert s.lower() in text.lower()
    if "realism_note" in expect:
        assert (data["report"]["realism_note"] is not None) is expect["realism_note"]
    if expect.get("intermediate_target"):
        assert "first target" in data["report"]["realism_note"]
    if expect.get("said_once"):
        assert text.count("first target") == 1
    for reason in inp.get("not_measured", {}).values():  # client text is never echoed
        assert reason not in text


# --- the LLM writer through the route, on a fake LLM --------------------------
class EchoLLM:
    """Answers with the baseline plan of the facts it is given; records what it saw."""

    def __init__(self, fail: str | None = None):
        self.calls, self.fail, self.seen = 0, fail, []

    def available(self, provider):
        return True

    def complete(self, provider, messages, schema):
        self.calls += 1
        self.seen.append(json.dumps(messages))
        if self.fail:
            raise LLMError(self.fail)
        facts = json.loads(messages[1]["content"])
        days = [
            {
                "label": f"Day {d['day']}",
                "items": [{"exercise_id": i, "reps": "8-12", "cue": "Steady reps."} for i in d["baseline_plan"]],
            }
            for d in facts["plan_days"]
        ]
        return json.dumps({"summary": "A steady plan.", "days": days})


def test_writer_enabled_serves_the_llm_text_and_keeps_the_planners_sets(monkeypatch):
    llm = EchoLLM()
    monkeypatch.setenv("LLM_WRITER_ENABLED", "true")
    monkeypatch.setattr(plan_route, "LLM_CLIENT", llm)
    data = client.post("/plan", json=payload()).json()
    assert (data["served_by"], data["fallback_reason"], llm.calls) == ("primary", None, 1)
    assert data["report"]["summary"] == "A steady plan."
    assert {s["sets"] for d in data["days"] for s in d} == {3}


def test_a_failing_provider_serves_the_deterministic_plan_with_a_closed_reason(monkeypatch):
    llm = EchoLLM(fail="rate_limited")
    monkeypatch.setenv("LLM_WRITER_ENABLED", "true")
    monkeypatch.setattr(plan_route, "LLM_CLIENT", llm)
    data = client.post("/plan", json=payload()).json()
    assert (data["served_by"], data["fallback_reason"]) == ("deterministic", "rate_limited")
    assert llm.calls <= 3 and data["report"] is not None


def test_client_text_never_reaches_the_model(monkeypatch):
    llm = EchoLLM()
    monkeypatch.setenv("LLM_WRITER_ENABLED", "true")
    monkeypatch.setattr(plan_route, "LLM_CLIENT", llm)
    gap = region_gap({"gap": {"chest": 0.03}, "not_measured": {"lats": "SECRET-CLIENT-TEXT"}})
    r = client.post("/plan", json=payload(gap=gap))
    assert r.status_code == 200
    assert llm.seen and not any("SECRET-CLIENT-TEXT" in s for s in llm.seen)
    assert "SECRET-CLIENT-TEXT" not in r.text  # nor echoed back anywhere in the response
    assert r.json()["report"]["gap"]["v_taper"] is None


def test_must_include_and_supportive_note_wait_for_a_text_rubric():
    pytest.skip("must_include / supportive_note in s01-s10 are rubric items on the final text, scored by the owner")
