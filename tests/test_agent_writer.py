"""The LLM writer (ADR-058) on a FAKE LLM: acceptance checks, rejection reasons, the call budget, and
the HTTP client against a mock transport. Nothing here calls a real provider, and nothing here is a
live pass rate."""

import json
import logging
from pathlib import Path

import httpx
import pytest

from app.agent.llm import HttpxLLM, LLMError
from app.agent.schema import WriterOutput
from app.agent.writer import Context, build_facts, deterministic_summary, write
from app.planner.params import LAYOUTS
from app.planner.plan import build
from app.rag import corpus as kb_corpus
from physique_schemas.models import REGIONS, GapProfile, RegionGap, UserProfile

FIXTURES = Path(__file__).parent / "fixtures"
BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip
CORPUS = kb_corpus.load()
BY_ID = {c["id"]: c for c in CORPUS.chunks}
PROFILE = UserProfile(**BASE_PROFILE)
CANDS = json.loads((FIXTURES / "planner_ranks_hybrid.json").read_text())["ranks"][
    "beginner|barbell,body only,cable,dumbbell,machine"
]
CHOSEN = ["shoulders", "chest", "upper_arm"]
GAP = GapProfile(
    analyzer="siamese_subtract",
    regions={
        r: RegionGap(delta={"shoulders": 0.03, "chest": 0.025, "upper_arm": 0.02}.get(r, 0.0), confidence=0.6)
        for r in REGIONS
    },
)
DET = build(PROFILE, CHOSEN, CORPUS.chunks, CANDS)
FOCUS = LAYOUTS[3]
FACTS, ALLOWED = build_facts(CHOSEN, GAP, None, FOCUS, CANDS, BY_ID, DET)
CTX = Context(FOCUS, CANDS, BY_ID, DET, ALLOWED)
SUMMARY = deterministic_summary(CHOSEN, 3, bool(DET.shortfalls))


def good() -> dict:
    return {
        "summary": "A steady plan that trains every muscle group.",
        "days": [
            {
                "label": f"Day {i + 1}",
                "items": [{"exercise_id": s.exercise_id, "reps": "8-12", "cue": "Controlled reps."} for s in day],
            }
            for i, day in enumerate(DET.days)
        ],
    }


class FakeLLM:
    def __init__(self, script: dict[str, list], available=("primary", "fallback")):
        self.script, self.avail, self.calls = {k: list(v) for k, v in script.items()}, available, []

    def available(self, provider):
        return provider in self.avail

    def complete(self, provider, messages, schema):
        self.calls.append(provider)
        item = self.script[provider].pop(0)
        if isinstance(item, LLMError):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def run(llm, enabled=True):
    return write(enabled=enabled, llm=llm, facts=FACTS, ctx=CTX, summary=SUMMARY)


def test_disabled_serves_the_deterministic_plan_without_a_call():
    llm = FakeLLM({})
    out = run(llm, enabled=False)
    assert (out.served_by, out.fallback_reason, llm.calls) == ("deterministic", "disabled", [])
    assert out.days == DET.days and out.summary == SUMMARY


def test_a_valid_output_is_served_by_the_primary():
    llm = FakeLLM({"primary": [good()]})
    out = run(llm)
    assert (out.served_by, out.fallback_reason, llm.calls) == ("primary", None, ["primary"])
    assert [[s.exercise_id for s in d] for d in out.days] == [[s.exercise_id for s in d] for d in DET.days]
    assert all(s.sets == 3 for d in out.days for s in d)  # sets come from the planner, not the model
    assert out.summary == good()["summary"]


def test_the_prompt_fits_the_free_tier_token_limit():
    """The first live prompt was 94,225 characters (~24k tokens) against Groq's 8,000 tokens a minute.
    20,000 characters (~5k tokens, ours) leaves room for the answer in the same budget."""
    from app.agent.writer import SYSTEM

    size = len(SYSTEM) + len(json.dumps(FACTS))
    assert size < 20_000, size
    assert FACTS["exercises"] and all(i in FACTS["exercises"] for d in FACTS["plan_days"] for i in d["baseline_plan"])


def test_how_text_is_the_instructions_not_the_metadata_header():
    """The first live run showed the model copying `Name. Targets x. Equipment: y. Level: z.` as a cue."""
    from app.agent.writer import how_text

    assert FACTS["exercises"]
    for entry in FACTS["exercises"].values():
        assert entry["how"], entry["name"]  # every exercise has instructions to cue from
        assert "Equipment:" not in entry["how"] and "Level:" not in entry["how"]
        assert not entry["how"].startswith(entry["name"])
    assert (
        how_text("Name. Targets chest. Equipment: body only. Level: beginner. compound push strength. Lie flat.")
        == "Lie flat."
    )
    assert how_text("no header here") == ""


def test_a_cue_that_is_the_metadata_header_is_rejected():
    bad = _mutate(lambda o: o["days"][0]["items"][0].update(cue="Thigh Abductor. Equipment: machine. Level: beginner"))
    out = run(FakeLLM({"primary": [bad]}, available=("primary",)))
    assert (out.served_by, out.fallback_reason) == ("deterministic", "check_failed")


def test_the_output_schema_has_no_sets_field():
    props = WriterOutput.model_json_schema()["$defs"]["WriterItem"]["properties"]
    assert "sets" not in props


def test_a_transient_failure_is_retried_once():
    llm = FakeLLM({"primary": [LLMError("timeout"), good()]})
    out = run(llm)
    assert (out.served_by, llm.calls) == ("primary", ["primary", "primary"])


def test_the_fallback_serves_and_records_why_the_primary_failed():
    bad = good()
    bad["days"][0]["items"][0]["exercise_id"] = "ex:Not_A_Candidate"
    llm = FakeLLM({"primary": [bad], "fallback": [good()]})
    out = run(llm)
    assert (out.served_by, out.fallback_reason, llm.calls) == ("fallback", "check_failed", ["primary", "fallback"])


def test_the_worst_case_is_three_calls_and_the_deterministic_plan():
    llm = FakeLLM({"primary": [LLMError("timeout"), LLMError("timeout")], "fallback": [LLMError("rate_limited")]})
    out = run(llm)
    assert len(llm.calls) == 3
    assert (out.served_by, out.fallback_reason) == ("deterministic", "rate_limited")


def test_no_provider_configured_is_disabled():
    out = run(FakeLLM({}, available=()))
    assert (out.served_by, out.fallback_reason) == ("deterministic", "disabled")


def _mutate(fn):
    out = good()
    fn(out)
    return out


REJECTIONS = {
    "not json": ("not json at all", "invalid_json"),
    "extra sets field": (_mutate(lambda o: o["days"][0]["items"][0].update(sets=4)), "invalid_json"),
    "unknown id": (_mutate(lambda o: o["days"][0]["items"][0].update(exercise_id="ex:Nope")), "check_failed"),
    "repeat in a day": (
        _mutate(lambda o: o["days"][0]["items"].append(dict(o["days"][0]["items"][0]))),
        "check_failed",
    ),
    "a day missing": (_mutate(lambda o: o["days"].pop()), "check_failed"),
    "more shortfalls than the planner": (
        _mutate(lambda o: [d.update(items=d["items"][:1]) for d in o["days"]]),
        "check_failed",
    ),
    "url in a cue": (_mutate(lambda o: o["days"][0]["items"][0].update(cue="See https://x.com")), "unsafe_text"),
    "markdown link": (_mutate(lambda o: o["days"][0]["items"][0].update(cue="[more](x)")), "unsafe_text"),
    "html": (_mutate(lambda o: o["days"][0]["items"][0].update(cue="<b>push</b>")), "unsafe_text"),
    "banned word": (_mutate(lambda o: o.update(summary="Your weak chest needs work.")), "unsafe_text"),
    "supplement": (_mutate(lambda o: o["days"][0]["items"][0].update(cue="Take creatine first")), "unsafe_text"),
    "regional claim": (
        _mutate(lambda o: o.update(summary="Your abs are more defined than your chest.")),
        "unsafe_text",
    ),
    "invented number": (_mutate(lambda o: o.update(summary="Expect results in 12 weeks.")), "unsafe_text"),
}


@pytest.mark.parametrize("name", sorted(REJECTIONS))
def test_a_bad_output_is_discarded_whole(name):
    bad, reason = REJECTIONS[name]
    llm = FakeLLM({"primary": [bad]}, available=("primary",))
    out = run(llm)
    assert (out.served_by, out.fallback_reason) == ("deterministic", reason)
    assert out.days == DET.days  # nothing from the bad output is used


def test_a_number_from_the_facts_is_allowed():
    ok = _mutate(lambda o: o.update(summary="Your shoulders gap is about 3% of your height."))
    assert run(FakeLLM({"primary": [ok]})).served_by == "primary"


# --- the HTTP client, on a mock transport ------------------------------------
@pytest.fixture
def env(monkeypatch):
    for k, v in {
        "LLM_BASE_URL": "https://primary.test/v1", "LLM_MODEL": "model-a", "GROQ_API_KEY": "SECRET-PRIMARY",
        "LLM_FALLBACK_BASE_URL": "https://fallback.test/v1", "LLM_FALLBACK_MODEL": "model-b",
        "OPENROUTER_API_KEY": "SECRET-FALLBACK",
    }.items():  # fmt: skip
        monkeypatch.setenv(k, v)


def ok_response(content="{}"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_the_request_is_openai_compatible_and_carries_the_key_only_in_the_header(env):
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers["authorization"], request.read()
        return ok_response("RESPONSE-MARKER")

    llm = HttpxLLM(transport=httpx.MockTransport(handler))
    assert (
        llm.complete("primary", [{"role": "user", "content": "PROMPT-MARKER"}], {"type": "object"}) == "RESPONSE-MARKER"
    )
    body = json.loads(seen["body"])
    assert seen["url"] == "https://primary.test/v1/chat/completions"
    assert seen["auth"] == "Bearer SECRET-PRIMARY" and "SECRET" not in seen["body"].decode()
    assert body["model"] == "model-a" and body["response_format"]["type"] == "json_schema"
    assert "provider" not in body


def test_the_fallback_asks_not_to_train_on_the_prompt(env):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.read())
        return ok_response()

    HttpxLLM(transport=httpx.MockTransport(handler)).complete("fallback", [], {})
    assert seen["body"]["provider"] == {"data_collection": "deny"}


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(429), "rate_limited"),
        (httpx.Response(413), "rate_limited"),  # Groq's tokens-per-minute answer, seen in the live probe
        (httpx.Response(500), "timeout"),
        (httpx.Response(503), "timeout"),
        (httpx.Response(400), "invalid_json"),
        (httpx.Response(200, text="not json"), "invalid_json"),
        (httpx.Response(200, json={"choices": []}), "invalid_json"),
        (ok_response("   "), "invalid_json"),
    ],
)
def test_provider_failures_map_to_the_closed_reasons(env, response, reason):
    llm = HttpxLLM(transport=httpx.MockTransport(lambda request: response))
    with pytest.raises(LLMError) as err:
        llm.complete("primary", [], {})
    assert err.value.reason == reason


def test_a_transport_error_is_a_timeout(env):
    def handler(request):
        raise httpx.ConnectTimeout("slow")

    with pytest.raises(LLMError) as err:
        HttpxLLM(transport=httpx.MockTransport(handler)).complete("primary", [], {})
    assert err.value.reason == "timeout"


def test_a_missing_key_is_disabled(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    llm = HttpxLLM()
    assert not llm.available("primary")
    with pytest.raises(LLMError) as err:
        llm.complete("primary", [], {})
    assert err.value.reason == "disabled"


def test_neither_the_key_nor_the_prompt_nor_the_response_is_logged(env, caplog):
    llm = HttpxLLM(transport=httpx.MockTransport(lambda request: ok_response("RESPONSE-MARKER")))
    with caplog.at_level(logging.DEBUG):
        llm.complete("primary", [{"role": "user", "content": "PROMPT-MARKER"}], {})
        with pytest.raises(LLMError):
            HttpxLLM(transport=httpx.MockTransport(lambda request: httpx.Response(500))).complete("primary", [], {})
    assert not any(s in caplog.text for s in ("SECRET", "PROMPT-MARKER", "RESPONSE-MARKER"))
    assert "outcome=ok" in caplog.text and "outcome=timeout" in caplog.text
