"""One-off diagnostic: why does the live writer end in `invalid_json`? (ADR-058)

    uv run --env-file .env python -m evals.probe_llm

Sends four requests to the PRIMARY provider and prints, for each, the HTTP status and what went wrong:
the provider's own error message for a 4xx, or for a 200 whether the content is JSON and which fields
of WriterOutput it breaks (location and type only). The prompts are synthetic or the frozen case p01;
the key is read from the environment and never printed. This is a dev diagnostic, not app logging:
it prints a truncated model answer, which the application itself never logs.
"""

import json
import sys
import time

import httpx
from pydantic import ValidationError

from evals import _harness as H


def _call(p, body: dict) -> tuple[int, str]:
    with httpx.Client(timeout=40) as client:
        r = client.post(
            f"{p.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {p.key}"},
        )
    return r.status_code, r.text


def _report(label: str, status: int, text: str, ctx=None) -> None:
    from app.agent.schema import WriterOutput
    from app.agent.writer import Reject, _accept

    print(f"\n[{label}] HTTP {status}")
    try:
        payload = json.loads(text)
    except ValueError:
        print("  body is not JSON:", text[:200])
        return
    if status != 200:
        err = payload.get("error", payload)
        print("  provider error:", str(err.get("message", err))[:400] if isinstance(err, dict) else str(err)[:400])
        return
    content = payload["choices"][0]["message"]["content"]
    print("  content (first 300 chars):", (content or "")[:300].replace("\n", " "))
    try:
        out = WriterOutput.model_validate_json(content)
        print("  -> valid WriterOutput")
    except ValidationError as e:
        print("  -> WriterOutput errors:", [(list(x["loc"]), x["type"]) for x in e.errors()][:8])
        return
    if ctx is not None:
        try:
            _accept(out, ctx)
            print("  -> ACCEPTED by the app's checks")
        except Reject as r:
            print("  -> REJECTED by the app's checks:", r.reason)
            _explain(out, ctx)


def _explain(out, ctx) -> None:
    """Which acceptance rule failed (ids and counts only)."""
    from app.planner.params import kb_json

    print("     days:", len(out.days), "expected", len(ctx.focus))
    for d, day in enumerate(out.days[: len(ctx.focus)]):
        allowed = {i for g in ctx.focus[d] for i in ctx.candidates.get(g, [])}
        ids = [it.exercise_id for it in day.items]
        stray = [i for i in ids if i not in allowed]
        print(f"     day {d + 1}: {len(ids)} items, not candidates: {stray}, repeats: {len(ids) - len(set(ids))}")
    print(
        "     deterministic shortfall groups:",
        [s.group for s in ctx.det.shortfalls],
        "| floors from",
        len(kb_json("baselines.json")["groups"]),
        "groups",
    )


def main() -> None:
    from app.agent.llm import HttpxLLM
    from app.agent.schema import WriterOutput
    from app.agent.writer import SYSTEM, Context, build_facts
    from app.planner.candidates import ranked_candidates
    from app.planner.params import LAYOUTS
    from app.planner.plan import build
    from app.planner.priorities import priorities
    from app.routes import plan as plan_route
    from physique_schemas.models import GapProfile, UserProfile

    p = HttpxLLM._provider("primary")
    if p is None:
        print("primary provider not configured: set LLM_BASE_URL, LLM_MODEL and GROQ_API_KEY")
        return
    print("provider base URL:", p.base_url, "| model:", p.model)

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    case_id, real_only = (args[0] if args else "p01"), "--real-only" in sys.argv
    case = next(c for c in H.load("plan.jsonl") if c["id"] == case_id)
    print("case:", case_id, "| days:", case["input"].get("profile", {}).get("days_per_week", 3))
    profile = UserProfile(**{**H.BASE_PROFILE, **case["input"].get("profile", {})})
    gap = GapProfile.model_validate(H.region_gap(case["input"]))
    chosen = priorities(gap)
    retrieval = plan_route.RETRIEVAL
    cands = ranked_candidates(profile, retrieval.search)
    by_id = {c["id"]: c for c in retrieval.corpus.chunks}
    det = build(profile, chosen, retrieval.corpus.chunks, cands)
    facts, allowed = build_facts(chosen, gap, None, LAYOUTS[profile.days_per_week], cands, by_id, det)
    ctx = Context(LAYOUTS[profile.days_per_week], cands, by_id, det, allowed)
    real = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(facts)}]
    tiny = [{"role": "user", "content": 'Reply with JSON only: {"summary": "hi", "days": []}'}]
    strict = {
        "type": "json_schema",
        "json_schema": {"name": "plan_writer", "strict": True, "schema": WriterOutput.model_json_schema()},
    }
    trials = {
        "A tiny prompt, strict json_schema": (tiny, strict),
        "B tiny prompt, json_object": (tiny, {"type": "json_object"}),
        "C real p01 prompt, strict json_schema (what the app sends)": (real, strict),
        "D real p01 prompt, json_object": (real, {"type": "json_object"}),
    }
    print("real prompt size (characters):", len(json.dumps(real)))
    for label, (messages, fmt) in trials.items():
        if real_only and not label.startswith("C"):
            continue
        body = {"model": p.model, "messages": messages, "temperature": 0, "response_format": fmt}
        _report(label, *_call(p, body), ctx=ctx if label[0] in "CD" else None)
        time.sleep(4)


if __name__ == "__main__":
    main()
