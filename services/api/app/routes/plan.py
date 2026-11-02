"""POST /plan: profile + gap (+ waist_g) in, the weekly plan and its report out (ADR-057, ADR-058).

No scans. Body composition is recomputed here from the profile; a client-supplied BodyComp is not
accepted. The deterministic planner always runs; the LLM writer (off unless LLM_WRITER_ENABLED=true)
may replace its exercise choices and wording, and only if its whole output passes the independent
checks. The safety layer re-checks the result. Any internal breach is a generic 500 with the detail in
the server log only; nothing partial is returned.
"""

import logging
import os
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app import security
from app.agent.llm import LLM, HttpxLLM
from app.agent.writer import (
    Context,
    build_facts,
    build_report,
    deterministic_summary,
    write,
)
from app.bodycomp import NOTE_ED, body_comp
from app.errors import safe_detail
from app.planner.candidates import ranked_candidates
from app.planner.check import check, weekly_credit
from app.planner.params import LAYOUTS, kb_json
from app.planner.plan import Shortfall, build
from app.planner.priorities import priorities
from app.planner.realism import clean_gap, realism_note, warnings
from app.rag.service import load_retrieval
from app.safety.rules import check_plan_safety
from physique_schemas.models import (
    BodyComp,
    PlanRequest,
    PlanResponse,
    PlanShortfall,
    PlanSlot,
    UserProfile,
)

log = logging.getLogger("plan")
router = APIRouter()
RETRIEVAL = load_retrieval()  # once, at startup (hybrid, or BM25 with a warning)
LLM_CLIENT: LLM = HttpxLLM()

NOTE_NO_WAIST = (
    "Calorie targets need a waist measurement. Add a tape measurement, or use the result of "
    "the scan, to see them; your training plan is unchanged."
)


def writer_enabled() -> bool:
    return os.environ.get("LLM_WRITER_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _body_comp(profile: UserProfile, waist_g: float | None) -> tuple[BodyComp | None, str]:
    """(body composition, note). Without any waist, the plan still works: an eating-disorder history
    gets its own supportive note and no numbers, anyone else gets the fixed no-waist note."""
    needs_waist = profile.tape_waist_cm is None and profile.sex != "not_specified"
    if needs_waist and waist_g is None:
        if profile.ed_history:
            return BodyComp(energy_measured=False, note=NOTE_ED), ""
        return None, NOTE_NO_WAIST
    return body_comp(profile, waist_g), ""


def _plan(req: PlanRequest) -> PlanResponse:
    profile = req.profile
    chosen = priorities(req.gap)
    candidates = ranked_candidates(profile, RETRIEVAL.search)
    by_id = {c["id"]: c for c in RETRIEVAL.corpus.chunks}
    det = build(profile, chosen, RETRIEVAL.corpus.chunks, candidates)
    comp, comp_note = _body_comp(profile, req.waist_g)
    check_plan_safety(profile, comp)

    focus = LAYOUTS[profile.days_per_week]
    facts, allowed = build_facts(chosen, req.gap, comp, focus, candidates, by_id, det)
    summary = deterministic_summary(chosen, len(focus), bool(det.shortfalls))
    written = write(
        enabled=writer_enabled(),
        llm=LLM_CLIENT,
        facts=facts,
        ctx=Context(focus, candidates, by_id, det, allowed),
        summary=summary,
    )

    if written.served_by == "deterministic":
        days, weekly, shortfalls = det.days, det.weekly, det.shortfalls
    else:  # the writer's plan: totals and shortfalls recomputed, never taken from the model
        credit = weekly_credit(written.days, by_id)
        floors = {g: float(e["min_sets_per_week"]) for g, e in kb_json("baselines.json")["groups"].items()}
        weekly = {g: credit.get(g, 0.0) for g in floors}
        shortfalls = [Shortfall(g, weekly[g], f) for g, f in floors.items() if weekly[g] < f]
        days = written.days
    check(days, shortfalls, by_id)  # the served plan, whichever path wrote it

    notes = warnings(req.gap, profile.sex, req.goal_sex)
    return PlanResponse(
        priorities=chosen,
        days=[[PlanSlot(exercise_id=s.exercise_id, name=s.name, sets=s.sets) for s in d] for d in days],
        weekly=weekly,
        shortfalls=[PlanShortfall(group=s.group, planned=s.planned, floor=s.floor) for s in shortfalls],
        note=det.note,
        body_comp=comp,
        body_comp_note=comp_note,
        retrieval=RETRIEVAL.mode,
        served_by=written.served_by,
        fallback_reason=written.fallback_reason,
        warnings=notes,
        report=build_report(
            written,
            clean_gap(req.gap),
            comp,
            chosen,
            realism_note(req.gap, chosen, profile.height_cm),
            by_id,
        ),
    )


@router.post("/plan", response_model=PlanResponse)
async def plan(request: Request) -> PlanResponse:
    security.limiter.check(security.client_key(request))
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "send application/json")
    body = await security.read_capped(request)
    try:
        req = PlanRequest.model_validate_json(body)
    except ValidationError as e:  # field path, type and our own message; never the rejected value
        errors = e.errors(include_input=False, include_url=False, include_context=False)
        raise HTTPException(422, safe_detail(errors)) from e

    start, rid = time.monotonic(), uuid.uuid4().hex[:8]
    try:
        out = await run_in_threadpool(_plan, req)
    except Exception:  # an internal invariant or safety breach: log the detail, return nothing partial
        log.exception("id=%s /plan failed an internal check", rid)
        raise HTTPException(500, "internal error") from None
    log.info(
        "id=%s ms=%d served_by=%s fallback_reason=%s retrieval=%s priorities=%s shortfalls=%d",
        rid, (time.monotonic() - start) * 1000, out.served_by, out.fallback_reason, out.retrieval,
        out.priorities, len(out.shortfalls),
    )  # fmt: skip
    return out
