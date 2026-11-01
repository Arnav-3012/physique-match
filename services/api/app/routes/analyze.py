"""POST /analyze: goal scan + current scan + profile in, gap + body composition out.

Silhouettes and numbers only (ADR-003). The gap never sees weight; only the current scan feeds
the weight-aware waist model (ADR-024). Logs carry the request id, latency and region deltas,
never masks (privacy-and-safety.md).
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app import security
from app.analyzer.geometric import GeometricAnalyzer
from app.bodycomp import body_comp, silhouette_waist_g
from physique_schemas.models import AnalyzeRequest, AnalyzeResponse

log = logging.getLogger("analyze")
router = APIRouter()
analyzer = GeometricAnalyzer()


def _run(req: AnalyzeRequest) -> AnalyzeResponse:
    gap = analyzer.compare(req.goal, req.current)
    # lean: the current scan is featurised twice (compare + waist); cache if latency matters
    waist_g = None if req.profile.tape_waist_cm is not None else silhouette_waist_g(req.current)
    return AnalyzeResponse(gap=gap, body_comp=body_comp(req.profile, waist_g))


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request) -> AnalyzeResponse:
    security.limiter.check(security.client_key(request))
    if not request.headers.get("content-type", "").startswith("application/json"):
        raise HTTPException(415, "send application/json")
    body = await security.read_capped(request)
    try:
        req = AnalyzeRequest.model_validate_json(body)
    except ValidationError as e:  # errors without the input: never echo a mask back
        detail = e.errors(include_input=False, include_url=False, include_context=False)
        raise HTTPException(422, detail) from e

    start, rid = time.monotonic(), uuid.uuid4().hex[:8]
    try:
        out = await run_in_threadpool(_run, req)
    except ValueError as e:  # decode_mask / body_comp messages are ours and carry no payload
        raise HTTPException(422, str(e)) from e
    deltas = {r: round(g.delta, 3) for r, g in out.gap.regions.items()}
    log.info("id=%s ms=%d deltas=%s", rid, (time.monotonic() - start) * 1000, deltas)
    return out
