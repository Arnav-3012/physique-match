"""/health: service is up + can each LLM provider be reached. Cheap GETs, cached, no quota burn."""

import asyncio
import os
import time

import httpx
from fastapi import APIRouter

router = APIRouter()

PROVIDERS = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/models"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/auth/key"),
}
TTL_S = 60
_cache: dict[str, tuple[float, dict]] = {}


async def _probe(client: httpx.AsyncClient, name: str) -> dict:
    env_var, url = PROVIDERS[name]
    key = os.environ.get(env_var, "").strip()  # a trailing newline in a secret breaks the header
    if not key:
        return {"status": "not_configured"}

    cached = _cache.get(name)
    if cached and time.monotonic() - cached[0] < TTL_S:
        return cached[1]

    start = time.monotonic()
    try:
        r = await client.get(url, headers={"Authorization": f"Bearer {key}"})
        result = {"status": "ok" if r.is_success else f"http_{r.status_code}"}
        result["latency_ms"] = round((time.monotonic() - start) * 1000)
    except httpx.HTTPError:
        result = {"status": "unreachable"}
    _cache[name] = (time.monotonic(), result)
    return result


@router.get("/health")
async def health() -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        results = await asyncio.gather(*(_probe(client, n) for n in PROVIDERS))
    return {"status": "ok", "providers": dict(zip(PROVIDERS, results, strict=True))}
