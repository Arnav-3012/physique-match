"""Request guards for /analyze: size cap before parsing, per-client rate limit (docs/architecture.md
section 5). Mask-level validation lives in analyzer/masks.py."""

import hashlib
import os
import time
from collections import deque
from typing import Protocol

import httpx
from fastapi import HTTPException, Request

from physique_schemas.models import MAX_PAYLOAD_B64

MAX_BODY_BYTES = 2 * MAX_PAYLOAD_B64 + 50_000  # goal scan + current scan + profile
MAX_KEYS = 10_000


class RateLimiter(Protocol):
    """What /analyze needs. In-memory now; an Upstash implementation replaces it before P7."""

    def check(self, key: str) -> None: ...


class SlidingWindowLimiter:
    # lean: in-memory, one process. Behind a proxy every client shares its IP; move to Redis
    # (Upstash) or key on a forwarded header if the API is ever scaled out or tunnelled (P7).
    def __init__(self, limit: int, window_s: float = 60.0):
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        now = time.monotonic()
        if len(self._hits) > MAX_KEYS:
            self._hits = {k: h for k, h in self._hits.items() if now - h[-1] < self.window_s}
        hits = self._hits.setdefault(key, deque())
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            retry = int(self.window_s - (now - hits[0])) + 1
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry)})
        hits.append(now)


class UpstashLimiter:
    """Fixed-window counter in Upstash Redis over its REST API, shared by every instance and surviving scale-to-zero
    (ADR-028). The key is a hash of the client address, never the address itself, and expires with its window.
    If Upstash is unreachable it falls back to the in-memory limiter instead of failing open."""

    def __init__(self, url: str, token: str, limit: int, fallback: RateLimiter, window_s: int = 60,
                 transport: httpx.BaseTransport | None = None):  # fmt: skip
        self.limit, self.window_s, self.fallback = limit, window_s, fallback
        self._http = httpx.Client(
            base_url=url.rstrip("/"),
            headers={"Authorization": f"Bearer {token.strip()}"},
            timeout=1.5,
            transport=transport,
        )

    def check(self, key: str) -> None:
        now = time.time()
        slot = int(now // self.window_s)
        name = f"rl:{hashlib.sha256(key.encode()).hexdigest()[:16]}:{slot}"
        try:
            r = self._http.post("/pipeline", json=[["INCR", name], ["EXPIRE", name, self.window_s * 2]])
            r.raise_for_status()
            count = int(r.json()[0]["result"])
        except Exception:  # network, auth, or a malformed answer
            self.fallback.check(key)
            return
        if count > self.limit:
            retry = self.window_s - int(now) % self.window_s
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(retry)})


def make_limiter() -> RateLimiter:
    local = SlidingWindowLimiter(int(os.environ.get("RATE_LIMIT_PER_MIN", "30")))
    url, token = os.environ.get("UPSTASH_REDIS_REST_URL", ""), os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
    return UpstashLimiter(url, token, local.limit, local) if url.strip() and token.strip() else local


limiter: RateLimiter = make_limiter()


def client_key(request: Request) -> str:
    """Behind Cloud Run every request comes from an internal address, so the client is read from X-Forwarded-For
    (the spike measured one client address in it). The LAST entry is the one Google's front end added; earlier
    entries are client-supplied and spoofable. Only trusted when TRUST_FORWARDED_FOR is set (the deployed service)."""
    if os.environ.get("TRUST_FORWARDED_FOR", "").strip().lower() in {"1", "true", "yes"}:
        forwarded = request.headers.get("x-forwarded-for", "").split(",")[-1].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


async def read_capped(request: Request, limit: int = MAX_BODY_BYTES) -> bytes:
    """Whole body, or 413 as soon as it passes `limit` (also when Content-Length is absent)."""
    declared = request.headers.get("content-length", "0")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, "payload too large")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, "payload too large")
        chunks.append(chunk)
    return b"".join(chunks)
