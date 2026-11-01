"""Request guards for /analyze: size cap before parsing, per-client rate limit (docs/architecture.md
section 5). Mask-level validation lives in analyzer/masks.py."""

import os
import time
from collections import deque
from typing import Protocol

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


limiter: RateLimiter = SlidingWindowLimiter(int(os.environ.get("RATE_LIMIT_PER_MIN", "30")))


def client_key(request: Request) -> str:
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
