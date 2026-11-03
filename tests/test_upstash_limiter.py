"""Upstash-backed rate limiter and the forwarded-address key (ADR-028). No network: httpx MockTransport."""

import json

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import security


def _request(headers: dict[str, str], host: str = "10.0.0.1") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (host, 1234),
    }
    return Request(scope)


class _Local:
    def __init__(self):
        self.calls = []

    def check(self, key: str) -> None:
        self.calls.append(key)


def _limiter(handler, limit=2):
    local = _Local()
    lim = security.UpstashLimiter(
        "https://x.upstash.io", " tok\n", limit, local, transport=httpx.MockTransport(handler)
    )
    return lim, local


def test_counts_up_to_the_limit_then_429_with_retry_after():
    seen = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"] == "Bearer tok"  # key stripped of whitespace
        body = json.loads(req.content)
        assert body[0][0] == "INCR" and body[1][0] == "EXPIRE"
        seen["n"] += 1
        return httpx.Response(200, json=[{"result": seen["n"]}, {"result": 1}])

    lim, local = _limiter(handler, limit=2)
    lim.check("1.2.3.4")
    lim.check("1.2.3.4")
    with pytest.raises(HTTPException) as e:
        lim.check("1.2.3.4")
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) >= 1
    assert local.calls == []


def test_the_redis_key_never_holds_the_address():
    keys = []

    def handler(req: httpx.Request) -> httpx.Response:
        keys.append(json.loads(req.content)[0][1])
        return httpx.Response(200, json=[{"result": 1}, {"result": 1}])

    lim, _ = _limiter(handler)
    lim.check("203.0.113.9")
    assert "203.0.113.9" not in keys[0] and keys[0].startswith("rl:")


@pytest.mark.parametrize("status", [401, 500])
def test_falls_back_to_memory_when_upstash_fails(status):
    lim, local = _limiter(lambda req: httpx.Response(status, json={"error": "x"}))
    lim.check("1.2.3.4")
    assert local.calls == ["1.2.3.4"]


def test_falls_back_when_upstash_is_unreachable():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    lim, local = _limiter(handler)
    lim.check("1.2.3.4")
    assert local.calls == ["1.2.3.4"]


def test_client_key_uses_the_last_forwarded_entry_only_when_trusted(monkeypatch):
    req = _request({"x-forwarded-for": "6.6.6.6, 198.51.100.7"})
    monkeypatch.delenv("TRUST_FORWARDED_FOR", raising=False)
    assert security.client_key(req) == "10.0.0.1"  # untrusted: the header is spoofable
    monkeypatch.setenv("TRUST_FORWARDED_FOR", "true")
    assert security.client_key(req) == "198.51.100.7"  # the entry Google's front end added, not the spoofed one
    assert security.client_key(_request({})) == "10.0.0.1"  # no header: the socket address


def test_make_limiter_is_memory_without_upstash_env(monkeypatch):
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    assert isinstance(security.make_limiter(), security.SlidingWindowLimiter)
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://x.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "t")
    assert isinstance(security.make_limiter(), security.UpstashLimiter)
