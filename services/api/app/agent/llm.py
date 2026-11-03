"""OpenAI-compatible chat completions over httpx (ADR-058; CLAUDE.md: the wire protocol, no SDK).

Providers come from the environment (.env.example). A failure is one of the closed reasons of
`FallbackReason`. The API key, the prompt and the raw response are never logged: only the provider
name, an outcome category and the latency.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger("llm")


class LLMError(Exception):
    """`reason` is a member of physique_schemas.models.FallbackReason."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class LLM(Protocol):
    def available(self, provider: str) -> bool: ...

    def complete(self, provider: str, messages: list[dict], schema: dict) -> str: ...


@dataclass(frozen=True)
class Provider:
    base_url: str
    model: str
    key: str
    extra: dict  # extra request fields, e.g. OpenRouter's no-train preference


class HttpxLLM:
    def __init__(self, transport: httpx.BaseTransport | None = None):
        self.transport = transport

    @staticmethod
    def _provider(name: str) -> Provider | None:
        env = os.environ
        if name == "primary":
            p = Provider(env.get("LLM_BASE_URL", ""), env.get("LLM_MODEL", ""), env.get("GROQ_API_KEY", "").strip(), {})
        else:  # ADR-017: the free fallback must not train on prompts
            p = Provider(
                env.get("LLM_FALLBACK_BASE_URL", ""),
                env.get("LLM_FALLBACK_MODEL", ""),
                env.get("OPENROUTER_API_KEY", "").strip(),
                {"provider": {"data_collection": "deny"}},
            )
        return p if p.base_url and p.model and p.key else None

    def available(self, provider: str) -> bool:
        return self._provider(provider) is not None

    def complete(self, provider: str, messages: list[dict], schema: dict) -> str:
        p = self._provider(provider)
        if p is None:
            raise LLMError("disabled")
        body = {
            "model": p.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "plan_writer", "strict": True, "schema": schema},
            },
            **p.extra,
        }
        timeout = float(os.environ.get("LLM_TIMEOUT_S", "20"))
        start = time.monotonic()
        outcome, content = "ok", ""
        try:
            with httpx.Client(timeout=timeout, transport=self.transport) as client:
                r = client.post(
                    f"{p.base_url.rstrip('/')}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {p.key}"},
                )
        except httpx.HTTPError:
            outcome = "timeout"
        else:
            if r.status_code in (413, 429):  # Groq answers 413 when a request exceeds its tokens-per-minute limit
                outcome = "rate_limited"
            elif r.status_code >= 500:
                outcome = "timeout"
            elif not r.is_success:
                outcome = "invalid_json"
            else:
                try:
                    content = r.json()["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        outcome = "invalid_json"
                except (ValueError, KeyError, IndexError, TypeError):
                    outcome = "invalid_json"
        log.info("provider=%s outcome=%s ms=%d", provider, outcome, (time.monotonic() - start) * 1000)
        if outcome != "ok":
            raise LLMError(outcome) from None
        return content
