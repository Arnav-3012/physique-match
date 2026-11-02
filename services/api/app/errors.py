"""Validation errors that carry no client-chosen text (ADR-054)."""

import re

_FIELD = re.compile(r"[a-z_]{1,24}")


def safe_detail(errors: list[dict]) -> list[dict]:
    """Field path, error type and our own message. An extra key's own name is dropped and odd path
    names are masked, so a hostile string never comes back to the caller. Expects
    `e.errors(include_input=False, include_url=False, include_context=False)`."""
    return [
        {
            "loc": [
                p if isinstance(p, int) or _FIELD.fullmatch(p) else "?"
                for p in (e["loc"][:-1] if e["type"] == "extra_forbidden" else e["loc"])
            ],
            "type": e["type"],
            "msg": e["msg"],
        }
        for e in errors
    ]
