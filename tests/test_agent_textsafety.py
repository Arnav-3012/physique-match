"""app/agent/textsafety.py repeats the frozen evals/patterns.py; every regex is pinned to it here, the
frozen false-positive fixtures run against the copy, and the new markup and number checks are tested."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.agent import textsafety as ts

EVALS = Path(__file__).resolve().parent.parent / "evals"
_spec = importlib.util.spec_from_file_location("frozen_patterns", EVALS / "patterns.py")
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)
FIXTURES = [json.loads(line) for line in (EVALS / "pattern_checks.jsonl").read_text().splitlines()]


def test_the_copy_equals_the_frozen_patterns():
    for mine, theirs in ((ts.BANNED, frozen.BANNED), (ts.REGIONAL_CLAIM, frozen.REGIONAL_CLAIM)):
        assert [(p.pattern, p.flags) for p in mine] == [(p.pattern, p.flags) for p in theirs]
    assert (ts.DRUG_WORDS.pattern, ts.DRUG_WORDS.flags) == (
        frozen.DRUG_WORDS.pattern,
        frozen.DRUG_WORDS.flags,
    )


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f["id"])
def test_frozen_fixtures_against_the_copy(fixture):
    patterns = {"banned": ts.BANNED, "regional": ts.REGIONAL_CLAIM, "drug": [ts.DRUG_WORDS]}
    hit = any(p.search(fixture["text"]) for p in patterns[fixture["check"]])
    assert hit is fixture["flag"]


@pytest.mark.parametrize(
    "text",
    [
        "See https://example.com for more.",
        "Go to www.example.com",
        "Read [this guide](page) first.",
        "Use <b>bold</b> form.",
        "Add a <br> here",
    ],
)
def test_urls_markdown_and_html_are_unsafe(text):
    assert not ts.text_is_safe(text)


@pytest.mark.parametrize("text", ["Keep your back flat and brace your core.", "Lower slowly, then press up."])
def test_plain_coaching_text_is_safe(text):
    assert ts.text_is_safe(text)


def test_numbers_are_compared_as_numbers():
    allowed = {3.0, 1.0, 2.0}
    assert ts.numbers_are_known("about 3% of your height on day 2", allowed)
    assert ts.numbers_are_known("no digits at all", allowed)
    assert not ts.numbers_are_known("for 12 weeks", allowed)
    assert not ts.numbers_are_known("3.5 percent", allowed)
