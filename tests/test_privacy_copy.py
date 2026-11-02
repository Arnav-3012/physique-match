"""The privacy statement and the no-storage rule (ADR-051). These read source files, so they live in pytest:
the web app's tsconfig has no Node types for a vitest file reader."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_SRC = ROOT / "apps" / "web" / "src"
FORBIDDEN_STORAGE = re.compile(
    r"localStorage|sessionStorage|indexedDB|\bcaches\.|createObjectURL|navigator\.serviceWorker"
)


def _copy() -> str:
    text = (WEB_SRC / "capture" / "privacyCopy.ts").read_text()
    match = re.search(r'PRIVACY_COPY =\s*"([^"\n]+)";', text)
    assert match, "PRIVACY_COPY must stay one double-quoted line"
    return match.group(1)


def test_the_copy_in_the_app_matches_the_doc_word_for_word():
    doc = (ROOT / "docs" / "privacy-and-safety.md").read_text()
    assert f"> {_copy()}" in doc


def test_the_copy_makes_no_claim_the_code_does_not_support():
    low = _copy().lower()
    for banned in ("nothing leaves", "never stored", "nothing is uploaded", "never sent"):
        assert banned not in low
    assert "does not store it" in low  # the server claim, checked against analyze.py in ADR-051
    assert "silhouette" in low and "joint positions" in low
    assert "by design" in low  # the storage sentence rests on a source scan and a manual check, not an observation


def test_the_limits_copy_in_the_app_matches_adr_044_word_for_word():
    text = (WEB_SRC / "lib" / "definitionCopy.ts").read_text()
    match = re.search(r'LIMITS_COPY =\s*"([^"\n]+)";', text)
    assert match, "LIMITS_COPY must stay one double-quoted line"
    assert f'"{match.group(1)}"' in (ROOT / "docs" / "decisions.md").read_text()


def test_no_web_source_uses_a_browser_storage_api():
    files = [p for p in WEB_SRC.rglob("*") if p.suffix in {".ts", ".tsx"} and ".test." not in p.name]
    assert len(files) >= 10  # the scan found the app, not an empty folder
    hits = [f"{p.relative_to(ROOT)}: {m.group(0)}" for p in files for m in FORBIDDEN_STORAGE.finditer(p.read_text())]
    assert hits == []


def test_the_analyze_route_writes_no_request_body_to_disk_or_a_database():
    code = (ROOT / "services" / "api" / "app" / "routes" / "analyze.py").read_text()
    for call in ("open(", "write_text", "write_bytes", "sqlite", "redis", "shelve", "pickle"):
        assert call not in code
