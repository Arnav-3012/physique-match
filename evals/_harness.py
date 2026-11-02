"""Shared by the plan and safety suites: the same pipeline the API runs, through FastAPI's TestClient.

`--llm off` serves the deterministic plan; `--llm live` turns the writer on and calls the real providers
(keys from the environment, e.g. `uv run --env-file .env ...`). Keys, prompts and responses are never
printed or written to the results.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "services" / "api"))
os.environ.setdefault("RATE_LIMIT_PER_MIN", "1000000")  # read when app.security is imported

BASE_PROFILE = dict(
    sex="male", height_cm=175, weight_kg=75, age=30, activity="moderate", level="beginner",
    goal="recomp", days_per_week=3, equipment=["barbell", "dumbbell", "cable", "machine", "body only"],
)  # fmt: skip
WAIST_G = 0.5  # the cases carry no waist; this harness supplies one (reported in the dev-log)


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (Path(__file__).parent / name).read_text().splitlines()]


def client():
    from fastapi.testclient import TestClient

    from app.main import app  # imports the retrieval once (hybrid when the model is present)

    return TestClient(app)


def regions():
    from physique_schemas.models import REGIONS

    return REGIONS


def region_gap(inp: dict) -> dict:
    gaps, skipped, conf = inp.get("gap", {}), inp.get("not_measured", {}), inp.get("confidence", {})
    return {
        "analyzer": "siamese_subtract",
        "regions": {
            r: {"delta": 0, "confidence": 0, "measured": False, "reason": skipped[r]}
            if r in skipped
            else {"delta": gaps.get(r, 0.0), "confidence": conf.get(r, 0.6)}
            for r in regions()
        },
    }


def payload(profile=None, waist_g=WAIST_G, gap=None, **extra) -> dict:
    return {
        "profile": {**BASE_PROFILE, **(profile or {})},
        "gap": gap or region_gap({}),
        "waist_g": waist_g,
        **extra,
    }


def post(c, body: dict, writer: bool):
    os.environ["LLM_WRITER_ENABLED"] = "true" if writer else "false"
    r = c.post("/plan", json=body)
    if writer:  # Groq's free tier: 8,000 tokens a minute (seen in the live probe) and a report is ~4-5k
        time.sleep(float(os.environ.get("EVALS_PAUSE_S", "35")))
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {}), r.text


def report_text(data: dict) -> str:
    rep = data["report"]
    parts = [rep["summary"], rep["realism_note"] or "", rep["disclaimer"], *data["warnings"]]
    for day in rep["plan"]:
        parts += [day["label"], *(e["cue"] for e in day["exercises"])]
    return " ".join(parts)
