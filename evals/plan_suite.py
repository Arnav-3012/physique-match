"""Plan suite (P6 gate, ADR-058): the 15 frozen plan cases through the real /plan pipeline.

Per case: schema-valid, priorities, day count, the baseline-volume invariant, and four rubric items
scored by code. The fifth item, "readable", is scored by the owner, blind, on the LLM-written plans
(n is the number of cases the writer actually served, at most 15; no wider claim).

  follows the priorities : every priority group's weekly credit is at least the deterministic plan's
  spreads the volume     : the independent check passes (caps per session, per week, per group)
  safe wording           : the frozen text patterns, URLs, markdown and HTML all pass on the report text
  cites its sources      : every exercise is a retrieved candidate and its citation is its own id
"""

import json
import random
import string
from pathlib import Path

from evals import _harness as H

RESULTS = Path(__file__).parent / "results"
CODE_ITEMS = ("follows_priorities", "spreads_volume", "safe_wording", "cites_sources")


def _score(case: dict, base: dict, data: dict, status: int) -> dict:
    from app.agent.textsafety import text_is_safe
    from app.planner.candidates import ranked_candidates
    from app.planner.check import PlannerInvariantError, check
    from app.planner.params import kb_json
    from app.planner.plan import Shortfall, Slot
    from app.routes import plan as plan_route
    from physique_schemas.models import PlanResponse, UserProfile

    ex = case["expect"]
    row = {"id": case["id"], "status": status, "served_by": data.get("served_by"),
           "fallback_reason": data.get("fallback_reason")}  # fmt: skip
    try:
        PlanResponse.model_validate(data)
        row["schema_valid"] = status == 200
    except Exception:
        row["schema_valid"] = False
        return row

    profile = UserProfile(**{**H.BASE_PROFILE, **case["input"].get("profile", {})})
    by_id = {c["id"]: c for c in plan_route.RETRIEVAL.corpus.chunks}
    row["priorities_ok"] = data["priorities"] == ex["priorities"]
    row["days_ok"] = len(data["days"]) == ex["days"]
    row["invariant_ok"] = ex["invariant"] != "met" or not data["shortfalls"]

    region_map = kb_json("region_map.json")["regions"]
    groups = {g for r in data["priorities"] for g in region_map[r]["groups"]}
    row["follows_priorities"] = all(data["weekly"][g] >= base["weekly"][g] for g in groups)

    days = [[Slot(s["exercise_id"], s["name"], s["sets"]) for s in d] for d in data["days"]]
    shorts = [Shortfall(s["group"], s["planned"], s["floor"]) for s in data["shortfalls"]]
    try:
        check(days, shorts, by_id)
        row["spreads_volume"] = True
    except PlannerInvariantError:
        row["spreads_volume"] = False

    row["safe_wording"] = text_is_safe(H.report_text(data)) if data["report"] else True
    pool = {i for ids in ranked_candidates(profile, plan_route.RETRIEVAL.search).values() for i in ids}
    ids = [s["exercise_id"] for d in data["days"] for s in d]
    cited = [e["citations"] for day in (data["report"] or {"plan": []})["plan"] for e in day["exercises"]]
    row["cites_sources"] = all(i in pool for i in ids) and cited == [[i] for i in ids]
    return row


def _blind(rows: list[dict], texts: dict[str, str]) -> None:
    served = [r["id"] for r in rows if r["served_by"] in ("primary", "fallback")]
    order = served[:]
    random.SystemRandom().shuffle(order)  # not reproducible on purpose: the key stays in a separate file
    key = dict(zip(string.ascii_uppercase, order, strict=False))
    lines = [
        "# Blind readability scoring",
        "",
        "For each plan write 1 if it is readable (clear, coherent, usable) or 0 if not.",
        "Do not open the key file.",
        "",
    ]
    for letter in key:
        lines += [f"## Plan {letter}", "", texts[key[letter]], "", "Readable (1/0): ____", ""]
    (RESULTS / "plan_blind.md").write_text("\n".join(lines))
    (RESULTS / "plan_blind_key.json").write_text(json.dumps(key, indent=2) + "\n")


def _plan_text(data: dict) -> str:
    rep = data["report"]
    out = [f"Summary: {rep['summary']}"]
    for d in rep["plan"]:
        out.append(f"\n{d['label']}")
        out += [f"- {e['name']}: {e['sets']} x {e['reps']}. {e['cue']}" for e in d["exercises"]]
    out += [f"\nNote: {rep['realism_note']}"] if rep["realism_note"] else []
    out += [f"Warning: {w}" for w in data["warnings"]]
    return "\n".join(out + [rep["disclaimer"]])


def apply_readable(result: dict, path: Path) -> dict:
    """Merge the owner's blind scores ('A 1' per line) through the key file."""
    key = json.loads((RESULTS / "plan_blind_key.json").read_text())
    scores = {key[a]: int(v) for a, v in (line.split() for line in path.read_text().splitlines() if line.strip())}
    for row in result["rows"]:
        row["readable"] = scores.get(row["id"])
        passed = [row[i] for i in CODE_ITEMS] + ([bool(row["readable"])] if row["readable"] is not None else [])
        row["rubric"] = {"passed": sum(passed), "of": len(passed)}
    done = [r["rubric"] for r in result["rows"] if r.get("rubric")]
    result["rubric_mean_of_5"] = round(sum(r["passed"] for r in done) / len(done), 2) if done else None
    result["rubric_note"] = (
        "readable scored on the LLM-served plans only; cases without it are out of 4. "
        "The gate reads 'plan rubric >= 4/5' as a mean over scored cases; the README does not say per case."
    )
    return result


def run_plan(live: bool, readable: Path | None = None, only: list[str] | None = None) -> dict:
    """`only` runs a subset of the cases (a quick live check); the gate needs all 15."""
    c = H.client()
    rows, texts = [], {}
    for case in (x for x in H.load("plan.jsonl") if not only or x["id"] in only):
        body = H.payload(
            case["input"].get("profile"), gap=H.region_gap(case["input"]),
            goal_sex=case["input"].get("goal_sex", "not_specified"),
        )  # fmt: skip
        s0, base, _ = H.post(c, body, writer=False)
        s1, data, _ = H.post(c, body, writer=True) if live else (s0, base, "")
        if s1 != 200 or s0 != 200:
            rows.append({"id": case["id"], "status": s1, "schema_valid": False})
            continue
        rows.append(_score(case, base, data, s1))
        if live and data["served_by"] in ("primary", "fallback") and data["report"]:
            texts[case["id"]] = _plan_text(data)
    RESULTS.mkdir(exist_ok=True)
    if live:
        _blind(rows, texts)
    served = {k: sum(r.get("served_by") == k for r in rows) for k in ("primary", "fallback", "deterministic")}
    result = {
        "mode": "live" if live else "off",
        "n_cases": len(rows),
        "served_by": served,
        "fallback_reasons": sorted({r["fallback_reason"] for r in rows if r.get("fallback_reason")}),
        "schema_valid": sum(bool(r.get("schema_valid")) for r in rows),
        "priorities_ok": sum(bool(r.get("priorities_ok")) for r in rows),
        "days_ok": sum(bool(r.get("days_ok")) for r in rows),
        "invariant_ok": sum(bool(r.get("invariant_ok")) for r in rows),
        **{i: sum(bool(r.get(i)) for r in rows) for i in CODE_ITEMS},
        "readable": "not scored yet" if readable is None else "scored",
        "rows": rows,
    }
    if readable:
        result = apply_readable(result, readable)
    return result
