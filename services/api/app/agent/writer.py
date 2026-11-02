"""The LLM writer (ADR-058): one structured call that picks exercises from the planner's candidate lists
by id and writes wording. Sets, caps and every number stay with the planner and BodyComp.

The model's whole output is accepted or discarded: it must parse, use only candidate ids that fit the
day's focus, pass the independent floor / cap check without more shortfalls than the deterministic plan,
and pass the text checks. Any miss serves the deterministic plan with a closed `fallback_reason`.
Prompts and responses are never logged.
"""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.agent.llm import LLM, LLMError
from app.agent.schema import WriterOutput
from app.agent.textsafety import NUMBER, numbers_are_known, text_is_safe
from app.planner.check import PlannerInvariantError, check, weekly_credit
from app.planner.params import SETS_PER_EXERCISE, kb_json
from app.planner.plan import Shortfall, Slot, WeeklyPlan
from physique_schemas.models import (
    BodyComp,
    GapProfile,
    PlanDay,
    PlannedExercise,
    Report,
)

MAX_CALLS = 3  # primary, one retry of a transient failure, the fallback (ADR-058)
MAX_RETRIES = 1
TRANSIENT = {"timeout", "rate_limited"}
# Ours, prompt size only; validation accepts any candidate id. Set after the first live probe: 6 options
# and 200 characters per exercise, repeated per day, made a 94,225-character prompt (about 24k tokens)
# against Groq's free-tier limit of 8,000 tokens a minute (HTTP 413). Now one catalogue, 3 options, 100 characters.
PROMPT_CANDIDATES_PER_GROUP = 3
HOW_CHARS = 100
DEFAULT_REPS = "8-12"  # ACSM 2009 for beginners (kb/docs/rep-ranges-failure.md): a typical range, not a rule
DEFAULT_CUE = "Controlled reps, in a range of motion you can manage with good form."
DISCLAIMER = "This is not medical advice. See a professional for injuries or health conditions."

SYSTEM = (
    "You write the wording for a weekly training plan from the facts given. Reply with JSON only, "
    "matching the schema. Each day in plan_days has a focus (muscle groups) and a baseline_plan of "
    "exercise ids. Choose exercises ONLY by exercise_id, from that day's baseline_plan or from "
    "options_by_group for the groups in its focus; never the same id twice in one day; use exactly the "
    "days given. Keep the baseline_plan unless another option fits better. 'exercises' gives each id's "
    "name and a short 'how'. Do not write any numbers in the summary, "
    "the day labels or the cues. Do not mention calories, body fat, supplements, drugs, links or "
    "formatting. Never say that one body part is more or less defined or lean than another. Use "
    "neutral, encouraging language. Each cue is your own short coaching sentence in plain words, based "
    "on the exercise's 'how' text, never copied from it and never containing names, equipment or level "
    "labels. The summary has at most three sentences."
)


class Reject(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Context:
    focus: list[frozenset[str]]
    candidates: dict[str, list[str]]
    by_id: dict[str, dict]
    det: WeeklyPlan
    allowed_numbers: set[float]


@dataclass
class Written:
    served_by: str
    fallback_reason: str | None
    days: list[list[Slot]]
    labels: list[str]
    reps: list[list[str]]
    cues: list[list[str]]
    summary: str


_HEADER = re.compile(r"Level: \w+\. [^.]*\. ")  # kb/ingest_exercises.py: "...Level: L. <style>. <instructions>"


def how_text(chunk_text: str) -> str:
    """The instructions part of an exercise chunk. The first live run gave the model the first 100
    characters of the chunk, which is only the metadata header, and it copied that into its cues."""
    parts = _HEADER.split(chunk_text, maxsplit=1)
    return parts[1][:HOW_CHARS] if len(parts) == 2 else ""


def build_facts(
    chosen: list[str],
    gap: GapProfile,
    comp: BodyComp | None,
    ctx_focus: list[frozenset[str]],
    candidates: dict[str, list[str]],
    by_id: dict[str, dict],
    det: WeeklyPlan,
) -> tuple[dict, set[float]]:
    """The facts block and the set of numbers the model's prose may use. Numbers and KB text only: no
    user free text and no client-supplied `reason` ever enters it."""
    options = {
        g: candidates.get(g, [])[:PROMPT_CANDIDATES_PER_GROUP]
        for g in sorted({g for groups in ctx_focus for g in groups})
    }
    days = [
        {"day": d + 1, "focus": sorted(groups), "baseline_plan": [s.exercise_id for s in det.days[d]]}
        for d, groups in enumerate(ctx_focus)
    ]
    shown = {i for ids in options.values() for i in ids} | {i for d in days for i in d["baseline_plan"]}
    exercises = {  # each exercise once, however many days or groups use it
        i: {"name": by_id[i]["name"], "primary": by_id[i]["muscle"], "how": how_text(by_id[i]["text"])}
        for i in sorted(shown)
    }
    numeric = {
        "days": len(ctx_focus),
        "priorities": [
            {
                "region": r,
                "gap_pct_of_height": round(gap.regions[r].delta * 100, 1),
                "reliability_pct": round(gap.regions[r].confidence * 100),
            }
            for r in chosen
        ],
        "groups_short_of_minimum": [s.group for s in det.shortfalls],
        "body_fat_range_pct": [comp.body_fat.low, comp.body_fat.high] if comp and comp.body_fat else None,
        "protein_g_range": [comp.protein.g_low, comp.protein.g_high] if comp and comp.protein else None,
        "energy_kcal": [
            {"goal": t.goal, "low": t.kcal_low, "high": t.kcal_high} for t in (comp.energy if comp else [])
        ],
    }
    allowed = {float(t) for t in NUMBER.findall(json.dumps(numeric))}
    allowed |= {float(i) for i in range(1, len(ctx_focus) + 1)}  # "Day 1" ...
    return {**numeric, "options_by_group": options, "plan_days": days, "exercises": exercises}, allowed


def _parse(raw: str) -> WriterOutput:
    try:
        return WriterOutput.model_validate_json(raw)
    except ValidationError:
        raise Reject("invalid_json") from None


def _accept(out: WriterOutput, ctx: Context):
    """Raise Reject unless the whole output passes; otherwise the parts of a Written."""
    if len(out.days) != len(ctx.focus):
        raise Reject("check_failed")
    days: list[list[Slot]] = []
    for d, day in enumerate(out.days):
        ids = [it.exercise_id for it in day.items]
        allowed = {i for g in ctx.focus[d] for i in ctx.candidates.get(g, [])}
        if len(set(ids)) != len(ids) or any(i not in allowed for i in ids):
            raise Reject("check_failed")
        days.append([Slot(i, ctx.by_id[i]["name"], SETS_PER_EXERCISE) for i in ids])

    floors = {g: float(e["min_sets_per_week"]) for g, e in kb_json("baselines.json")["groups"].items()}
    week = weekly_credit(days, ctx.by_id)
    short = [Shortfall(g, week.get(g, 0.0), f) for g, f in floors.items() if week.get(g, 0.0) < f]
    if {s.group for s in short} - {s.group for s in ctx.det.shortfalls}:
        raise Reject("check_failed")  # never worse than the deterministic plan
    try:
        check(days, short, ctx.by_id)
    except PlannerInvariantError:
        raise Reject("check_failed") from None

    cues = [it.cue for d in out.days for it in d.items]
    if any("equipment:" in c.lower() or "level:" in c.lower() for c in cues):
        raise Reject("check_failed")  # a cue that is the KB metadata header, not a coaching sentence
    prose = [out.summary, *(d.label for d in out.days), *cues]
    if not all(text_is_safe(s) and numbers_are_known(s, ctx.allowed_numbers) for s in prose):
        raise Reject("unsafe_text")
    return Written(
        served_by="",
        fallback_reason=None,
        days=days,
        labels=[d.label for d in out.days],
        reps=[[it.reps for it in d.items] for d in out.days],
        cues=[[it.cue for it in d.items] for d in out.days],
        summary=out.summary,
    )


def deterministic_summary(chosen: list[str], n_days: int, has_shortfall: bool) -> str:
    text = f"Your {n_days}-day plan covers every tracked muscle group where your setup allows."
    if chosen:
        text += f" Priority regions: {', '.join(r.replace('_', ' ') for r in chosen)}."
    if has_shortfall:
        text += " Some muscle groups fall short of their typical weekly minimum; they are listed."
    return text


def deterministic(det: WeeklyPlan, reason: str, summary: str) -> Written:
    return Written(
        served_by="deterministic",
        fallback_reason=reason,
        days=det.days,
        labels=[f"Day {i + 1}" for i in range(len(det.days))],
        reps=[[DEFAULT_REPS] * len(d) for d in det.days],
        cues=[[DEFAULT_CUE] * len(d) for d in det.days],
        summary=summary,
    )


def write(*, enabled: bool, llm: LLM, facts: dict, ctx: Context, summary: str) -> Written:
    """At most MAX_CALLS calls and MAX_RETRIES retries per report. The deterministic result carries
    the last failure; a fallback result carries why the primary failed."""
    if not enabled:
        return deterministic(ctx.det, "disabled", summary)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(facts)},
    ]
    schema = WriterOutput.model_json_schema()
    calls, retries, last, first = 0, MAX_RETRIES, None, None
    for provider in ("primary", "fallback"):
        if not llm.available(provider):
            last = last or "disabled"  # an unconfigured provider never hides an earlier failure
        else:
            while calls < MAX_CALLS:
                calls += 1
                try:
                    written = _accept(_parse(llm.complete(provider, messages, schema)), ctx)
                except LLMError as e:
                    last = e.reason
                    if e.reason in TRANSIENT and retries > 0 and calls < MAX_CALLS:
                        retries -= 1
                        continue
                    break
                except Reject as e:
                    last = e.reason
                    break
                written.served_by = provider
                written.fallback_reason = first if provider == "fallback" else None
                return written
        first = first or last
    return deterministic(ctx.det, last or "disabled", summary)


def build_report(
    written: Written,
    gap: GapProfile,
    comp: BodyComp | None,
    chosen: list[str],
    realism: str | None,
    by_id: dict[str, dict],
) -> Report | None:
    """None when a day has no exercise (Report needs at least one per day)."""
    if not written.days or any(not d for d in written.days):
        return None
    plan = [
        PlanDay(
            label=written.labels[i],
            exercises=[
                PlannedExercise(
                    name=s.name,
                    muscle=by_id[s.exercise_id]["muscle"][0],
                    sets=s.sets,
                    reps=written.reps[i][j],
                    cue=written.cues[i][j],
                    citations=[s.exercise_id],
                )
                for j, s in enumerate(day)
            ],
        )
        for i, day in enumerate(written.days)
    ]
    return Report(
        summary=written.summary,
        priorities=chosen,
        gap=gap,
        body_comp=comp,
        plan=plan,
        realism_note=realism,
        disclaimer=DISCLAIMER,
    )
