"""Ranked candidate lists for the planner, built outside it (ADR-055): the planner never calls the
retriever. Equipment is filtered on the closed checklist values directly, not through
`query.EQUIPMENT`; an empty list means body only, never "no filter"."""

from collections.abc import Callable

from app.planner.params import TOP_K, kb_json
from app.rag.query import levels_up_to
from app.rag.retrieve import Filters
from physique_schemas.models import UserProfile

Search = Callable[[str, Filters], list[str]]  # query, filters -> chunk ids, best first


def ranked_candidates(profile: UserProfile, search: Search) -> dict[str, list[str]]:
    gear = tuple(profile.equipment) or ("body only",)
    levels = levels_up_to(profile.level)
    return {
        group: search(f"{group} exercise", Filters(muscle=(group,), equipment=gear, level=levels))[:TOP_K]
        for group in kb_json("baselines.json")["groups"]
    }
