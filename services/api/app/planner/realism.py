"""Code-written notes for a report (ADR-058): the realism note and the fixed warnings. No LLM, no
client text: unmeasured regions are named from the region enum, never from `RegionGap.reason`."""

from physique_schemas.models import UPPER_BACK_REASON, GapProfile

REALISM_GAP = 0.05  # largest priority gap, as a fraction of height: our rule, untuned (ADR-058)
INTERMEDIATE_FRACTION = 0.5  # the first target is this share of that gap: ours, untuned

OTHER_SEX = "The goal physique is the other sex from yours, so absolute size gaps may not be reachable."


NOT_MEASURED_REASON = "Not measured in this scan."


def clean_gap(gap: GapProfile) -> GapProfile:
    """The gap as the report may carry it: client text (`reason`, the V-taper wording) never comes back.
    Unmeasured regions get a fixed reason; the V-taper, which is display-only, is left out."""
    regions = {
        r: g
        if g.measured
        else g.model_copy(update={"reason": UPPER_BACK_REASON if r == "upper_back" else NOT_MEASURED_REASON})
        for r, g in gap.regions.items()
    }
    return gap.model_copy(update={"regions": regions, "v_taper": None})


def realism_note(gap: GapProfile, chosen: list[str], height_cm: float) -> str | None:
    """Shown once, when the largest priority gap is at least REALISM_GAP. A negative gap never counts
    (priorities hold positive gaps only)."""
    if not chosen:
        return None
    top = max(gap.regions[r].delta for r in chosen)
    if round(top, 6) < REALISM_GAP:
        return None
    half = top * INTERMEDIATE_FRACTION
    return (
        "Goal photos are often posed, lit, pumped or edited, so this gap may be larger than natural "
        "training reaches. A sensible first target is about half of the largest gap: "
        f"{half * 100:.1f}% of your height, about {half * height_cm:.0f} cm at your height. "
        "Reassess after that."
    )


def warnings(gap: GapProfile, user_sex: str, goal_sex: str) -> list[str]:
    out = []
    if "not_specified" not in (user_sex, goal_sex) and user_sex != goal_sex:
        out.append(OTHER_SEX)
    missing = [r for r, g in gap.regions.items() if not g.measured]
    if missing:
        names = [{"lats": "back width (estimate)"}.get(r, r.replace("_", " ")) for r in missing]
        out.append(f"Not measured in this scan: {', '.join(names)}.")
    if "upper_back" in missing:
        out.append(f"Upper back: {UPPER_BACK_REASON}")
    return out
