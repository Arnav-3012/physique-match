"""Priority regions: the rule fixed in evals/p6-README.md (ADR-054). Placeholders, never tuned after
seeing plans; the minimum gap is untested."""

from physique_schemas.models import REGIONS, GapProfile

MIN_GAP = 0.010  # 1% of height
MAX_PRIORITIES = 3
# lats is unranked (ADR-037), upper_back is never measured, abdominal work does not shrink girth.
NOT_RANKED = frozenset({"lats", "upper_back", "waist"})


def priorities(gap: GapProfile) -> list[str]:
    """Up to 3 measured regions with the largest positive delta, at least MIN_GAP; ties go to the
    higher confidence, then the order of REGIONS. A negative delta is never a growth target."""
    ranked = sorted(
        (-g.delta, -g.confidence, REGIONS.index(r), r)
        for r, g in gap.regions.items()
        if g.measured and r not in NOT_RANKED and round(g.delta, 6) >= MIN_GAP
    )
    return [r for *_, r in ranked[:MAX_PRIORITIES]]
