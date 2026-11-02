"""Definition status from the capture scope (ADR-051, ADR-053 Amendment 4).

The demo model has two heads, an overall definition ("leanness & muscularity (estimate)") that is dominated by the
abdomen, and adiposity. So the `abs` entry stands for the overall definition score, and every other region is N/A
(`not_measured_not_in_demo`); regional heads are TBD after demo. A scan that does not show the abdomen (the clothed
scan, the arms-shoulders-legs scan) has no definition score, because the model is trained on bare-body renders and a
covered region would be noise shown as a number. N/A always carries a reason, never a number, never read as zero.
Precedence: age gate, then scope, then the goal photo.
"""

from collections.abc import Collection

from physique_schemas.models import (
    DEFINITION_REGIONS,
    MIN_ADULT_AGE,
    DefinitionRegionStatus,
    DefinitionScope,
)

REASON_CLOTHED = "Clothed scan: definition is not scored."
REASON_NO_ABDOMEN = "This scan does not show the abdomen, so definition is not scored."
REASON_AGE = "This option is for adults."
REASON_GOAL = "The goal photo does not show the abdomen, so there is no goal definition to compare."
REASON_REGION = "The demo scores one overall definition from the abdomen, not this region."
REASON_BACK = "The demo does not score the back."


def _na(status: str, reason: str) -> DefinitionRegionStatus:
    return DefinitionRegionStatus(status=status, reason=reason)


def definition_statuses(
    scope: DefinitionScope, age: int | None, goal_not_shown: Collection[str]
) -> dict[str, DefinitionRegionStatus]:
    """One status per definition region. `measured` on `abs` means the overall score is eligible."""
    adult = age is not None and age >= MIN_ADULT_AGE
    out = {}
    for region in DEFINITION_REGIONS:
        if scope == "full_body" and not adult:
            out[region] = _na("not_measured_age_gate", REASON_AGE)
        elif scope == "none":
            out[region] = _na("not_measured_covered", REASON_CLOTHED)
        elif scope == "arms_shoulders_legs":
            out[region] = _na("not_measured_covered", REASON_NO_ABDOMEN)
        elif region == "abs":
            if "abs" in goal_not_shown:
                out[region] = _na("not_measured_goal_covers", REASON_GOAL)
            else:
                out[region] = DefinitionRegionStatus(status="measured")
        elif region == "back":
            out[region] = _na("not_measured_not_in_demo", REASON_BACK)
        else:
            out[region] = _na("not_measured_not_in_demo", REASON_REGION)
    return out
