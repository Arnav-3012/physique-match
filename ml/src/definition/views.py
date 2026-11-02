"""The one view-to-head map of the 8-head definition model (ADR-053 Amendment 5). Stdlib only.

Training (the loss mask), evaluation (which samples count for a head) and inference (which heads an image may
show) all read HEAD_VIEWS through valid_mask / valid_heads. Pre-registered; not tuned after results.
"""

HEADS8 = ("definition", "abs", "chest", "shoulders", "arms", "back", "legs", "adiposity")
REGIONS = ("abs", "chest", "shoulders", "arms", "back", "legs")
VIEWS = ("front", "back")
HEAD_VIEWS = {
    "definition": ("front",),
    "abs": ("front",),
    "chest": ("front",),
    "shoulders": ("front",),
    "arms": VIEWS,
    "back": ("back",),
    "legs": VIEWS,
    "adiposity": VIEWS,
}


def valid_mask(view: str) -> tuple[bool, ...]:
    """Which of the 8 heads are valid for an image of this view, in HEADS8 order."""
    return tuple(view in HEAD_VIEWS[h] for h in HEADS8)


def valid_heads(view: str) -> tuple[str, ...]:
    return tuple(h for h, ok in zip(HEADS8, valid_mask(view), strict=False) if ok)
