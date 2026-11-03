"""Scale-free silhouette features per region (ADR-024). No weight, no cm: every value is a length
divided by the view's own body height in px, so a goal photo and a current scan are comparable.

Front/back masks give widths at the rows from `rows.py`; the side mask gives depths at the same
body fractions. Chest, waist, hip, thigh and calf combine both into an ellipse girth.
A side row is skipped (NaN) when its depth spikes against the rows around it, which is how a
protruding arm or hand looks in profile.
"""

import numpy as np

from app.analyzer.rows import Rows, find_rows, row_runs
from ml.src.preprocess import H, body_px

# Starting guesses, tuned on the BodyM train split only. Not facts.
# Body heights below the armpit. Swept on train (ml/src/sweep_windows.py): the armpit window gives
# rho .83 vs bicep. It also scores .78 vs forearm, but that is general arm size, not the forearm,
# so the forearm keeps its own window (rho ~.38) and is reported as low confidence.
UPPER_ARM_ROWS = (0.0, 0.07)
FOREARM_ROWS = (0.12, 0.19)
MIN_ARM_ROWS = 8  # rows with separated arms needed for a thickness
THIGH_RANGE = (0.02, 0.25)  # fractions of the crotch -> feet distance; the widest row is used
CALF_RANGE = (0.50, 0.80)
LEG_RATIO = 0.6  # second-widest run counts as a second leg above this fraction of the widest
SIDE_CLEAN_HALF = 0.06  # body heights each side used as the local reference depth
SIDE_CLEAN_TOL = 0.15  # |depth - local median| / local median above this -> skip the row
BAND = 2  # rows each side averaged (median) around a feature row

GIRTH_REGIONS = ("chest", "waist", "hip", "thigh", "calf")
REGION_FEATURE = {
    "shoulders": "shoulder_w",
    "chest": "chest_g",
    "waist": "waist_g",
    "hips": "hip_g",
    "upper_arm": "upper_arm_w",
    "forearm": "forearm_w",
    "thigh": "thigh_w",  # width beat the ellipse girth on val (.68 vs .61, .62 vs .60): the
    "calf": "calf_w",  # side depth of the legs is polluted by hands and clipped feet
    "upper_back": "upper_back_w",
    "lats": "lats_w",
}
# Front-only goal (ADR-025/033): the side-dependent regions fall back to their front width.
WIDTH_FEATURE = REGION_FEATURE | {"chest": "chest_w", "waist": "waist_w", "hips": "hip_w"}


def girth(width: float, depth: float) -> float:
    """Ramanujan's ellipse perimeter from full width and depth."""
    a, b = width / 2, depth / 2
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def _span(runs):
    return runs[-1][1] - runs[0][0] if runs else np.nan


def _widest(runs):
    return max((b - a for a, b in runs), default=np.nan)


def arm_thickness(mask: np.ndarray, r0: int, r1: int) -> float:
    """Mean arm thickness in px over rows r0..r1, independent of the A-pose angle.

    thickness = area / path length: a band at angle t has horizontal width w/cos(t) and a path of
    1/cos(t) per row. Uses the outer two runs of rows where the arms are separated.
    """
    sides = []
    runs = [row_runs(mask[r]) for r in range(r0, r1)]
    for pick in (0, -1):  # left arm = first run, right arm = last run
        arm = [(i, rr[pick]) for i, rr in enumerate(runs) if len(rr) >= 3]
        if len(arm) < MIN_ARM_ROWS:
            continue
        rows, spans = np.array([i for i, _ in arm]), np.array([s for _, s in arm], float)
        centre, width = spans.mean(axis=1), spans[:, 1] - spans[:, 0]
        step = np.hypot(np.diff(rows), np.diff(centre))
        sides.append(width.sum() / step.sum() * (len(step)) / len(width))
    return float(np.mean(sides)) if sides else np.nan


def _leg(runs):  # mean of the two leg runs; one fused run counts as two legs
    w = sorted((b - a for a, b in runs), reverse=True)
    if not w:
        return np.nan
    return (w[0] + w[1]) / 2 if len(w) >= 2 and w[1] >= LEG_RATIO * w[0] else w[0] / 2


def _median(values) -> float:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return float(np.median(v)) if v.size else np.nan


def _at(mask: np.ndarray, row: int, pick) -> float:
    rows = range(max(row - BAND, 0), min(row + BAND + 1, H))
    return _median([pick(row_runs(mask[r])) for r in rows])


def _peak(mask: np.ndarray, r0: int, r1: int, pick) -> tuple[int, float]:
    """(row, value) of the largest `pick` over rows r0..r1; the row ties width to side depth."""
    rows = list(range(r0, r1 + 1, 2))
    values = np.array([_at(mask, r, pick) for r in rows])
    if np.isnan(values).all():
        return (r0 + r1) // 2, np.nan
    i = int(np.nanargmax(values))
    return rows[i], float(values[i])


def front_widths(mask: np.ndarray) -> tuple[dict[str, float], Rows, dict[str, int]]:
    """Widths / body height at the landmark rows of a front or back mask."""
    rows, bpx = find_rows(mask), body_px(mask)
    leg = rows.feet - rows.crotch
    where = {
        "shoulder": (rows.shoulder, _span),
        "chest": (rows.chest, _widest),
        "waist": (rows.waist, _widest),
        "hip": (rows.hip, _widest),
    }
    widths = {f"{n}_w": _at(mask, r, pick) / bpx for n, (r, pick) in where.items()}
    at_row = {n: r for n, (r, _) in where.items()}
    for name, span in {"thigh": THIGH_RANGE, "calf": CALF_RANGE}.items():
        row, value = _peak(mask, *(rows.crotch + round(f * leg) for f in span), _leg)
        widths[f"{name}_w"], at_row[name] = value / bpx, row
    for name, (lo, hi) in {"upper_arm": UPPER_ARM_ROWS, "forearm": FOREARM_ROWS}.items():
        r0, r1 = (rows.armpit + round(f * bpx) for f in (lo, hi))
        widths[f"{name}_w"] = arm_thickness(mask, r0, r1) / bpx
    return widths, rows, at_row


def side_row(front_row: int, crown_f: int, bpx_f: float, side: np.ndarray) -> int:
    """Row of the side mask at the same fraction of body height as `front_row` in the front."""
    return round((H - body_px(side)) / 2 + (front_row - crown_f) / bpx_f * body_px(side))


def side_depths(mask: np.ndarray, front_rows: dict[str, int], crown_f: int, bpx_f: float) -> dict[str, float]:
    """Depth / body height at the body fractions of the front rows; NaN where it spikes."""
    bpx = body_px(mask)
    depth = np.array([_widest(row_runs(r)) for r in mask], float)
    half = round(SIDE_CLEAN_HALF * bpx)
    out = {}
    for name in GIRTH_REGIONS:
        row = side_row(front_rows[name], crown_f, bpx_f, mask)
        local = _median(depth[max(row - half, 0) : row + half + 1])
        d = _median(depth[max(row - BAND, 0) : row + BAND + 1])
        clean = abs(d - local) <= SIDE_CLEAN_TOL * local
        out[f"{name}_d"] = d / bpx if clean else np.nan
    return out


def extract(front: np.ndarray, side: np.ndarray | None = None, back: np.ndarray | None = None) -> dict[str, float]:
    """All features for one person. Missing views give NaN (never a guess).

    `side` must not be a clipped BodyM mask (ml.src.data.bodym.clipped); the caller drops those.
    """
    widths, rows, where = front_widths(front)
    out = dict(widths)
    bpx = body_px(front)
    depths = (
        side_depths(side, where, rows.crown, bpx) if side is not None else {f"{n}_d": np.nan for n in GIRTH_REGIONS}
    )
    out |= depths
    for n in GIRTH_REGIONS:
        out[f"{n}_g"] = girth(widths[f"{n}_w"], depths[f"{n}_d"])

    if back is not None:  # upper back / lats come from the back view when we have one
        b_widths, _, _ = front_widths(back)
        out["upper_back_w"], out["lats_w"] = b_widths["shoulder_w"], b_widths["chest_w"]
    else:
        out["upper_back_w"], out["lats_w"] = np.nan, np.nan
    out["arms_separated"] = float(rows.arms_separated)
    out["crotch_found"] = float(rows.crotch_found)
    return out
