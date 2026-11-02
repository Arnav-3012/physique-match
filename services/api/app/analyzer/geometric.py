"""Geometric analyzer, baseline A (ADR-001, ADR-024). Silhouette in, GapProfile out.

delta = slope * (goal feature - current feature), in body heights. Weight is not an input: a goal
photo has none, and the two people must sit on the same scale. Confidence is the validated val
rho of the region's feature, damped when the row finder fell back to a prior. A goal without a side
view uses the width-only fits for both people (ADR-033), so no view being absent raises.
upper_back and lats are never measured here: the back-view widths were unvalidated (ADR-037).
"""

import json
from pathlib import Path

import numpy as np

from app.analyzer.features import extract
from app.analyzer.masks import decode_scan, has_side_view, side_view
from physique_schemas.models import REGIONS, UPPER_BACK_REASON, GapProfile, RegionGap, Scan

COEF = json.loads((Path(__file__).parent / "gap_coef.json").read_text())
COEF_WIDTHS = json.loads((Path(__file__).parent / "gap_coef_widths.json").read_text())
RHO_FLOOR, RHO_SPAN = 0.3, 0.6  # confidence 0 at rho 0.3 (noise), 1 at rho 0.9
PRIOR_DAMPING = 0.7  # per fallback flag (arms not separated / crotch not found)


class GeometricAnalyzer:
    def compare(self, goal: Scan, current: Scan) -> GapProfile:
        g, c = self._features(goal), self._features(current)
        damping = min(self._damping(g), self._damping(c))
        front_only = not has_side_view(goal)
        coef = COEF_WIDTHS if front_only else COEF
        blank = RegionGap(delta=0.0, confidence=0.0, measured=False)
        regions = {}
        for region in REGIONS:
            if region not in coef:  # upper_back, lats: retired here (ADR-036, ADR-037 Amendment 2)
                reason = UPPER_BACK_REASON if region == "upper_back" else None
                regions[region] = blank.model_copy(update={"reason": reason})
                continue
            k = coef[region]
            feat = g[k["feature"]], c[k["feature"]]
            if np.isnan(feat).any():
                regions[region] = blank
                continue
            conf = float(np.clip((k["rho_val"] - RHO_FLOOR) / RHO_SPAN, 0, 1))
            delta = float(np.clip(k["slope"] * (feat[0] - feat[1]), -1, 1))
            regions[region] = RegionGap(delta=delta, confidence=conf * damping)
        return GapProfile(analyzer="geometric", regions=regions, goal_front_only=front_only)

    @staticmethod
    def _features(scan: Scan) -> dict[str, float]:
        masks = decode_scan(scan)
        return extract(masks["front"], side_view(masks))

    @staticmethod
    def _damping(f: dict[str, float]) -> float:
        return PRIOR_DAMPING ** (2 - f["arms_separated"] - f["crotch_found"])
