"""Geometric analyzer, baseline A (ADR-001, ADR-024). Silhouette in, GapProfile out.

delta = slope * (goal feature - current feature), in body heights. Weight is not an input: a goal
photo has none, and the two people must sit on the same scale. Confidence is the validated val
rho of the region's feature, damped when the row finder fell back to a prior.
"""

import json
from pathlib import Path

import numpy as np

from app.analyzer.features import extract
from app.analyzer.masks import decode_mask
from physique_schemas.models import REGIONS, GapProfile, RegionGap, Scan

COEF = json.loads((Path(__file__).parent / "gap_coef.json").read_text())
RHO_FLOOR, RHO_SPAN = 0.3, 0.6  # confidence 0 at rho 0.3 (noise), 1 at rho 0.9
UNVALIDATED = {"upper_back": "upper_back_w", "lats": "lats_w"}  # back view, no BodyM label
UNVALIDATED_CONF = 0.3
PRIOR_DAMPING = 0.7  # per fallback flag (arms not separated / crotch not found)


class GeometricAnalyzer:
    def compare(self, goal: Scan, current: Scan) -> GapProfile:
        g, c = self._features(goal), self._features(current)
        damping = min(self._damping(g), self._damping(c))
        regions = {}
        for region in REGIONS:
            if region in COEF:
                k = COEF[region]
                scale = k["slope"]
                conf = float(np.clip((k["rho_val"] - RHO_FLOOR) / RHO_SPAN, 0, 1))
                feat = k["feature"]
            else:
                scale, conf, feat = 1.0, UNVALIDATED_CONF, UNVALIDATED[region]
            pair = g[feat], c[feat]
            if np.isnan(pair).any():
                regions[region] = RegionGap(delta=0.0, confidence=0.0)
                continue
            delta = float(np.clip(scale * (pair[0] - pair[1]), -1, 1))
            regions[region] = RegionGap(delta=delta, confidence=conf * damping)
        return GapProfile(analyzer="geometric", regions=regions)

    @staticmethod
    def _features(scan: Scan) -> dict[str, float]:
        masks = {name: decode_mask(v.mask_png_b64) for name, v in scan.views.items()}
        return extract(masks["front"], masks["left"], masks.get("back"))

    @staticmethod
    def _damping(f: dict[str, float]) -> float:
        return PRIOR_DAMPING ** (2 - f["arms_separated"] - f["crotch_found"])
