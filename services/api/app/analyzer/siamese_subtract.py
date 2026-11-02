"""SiameseSubtractAnalyzer, the shipped analyzer (ADR-035): one ONNX encoder predicts each person's
proportions (measurement / height) and the gap is goal minus current, so it is exactly
antisymmetric. Confidence is the region reliability from val: fixed per region and slice, not per
scan. upper_back and lats have no BodyM label, so they come from the geometric analyzer; without a
back view on both scans they are reported as not measured.

Both scans are encoded on every request (one encode is ~16 ms on one thread) and nothing about a
scan is kept between requests.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.analyzer.geometric import GeometricAnalyzer
from app.analyzer.masks import decode_scan, side_view
from ml.src.calibrate import reliability
from ml.src.preprocess import model_input
from physique_schemas.models import GapProfile, RegionGap, Scan

ONNX_PATH = Path(__file__).parent / "gap_regressor.onnx"
META_PATH = Path(__file__).parent / "gap_regressor.json"
BACK_VIEW_REGIONS = ("upper_back", "lats")


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1  # 1 vCPU, as measured
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class SiameseSubtractAnalyzer:
    def __init__(self, session=None, meta: dict | None = None, onnx_path: Path = ONNX_PATH):
        self.session = session or _session(onnx_path)
        self.meta = meta or json.loads(META_PATH.read_text())
        self.geometric = GeometricAnalyzer()

    def compare(self, goal: Scan, current: Scan) -> GapProfile:
        goal_masks, current_masks = decode_scan(goal), decode_scan(current)  # validates every view
        front_only = side_view(goal_masks) is None
        g, c = self._proportions(goal_masks), self._proportions(current_masks)
        regions = {
            region: RegionGap(
                delta=float(np.clip(g[k] - c[k], -1, 1)),
                confidence=self._reliability(region, front_only),
                goal_ratio=float(g[k]),
                current_ratio=float(c[k]),
            )
            for k, region in enumerate(self.meta["output"]["regions"])
        }
        regions |= self._back_view_regions(goal, current)
        return GapProfile(analyzer="siamese_subtract", regions=regions, goal_front_only=front_only)

    def _proportions(self, masks: dict[str, np.ndarray]) -> np.ndarray:
        x = model_input(masks["front"], side_view(masks))[None]
        return self.session.run(None, {"x": x})[0][0]

    def _reliability(self, region: str, front_only: bool) -> float:
        s = self.meta["calibration"][region]
        sigma_goal = s["sigma_front"] if front_only else s["sigma_full"]
        return float(reliability(sigma_goal, s["sigma_full"], s["sigma_prior"]))

    def _back_view_regions(self, goal: Scan, current: Scan) -> dict[str, RegionGap]:
        if "back" not in goal.views or "back" not in current.views:
            blank = RegionGap(delta=0.0, confidence=0.0, measured=False)
            return dict.fromkeys(BACK_VIEW_REGIONS, blank)
        gap = self.geometric.compare(goal, current)
        return {r: gap.regions[r] for r in BACK_VIEW_REGIONS}
