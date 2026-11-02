"""SiameseSubtractAnalyzer, the shipped analyzer (ADR-035): one ONNX encoder predicts each person's
proportions (measurement / height) and the gap is goal minus current, so it is exactly
antisymmetric. Confidence is the region reliability from val: fixed per region and slice, not per
scan. upper_back and lats have no BodyM label and are returned as not measured: upper_back stays
that way (ADR-037 Amendment 2) and lats is filled in by the route (`app/analyzer/lats.py`) from
this analyzer's chest, shoulders and waist (ADR-036), under the sex rule of ADR-037 Amendment 3.

`GapProfile.v_taper` (shoulders / waist, ADR-038) is derived from the same per-person ratios.

Both scans are encoded on every request (one encode is ~16 ms on one thread) and nothing about a
scan is kept between requests.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.analyzer.masks import decode_scan, side_view
from ml.src.calibrate import reliability
from ml.src.preprocess import model_input
from physique_schemas.models import UPPER_BACK_REASON, GapProfile, RegionGap, Scan, VTaper

ONNX_PATH = Path(__file__).parent / "gap_regressor.onnx"
META_PATH = Path(__file__).parent / "gap_regressor.json"
V_TAPER_NOT_MEASURED = "Frame V-taper needs both shoulders and waist to be measured."


def _reliability(sigmas: dict, front_only: bool) -> float:
    sigma_goal = sigmas["sigma_front"] if front_only else sigmas["sigma_full"]
    return float(reliability(sigma_goal, sigmas["sigma_full"], sigmas["sigma_prior"]))


def v_taper(regions: dict[str, RegionGap], front_only: bool, block: dict | None) -> VTaper:
    """Frame V-taper (ADR-038): shoulders / waist per person, from B's per-person ratios. `block` is
    the `v_taper` entry of gap_regressor.json (val sigmas); without it there is no reliability."""
    shoulders, waist = regions["shoulders"], regions["waist"]
    measured = shoulders.measured and waist.measured
    if not measured or min(waist.goal_ratio, waist.current_ratio) <= 0:
        return VTaper(measured=False, reason=V_TAPER_NOT_MEASURED)
    goal = shoulders.goal_ratio / waist.goal_ratio
    current = shoulders.current_ratio / waist.current_ratio
    return VTaper(
        goal_index=goal,
        current_index=current,
        delta=float(np.clip(goal - current, -1, 1)),
        reliability=_reliability(block, front_only) if block else None,
    )


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1  # 1 vCPU, as measured
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class SiameseSubtractAnalyzer:
    def __init__(self, session=None, meta: dict | None = None, onnx_path: Path = ONNX_PATH):
        self.session = session or _session(onnx_path)
        self.meta = meta or json.loads(META_PATH.read_text())

    def compare(self, goal: Scan, current: Scan) -> GapProfile:
        goal_masks, current_masks = decode_scan(goal), decode_scan(current)  # validates every view
        front_only = side_view(goal_masks) is None
        g, c = self._proportions(goal_masks), self._proportions(current_masks)
        regions = {
            region: RegionGap(
                delta=float(np.clip(g[k] - c[k], -1, 1)),
                confidence=_reliability(self.meta["calibration"][region], front_only),
                goal_ratio=float(g[k]),
                current_ratio=float(c[k]),
            )
            for k, region in enumerate(self.meta["output"]["regions"])
        }
        regions |= {
            "upper_back": RegionGap(delta=0.0, confidence=0.0, measured=False,
                                    reason=UPPER_BACK_REASON),
            "lats": RegionGap(delta=0.0, confidence=0.0, measured=False),
        }  # fmt: skip
        return GapProfile(
            analyzer="siamese_subtract",
            regions=regions,
            goal_front_only=front_only,
            v_taper=v_taper(regions, front_only, self.meta.get("v_taper")),
        )

    def _proportions(self, masks: dict[str, np.ndarray]) -> np.ndarray:
        x = model_input(masks["front"], side_view(masks))[None]
        return self.session.run(None, {"x": x})[0][0]
