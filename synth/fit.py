"""The fitted map from slider values to a waist-girth proxy of adiposity (ADR-053). Pure Python.

The MPFB `weight` slider is a mass slider, not a fat slider. The adiposity score is therefore defined from what
the mesh does: the waist girth that `weight` adds at fixed `muscle` (a partial effect), read from a lookup table
measured on the muscle x weight probe grid only (synth/probe.py, never on training identities) and bilinearly
interpolated. It is a **girth proxy, not body fat %**. The tables, the clamps and the check results live in
synth/plan/fit.json (written by `python -m synth.fit_labels`); its coefficient hash is frozen in the ADR before
render.py starts.
"""

import hashlib
import json
import math
from pathlib import Path

FIT_PATH = Path(__file__).parent / "plan" / "fit.json"
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)  # the probe grid, per slider
LATTICE = 21  # points per slider for the normaliser and the monotone checks


def girth_cm(width_cm: float, depth_cm: float) -> float:
    """Elliptical girth (Ramanujan) of a torso band from its width and depth."""
    a, b = width_cm / 2, depth_cm / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _interp(table: list[list[float]], muscle: float, weight: float) -> float:
    """Bilinear lookup on the 5 x 5 probe grid; rows = muscle, columns = weight, both 0..1."""

    def cell(x: float) -> tuple[int, float]:
        x = min(max(x, 0.0), 1.0) * (len(LEVELS) - 1)
        k = min(int(x), len(LEVELS) - 2)
        return k, x - k

    (i, fm), (j, fw) = cell(muscle), cell(weight)
    low = table[i][j] * (1 - fw) + table[i][j + 1] * fw
    high = table[i + 1][j] * (1 - fw) + table[i + 1][j + 1] * fw
    return low * (1 - fm) + high * fm


def lattice(lo_hi: list[float]) -> list[float]:
    lo, hi = lo_hi
    return [lo + (hi - lo) * k / (LATTICE - 1) for k in range(LATTICE)]


class Fit:
    def __init__(self, data: dict):
        self.data = data

    def girth(self, sex: str, muscle: float, weight: float) -> float:
        return _interp(self.data["girth"][sex], muscle, weight)

    def raw_added(self, sex: str, muscle: float, weight: float) -> float:
        """Girth (cm) that weight adds over the lowest allowed weight, at this muscle."""
        w_lo = self.data["clamps"][sex]["weight"][0]
        return self.girth(sex, muscle, weight) - self.girth(sex, muscle, w_lo)

    def adiposity(self, sex: str, muscle: float, weight: float) -> float:
        """Normalised adiposity proxy in [0, 1] (per sex: 1 = the most girth weight adds inside the clamps)."""
        return min(max(self.raw_added(sex, muscle, weight) / self.data["a_ref"][sex], 0.0), 1.0)


def reference_added(girth: dict, clamps: dict) -> dict[str, float]:
    """Per sex, the largest girth added by weight over the clamped lattice: the normaliser."""
    probe = Fit({"girth": girth, "clamps": clamps})
    return {
        sex: max(
            probe.raw_added(sex, m, w) for m in lattice(clamps[sex]["muscle"]) for w in lattice(clamps[sex]["weight"])
        )
        for sex in girth
    }


def build(girth: dict, clamps: dict) -> dict:
    """The fit dict (without checks or hash) from the probe girth tables and the clamps."""
    return {"girth": girth, "clamps": clamps, "a_ref": reference_added(girth, clamps)}


def coefficient_hash(data: dict) -> str:
    """SHA-256 of the coefficients: the tables, the clamps and the normaliser (not the check report)."""
    core = {k: data[k] for k in ("girth", "clamps", "a_ref")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path: Path = FIT_PATH) -> Fit:
    if not path.exists():
        raise SystemExit(f"{path} not found: run `uv run python -m synth.fit_labels <probe_measures.csv>` first")
    return Fit(json.loads(path.read_text()))
