"""Definition latent and its effects (ADR-049, amended by ADR-049 Amendment 2 and ADR-053). Pure Python: the
shader code, the coverage check and the tests share one definition. Every constant is fixed before any render
and is not tuned after.

Positions m (muscle) and f (weight) are in [0, 1] inside the clamped slider range. The `a` that hide() and
d_eff() take is NOT a slider position: it is the normalised adiposity proxy (0..1) from the fitted waist-girth
lookup (synth/fit.py): the girth the weight slider adds at fixed muscle. A girth proxy, not body fat %.
"""

import math
from bisect import bisect_left
from statistics import quantiles

REGIONS = ("abs", "chest", "shoulders", "arms", "back", "legs")
LEAN_F, SOFT_F = (
    1 / 3,
    0.8,
)  # hide(a) = 1 up to LEAN_F, 0 from SOFT_F, on the normalised adiposity proxy a
CAP_BASE = 0.25  # cap(m) = CAP_BASE + (1 - CAP_BASE) * m: low muscle caps the cues
JITTER_SD = 0.12  # per-region muscle jitter around the cell's muscle
BUMP_SIGMA = 0.35  # log-normal spread of the bump factor (mean 1)
BUMP_CAP = 1.0


def clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def hide(a: float) -> float:
    """High adiposity hides cues: 1 for a <= LEAN_F, linear down to 0 at SOFT_F."""
    return clip01((SOFT_F - a) / (SOFT_F - LEAN_F))


def muscle_cap(m_r: float) -> float:
    return CAP_BASE + (1 - CAP_BASE) * m_r


def regional_muscle(m: float, jitter: float) -> float:
    return clip01(m + jitter)


def d_eff(d: float, a: float, m_r: float) -> float:
    return min(d * hide(a), muscle_cap(m_r))


def bump_strength(d_eff_r: float, z: float) -> float:
    """Texture strength: d_eff times a log-normal factor (mean 1), so it is not the label."""
    return min(d_eff_r * math.exp(BUMP_SIGMA * z - BUMP_SIGMA**2 / 2), BUMP_CAP)


def flat_score(m_r: float, a: float) -> float:
    """Bulk-plus-leanness proxy for the flat set (no bump): muscular and lean scores high."""
    return m_r * (1 - a)


def target_signed(m_r: float) -> tuple[str, float]:
    """Regional muscle target: ('incr' | 'decr', weight) from s = 2 m_r - 1."""
    s = 2 * m_r - 1
    return ("incr" if s > 0 else "decr", abs(s))


def bin_edges(values: list[float], classes: int = 5) -> tuple[float, ...]:
    """Quantile edges (20/40/60/80% for 5 classes) over the training partition of the planned sequence."""
    return tuple(quantiles(values, n=classes, method="inclusive"))


def label(value: float, edges: tuple[float, ...]) -> int:
    """1 + the number of edges strictly below the value; with 4 edges this is a class 1-5. A value equal to an
    edge goes to the LOWER class (ADR-045 tie rule): high fat gives d_eff = 0 for about 20% of identities, a
    point mass that sits exactly on the 20% edge and would otherwise empty class 1."""
    return 1 + bisect_left(edges, value)
