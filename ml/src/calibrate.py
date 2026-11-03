"""Region reliability inputs for the shipped model, from val residuals (ADR-033, ADR-035; no MC
dropout). The UI and the model card call the result "region reliability": fixed per region and
slice, the same for every scan, not a per-scan confidence.

Per region, over val subjects: the residual std of the per-person prediction with the side view
(sigma_full) and with it blanked (sigma_front). sigma_prior is the std of the true gap over every
ordered val pair, the same units as the gap error; the truth does not depend on the slice, so it is
the same for both. The reliability of a request:
    sigma_gap   = sqrt(sigma_goal^2 + sigma_current^2)   goal = front if no side view, else full
    reliability = clip(1 - sigma_gap / sigma_prior, 0, 1)
The sum assumes independent errors, which overstates sigma_gap when both people share a bias.
Numpy only. Val also picked the epoch, so these sigmas are slightly optimistic (model card).
"""

import numpy as np

from ml.src.data.regions import LABELLED_REGIONS


def _sigmas(full: np.ndarray, front: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per column of (subjects, columns) arrays: residual stds and the true-gap std."""
    off_diagonal = ~np.eye(len(y), dtype=bool)
    pair_gaps = (y[:, None, :] - y[None, :, :])[off_diagonal]  # (ordered pairs, columns)
    return {
        "sigma_full": (full - y).std(axis=0, ddof=1),
        "sigma_front": (front - y).std(axis=0, ddof=1),
        "sigma_prior": pair_gaps.std(axis=0),
    }


def calibrate(full: np.ndarray, front: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    """full, front, y: (subjects, regions) predictions and truth, measurement / height."""
    s = _sigmas(full, front, y)
    return {region: {name: float(v[k]) for name, v in s.items()} for k, region in enumerate(LABELLED_REGIONS)}


def index_sigmas(full: np.ndarray, front: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """The same three numbers for the frame V-taper index, shoulders / waist (ADR-038): residual
    of the predicted index against the true one, and the std of the true index gap."""
    shoulders, waist = (LABELLED_REGIONS.index(r) for r in ("shoulders", "waist"))

    def index(a: np.ndarray) -> np.ndarray:
        return (a[:, shoulders] / a[:, waist])[:, None]

    return {name: float(v[0]) for name, v in _sigmas(index(full), index(front), index(y)).items()}


def reliability(sigma_goal, sigma_current, sigma_prior):
    """Floats or arrays; 1 = gap error negligible next to the spread, 0 = no better than it."""
    sigma_gap = np.sqrt(np.square(sigma_goal) + np.square(sigma_current))
    return np.clip(1 - sigma_gap / sigma_prior, 0, 1)
