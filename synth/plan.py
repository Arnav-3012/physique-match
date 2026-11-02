"""The planned identity sequence, the split and the lighting slice (ADR-045). Plain Python + scipy, no Blender.

    uv run python -m synth.plan            # writes synth/plan/identities.csv and prints its SHA-256

The whole planned sequence (N_PLANNED identities) is drawn here before any render, so the split, the label
edges (synth/coverage.py) and the lighting slice are fixed by the sliders alone. Rendering walks the file in
order and stops at a whole round of 18 cells, so every rendered prefix is balanced. Positions m (muscle) and
f (weight) are in [0, 1] inside the clamped slider range. Since ADR-053 the clamps and the fitted adiposity
lookup (synth/plan/fit.json, from synth/fit_labels.py) are inputs: each row carries the slider values, the
fitted adiposity proxy `adip` and the fit's hash, so labels are regenerable from params.csv alone.
"""

import argparse
import csv
import hashlib
import warnings
from pathlib import Path

from scipy.stats import norm, qmc

from synth import fit as fitmod
from synth import latents

SEED = 20261102
N_CELLS = 18  # 2 sexes x 3 muscle x 3 fat, round-robin: identity i is in cell i % 18
N_PLANNED = 3996  # 4000 rounded down to whole rounds (ADR-047 Amendment 1 item A)
N_MIN = 600  # ADR-047 item 8; >= 100 held out
SEXES = (
    "female",
    "male",
)  # gender 0 / 1 in MPFB; consecutive cells alternate sex, so any prefix is near-balanced
JITTERED = (
    "chest",
    "shoulders",
    "arms",
    "back",
    "legs",
)  # abs has no muscle target: m_abs = m (ADR-049 item 1)
DIMS = (  # Sobol dimensions per identity, in this order (synth/grid.yaml)
    "height", "proportions", "skin_tone", "muscle_in_cell", "fat_in_cell", "d",
    *(f"j_{r}" for r in JITTERED),
    *(f"z_{r}" for r in latents.REGIONS),
    "bump_seed",
)  # fmt: skip

# Rig roles, pre-registered (ADR-050 Amendment 1): both are TRAINING rigs, the lighting slice uses another one.
TRAIN_RIGS = ("key_left", "key_right")
SLICE_RIG = "top"
SLICE_N = 50  # the first 50 held-out identities in sequence order get a second render under SLICE_RIG

# Split (ADR-045): inside each cell, consecutive blocks of 10 identities hold exactly 7 train / 1 val / 2 test,
# chosen by a SHA-256 hash of the identity id. Stratified by cell, and balanced at every whole round >= 10.
BLOCK = ("train",) * 7 + ("val",) + ("test",) * 2
SPLIT_SALT = "physique-p4-split-v1"

# Advisory tag per (muscle level, fat level) cell, ADR-047 Amendment 1 item C; overlaps allowed, not a label.
ARCHETYPE = {
    (0, 0): "skinny", (1, 0): "lean", (2, 0): "shredded_muscular",
    (0, 1): "unnamed", (1, 1): "average", (2, 1): "athletic_very_muscular",
    (0, 2): "soft_overweight", (1, 2): "unnamed", (2, 2): "very_muscular",
}  # fmt: skip
# "bulky" (high muscle + a soft waist) is UNSUPPORTED inside the clamps: the generator cannot make it (ADR-053
# Amendment 2). The tag was renamed; no label or split depends on it. Keep the word out of the UI and README.

PLAN_PATH = Path(__file__).parent / "plan" / "identities.csv"


def _clip(u: float) -> float:
    return min(max(u, 1e-9), 1 - 1e-9)


def _normal(u: float) -> float:
    return float(norm.ppf(_clip(u)))


def _cell(i: int) -> tuple[int, str, int, int]:
    c = i % N_CELLS
    g = c // 2  # 0..8 inside the sex
    return c, SEXES[c % 2], g // 3, g % 3


def _row(i: int, u: dict[str, float], fit: fitmod.Fit) -> dict:
    c, sex, mi, fi = _cell(i)
    m, f, d = (mi + u["muscle_in_cell"]) / 3, (fi + u["fat_in_cell"]) / 3, u["d"]
    clamp = fit.data["clamps"][sex]
    muscle_slider = clamp["muscle"][0] + m * (clamp["muscle"][1] - clamp["muscle"][0])
    weight_slider = clamp["weight"][0] + f * (clamp["weight"][1] - clamp["weight"][0])
    adip = fit.adiposity(sex, muscle_slider, weight_slider)  # fitted girth proxy, ADR-053
    m_r = {"abs": m} | {r: latents.regional_muscle(m, latents.JITTER_SD * _normal(u[f"j_{r}"])) for r in JITTERED}
    row = {
        "index": i, "id": f"id{i:05d}", "cell": c, "sex": sex, "gender": float(c % 2),
        "muscle_level": mi, "fat_level": fi, "round": i // N_CELLS, "archetype": ARCHETYPE[mi, fi],
        "rig": TRAIN_RIGS[(i // N_CELLS) % 2],
        "height": u["height"], "proportions": u["proportions"], "skin_u": u["skin_tone"],
        "m": m, "f": f, "d": d, "bump_seed": int(_clip(u["bump_seed"]) * 2**31),
        "muscle_slider": muscle_slider, "weight_slider": weight_slider, "adip": adip,
        "fit_sha256": fit.data.get("sha256", ""),
    }  # fmt: skip
    for r in latents.REGIONS:
        d_eff = latents.d_eff(d, adip, m_r[r])
        row[f"m_{r}"] = m_r[r]
        row[f"d_eff_{r}"] = d_eff
        row[f"bump_{r}"] = latents.bump_strength(d_eff, _normal(u[f"z_{r}"]))
        row[f"flat_{r}"] = latents.flat_score(m_r[r], adip)
    return row


def _hash(identity: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{identity}".encode()).hexdigest()


def _assign_splits(rows: list[dict]) -> None:
    for c in range(N_CELLS):
        cell_rows = [r for r in rows if r["cell"] == c]
        for b in range(0, len(cell_rows), len(BLOCK)):
            block = sorted(cell_rows[b : b + len(BLOCK)], key=lambda r: _hash(r["id"]))
            for rank, r in enumerate(block):
                r["split"] = BLOCK[rank]
    first_tests = [r for r in rows if r["split"] == "test"][:SLICE_N]
    for r in rows:
        r["slice_top"] = 0
    for r in first_tests:
        r["slice_top"] = 1


def build_plan(n: int = N_PLANNED, fit: fitmod.Fit | None = None) -> list[dict]:
    """The first n identities of the planned sequence (n = N_PLANNED for the real file). `fit` defaults to
    synth/plan/fit.json; a test passes its own."""
    fit = fit or fitmod.load()
    per_cell = [list(range(c, n, N_CELLS)) for c in range(N_CELLS)]
    rows: dict[int, dict] = {}
    for c, indices in enumerate(per_cell):
        engine = qmc.Sobol(d=len(DIMS), scramble=True, seed=SEED + c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # n per cell is not a power of 2; balance is approximate between
            points = engine.random(len(indices))
        for i, p in zip(indices, points, strict=True):
            rows[i] = _row(i, dict(zip(DIMS, map(float, p), strict=True)), fit)
    # 6 decimals: the CSV is the contract, so the edges and labels see exactly the values render.py reads
    ordered = [{k: round(v, 6) if isinstance(v, float) else v for k, v in rows[i].items()} for i in range(n)]
    _assign_splits(ordered)
    return ordered


def write_plan(path: Path = PLAN_PATH) -> str:
    rows = build_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PLAN_PATH)
    args = ap.parse_args()
    print(f"{N_PLANNED} identities -> {args.out}\nsha256 {write_plan(args.out)}")
