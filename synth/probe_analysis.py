"""What do the weight and muscle sliders do to the mesh? (ADR-053, before the label mapping.)

    uv run python synth/probe_analysis.py synth/out/probe/probe_measures.csv

Per sex: the measured landmark widths, depths and volume at fixed muscle (varying weight) and at fixed weight
(varying muscle), and a least-squares slope per unit of each slider over the 25 probe points. Widths and depths are
torso bands with the arms excluded; the band heights are approximate landmarks, so read the changes, not the
absolute sizes. Nothing here defines a label: it is the evidence the adiposity and definition-cap mapping is built
from, and it is pasted into docs/logs/dev-log.md.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

MEASURES = [
    "volume_l",
    "waist_w_cm",
    "waist_d_cm",
    "chest_w_cm",
    "chest_d_cm",
    "hip_w_cm",
    "hip_d_cm",
    "thigh_w_cm",
]

ap = argparse.ArgumentParser()
ap.add_argument("csv_path", type=Path)
args = ap.parse_args()

with args.csv_path.open(newline="") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    for k in MEASURES + ["muscle", "weight"]:
        r[k] = float(r[k])
    r["waist_over_chest"] = r["waist_w_cm"] / r["chest_w_cm"]
cols = [*MEASURES, "waist_over_chest"]


def at(sex, **fixed):
    return sorted(
        (r for r in rows if r["sex"] == sex and all(r[k] == v for k, v in fixed.items())),
        key=lambda r: (r["muscle"], r["weight"]),
    )


for sex in ("female", "male"):
    mine = [r for r in rows if r["sex"] == sex]
    print(f"\n== {sex} ==")
    for title, fixed, vary in (("weight 0..1 at muscle 0.5", {"muscle": 0.5}, "weight"),
                               ("muscle 0..1 at weight 0.5", {"weight": 0.5}, "muscle")):  # fmt: skip
        print(f"-- {title}: " + " | ".join(f"{vary}={r[vary]:.2f}" for r in at(sex, **fixed)))
        for c in cols:
            vals = [r[c] for r in at(sex, **fixed)]
            print(f"   {c:<17}" + " ".join(f"{v:8.2f}" for v in vals) + f"   change {vals[-1] - vals[0]:+.2f}")
    x = np.array([[1.0, r["muscle"], r["weight"]] for r in mine])
    print("-- least-squares slope per unit slider over the 25 points (muscle, weight):")
    for c in cols:
        coef, *_ = np.linalg.lstsq(x, np.array([r[c] for r in mine]), rcond=None)
        print(f"   {c:<17} muscle {coef[1]:+8.3f}   weight {coef[2]:+8.3f}")
