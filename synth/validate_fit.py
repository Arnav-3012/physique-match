"""Fitted vs measured waist girth on the rendered identities (ADR-053). Validation only: never relabels anything.

    uv run python -m synth.validate_fit synth/out/render/params.csv

The fit (synth/plan/fit.json) was made on the probe grid at default height and proportions; the rendered
identities also vary height, proportions and the regional muscle jitter, so part of the error is that variation.
A large error goes into the model card as a known limit of the adiposity label.
"""

import argparse
import csv
import statistics
from pathlib import Path

from synth import fit as fitmod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    args = ap.parse_args()
    f = fitmod.load()
    with args.params.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for sex in ("female", "male"):
        errs, pairs = [], []
        for r in rows:
            if r["sex"] != sex:
                continue
            fitted = f.girth(sex, float(r["muscle_slider"]), float(r["weight_slider"]))
            measured = fitmod.girth_cm(float(r["waist_w_cm"]), float(r["waist_d_cm"]))
            errs.append(measured - fitted)
            pairs.append((fitted, measured))
        if not errs:
            continue
        corr = statistics.correlation(*zip(*pairs, strict=True)) if len(errs) > 2 else float("nan")
        mae = statistics.mean(abs(e) for e in errs)
        print(f"{sex}: n {len(errs)}  bias {statistics.mean(errs):+.2f} cm  MAE {mae:.2f} cm  "
              f"max |err| {max(abs(e) for e in errs):.2f} cm  correlation {corr:.3f}")  # fmt: skip
    print(f"fit sha256 {f.data['sha256']}; params.csv rows carry {sorted({r['fit_sha256'] for r in rows})}")


if __name__ == "__main__":
    main()
