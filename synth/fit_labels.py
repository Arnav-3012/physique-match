"""Build the girth lookup from the probe grid, run the pre-registered checks, write synth/plan/fit.json (ADR-053).

    uv run python -m synth.fit_labels synth/out/probe/probe_measures.csv

Fitted on the 5 x 5 muscle x weight probe grid only, never on training identities. The checks are written in
ADR-053 before this runs; any failure exits non-zero and writes nothing, and the form is changed before the render.
The clamps come from synth/clamps.json (frozen by the user in the ADR before render.py starts).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from synth import fit, latents

CLAMPS_PATH = Path(__file__).parent / "clamps.json"
TOL_CM = 0.05  # probe measurements are rounded to 0.01 cm; allow that much non-monotone noise
SEXES = ("female", "male")


def read_probe(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for sex in SEXES:
        mine = sorted(
            (r for r in rows if r["sex"] == sex),
            key=lambda r: (float(r["muscle"]), float(r["weight"])),
        )
        if len(mine) != len(fit.LEVELS) ** 2:
            raise SystemExit(f"{sex}: {len(mine)} probe rows, expected {len(fit.LEVELS) ** 2}")
        out[sex] = mine
    return out


def grid(rows: list[dict], key) -> list[list[float]]:
    n = len(fit.LEVELS)
    return [[key(rows[i * n + j]) for j in range(n)] for i in range(n)]


def nondecreasing(xs: list[float], tol: float = 0.0) -> bool:
    return all(b >= a - tol for a, b in zip(xs, xs[1:], strict=False))


def run_checks(probe: dict, data: dict) -> tuple[dict, list[str]]:
    """The pre-registered checks. Returns (report, failures)."""
    f = fit.Fit(data)
    report, failures = {}, []
    for sex in SEXES:
        c = data["clamps"][sex]
        if not all(0.0 <= lo < hi <= 1.0 for lo, hi in (c["muscle"], c["weight"])):
            failures.append(f"{sex}: clamps must satisfy 0 <= lo < hi <= 1, got {c}")
        table = data["girth"][sex]
        c1 = all(nondecreasing(row, TOL_CM) for row in table)  # girth rises with weight at every probe muscle
        ms, ws = fit.lattice(c["muscle"]), fit.lattice(c["weight"])
        adip = [[f.adiposity(sex, m, w) for w in ws] for m in ms]
        c2 = all(nondecreasing(row, 1e-9) for row in adip) and abs(max(map(max, adip)) - 1.0) < 1e-9
        c3 = all(nondecreasing([-latents.hide(a) for a in row], 1e-12) for row in adip)  # hide falls as weight rises
        caps = [latents.muscle_cap(m) for m in ms]
        c4 = nondecreasing(caps, 1e-12)  # the cap rises with muscle
        thigh = grid(probe[sex], lambda r: float(r["thigh_w_cm"]))
        chest = grid(probe[sex], lambda r: float(r["chest_d_cm"]))
        by_weight = lambda t, j: [t[i][j] for i in range(len(fit.LEVELS))]  # noqa: E731
        candidate = all(nondecreasing(by_weight(t, j), TOL_CM) for t in (thigh, chest) for j in range(len(fit.LEVELS)))
        report[sex] = {
            "C1_girth_monotone_in_weight_at_each_probe_muscle": c1,
            "C2_adiposity_monotone_in_weight_and_spans_0_1": c2,
            "C3_hide_monotone_down_in_adiposity": c3,
            "C4_cap_monotone_up_in_muscle": c4,
            "cap_calibrated_to_measured_muscle_effect_would_be_monotone": candidate,
        }
        for name in list(report[sex])[:4]:
            if not report[sex][name]:
                failures.append(f"{sex}: {name} failed")
    return report, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_csv", type=Path)
    ap.add_argument("--clamps", type=Path, default=CLAMPS_PATH)
    ap.add_argument("--out", type=Path, default=fit.FIT_PATH)
    ap.add_argument("--force", action="store_true", help="overwrite an existing fit.json (needs a new ADR)")
    args = ap.parse_args()
    probe = read_probe(args.probe_csv)
    clamps = {s: v for s, v in json.loads(args.clamps.read_text()).items() if s in SEXES}
    girth = {
        s: grid(
            probe[s],
            lambda r: round(fit.girth_cm(float(r["waist_w_cm"]), float(r["waist_d_cm"])), 3),
        )
        for s in SEXES
    }
    data = fit.build(girth, clamps)
    report, failures = run_checks(probe, data)
    for sex in SEXES:
        print(f"== {sex}: waist girth (cm), rows muscle 0..1, columns weight 0..1")
        for row in girth[sex]:
            print("   " + " ".join(f"{v:7.2f}" for v in row))
        print(f"   clamps {clamps[sex]}")
        print(f"   a_ref (largest girth added by weight inside the clamps) {data['a_ref'][sex]:.2f} cm")
        for name, ok in report[sex].items():
            print(f"   {'PASS' if ok else 'FAIL'}  {name}")
    if failures:
        print("\nFAILED (change the form before the render; nothing written):\n  " + "\n  ".join(failures))
        return 1
    data["checks"] = report
    data["cap_form"] = "position: cap = CAP_BASE + (1 - CAP_BASE) * regional muscle position (latents.muscle_cap)"
    data["sha256"] = fit.coefficient_hash(data)
    print(f"\ncoefficient sha256 {data['sha256']}")
    if args.out.exists() and not args.force:
        print(f"{args.out} exists: not overwritten (use --force with a new ADR)")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
