"""Post-render coverage report from params.csv and the frozen edges.json (ADR-047 item E, ADR-053). Read-only.

    uv run python -m synth.post_render synth/out/render/params.csv

Same checks as `synth.coverage`, but class shares use the edges fixed before the render (never recomputed),
plus counts per sex x muscle x fat cell per split and the render timings. Paste the output into dev-log.md.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from synth import coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("params", type=Path)
    ap.add_argument("--edges", type=Path, default=coverage.EDGES_PATH)
    args = ap.parse_args()
    edges = {h: tuple(e) for h, e in json.loads(args.edges.read_text()).items()}
    with args.params.open(newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = [{k: _num(v) for k, v in r.items()} for r in raw]

    lines, failures = coverage.check(rows, edges)
    print("\n".join(lines))

    print("\nidentities per sex x muscle x fat cell (train/val/test):")
    cells = Counter((r["sex"], r["muscle_level"], r["fat_level"], r["split"]) for r in rows)
    for sex in sorted({r["sex"] for r in rows}):
        for m in range(3):
            row = []
            for f in range(3):
                row.append("/".join(str(cells[(sex, m, f, s)]) for s in ("train", "val", "test")))
            print(f"  {sex:<7} muscle {m}: fat 0,1,2 = " + "  ".join(row))

    wall = [r["wall_s"] for r in rows]
    print(
        f"\nper identity wall_s: median {statistics.median(wall):.2f}, max {max(wall):.2f}, "
        f"total {sum(wall) / 3600:.2f} h"
    )
    print(f"devices: {dict(Counter(r['device'] for r in rows))}")
    print(f"fit hashes: {sorted({r['fit_sha256'] for r in rows})}")
    print(f"slice rendered: {sum(r['slice_rendered'] for r in rows)} of {sum(r['slice_top'] for r in rows)} planned")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nall coverage checks passed")
    return 0


def _num(value: str):
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    sys.exit(main())
