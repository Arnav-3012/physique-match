"""Own-scan ordering check (P3 gate): one browser capture as the goal, another as the current scan.
It only asks whether the ranking of regions looks sensible; it is not a metric. Run:
  PYTHONPATH=services/api uv run python -m ml.src.sanity_ordering \
      --goal ml/data/parity/<capture> --current ml/data/parity/<capture> --height-cm 175
Add --goal-front-only to drop the goal's side view. Masks are read from the capture folders
(gitignored); keypoints are not used by the analyzers, so they are filled with zeros.
"""

import argparse
import base64
from pathlib import Path

from app.analyzer.geometric import GeometricAnalyzer
from app.analyzer.siamese_subtract import SiameseSubtractAnalyzer
from physique_schemas.models import N_KEYPOINTS, Keypoint, Scan, ViewData

KEYPOINTS = [Keypoint(x=0, y=0, z=0, visibility=0)] * N_KEYPOINTS


def load_scan(folder: Path, views: tuple[str, ...] = ("front", "left", "back", "right")) -> Scan:
    found = {v: folder / f"{v}.png" for v in views if (folder / f"{v}.png").exists()}
    return Scan(
        views={
            v: ViewData(mask_png_b64=base64.b64encode(p.read_bytes()).decode(), keypoints=KEYPOINTS)
            for v, p in found.items()
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", type=Path, required=True)
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--height-cm", type=float, required=True)
    ap.add_argument("--goal-front-only", action="store_true")
    args = ap.parse_args()

    goal = load_scan(args.goal, ("front",) if args.goal_front_only else ("front", "left", "back"))
    current = load_scan(args.current)
    shipped = SiameseSubtractAnalyzer().compare(goal, current)
    geometric = GeometricAnalyzer().compare(goal, current)

    print(f"goal {args.goal.name} ({sorted(goal.views)}), current {args.current.name}, "
          f"front-only goal: {shipped.goal_front_only}")  # fmt: skip
    print(f"{'region':10s} {'delta':>7s} {'cm':>6s} {'reliab':>6s} {'goal':>6s} {'current':>7s} "
          f"{'geom':>7s}   (cm = delta x your height; "
          "goal, current = measurement / height)")  # fmt: skip
    ranked = sorted(shipped.regions.items(), key=lambda kv: -kv[1].delta)
    for region, r in ranked:
        if not r.measured:
            print(f"{region:10s} not measured (no view of it on both scans)")
            continue
        goal_ratio = f"{r.goal_ratio:.3f}" if r.goal_ratio is not None else "-"
        current_ratio = f"{r.current_ratio:.3f}" if r.current_ratio is not None else "-"
        print(f"{region:10s} {r.delta:7.3f} {r.delta * args.height_cm:6.1f} {r.confidence:6.2f} "
              f"{goal_ratio:>6s} {current_ratio:>7s} "
              f"{geometric.regions[region].delta:7.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
