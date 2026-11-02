"""Resolution check before the big render (ADR-050 item 3): do the abs cues survive the model's input size?

Takes the bump contact-sheet tiles and shows each one three ways, side by side, all enlarged to the original
size WITHOUT smoothing so the pixels the model would see are visible: the 384 px original, the same tile at
256 px height, and at 224 px height (bilinear, as the browser's canvas resamples).

    uv run python synth/downsample_check.py synth/out/spike/bump2

Rule written in ADR-050: if the abs rows are not readable as at least three separate horizontal grooves at
256 px on the high-d, low-fat tiles, STOP and tell the user before rendering.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

TILES = ["bump_male_d2_f0_front", "bump_male_d1_f0_front", "bump_female_d2_f0_front"]
HEIGHTS = (384, 256, 224)

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
args = ap.parse_args()

rows = []
for stem in TILES:
    src = Image.open(args.folder / f"{stem}.png").convert("RGBA")
    flat = Image.new("RGBA", src.size, (128, 128, 128, 255))
    flat.alpha_composite(src)
    base = flat.convert("RGB")
    w, h = base.size
    cells = []
    for height in HEIGHTS:
        small = base.resize((round(w * height / h), height), Image.Resampling.BILINEAR)
        big = small.resize((w, h), Image.Resampling.NEAREST)  # blow the pixels up, no smoothing
        ImageDraw.Draw(big).text((4, 4), f"{stem.removeprefix('bump_')} @ {height}px", fill=(200, 30, 30))
        cells.append(big)
    rows.append(cells)

w, h = rows[0][0].size
sheet = Image.new("RGB", (len(HEIGHTS) * w, len(rows) * h))
for r, cells in enumerate(rows):
    for c, cell in enumerate(cells):
        sheet.paste(cell, (c * w, r * h))
out = args.folder / "resolution_check.jpg"
sheet.save(out, quality=92)
print(f"{len(rows)} tiles x {len(HEIGHTS)} sizes -> {out}")
