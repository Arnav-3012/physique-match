"""Tile PNGs (RGBA, transparent background) on grey into one captioned sheet to look at.

    uv run python synth/contact_sheet.py synth/out/spike/metal
    uv run python synth/contact_sheet.py synth/out/spike/grid --pattern "grid_female_*.png" \\
        --cols 3 --name grid_female.jpg
    uv run python synth/contact_sheet.py synth/out/spike/grid --pattern "grid_male_*.png" --cols 3 --name grid_male.jpg

Files are tiled in name order, so grid_<sex>_m<muscle>_f<fat> reads as rows = muscle (low to high, top to
bottom) and columns = fat (low to high, left to right) with --cols 3.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--pattern", default="*.png")
ap.add_argument("--cols", type=int, default=6)
ap.add_argument("--name", default="contact_sheet.jpg")
args = ap.parse_args()

files = sorted(f for f in args.folder.glob(args.pattern) if f.suffix == ".png")
if not files:
    raise SystemExit(f"no PNGs matching {args.pattern} in {args.folder}")
w, h = Image.open(files[0]).size
rows = -(-len(files) // args.cols)
sheet = Image.new("RGB", (args.cols * w, rows * h), (128, 128, 128))
draw = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    x, y = (i % args.cols) * w, (i // args.cols) * h
    tile = Image.open(f).convert("RGBA")
    sheet.paste(tile, (x, y), tile)
    draw.text((x + 4, y + 4), f.stem.removeprefix("grid_"), fill=(200, 30, 30))
out = args.folder / args.name
sheet.save(out, quality=88)
print(f"{len(files)} tiles -> {out}")
