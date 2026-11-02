"""Class-exemplar sheets: per sex, one row per head, one column per class 1-5 (ADR-053 Amendment 3).

    uv run python synth/exemplar_sheet.py synth/out/exemplars --rig key_left

Reads synth/plan/exemplars.json and the renders written by
`render.py --ids synth/plan/exemplar_ids.txt --rig-override key_left --sets bump --out synth/out/exemplars`.
Front view for every head but `def_back`, which uses the back view. If adjacent classes look identical in a row,
that is reported to the user before the full render.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

EXEMPLARS = Path(__file__).parent / "plan" / "exemplars.json"

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path)
ap.add_argument("--rig", default="key_left")
args = ap.parse_args()

selection = json.loads(EXEMPLARS.read_text())
for sex, heads in selection.items():
    first = next(e for h in heads.values() for e in h.values() if e)
    w, h = Image.open(args.folder / "images" / f"{first['id']}_bump_{args.rig}_front.png").size
    sheet = Image.new("RGB", (5 * w, len(heads) * h), (128, 128, 128))
    draw = ImageDraw.Draw(sheet)
    for row, (head, classes) in enumerate(heads.items()):
        view = "back" if head == "def_back" else "front"
        for c in range(1, 6):
            e = classes[str(c)]
            x, y = (c - 1) * w, row * h
            if e is None:
                draw.text((x + 4, y + 4), f"{head} class {c}: EMPTY", fill=(200, 30, 30))
                continue
            tile = Image.open(args.folder / "images" / f"{e['id']}_bump_{args.rig}_{view}.png").convert("RGBA")
            sheet.paste(tile, (x, y), tile)
            label = (
                f"{sex} {head} c{c} {e['id']} v={e['value']:.2f} m={e['muscle_slider']:.2f} w={e['weight_slider']:.2f}"
            )
            draw.text((x + 4, y + 4), label, fill=(200, 30, 30))
    out = args.folder / f"exemplar_sheet_{sex}.jpg"
    sheet.save(out, quality=90)
    print(f"{sex}: {len(heads)} heads x 5 classes -> {out}")
