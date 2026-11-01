"""Draw where the widths and depths are measured (BodyM train only, no test data).
Run: PYTHONPATH=services/api uv run python -m ml.src.show_measurements

Each measurement row is a thin line; each dot is an edge of a white run on that row, and the
distance between two dots is the width (front) or depth (side) before dividing by body height.
Front rows: shoulders, chest, waist, hip, upper arm, forearm, thigh, calf. Side: the same
body fractions for chest, waist, hip, thigh, calf. Output: ml/data/preview/measurements.png
"""

import numpy as np
from PIL import Image, ImageDraw

from app.analyzer.features import GIRTH_REGIONS, front_widths, side_row
from app.analyzer.rows import row_runs
from ml.src.data.bodym import ROOT, clipped, load_mask
from ml.src.data.split import load_split_table
from ml.src.preprocess import body_px, from_bodym

N = 4
COLOURS = {
    "shoulder": (255, 80, 80), "chest": (255, 160, 0), "waist": (80, 220, 80),
    "hip": (80, 140, 255), "upper_arm": (220, 80, 255), "forearm": (80, 230, 230),
    "thigh": (255, 230, 0), "calf": (255, 120, 180),
}  # fmt: skip


def draw(mask: np.ndarray, rows: dict[str, int]) -> Image.Image:
    img = Image.fromarray(np.repeat(mask[..., None] * 90, 3, axis=2).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for name, row in rows.items():
        colour = COLOURS[name]
        d.line([(0, row), (mask.shape[1], row)], fill=colour, width=1)
        for a, b in row_runs(mask[row]):
            for x in (a, b):
                d.ellipse([x - 3, row - 3, x + 3, row + 3], fill=colour)
    return img


def main() -> None:
    df = load_split_table("train").drop_duplicates("subject_id").sample(N * 2, random_state=1)
    tiles = []
    for _, r in df.iterrows():
        side_raw = load_mask(r.side_path)
        if clipped(side_raw) or len(tiles) >= N:
            continue
        front, side = from_bodym(load_mask(r.front_path)), from_bodym(side_raw)
        _, rows, where = front_widths(front)
        s_rows = {n: side_row(where[n], rows.crown, body_px(front), side) for n in GIRTH_REGIONS}
        tiles.append((draw(front, where), draw(side, {n: s_rows[n] for n in GIRTH_REGIONS})))
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (w * 2 * 2, h * (len(tiles) + 1) // 2))
    for i, (f, s) in enumerate(tiles):
        sheet.paste(f, ((i % 2) * w * 2, (i // 2) * h))
        sheet.paste(s, ((i % 2) * w * 2 + w, (i // 2) * h))
    path = ROOT.parent / "preview" / "measurements.png"
    sheet.save(path)
    print("->", path, "| colours:", ", ".join(COLOURS))


if __name__ == "__main__":
    main()
