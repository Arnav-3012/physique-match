"""Parity: browser masks vs BodyM through the same preprocess (docs/phase-plan.md P2, gate 6).
Run: PYTHONPATH=services/api uv run python -m ml.src.parity

Masks: ml/data/parity/<device>_<clothes>_<distance>/{front,left,back,right}.png (browser output).
Reference: BodyM TRAIN only. |z| < 2 on a stat counts as comparable.
"""

import numpy as np
import pandas as pd
from PIL import Image

from app.analyzer.rows import find_rows
from ml.src.data.bodym import ROOT, clipped, load_mask
from ml.src.data.split import load_split_table
from ml.src.preprocess import H, body_px, from_bodym, from_browser

PARITY = ROOT.parent / "parity"
N_REF = 300
SEED = 0
GRID = np.linspace(0.20, 0.95, 60)  # body-height fractions below the crown, for the width profile
ROW_NAMES = ("armpit", "waist", "hip", "crotch")


def stats(mask: np.ndarray, view: str) -> dict:
    height = body_px(mask)
    crown = (H - height) / 2
    cols = np.flatnonzero(mask.any(axis=0))
    out = {
        "cut_u": (np.flatnonzero(mask.any(axis=1))[0] - crown) / height,
        "area": mask.mean(),
        "bbox_w": (cols[-1] - cols[0] + 1) / height,
        "profile": np.interp(GRID, (np.arange(H) - crown) / height, mask.sum(axis=1) / height),
    }
    if view in ("front", "back"):
        rows = find_rows(mask)
        out |= {n: (getattr(rows, n) - rows.crown) / height for n in ROW_NAMES}
        out |= {"arms_separated": rows.arms_separated, "crotch_found": rows.crotch_found}
    return out


def reference(view_col: str, kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_split_table("train").drop_duplicates("subject_id").sample(N_REF, random_state=SEED)
    recs, profiles = [], []
    for p in df[view_col]:
        raw = load_mask(p)
        if kind == "side" and clipped(raw):
            continue
        s = stats(from_bodym(raw), kind)
        profiles.append(s.pop("profile"))
        recs.append(s)
    return pd.DataFrame(recs), np.mean(profiles, axis=0)


def main() -> None:
    refs = {"front": reference("front_path", "front"), "side": reference("side_path", "side")}
    for k, (r, _) in refs.items():
        print(f"BodyM {k} reference: n={len(r)}, mean/std")
        print(r[["cut_u", "area", "bbox_w"]].agg(["mean", "std"]).round(3).to_string())

    records, tiles = [], []
    for d in sorted(p for p in PARITY.iterdir() if p.is_dir()):  # one folder per capture
        row_tiles = []
        for view in ("front", "left", "back", "right"):
            f = d / f"{view}.png"
            if not f.exists():
                continue
            mask = from_browser(np.asarray(Image.open(f).convert("L")) > 127)
            kind = "side" if view in ("left", "right") else "front"
            ref, ref_profile = refs[kind]
            s = stats(mask, kind)
            z = {k: (s[k] - ref[k].mean()) / ref[k].std() for k in ("cut_u", "area", "bbox_w")}
            records.append(
                {"capture": d.name, "view": view}
                | {k: round(s[k], 3) for k in ("cut_u", "area", "bbox_w")}
                | {f"z_{k}": round(v, 1) for k, v in z.items()}
                | {"profile_r": round(np.corrcoef(s["profile"], ref_profile)[0, 1], 3)}
                | {k: s[k] for k in s if k in ROW_NAMES or k in ("arms_separated", "crotch_found")}
            )
            row_tiles.append(mask.astype(np.uint8) * 255)
        tiles.append(np.hstack(row_tiles))

    out = pd.DataFrame(records)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n== browser masks vs BodyM ==")
    print(out.round(3).to_string(index=False))
    fb = out[out.view.isin(["front", "back"])]
    print(f"\narms_separated rate (front/back): {fb.arms_separated.mean():.2f} of {len(fb)}")
    print(f"crotch_found rate (front/back):   {fb.crotch_found.mean():.2f} of {len(fb)}")
    off = (out.filter(like="z_").abs() >= 2).any(axis=1)
    print(f"masks with any |z| >= 2: {off.sum()}/{len(out)}")

    first = load_split_table("train").iloc[0]
    ref_masks = [from_bodym(load_mask(p)) for p in (first.front_path, first.side_path)]
    ref_row = np.hstack(ref_masks).astype(np.uint8) * 255
    width = max(t.shape[1] for t in [ref_row, *tiles])
    pad = [np.pad(t, ((0, 0), (0, width - t.shape[1]))) for t in [ref_row, *tiles]]
    sheet = Image.fromarray(np.vstack(pad))
    path = ROOT.parent / "preview" / "parity.png"
    sheet.resize((sheet.width // 2, sheet.height // 2)).save(path)
    print("sheet (row 1 = BodyM, then one row per capture) ->", path)


if __name__ == "__main__":
    main()
