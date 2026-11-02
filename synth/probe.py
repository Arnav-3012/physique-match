"""Probe sheet before the clamps: muscle x weight on a 5 x 5 grid over the full 0-1 range, per sex, front and back,
default height and proportions, one rig, flat skin. Probe points only, NOT clamps: the user looks at the sheets and
picks the clamps by rules written before looking (docs/decisions.md, ADR-053). Also measures, per body, the torso
widths and depths and the mesh volume, so the effect of the weight and muscle sliders on waist, chest and mass is
measured from the mesh (synth/probe_analysis.py) before the label mapping, not assumed from the slider names.

    /Applications/Blender.app/Contents/MacOS/Blender -b -P synth/probe.py -- --cycles-device METAL
    uv run python synth/contact_sheet.py synth/out/probe --pattern "probe_female_*_front.png" --cols 5 --name sheet_female_front.jpg
    (and _back, and male: four sheets; rows = muscle low to high, columns = weight low to high)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

HERE = Path(globals().get("__file__", "synth/probe.py")).resolve().parent
sys.path.insert(0, str(HERE))  # Blender does not put the script's folder on sys.path
import bump  # noqa: E402
import rig  # noqa: E402

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
VIEWS = {"front": 0.0, "back": 3.141592653589793}
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("synth/out/probe"))
ap.add_argument("--size", type=int, default=320, help="render height in px; width = 3/4 of it")
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--cycles-device", default="METAL")
ap.add_argument("--rig", default="key_left")
args, _ = ap.parse_known_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])


def say(key, value):
    print(f"PROBE_{key} {json.dumps(value, default=str)}", flush=True)


def volume_litres(obj, world):
    """Signed-volume sum over the evaluated triangles, in litres. Relative values only: the MPFB mesh may carry
    helper geometry and need not be watertight. NaN if the triangle call is unavailable."""
    try:
        mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        mesh.calc_loop_triangles()
        tri = np.empty((len(mesh.loop_triangles), 3), np.int32)
        mesh.loop_triangles.foreach_get("vertices", tri.ravel())
        a, b, c = (world[tri[:, k]] for k in range(3))
        return round(float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6 * 1000), 2)
    except Exception:  # noqa: BLE001 - the probe reports NaN rather than stopping
        return float("nan")


def main():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    svc, _, mpfb_version = rig.load_mpfb()
    default = svc["TargetService"].get_default_macro_info_dict()
    args.out.mkdir(parents=True, exist_ok=True)
    rig.setup_camera()
    rig.configure_render(args.size, args.samples)
    device = rig.cycles_device(args.cycles_device.upper())
    say("DEVICE", device)
    if not device["ok"]:
        sys.exit("abort: the requested device is not enabled")
    lights = rig.add_rig(args.rig)
    tone = rig.skin_tone(0.3)
    rows, t0 = [], time.perf_counter()
    for sex, gender in (("female", 0.0), ("male", 1.0)):  # MakeHuman: gender 0 = female, 1 = male
        for mi, muscle in enumerate(LEVELS):
            for wi, weight in enumerate(LEVELS):
                macros = {
                    **default,
                    "gender": gender,
                    "muscle": muscle,
                    "weight": weight,
                }  # height, proportions: default
                body = rig.build_body(svc, macros, {r: muscle for r in rig.REGIONAL})
                co = rig.mesh_co(body)
                world = rig.world_coords(body, co)
                mat, _ = bump.skin_material(co, None, 0.0, tone)
                bump.assign(body, mat)
                for view, angle in VIEWS.items():
                    body.rotation_euler.z = angle
                    rig.scene.render.filepath = str(args.out / f"probe_{sex}_m{mi}_w{wi}_{view}.png")
                    bpy.ops.render.render(write_still=True)
                rows.append({"sex": sex, "muscle": muscle, "weight": weight, "volume_l": volume_litres(body, world),
                             **rig.torso_measures(world), **rig.width_profile(body, co)})  # fmt: skip
                bpy.data.materials.remove(mat)
                rig.remove([body])
    rig.remove(lights)
    with (args.out / "probe_measures.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    say("DONE", {"out": str(args.out), "bodies": len(rows), "images": 2 * len(rows), "mpfb": mpfb_version,
                 "blender": bpy.app.version_string, "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip


main()
