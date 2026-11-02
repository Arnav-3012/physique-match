"""Batch render of the planned identities (ADR-045, ADR-047 Amendment 1, ADR-049, ADR-050). Runs inside Blender:

    /Applications/Blender.app/Contents/MacOS/Blender -b -P synth/render.py -- \\
        --cycles-device METAL --seq-start 0 --count 216

Reads synth/plan/identities.csv (written by `uv run python -m synth.plan`; log its SHA-256 and the frozen clamps and
fit hash in docs/logs/dev-log.md BEFORE the first batch). The clamps are applied in plan.py (ADR-053): each plan
row holds the slider values muscle_slider and weight_slider that this script uses. One process = one batch of whole 18-cell rounds (>= 200 identities:
216 = 12 rounds); run the next batch with --seq-start 216, 432, ... The wall-clock budget is kept across batches in
<out>/budget.json. Per identity the body is built once; the bump set and the flat set (same body, same seeds, bump
strength 0) are rendered under the identity's training rig, both views; the first 50 held-out identities also get
the bump set under the reserved rig `top` (ADR-045 item 4). The device is checked before the first render and again
after it; anything but METAL aborts. Resumable: identities already in params.csv are skipped.

Writes <out>/images/<id>_<set>_<rig>_<view>.png (transparent RGBA), <out>/params.csv (plan row + render fields,
per ADR-026 as amended) and <out>/profile.csv (width and depth at 20 height bands).
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import bpy

HERE = Path(globals().get("__file__", "synth/render.py")).resolve().parent
sys.path.insert(0, str(HERE))  # Blender does not put the script's folder on sys.path
import bump  # noqa: E402
import cues  # noqa: E402
import latents  # noqa: E402
import rig  # noqa: E402

FROZEN_PLAN_SHA256 = "f839cba9ab55a0132bfe47b1fe269f2bb79d3582180135a4cb961b9ab6585416"  # ADR-053 Amendment 3
ROUND = 18  # cells per round (synth/plan.py N_CELLS); batches and the stop rule use whole rounds
N_CAP = 3996
SLICE_RIG = "top"  # the reserved lighting-slice rig, never a training rig (ADR-050 Amendment 1)
VIEWS = {"front": 0.0, "back": math.pi}
RENDER_FIELDS = [
    "blender", "mpfb", "clothing", "skin", "skin_r", "skin_g", "skin_b", "bump_distance", "size", "samples",
    "device", "waist_w_cm", "waist_d_cm", "setup_s", "cues_s", "view_s", "wall_s", "slice_rendered",
]  # fmt: skip


ap = argparse.ArgumentParser()
ap.add_argument("--plan", type=Path, default=HERE / "plan" / "identities.csv")
ap.add_argument("--out", type=Path, default=Path("synth/out/render"))
ap.add_argument("--seq-start", type=int, default=0, help="first identity index; a multiple of 18")
ap.add_argument("--count", type=int, default=216, help="identities in this batch; a multiple of 18")
ap.add_argument("--budget-hours", type=float, default=3.0, help="total across batches; stops at a round end")
ap.add_argument("--views", default="front,back")
ap.add_argument("--size", type=int, default=384, help="render height in px; width = 3/4 of it")
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--cycles-device", default="METAL")
ap.add_argument("--bump-distance", type=float, default=0.008, help="frozen in ADR-049 Amendment 1")
ap.add_argument(
    "--ids",
    type=Path,
    default=None,
    help="render only the ids in this file (one per line), e.g. the class exemplars",
)
ap.add_argument(
    "--rig-override",
    default="",
    help="one rig for every identity (exemplar sheets compare classes under one light)",
)
ap.add_argument("--sets", default="bump,flat", help="which sets to render: bump, flat or both")
ap.add_argument(
    "--allow-unfrozen-plan",
    action="store_true",
    help="skip the frozen plan hash check (tests only)",
)
args, _ = ap.parse_known_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
if args.seq_start % ROUND or args.count % ROUND:
    sys.exit(f"abort: --seq-start and --count must be multiples of {ROUND}")


def say(key, value):
    print(f"RENDER_{key} {json.dumps(value, default=str)}", flush=True)


def metal_active() -> bool:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    return any(d.use and d.type == args.cycles_device.upper() for d in prefs.devices)


def profile_keys():
    return [f"{p}{k:02d}" for p in ("w", "d") for k in range(rig.PROFILE_ROWS)]


def purge():
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except RuntimeError:  # not available in this context; the leak is small per identity
        pass


def shoot(body, row, set_name, rig_name, views, times):
    for view in views:
        name = f"{row['id']}_{set_name}_{rig_name}_{view}.png"
        rig.scene.render.filepath = str(args.out / "images" / name)
        body.rotation_euler.z = VIEWS[view]
        t = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(time.perf_counter() - t)


def render_identity(svc, default, row, mpfb_version, device):
    t0 = time.perf_counter()
    muscle, weight = (
        float(row["muscle_slider"]),
        float(row["weight_slider"]),
    )  # clamps are applied in plan.py
    macros = {**default, "gender": float(row["gender"]), "muscle": muscle, "weight": weight,
              "height": float(row["height"]), "proportions": float(row["proportions"])}  # fmt: skip
    body = rig.build_body(svc, macros, {r: float(row[f"m_{r}"]) for r in rig.REGIONAL})
    co = rig.mesh_co(body)
    profile = rig.width_profile(body, co)
    torso = rig.torso_measures(rig.world_coords(body, co))  # waist width / depth: validation of the fit only
    rig_name = args.rig_override or row["rig"]
    lights = rig.add_rig(rig_name)
    setup_s = time.perf_counter() - t0

    tone = rig.skin_tone(float(row["skin_u"]))
    amp = {r: float(row[f"bump_{r}"]) for r in latents.REGIONS}
    views, times, cues_s = args.views.split(","), [], 0.0
    sets = args.sets.split(",")
    sliced = row["slice_top"] == "1" and args.ids is None and "bump" in sets

    if "bump" in sets:
        t = time.perf_counter()
        maps = {v: cues.height_map(v, amp, seed=int(row["bump_seed"])) for v in VIEWS}
        mat, imgs = bump.skin_material(co, maps, args.bump_distance, tone)
        cues_s += time.perf_counter() - t
        bump.assign(body, mat)
        shoot(body, row, "bump", rig_name, views, times)
        if sliced:  # the reserved lighting-slice rig: same body, same material, bump set only (ADR-045 item 4)
            rig.remove(lights)
            lights = rig.add_rig(SLICE_RIG)
            shoot(body, row, "bump", SLICE_RIG, views, times)
            rig.remove(lights)
            lights = rig.add_rig(rig_name)
        bump.discard(mat, imgs)

    if "flat" in sets:
        flat, _ = bump.skin_material(co, None, args.bump_distance, tone)
        bump.assign(body, flat)
        shoot(body, row, "flat", rig_name, views, times)
        bpy.data.materials.remove(flat)

    rig.remove(lights + [body])
    purge()
    wall_s = time.perf_counter() - t0
    extra = {
        "blender": bpy.app.version_string, "mpfb": mpfb_version, "clothing": "none", "skin": "principled_bsdf",
        "skin_r": round(tone[0], 4), "skin_g": round(tone[1], 4), "skin_b": round(tone[2], 4),
        "bump_distance": args.bump_distance, "size": args.size, "samples": args.samples,
        "device": json.dumps(device["active"]), "waist_w_cm": torso["waist_w_cm"], "waist_d_cm": torso["waist_d_cm"],
        "setup_s": round(setup_s, 3), "cues_s": round(cues_s, 3), "view_s": round(statistics.mean(times), 3),
        "wall_s": round(wall_s, 3), "slice_rendered": int(sliced),
    }  # fmt: skip
    return extra, profile, {"setup_s": setup_s, "view_s": statistics.mean(times), "wall_s": wall_s}


def main():
    for o in list(bpy.data.objects):  # not read_factory_settings: that resets preferences and MPFB with them
        bpy.data.objects.remove(o, do_unlink=True)
    plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
    if plan_sha != FROZEN_PLAN_SHA256 and not args.allow_unfrozen_plan:
        sys.exit(f"abort: {args.plan} hash {plan_sha} is not the frozen plan hash {FROZEN_PLAN_SHA256} (ADR-053)")
    svc, _, mpfb_version = rig.load_mpfb()
    default = svc["TargetService"].get_default_macro_info_dict()
    with args.plan.open(newline="") as fh:
        plan = list(csv.DictReader(fh))
    if args.ids:  # a named subset (the class exemplars): no round rule, no budget
        wanted = set(args.ids.read_text().split())
        batch = [r for r in plan if r["id"] in wanted]
        if len(batch) != len(wanted):
            sys.exit(f"abort: {len(wanted) - len(batch)} ids in {args.ids} are not in the plan")
    else:
        batch = plan[args.seq_start : args.seq_start + args.count]
    if not batch:
        sys.exit("abort: nothing in the plan at that --seq-start")

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rig.setup_camera()
    rig.configure_render(args.size, args.samples)
    device = rig.cycles_device(args.cycles_device.upper())
    say("DEVICE", device)
    if args.cycles_device.upper() != "METAL" or not device["ok"]:
        sys.exit("abort: the render must run on METAL (ADR-047 item 2); it is not enabled")

    params_path, profile_path, budget_path = (
        args.out / "params.csv",
        args.out / "profile.csv",
        args.out / "budget.json",
    )
    done = set()
    if params_path.exists():
        with params_path.open(newline="") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
    spent = json.loads(budget_path.read_text())["elapsed_s"] if budget_path.exists() else 0.0
    hard_stop_s = args.budget_hours * 3600
    t_run = time.perf_counter()
    fields = list(batch[0]) + RENDER_FIELDS
    new_params = not params_path.exists()
    new_profile = not profile_path.exists()
    stats, stopped = [], False

    with params_path.open("a", newline="") as pf, profile_path.open("a", newline="") as qf:
        pw = csv.DictWriter(pf, fieldnames=fields, lineterminator="\n")
        qw = csv.DictWriter(qf, fieldnames=["id", *profile_keys()], lineterminator="\n")
        if new_params:
            pw.writeheader()
        if new_profile:
            qw.writeheader()
        for row in batch:
            elapsed = spent + time.perf_counter() - t_run
            if row["id"] in done:
                continue
            if args.ids is None and elapsed >= hard_stop_s and int(row["index"]) % ROUND == 0:
                stopped = True  # stop only at a round boundary, so the rendered prefix is balanced
                break
            extra, profile, s = render_identity(svc, default, row, mpfb_version, device)
            if not stats and not metal_active():  # the first image: the enabled device, read again after it
                sys.exit("abort: METAL is not active after the first image")
            stats.append(s)
            pw.writerow({**row, **extra})
            qw.writerow({"id": row["id"], **profile})
            pf.flush()
            qf.flush()
            elapsed = spent + time.perf_counter() - t_run
            budget_path.write_text(json.dumps({"elapsed_s": elapsed}))
            say(
                "IDENTITY",
                {
                    "id": row["id"],
                    "cell": row["cell"],
                    "rig": args.rig_override or row["rig"],
                    **s,
                    "hours": round(elapsed / 3600, 3),
                },
            )

    steady = stats[1:] or stats  # the first identity carries the Metal kernel compile
    if steady:
        setup_s = statistics.median(s["setup_s"] for s in steady)
        view_s = statistics.median(s["view_s"] for s in steady)
        wall_s = statistics.median(s["wall_s"] for s in steady)
        n = int((hard_stop_s - 100 * view_s) // wall_s) // ROUND * ROUND  # 100 = the 50 slice identities x 2 views
        say("SIZING", {"setup_s": round(setup_s, 3), "view_s": round(view_s, 3), "wall_s_per_identity": round(wall_s, 3),
                       "N_within_budget": min(n, N_CAP), "rendered_this_batch": len(stats), "stopped_at_budget": stopped,
                       "note": "wall_s includes cue maps, both sets and the slice; measured on the real path"})  # fmt: skip
    say("DONE", str(args.out))


main()
