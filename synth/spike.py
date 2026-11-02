"""Render-budget spike (ADR-047). Runs inside Blender's python (stdlib + bpy only), one process per device setting:

    blender -b -P synth/spike.py -- --cycles-device METAL --tag metal
    blender -b -P synth/spike.py -- --tag cpu

Prints, as SPIKE_* lines and in <out>/<tag>/summary.json:
  * setup_s (build body + sliders + rig) and view_s (one render) separately, per engine, with the first
    identity's views kept apart as warm-up (Metal kernel compile lands there);
  * the device Cycles actually has enabled, and N for 2 h / 4 h from N = budget / (setup_s + 2 * view_s);
  * what the installed MPFB exposes: macro-looking calls, shape keys, .target files, clothes, versions;
  * whether a muscle / weight call changes the mesh (dimension delta), so "applied" is checked, not assumed.
Every MPFB call is wrapped: a failure is reported, the timing run continues on the default body.
Nothing here is a result until the numbers are pasted back. Contact sheet: synth/contact_sheet.py.
"""

import argparse
import importlib
import inspect
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import addon_utils
import bpy
from mathutils import Vector

HERE = Path(globals().get("__file__", "synth/spike.py")).resolve().parent
sys.path.insert(0, str(HERE))  # Blender does not put the script's folder on sys.path
import bump  # noqa: E402
import cues  # noqa: E402
import latents  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="run")
ap.add_argument("--out", default="synth/out/spike")
ap.add_argument("--n", type=int, default=6, help="identities per engine (>= 2: the first is warm-up)")
ap.add_argument("--size", type=int, default=320, help="render height in px; width = 3/4 of it")
ap.add_argument("--samples", type=int, default=24)
ap.add_argument("--cycles-device", default="", help="METAL for the GPU; empty = CPU")
ap.add_argument("--engines", default="CYCLES,EEVEE")
ap.add_argument(
    "--grid",
    action="store_true",
    help="render the muscle x fat contact-sheet grid per sex, skip the timing run",
)
ap.add_argument("--bump-grid", action="store_true", help="the ONE bump contact-sheet round (ADR-049 item 8)")
ap.add_argument(
    "--bump-distance",
    type=float,
    default=0.004,
    help="Bump node Distance (m) for the 1x strip image",
)
args, _ = ap.parse_known_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])

BODY_H = 1.75  # every body is scaled to this height so one light rig fits all
RIGS = {  # (light type, position as multiples of BODY_H, watts); cheap, three of them
    "key_left": [("AREA", (-1.2, -1.4, 1.3), 500), ("AREA", (1.5, -1.0, 0.8), 120)],
    "key_right": [("AREA", (1.2, -1.4, 1.3), 500), ("AREA", (-1.5, -1.0, 0.8), 120)],
    "top": [("AREA", (0.0, -0.8, 1.9), 700), ("AREA", (0.0, -1.5, 0.4), 80)],
}
RIG_SIZE = {
    "key_left": 1.5,
    "key_right": 0.4,
    "top": 1.5,
}  # ADR-050 item 2: key_right has small lights, harder shadows
BUDGETS_H = (2, 4)
scene = bpy.context.scene
log = {"args": vars(args), "blender": bpy.app.version_string}


def say(key, value):
    print(f"SPIKE_{key} {json.dumps(value, default=str)}", flush=True)
    log[key] = value


def attempt(label, fn):
    """Run fn; on failure report it and return None so the spike keeps going."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a spike reports, it does not stop
        say(f"ERROR_{label}", f"{type(e).__name__}: {e}")
        return None


# --- MPFB discovery ---------------------------------------------------------
def load_mpfb():
    mod = next((m for m in addon_utils.modules() if "mpfb" in m.__name__.lower()), None)
    if mod is None:
        say("MPFB", "not found: install MPFB (Blender extension) first")
        return None
    # default_set=True puts it in the (in-memory, unsaved) preferences: MPFB reads its settings from there
    if addon_utils.enable(mod.__name__, default_set=True, persistent=False) is None:
        say("MPFB", f"found as {mod.__name__} but enable() failed; see the traceback above")
        return None
    root = Path(sys.modules[mod.__name__].__file__).parent
    svc = {}
    for cls, path in (("HumanService", "humanservice"), ("TargetService", "targetservice")):
        m = attempt(f"import_{cls}", lambda p=path: importlib.import_module(f"{mod.__name__}.services.{p}"))
        if m:
            svc[cls] = getattr(m, cls)
    manifest = root / "blender_manifest.toml"
    say(
        "MPFB",
        {
            "module": mod.__name__,
            "root": str(root),
            "manifest": manifest.read_text()[:400] if manifest.exists() else None,
        },
    )
    return svc, root


def describe(svc, root):
    out = {}
    for name, cls in svc.items():
        out[name] = {
            m: str(inspect.signature(getattr(cls, m)))
            for m in dir(cls)
            if not m.startswith("_")
            and re.search(r"create|macro|target|muscle|weight|proportion|clothes|skin", m, re.I)
        }
    say("MPFB_CALLS", out)

    groups = {}
    for t in root.rglob("*.target*"):
        groups[str(t.parent.relative_to(root))] = groups.get(str(t.parent.relative_to(root)), 0) + 1
    say("MPFB_TARGET_DIRS", dict(sorted(groups.items())[:120]))

    roots = {root}
    for fn in (
        lambda: Path(bpy.utils.user_resource("CONFIG")),
        lambda: Path(bpy.utils.user_resource("DATAFILES")),
    ):
        p = attempt("user_dir", fn)
        if p and p.exists():
            roots.add(p)
    clothes = sorted({str(f) for r in roots for f in r.rglob("*.mhclo")})
    say(
        "MPFB_CLOTHES",
        {
            "count": len(clothes),
            "first": [Path(c).name for c in clothes[:25]],
            "roots": [str(r) for r in roots],
        },
    )


# --- body, rig, camera ------------------------------------------------------
def dims(obj):
    bpy.context.view_layer.update()
    return tuple(round(x, 5) for x in obj.dimensions)


TARGET_ROOT = None  # <mpfb>/data/targets, set in main()
TONE = (
    0.45,
    0.25,
    0.18,
    1.0,
)  # placeholder skin colour; the MPFB assets pack was not found (ADR-049)
REGIONAL = {  # region -> target files (`{s}` = l / r); `abs` has no muscle target (ADR-049 item 1)
    "chest": ["torso/torso-muscle-pectoral"],
    "back": ["torso/torso-muscle-dorsi"],
    "shoulders": ["arms/{s}-upperarm-shoulder-muscle"],
    "arms": ["arms/{s}-upperarm-muscle", "arms/{s}-lowerarm-muscle"],
    "legs": ["legs/{s}-upperleg-muscle", "legs/{s}-lowerleg-muscle"],
}


def apply_regional(svc, obj, m_r):
    """Load the regional muscle targets at weight |2 m_r - 1| as incr / decr (ADR-049 item 1)."""
    for region, patterns in REGIONAL.items():
        kind, weight = latents.target_signed(m_r[region])
        if weight == 0:
            continue
        for pat in patterns:
            for side in ("l", "r") if "{s}" in pat else (None,):
                folder, base = f"{pat.format(s=side)}-{kind}".split("/")
                path = next((TARGET_ROOT / folder).glob(f"{base}.target*"))
                svc["TargetService"].load_target(obj, str(path), weight=weight)


def build_body(svc, macros=None, regional=None):
    """MPFB human (default sliders, or `macros` = a macro dict, plus regional muscle levels `regional` = {region: m_r}),
    scaled to BODY_H and put on the floor at the origin. None without MPFB."""
    if "HumanService" not in svc:
        return None
    obj = svc["HumanService"].create_human(macro_detail_dict=macros) or bpy.context.active_object
    if regional:
        apply_regional(svc, obj, regional)
    s = BODY_H / max(obj.dimensions.z, 1e-6)
    obj.scale = (s, s, s)
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    obj.location -= Vector((sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8, min(c.z for c in corners)))
    bpy.context.view_layer.update()
    return obj


def mesh_co(obj):
    """Evaluated vertex coordinates (n, 3): shape keys and modifiers included."""
    import numpy as np

    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    co = np.empty((len(mesh.vertices), 3), np.float64)
    mesh.vertices.foreach_get("co", co.ravel())
    return co


def displacement_cm(co0, co1):
    """Max and mean vertex displacement between two bodies, in cm at a 175 cm body."""
    import numpy as np

    scale = 175.0 / np.ptp(co0[:, 2])
    d = np.linalg.norm(co1 - co0, axis=1) * scale
    return {"max_cm": round(float(d.max()), 2), "mean_cm": round(float(d.mean()), 3)}


def new_body(svc, macros=None):
    obj = svc["HumanService"].create_human(macro_detail_dict=macros)
    bpy.context.view_layer.update()
    return obj


def probe_sliders(svc, root):
    """Which macro sliders and which per-region targets really move the mesh? Throwaway bodies, effect-checked.

    Macro sliders: every numeric key of MPFB's default macro dict, set to 0 and to 1 through create_human.
    Regional targets: the arms / legs / torso / stomach / hip / ... target files, loaded at weight 1 with load_target.
    """
    TS = svc["TargetService"]
    default = TS.get_default_macro_info_dict()
    say("MACRO_DEFAULTS", default)
    base_obj = new_body(svc)
    base = mesh_co(base_obj)
    bpy.data.objects.remove(base_obj, do_unlink=True)

    macro = {}
    for key, v in default.items():
        if not isinstance(v, int | float):
            continue
        cos = []
        for val in (0.0, 1.0):
            obj = attempt(f"macro_{key}", lambda k=key, v=val: new_body(svc, {**default, k: v}))
            if obj:
                cos.append(mesh_co(obj))
                bpy.data.objects.remove(obj, do_unlink=True)
        macro[key] = displacement_cm(cos[0], cos[1]) if len(cos) == 2 else None  # slider at 0 vs at 1

    tdir = root / "data" / "targets"
    regions = ("arms", "legs", "torso", "stomach", "hip", "neck", "pelvis", "buttocks")
    names = {d: sorted(p.name.split(".")[0] for p in (tdir / d).glob("*.target*")) for d in regions}
    say("REGIONAL_TARGETS", names)
    regional = {}
    pick = re.compile(r"muscle|fat|weight|girth|bulk|torso-vshape|scale-depth|scale-horiz|trunk", re.I)
    for d, files in names.items():
        for name in [n for n in files if pick.search(n)][:6]:
            path = next((tdir / d).glob(f"{name}.target*"))
            obj = new_body(svc)
            attempt(f"load_{name}", lambda o=obj, p=path: TS.load_target(o, str(p), weight=1.0))
            regional[f"{d}/{name}"] = displacement_cm(base, mesh_co(obj))  # target at weight 1 vs the default body
            bpy.data.objects.remove(obj, do_unlink=True)
    return {"macro_0_vs_1": macro, "regional_targets_weight_1": regional}


def render_grid(svc, out_dir):
    """Front view of the 3 x 3 muscle x fat grid per sex at fixed height and proportions (ADR-047 Amendment 1, F).
    Levels are placeholders: the sheet is how the user judges them, and the clamps come from it."""
    default = svc["TargetService"].get_default_macro_info_dict()
    missing = {"gender", "muscle", "weight", "height", "proportions"} - default.keys()
    if missing:
        say("ERROR_grid", f"macro dict has no {sorted(missing)}; keys are {sorted(default)}")
        return
    levels = (0.15, 0.5, 0.85)
    say("DEVICE_GRID", set_engine("CYCLES"))
    lights = add_rig("key_left")
    for sex, gender in (("female", 0.0), ("male", 1.0)):  # MakeHuman: gender 0 = female, 1 = male
        for mi, m in enumerate(levels):
            for fi, f in enumerate(levels):
                macros = {
                    **default,
                    "gender": gender,
                    "muscle": m,
                    "weight": f,
                    "height": 0.5,
                    "proportions": 0.5,
                }
                body = build_body(svc, macros)
                scene.render.filepath = str(out_dir / f"grid_{sex}_m{mi}_f{fi}.png")
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(body, do_unlink=True)
    for o in lights:
        bpy.data.objects.remove(o, do_unlink=True)
    say(
        "GRID",
        {
            "levels": levels,
            "files": "grid_<sex>_m<muscle idx>_f<fat idx>.png; rows = muscle, cols = fat in the sheet",
        },
    )


def render_bump_grid(svc, out_dir):
    """The one contact-sheet round (ADR-049 item 8), per sex: definition latent d x fat at fixed height and
    proportions, muscle fixed high (0.85) so the cap does not hide the cues, front and back, one rig; a strip at
    0.5x / 1x / 2x of the Bump Distance; render time with the bump against the flat skin. d, fat and muscle are
    placeholders on the full slider range (clamps TBD), not a result."""
    default = svc["TargetService"].get_default_macro_info_dict()
    say("DEVICE_BUMP", set_engine("CYCLES"))
    lights = add_rig("key_left")
    levels, m = (0.15, 0.5, 0.85), 0.85
    regional = {r: m for r in REGIONAL}
    times = {"bump": [], "flat": []}

    def shoot(body, amp, distance, stem, views=(("front", 0.0), ("back", math.pi)), flat=False):
        maps = None if flat else {v: cues.height_map(v, amp, seed=7) for v in ("front", "back")}
        mat, imgs = bump.skin_material(mesh_co(body), maps, distance, TONE)
        bump.assign(body, mat)
        for view, angle in views:
            body.rotation_euler.z = angle
            scene.render.filepath = str(out_dir / f"{stem}_{view}.png")
            t = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            times["flat" if flat else "bump"].append(time.perf_counter() - t)
        bump.discard(mat, imgs)

    for sex, gender in (("female", 0.0), ("male", 1.0)):
        for di, d in enumerate(levels):
            for fi, f in enumerate(levels):
                macros = {
                    **default,
                    "gender": gender,
                    "muscle": m,
                    "weight": f,
                    "height": 0.5,
                    "proportions": 0.5,
                }
                body = build_body(svc, macros, regional)
                amp = {r: latents.bump_strength(latents.d_eff(d, f, m), 0.0) for r in latents.REGIONS}
                shoot(body, amp, args.bump_distance, f"bump_{sex}_d{di}_f{fi}")
                if (di, fi) == (
                    2,
                    0,
                ):  # high d, lean: the strip, then the flat timing on the same body
                    for mult in (0.5, 1.0, 2.0):
                        shoot(
                            body,
                            amp,
                            args.bump_distance * mult,
                            f"strip_{sex}_x{mult}",
                            views=(("front", 0.0),),
                        )
                    if sex == "male":
                        for _ in range(3):
                            shoot(body, amp, args.bump_distance, "flat_timing", flat=True)
                bpy.data.objects.remove(body, do_unlink=True)
    t = time.perf_counter()
    for k in range(5):
        for view in ("front", "back"):
            cues.height_map(view, {r: 0.8 for r in latents.REGIONS}, seed=k)
    maps_s = (time.perf_counter() - t) / 5
    med = lambda xs: round(statistics.median(xs[2:] or xs), 3)  # noqa: E731 - first renders carry shader compile
    for o in lights:
        bpy.data.objects.remove(o, do_unlink=True)
    say(
        "BUMP_TIMING",
        {
            "view_s_bump": med(times["bump"]),
            "view_s_flat": med(times["flat"]),
            "maps_s_per_identity_both_views": round(maps_s, 3),
            "n_bump": len(times["bump"]),
            "bump_distance": args.bump_distance,
        },
    )
    say(
        "BUMP_GRID",
        {
            "levels_d_and_fat": levels,
            "muscle": m,
            "files": "bump_<sex>_d<i>_f<j>_<view>.png, strip_<sex>_x<mult>_front.png",
        },
    )


def add_rig(name):
    target = bpy.data.objects.new("aim", None)
    target.location = (0, 0, BODY_H / 2)
    scene.collection.objects.link(target)
    made = [target]
    for kind, pos, watts in RIGS[name]:
        light = bpy.data.objects.new("light", bpy.data.lights.new("light", kind))
        light.data.energy = watts
        light.data.size = RIG_SIZE[name]
        light.location = [p * BODY_H for p in pos]
        c = light.constraints.new("TRACK_TO")
        c.target, c.track_axis, c.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
        scene.collection.objects.link(light)
        made.append(light)
    return made


def setup_camera():
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type, cam.data.ortho_scale, cam.data.clip_end = "ORTHO", BODY_H * 1.12, 50
    cam.location, cam.rotation_euler = (0, -10, BODY_H / 2), (math.pi / 2, 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam


# --- engines ----------------------------------------------------------------
def set_engine(name):
    scene.render.resolution_y = args.size
    scene.render.resolution_x = args.size * 3 // 4
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGBA"
    if name == "CYCLES":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        return cycles_device()
    ids = [e.identifier for e in scene.render.bl_rna.properties["engine"].enum_items]
    scene.render.engine = next(i for i in ids if i.startswith("BLENDER_EEVEE"))
    for attr in ("taa_render_samples",):
        if hasattr(scene.eevee, attr):
            setattr(scene.eevee, attr, args.samples)
    return {"engine": scene.render.engine}


def cycles_device():
    """Enable the requested device and report what is really enabled. {'ok': False} = skip this engine."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    kind = args.cycles_device.upper()
    if kind:
        prefs.compute_device_type = kind
        attempt("refresh_devices", prefs.refresh_devices)
        for d in prefs.devices:
            d.use = d.type == kind
    scene.cycles.device = "GPU" if kind else "CPU"
    active = [(d.name, d.type) for d in prefs.devices if d.use]
    info = {
        "requested": kind or "CPU",
        "scene_device": scene.cycles.device,
        "compute_device_type": prefs.compute_device_type,
        "active": active,
    }
    info["ok"] = (not kind) or any(t == kind for _, t in active)
    return info


def time_engine(name, svc, out_dir):
    info = set_engine(name)
    say(f"DEVICE_{name}", info)
    if info.get("ok") is False:
        say(f"SKIPPED_{name}", "requested GPU device is not enabled")
        return
    rig_names, rows = list(RIGS), []
    for i in range(args.n):
        t0 = time.perf_counter()
        body = attempt("build_body", lambda: build_body(svc))
        lights = add_rig(rig_names[i % len(rig_names)])
        setup_s = time.perf_counter() - t0
        views = []
        for view, angle in (("front", 0.0), ("back", math.pi)):
            if body:
                body.rotation_euler.z = angle
            scene.render.filepath = str(out_dir / f"{name.lower()}_id{i}_{view}.png")
            t = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            views.append(time.perf_counter() - t)
        rows.append({"id": i, "rig": rig_names[i % len(rig_names)], "setup_s": setup_s, "view_s": views})
        for o in lights + ([body] if body else []):
            bpy.data.objects.remove(o, do_unlink=True)
    steady = rows[1:] or rows  # identity 0 carries the one-off kernel compile / shader cache
    setup_s = statistics.median(r["setup_s"] for r in steady)
    view_s = statistics.median(v for r in steady for v in r["view_s"])
    sized = {}
    for h in BUDGETS_H:
        n = int(h * 3600 / (setup_s + 2 * view_s))
        sized[f"{h}h"] = {"N": n, "largest_pow2": 2 ** int(math.log2(n)) if n else 0}
    say(
        f"TIMING_{name}",
        {
            "setup_s": round(setup_s, 3),
            "view_s": round(view_s, 3),
            "warmup_views_s": [round(v, 2) for v in rows[0]["view_s"]],
            "warmup_setup_s": round(rows[0]["setup_s"], 2),
            "N": sized,
            "per_identity": rows,
            "note": "excludes compositing and MediaPipe mask time (post) and the ~50 extra held-out-rig identities (ADR-045)",
        },
    )


def main():
    for o in list(
        bpy.data.objects
    ):  # not read_factory_settings: that resets preferences and the MPFB extension with them
        bpy.data.objects.remove(o, do_unlink=True)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    found = attempt("load_mpfb", load_mpfb)
    svc = {}
    if found:
        svc, root = found
        global TARGET_ROOT
        TARGET_ROOT = root / "data" / "targets"
        attempt("describe", lambda: describe(svc, root))
        say("SLIDER_PROBE", attempt("probe_sliders", lambda: probe_sliders(svc, root)))
    setup_camera()
    if args.bump_grid and svc:
        attempt("render_bump_grid", lambda: render_bump_grid(svc, out_dir))
    elif args.grid and svc:
        attempt("render_grid", lambda: render_grid(svc, out_dir))
    else:
        for name in args.engines.split(","):
            attempt(f"engine_{name}", lambda n=name: time_engine(n, svc, out_dir))
    (out_dir / "summary.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"SPIKE_DONE {out_dir}/summary.json", flush=True)


main()
