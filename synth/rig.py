"""Blender helpers for render.py: MPFB loading, body build, light rigs, camera, engine and device.

Runs inside Blender (bpy + numpy). Copied from synth/spike.py, which is left untouched because it was run: if a
constant here changes (RIGS, RIG_SIZE, BODY_H, REGIONAL), the spike's numbers no longer describe the render.
"""

import importlib
import math
import re
import sys
from pathlib import Path

import addon_utils
import bpy
import latents  # found through the sys.path entry render.py adds for this folder
import numpy as np
from mathutils import Vector

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
}  # ADR-050 item 2; roles in ADR-050 Amendment 1
REGIONAL = {  # region -> target files (`{s}` = l / r); `abs` has no muscle target (ADR-049 item 1)
    "chest": ["torso/torso-muscle-pectoral"],
    "back": ["torso/torso-muscle-dorsi"],
    "shoulders": ["arms/{s}-upperarm-shoulder-muscle"],
    "arms": ["arms/{s}-upperarm-muscle", "arms/{s}-lowerarm-muscle"],
    "legs": ["legs/{s}-upperleg-muscle", "legs/{s}-lowerleg-muscle"],
}
SKIN_LIGHT, SKIN_DARK = (
    (0.55, 0.34, 0.25),
    (0.12, 0.06, 0.045),
)  # linear Base Color; PLACEHOLDERS, look at batch 1
PROFILE_ROWS = 20  # ADR-047 condition (c): width (x) and depth (y) at 20 height fractions; rows chosen later

scene = bpy.context.scene
TARGET_ROOT: Path | None = None


def load_mpfb():
    """(services, MPFB root, MPFB version). Raises: a render must not fall back to the default body."""
    mod = next((m for m in addon_utils.modules() if "mpfb" in m.__name__.lower()), None)
    if mod is None:
        raise SystemExit("MPFB not found: install the MPFB Blender extension")
    if addon_utils.enable(mod.__name__, default_set=True, persistent=False) is None:
        raise SystemExit(f"MPFB found as {mod.__name__} but enable() failed")
    root = Path(sys.modules[mod.__name__].__file__).parent
    svc = {}
    for cls, path in (("HumanService", "humanservice"), ("TargetService", "targetservice")):
        svc[cls] = getattr(importlib.import_module(f"{mod.__name__}.services.{path}"), cls)
    manifest = root / "blender_manifest.toml"
    found = re.search(r'^version\s*=\s*"([^"]+)"', manifest.read_text(), re.M) if manifest.exists() else None
    global TARGET_ROOT
    TARGET_ROOT = root / "data" / "targets"
    return svc, root, found.group(1) if found else "unknown"


def skin_tone(u: float) -> tuple[float, float, float, float]:
    return (*(a + u * (b - a) for a, b in zip(SKIN_LIGHT, SKIN_DARK, strict=True)), 1.0)


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


def build_body(svc, macros, regional):
    """MPFB human from a macro dict plus regional muscle levels, scaled to BODY_H, on the floor at the origin."""
    obj = svc["HumanService"].create_human(macro_detail_dict=macros) or bpy.context.active_object
    apply_regional(svc, obj, regional)
    s = BODY_H / max(obj.dimensions.z, 1e-6)
    obj.scale = (s, s, s)
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    centre = Vector((sum(c.x for c in corners) / 8, sum(c.y for c in corners) / 8, min(c.z for c in corners)))
    obj.location -= centre
    bpy.context.view_layer.update()
    return obj


def mesh_co(obj):
    """Evaluated local vertex coordinates (n, 3): shape keys and modifiers included."""
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    co = np.empty((len(mesh.vertices), 3), np.float64)
    mesh.vertices.foreach_get("co", co.ravel())
    return co


def world_coords(obj, co):
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


# Height fractions of the torso landmarks (approximate, from the cue landmarks in cues.py; not validated).
LANDMARKS = {"hip": 0.52, "waist": 0.62, "chest": 0.72, "thigh": 0.40}
TORSO_HALF_X = 0.14  # body heights: keeps the A-pose arms out of the torso width. At the chest band this cut is
# wider than the torso, so chest_w_cm just reads the cut (about 49 cm) and is NOT a measurement (ADR-053).


def torso_measures(world):
    """Width (x range) and depth (y range) in cm at each landmark band, arms excluded by |x| < TORSO_HALF_X * H."""
    out = {}
    for name, frac in LANDMARKS.items():
        keep = (np.abs(world[:, 2] / BODY_H - frac) < 0.01) & (np.abs(world[:, 0]) < TORSO_HALF_X * BODY_H)
        pts = world[keep]
        out[f"{name}_w_cm"] = round(float(np.ptp(pts[:, 0]) * 100), 2) if len(pts) else float("nan")
        out[f"{name}_d_cm"] = round(float(np.ptp(pts[:, 1]) * 100), 2) if len(pts) else float("nan")
    return out


def width_profile(obj, co=None):
    """Front width (x range) and side depth (y range) in cm per height band, in world space, A-pose arms included
    at the arm heights (stated in ADR-045; the lat-flare ADR chooses its rows and handles that)."""
    co = mesh_co(obj) if co is None else co
    m = np.array(obj.matrix_world)
    world = co @ m[:3, :3].T + m[:3, 3]
    band = np.minimum((world[:, 2] / BODY_H * PROFILE_ROWS).astype(int), PROFILE_ROWS - 1)
    out = {}
    for k in range(PROFILE_ROWS):
        pts = world[band == k]
        out[f"w{k:02d}"] = round(float(np.ptp(pts[:, 0]) * 100), 2) if len(pts) else 0.0
        out[f"d{k:02d}"] = round(float(np.ptp(pts[:, 1]) * 100), 2) if len(pts) else 0.0
    return out


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


def remove(objects):
    for o in objects:
        bpy.data.objects.remove(o, do_unlink=True)


def setup_camera():
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.type, cam.data.ortho_scale, cam.data.clip_end = "ORTHO", BODY_H * 1.12, 50
    cam.location, cam.rotation_euler = (0, -10, BODY_H / 2), (math.pi / 2, 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam


def cycles_device(kind: str):
    """Enable the requested device and report what is really enabled; info["ok"] is False if it is not `kind`."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = kind
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type == kind
    scene.cycles.device = "GPU"
    active = [(d.name, d.type) for d in prefs.devices if d.use]
    return {"requested": kind, "compute_device_type": prefs.compute_device_type, "active": active,
            "ok": any(t == kind for _, t in active)}  # fmt: skip


def configure_render(size: int, samples: int):
    scene.render.resolution_y = size
    scene.render.resolution_x = size * 3 // 4
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format, scene.render.image_settings.color_mode = "PNG", "RGBA"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
