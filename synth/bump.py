"""Skin material with procedural definition cues (ADR-049). Runs inside Blender (bpy + numpy).

The numpy height maps of `cues.py` are sampled through the body's own object-space coordinates (height fraction
and x), not its UV layout; front and back maps are blended by the side of the body (y), then a Bump node feeds
the Principled BSDF normal. No geometry is added. Without `maps` the material is the flat skin (no Bump node).
"""

import bpy
import cues
import numpy as np

# Body heights over which the front map hands over to the back map across y. Round 1 used 0.06: a limb is only a
# few cm thick, so both its surfaces sat near the midpoint and got half of each map (ADR-049 Amendment 1).
SIDE_BLEND = 0.01


def _image(name, arr):
    img = bpy.data.images.new(name, cues.W, cues.H, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((cues.H, cues.W, 4), np.float32)
    rgba[..., :3] = arr[..., None]
    img.pixels.foreach_set(rgba.ravel())  # row 0 = bottom, as cues.py lays it out
    img.update()
    return img


def _math(nodes, op, a=None, b=None, c=None, clamp=False):
    n = nodes.new("ShaderNodeMath")
    n.operation, n.use_clamp = op, clamp
    for i, v in enumerate((a, b, c)):
        if isinstance(v, int | float):
            n.inputs[i].default_value = v
    return n


def skin_material(co, maps, distance, tone, name="skin"):
    """co: evaluated local vertex coordinates (n, 3) of the body; maps: {"front": arr, "back": arr} or None."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out, bsdf = nt.nodes.new("ShaderNodeOutputMaterial"), nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = tone
    bsdf.inputs["Roughness"].default_value = 0.55
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if maps is None:
        return mat, []

    zmin, zmax = co[:, 2].min(), co[:, 2].max()
    h = zmax - zmin
    xc = (co[:, 0].min() + co[:, 0].max()) / 2
    torso = (co[:, 2] > zmin + 0.45 * h) & (co[:, 2] < zmin + 0.75 * h) & (np.abs(co[:, 0] - xc) < 0.1 * h)
    yc = (co[torso, 1].min() + co[torso, 1].max()) / 2  # the torso's mid-depth separates front from back

    link, nodes = nt.links.new, nt.nodes
    sep = nodes.new("ShaderNodeSeparateXYZ")
    link(nodes.new("ShaderNodeTexCoord").outputs["Object"], sep.inputs["Vector"])
    u = _math(nodes, "MULTIPLY_ADD", None, 1 / (h * cues.XSPAN), 0.5 - xc / (h * cues.XSPAN))
    v = _math(nodes, "MULTIPLY_ADD", None, 1 / h, -zmin / h)
    link(sep.outputs["X"], u.inputs[0])
    link(sep.outputs["Z"], v.inputs[0])
    uv = nodes.new("ShaderNodeCombineXYZ")
    link(u.outputs[0], uv.inputs["X"])
    link(v.outputs[0], uv.inputs["Y"])

    images, tex = [], {}
    for view in ("front", "back"):
        img = _image(f"{name}_{view}", maps[view])
        images.append(img)
        t = nodes.new("ShaderNodeTexImage")
        t.image, t.interpolation, t.extension = (
            img,
            "Linear",
            "EXTEND",
        )  # outside the map = flat skin
        link(uv.outputs["Vector"], t.inputs["Vector"])
        tex[view] = t
    side = _math(nodes, "MULTIPLY_ADD", None, 1 / (SIDE_BLEND * h), 0.5 - yc / (SIDE_BLEND * h), clamp=True)
    link(sep.outputs["Y"], side.inputs[0])  # 0 = front side (-y), 1 = back side
    diff = _math(nodes, "SUBTRACT")
    link(tex["back"].outputs["Color"], diff.inputs[0])
    link(tex["front"].outputs["Color"], diff.inputs[1])
    mix = _math(nodes, "MULTIPLY_ADD")
    link(diff.outputs[0], mix.inputs[0])
    link(side.outputs[0], mix.inputs[1])
    link(tex["front"].outputs["Color"], mix.inputs[2])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = distance
    link(mix.outputs[0], bump.inputs["Height"])
    link(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat, images


def assign(body, mat):
    body.data.materials.clear()
    body.data.materials.append(mat)
    for p in body.data.polygons:
        p.material_index = 0


def discard(mat, images):
    bpy.data.materials.remove(mat)
    for img in images:
        bpy.data.images.remove(img)
