"""Procedural definition cues as a height map (ADR-049 item 5). Pure numpy, so it is unit-tested and plottable.

Map frame: row 0 = the feet, rows rise with height (the order Blender stores image pixels); columns run along x,
centred on the body midline and XSPAN body heights wide. 1.0 = flat skin, grooves go below it. Landmarks are
approximate fractions of body height read off the spike renders (|x| from the midline, z from the feet), jittered
per identity; they are starting points the one contact-sheet round judges, not anatomy.
"""

import numpy as np

W, H = 768, 1024  # W even: the left half mirrors the right
XSPAN = 0.72  # map width in body heights (the hands reach about 0.31 from the midline)
GROOVE_W = 0.0045  # Gaussian width of a groove, body heights (about 8 mm at 1.75 m)
WOBBLE = 0.004  # smooth displacement of the lines, body heights
POINT_JITTER = 0.003
ARM_S, ARM_E = (0.117, 0.80), (0.307, 0.58)  # shoulder joint and hand, A-pose

# (region, view, polyline of (|x|, z), relative strength)
STATIC_LINES = [
    ("abs", "front", [(0.0, 0.50), (0.0, 0.71)], 1.0),  # linea alba
    ("abs", "front", [(0.048, 0.52), (0.052, 0.69)], 0.8),  # lateral border
    ("chest", "front", [(0.012, 0.70), (0.05, 0.688), (0.10, 0.715)], 1.0),  # lower pec border
    ("chest", "front", [(0.0, 0.70), (0.0, 0.80)], 0.6),  # sternum
    ("shoulders", "front", [(0.088, 0.775), (0.118, 0.725)], 0.8),  # deltoid / pec
    ("shoulders", "back", [(0.100, 0.79), (0.127, 0.73)], 0.8),
    ("back", "back", [(0.0, 0.52), (0.0, 0.80)], 1.0),  # spine groove
    ("back", "back", [(0.03, 0.775), (0.058, 0.705), (0.046, 0.665)], 0.9),  # scapula
    ("back", "back", [(0.098, 0.70), (0.088, 0.58)], 0.7),  # lat edge
    ("legs", "front", [(0.088, 0.30), (0.072, 0.44)], 0.9),  # outer quad line
    ("legs", "front", [(0.036, 0.32), (0.042, 0.43)], 0.6),  # inner quad line
    ("legs", "back", [(0.058, 0.28), (0.058, 0.45)], 0.9),  # hamstring split
]


def smooth_noise(rng, shape, cells):
    """Bilinearly upsampled Gaussian grid, roughly unit variance, values in about [-2, 2]."""
    gh, gw = cells
    g = rng.standard_normal((gh, gw))
    r = np.linspace(0, gh - 1, shape[0])
    c = np.linspace(0, gw - 1, shape[1])
    r0, c0 = np.minimum(r.astype(int), gh - 2), np.minimum(c.astype(int), gw - 2)
    fr, fc = (r - r0)[:, None], (c - c0)[None, :]
    a, b = g[r0][:, c0], g[r0][:, c0 + 1]
    c_, d = g[r0 + 1][:, c0], g[r0 + 1][:, c0 + 1]
    return (a * (1 - fc) + b * fc) * (1 - fr) + (c_ * (1 - fc) + d * fc) * fr


def _seg_dist(x, z, p0, p1):
    d = np.subtract(p1, p0)
    t = np.clip(((x - p0[0]) * d[0] + (z - p0[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(x - (p0[0] + t * d[0]), z - (p0[1] + t * d[1]))


def _arm_line(t0, t1, offset):
    """Line along the arm axis between fractions t0..t1 of shoulder -> hand, shifted sideways by `offset`."""
    s, e = np.array(ARM_S), np.array(ARM_E)
    n = np.array([e[1] - s[1], s[0] - e[0]])
    n = n / np.hypot(*n) * offset
    return [tuple(s + t * (e - s) + n) for t in (t0, t1)]


def _lines(rng):
    """Every line of both views with its jitter drawn in a fixed order, so the same seed gives consistent
    front and back maps whatever the view or the amplitudes."""
    lines = list(STATIC_LINES)
    step = 0.036 + 0.004 * rng.standard_normal()
    for k in range(4 if rng.random() < 0.5 else 3):  # abdominal rows
        z = 0.545 + k * step
        lines.append(("abs", "front", [(0.0, z), (0.025, z + 0.002), (0.048, z)], 0.9))
    lines += [
        ("arms", "front", _arm_line(0.2, 0.6, 0.0), 0.8),  # bicep separation
        ("arms", "front", _arm_line(0.66, 0.95, 0.003), 0.5),  # forearm
        ("arms", "back", _arm_line(0.22, 0.58, 0.006), 0.8),  # triceps horseshoe
        ("arms", "back", _arm_line(0.22, 0.58, -0.006), 0.8),
    ]
    out = []
    for region, view, pts, base in lines:
        pts = [(x + rng.normal(0, POINT_JITTER) if x > 0.01 else x, z + rng.normal(0, POINT_JITTER)) for x, z in pts]
        out.append((region, view, pts, base * rng.uniform(0.7, 1.3), GROOVE_W * rng.uniform(0.8, 1.25)))
    return out


def height_map(view: str, amp: dict[str, float], seed: int) -> np.ndarray:
    """(H, W) float32 in [0, 1]: 1 - the sum of the grooves of `view`, each scaled by its region's amp (0..1)."""
    rng = np.random.default_rng(seed)
    half = W // 2
    wx, wz = smooth_noise(rng, (H, half), (8, 4)), smooth_noise(rng, (H, half), (8, 4))
    xs = (np.arange(half) + 0.5) / half * (XSPAN / 2)  # x >= 0, body heights
    zs = (np.arange(H) + 0.5) / H
    depth = np.zeros((H, half), np.float32)
    lines = _lines(rng)
    gain_left, gain_right = rng.uniform(0.9, 1.1, 2)
    for region, line_view, pts, strength, width in lines:
        if line_view != view or amp.get(region, 0.0) <= 0:
            continue
        for p0, p1 in zip(pts[:-1], pts[1:], strict=False):
            pad = 4 * width + WOBBLE * 3 + POINT_JITTER
            c0 = max(0, int((min(p0[0], p1[0]) - pad) / (XSPAN / 2) * half))
            c1 = min(half, int((max(p0[0], p1[0]) + pad) / (XSPAN / 2) * half) + 1)
            r0 = max(0, int((min(p0[1], p1[1]) - pad) * H))
            r1 = min(H, int((max(p0[1], p1[1]) + pad) * H) + 1)
            if c0 >= c1 or r0 >= r1:
                continue
            x = xs[None, c0:c1] + WOBBLE * wx[r0:r1, c0:c1]
            z = zs[r0:r1, None] + WOBBLE * wz[r0:r1, c0:c1]
            d = _seg_dist(x, z, p0, p1)
            depth[r0:r1, c0:c1] += amp[region] * strength * np.exp(-((d / width) ** 2))
    both = np.concatenate([depth[:, ::-1] * gain_left, depth * gain_right], axis=1)
    return (1.0 - np.clip(both, 0.0, 1.0)).astype(np.float32)
