"""Claudeguy — the capstone creature's volumetric anatomy (work in progress).

Aesthetic brief (Max's claymation reference): soft clay flower-creature facing
the viewer. Plump terracotta petals radiating from a cream face disk, two
enormous glossy bulging eyes (asymmetric — wide-eyed-and-bushy-tailed), the
little omega-wobble mouth. Handmade irregularity everywhere; nothing perfect.

Authored in 32-grid units, scaled by K like every 3D creature — but this one
is designed to be trained at 64^3.

Preview: BONSAI_GRID3=64 python3 claudeguy3d.py
"""

import numpy as np

from target3d import GRID3, SCALE, SS, _sphere, _swept

K = SCALE
N_PETALS = 12

PETAL = (0.85, 0.47, 0.34)      # terracotta
PETAL_DEEP = (0.72, 0.36, 0.26)  # shadowed petal backs
FACE = (0.96, 0.93, 0.86)        # warm cream
EYE_WHITE = (0.99, 0.99, 0.97)
PUPIL = (0.08, 0.07, 0.07)
MOUTH = (0.30, 0.20, 0.16)

C = GRID3 / 2.0


def _ellipsoid(vol, cx, cy, cz, rx, ry, rz, color, soft=0.8):
    """Axis-aligned soft ellipsoid on the supersampled canvas (target3d convention)."""
    from target3d import _coords
    x, y, z = _coords()
    d = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2)
    m = np.clip((1.0 - d) * min(rx, ry, rz) / soft, 0, 1)
    a = m[..., None]
    vol[..., 3:4] = np.maximum(vol[..., 3:4], a)
    np.copyto(vol[..., 0:3], np.array(color, np.float32), where=a > 0.15)


FEATURE_SCALE = 1.45  # Max: features were too small; scale offsets and radii together


def _stroke_on_face(vol, pts2d, fz_off, r, color):
    """Sweep small spheres along 2D face-plane points [(x_off, y_off), ...]."""
    F = FEATURE_SCALE
    fy = 17.0 * K
    for (ox, oy) in pts2d:
        _sphere(vol, C + ox * F * K, fy + oy * F * K, C + 2.0 * K + fz_off * K, r * F * K, color, soft=0.3 * K)


# Expression presets, straight from the emotion anchor sheet.
# eyes: balls | happy (^^) | squeeze (><) | sadclosed | spiral
# mouth: w (:3) | frown | wavy | flat
# brows: none | angry (inner down) | plead (inner up)
EXPRESSIONS = {
    "neutral":    dict(eyes="balls", mouth="w", brows="none", droop=0.0, pupil=1.0, look=(0, 0)),
    "serene":     dict(eyes="happy", mouth="w", brows="none", droop=0.0, pupil=1.0, look=(0, 0)),
    "wince":      dict(eyes="squeeze", mouth="wavy", brows="none", droop=0.15, pupil=1.0, look=(0, 0)),
    "angry":      dict(eyes="balls", mouth="frown", brows="angry", droop=0.0, pupil=0.75, look=(0, -0.4)),
    "pleading":   dict(eyes="balls", mouth="flat", brows="plead", droop=0.25, pupil=1.35, look=(0, 1.0)),
    "melancholy": dict(eyes="sadclosed", mouth="frown", brows="plead", droop=0.6, pupil=1.0, look=(0, 0)),
    "dizzy":      dict(eyes="spiral", mouth="wavy", brows="none", droop=0.3, pupil=1.0, look=(0, 0)),
}




# --- Sheet-accurate expression painter (small, clustered features; sheet ratios) ---
STROKE = (0.12, 0.10, 0.10)
TAN = (0.82, 0.77, 0.60)
HATBROWN = (0.33, 0.22, 0.14)
GREEN = (0.45, 0.72, 0.42)
GREEN_D = (0.32, 0.55, 0.30)

SHEET_DROOP = {"melancholy": 0.55, "modecollapse": 0.3}
SHEET_NAMES = ["wideeyed", "serene", "wince", "angry", "mindblown", "monocle",
               "pleading", "suspicious", "cowboy", "melancholy", "disguise", "dizzy",
               "modecollapse"]


def _arc(cx, cy, r, a0, a1, n=11):
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(a0, a1, n)]


def _line(p0, p1, n=9):
    return [(p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)
            for t in np.linspace(0, 1, n)]


def _dot(vol, ox, oy, r, color=STROKE, zoff=5.6):
    F = FEATURE_SCALE
    fy = 17.0 * K
    _sphere(vol, C + ox * F * K, fy + oy * F * K, C + 2.0 * K + zoff * K, r * F * K, color, soft=0.3 * K)


def _wmouth(vol, w=1.3, y=-1.6, zoff=5.6):
    for t in np.linspace(-1, 1, 13):
        _stroke_on_face(vol, [(w * t * 2.0 / 2, y - 0.55 * abs(np.sin(np.pi * t)))], zoff, 0.38, STROKE)


def _paint_sheet(vol, name):
    fy = 17.0 * K
    s = _stroke_on_face
    if name == "wideeyed":
        _sphere(vol, C - 3.9 * K, fy + 2.4 * K, C + 2.0 * K + 4.4 * K, 4.0 * K, EYE_WHITE, soft=0.6 * K)
        _sphere(vol, C + 3.9 * K, fy + 3.0 * K, C + 2.0 * K + 4.8 * K, 4.6 * K, EYE_WHITE, soft=0.6 * K)
        _sphere(vol, C - 3.9 * K, fy + 2.2 * K, C + 2.0 * K + 7.8 * K, 1.8 * K, PUPIL, soft=0.4 * K)
        _sphere(vol, C + 3.9 * K, fy + 2.8 * K, C + 2.0 * K + 8.8 * K, 2.0 * K, PUPIL, soft=0.4 * K)
        for t in np.linspace(-1.0, 1.0, 15):
            _stroke_on_face(vol, [(1.8 * t, -2.9 - 0.72 * abs(np.sin(np.pi * t)))], 6.0, 0.42, MOUTH)
    elif name == "serene":
        for sx in (-2.3, 2.3):
            s(vol, _arc(sx, 1.3, 1.3, 0.5, np.pi - 0.5), 5.6, 0.42, STROKE)
        _wmouth(vol)
    elif name == "wince":
        for sx in (-2.3, 2.3):
            sd = 1 if sx > 0 else -1
            s(vol, _line((sx - sd * 1.1, 2.8), (sx + sd * 1.1, 1.8)), 5.6, 0.4, STROKE)
            s(vol, _line((sx - sd * 1.1, 0.8), (sx + sd * 1.1, 1.8)), 5.6, 0.4, STROKE)
        for t in np.linspace(-1, 1, 15):
            s(vol, [(1.5 * t, -1.8 + 0.35 * np.sin(3 * np.pi * t))], 5.6, 0.36, STROKE)
    elif name == "angry":
        _dot(vol, -2.0, 1.5, 0.62); _dot(vol, 2.0, 1.5, 0.62)
        s(vol, _line((-3.0, 3.4), (-1.2, 2.6)), 5.5, 0.42, STROKE)
        s(vol, _line((3.0, 3.4), (1.2, 2.6)), 5.5, 0.42, STROKE)
        s(vol, _arc(0, -3.2, 1.5, 0.45, np.pi - 0.45), 5.6, 0.4, STROKE)
    elif name == "mindblown":
        _dot(vol, -2.2, 0.2, 0.6); _dot(vol, 2.2, 0.2, 0.6)
        _dot(vol, 0, -2.0, 1.0, STROKE, 5.8)
        for i in range(6):  # jagged crack line across the upper face
            s(vol, [(-3.6 + i * 1.45, 2.6 + (0.7 if i % 2 else -0.2))], 5.2, 0.5, STROKE)
        _sphere(vol, C, fy + 6.6 * K, C + 3.0 * K, 1.7 * K, TAN, soft=0.6 * K)
        _sphere(vol, C, fy + 8.4 * K, C + 3.0 * K, 2.0 * K, TAN, soft=0.6 * K)
        for (px, py, pr) in ((0, 10.8, 3.4), (-3.6, 9.6, 2.6), (3.6, 9.8, 2.7), (0, 12.6, 2.3)):
            _sphere(vol, C + px * K, fy + py * K, C + 3.0 * K, pr * K, TAN, soft=0.9 * K)
        for (px, py) in ((-8.5, 10.0), (8.5, 9.6), (-10.5, 6.0), (10.5, 6.5)):
            _sphere(vol, C + px * K, fy + py * K, C + 2.0 * K, 1.3 * K, PETAL, soft=0.5 * K)
    elif name == "monocle":
        _dot(vol, -2.0, 1.7, 0.62); _dot(vol, 2.0, 1.7, 0.62)
        s(vol, _arc(2.0, 1.7, 1.75, 0, 2 * np.pi, 21), 5.4, 0.34, STROKE)
        s(vol, _line((2.6, 0.2), (4.4, -3.2), 11), 5.2, 0.28, STROKE)
        s(vol, _line((-3.0, 3.1), (-1.2, 2.8)), 5.5, 0.4, STROKE)
        s(vol, _line((-1.0, -1.9), (1.0, -1.9)), 5.6, 0.38, STROKE)
    elif name == "pleading":
        for sx in (-2.2, 2.2):
            _dot(vol, sx, 2.0, 1.15)
            _dot(vol, sx - 0.45, 2.5, 0.4, EYE_WHITE, 6.3)
        s(vol, _arc(-2.3, 3.9, 1.4, 0.5, np.pi - 0.7), 5.4, 0.4, STROKE)
        s(vol, _arc(2.3, 3.9, 1.4, 0.7, np.pi - 0.5), 5.4, 0.4, STROKE)
        s(vol, _arc(0, -3.1, 1.0, 0.5, np.pi - 0.5), 5.6, 0.36, STROKE)
    elif name == "suspicious":
        _dot(vol, -2.5, 1.8, 0.62); _dot(vol, 1.5, 1.8, 0.62)
        s(vol, _line((-3.2, 3.3), (-1.6, 3.5)), 5.5, 0.4, STROKE)
        s(vol, _line((1.0, 3.6), (2.8, 3.2)), 5.5, 0.4, STROKE)
        nose = [(0.3 + t * 3.6, 0.5 - t * 1.1) for t in np.linspace(0, 1, 10)]
        for i, (nx, ny) in enumerate(nose):
            tt = i / 9.0
            _dot(vol, nx, ny, 0.75 - 0.33 * tt, (0.93, 0.87, 0.76), 5.8 + 2.2 * tt)
        s(vol, _line((-1.6, -2.0), (0.4, -2.1)), 5.6, 0.38, STROKE)
    elif name == "cowboy":
        _dot(vol, -2.0, 1.9, 0.62); _dot(vol, 2.0, 1.9, 0.62)
        _wmouth(vol)
        _ellipsoid(vol, C, fy + 8.2 * K, C + 2.2 * K, 9.6 * K, 1.1 * K, 6.0 * K, HATBROWN, soft=0.5 * K)
        _ellipsoid(vol, C, fy + 11.0 * K, C + 2.2 * K, 4.6 * K, 3.2 * K, 4.2 * K, HATBROWN, soft=0.6 * K)
        _sphere(vol, C - 9.2 * K, fy + 9.5 * K, C + 2.2 * K, 1.2 * K, HATBROWN, soft=0.4 * K)
        _sphere(vol, C + 9.2 * K, fy + 9.5 * K, C + 2.2 * K, 1.2 * K, HATBROWN, soft=0.4 * K)
    elif name == "melancholy":
        for sx in (-2.3, 2.3):
            s(vol, _arc(sx, 2.1, 1.35, np.pi + 0.5, 2 * np.pi - 0.5), 5.5, 0.42, STROKE)
        s(vol, _arc(-2.3, 3.6, 1.2, 0.6, np.pi - 0.8, 7), 5.3, 0.36, STROKE)
        s(vol, _arc(2.3, 3.6, 1.2, 0.8, np.pi - 0.6, 7), 5.3, 0.36, STROKE)
        s(vol, _arc(0, -3.4, 1.4, 0.5, np.pi - 0.5), 5.6, 0.4, STROKE)
    elif name == "disguise":
        for sx in (-2.2, 2.2):
            s(vol, _arc(sx, 1.7, 1.6, 0, 2 * np.pi, 19), 5.5, 0.34, STROKE)
            s(vol, _arc(sx, 3.7, 1.35, 0.4, np.pi - 0.4), 5.4, 0.55, STROKE)
        s(vol, _line((-0.6, 1.9), (0.6, 1.9), 5), 5.5, 0.3, STROKE)
        _dot(vol, 0, -0.4, 0.7, (0.90, 0.82, 0.70), 6.2)
        _ellipsoid(vol, C, fy - 2.8 * K, C + 8.0 * K, 3.8 * K, 1.2 * K, 0.8 * K, (0.25, 0.22, 0.20), soft=0.4 * K)
    elif name == "dizzy":
        for sx in (-2.2, 2.2):
            spiral = [((sx + (0.3 + 1.15 * t) * np.cos(3.5 * np.pi * t)),
                       (1.7 + (0.3 + 1.15 * t) * np.sin(3.5 * np.pi * t))) for t in np.linspace(0, 1, 19)]
            s(vol, spiral, 5.5, 0.34, STROKE)
        for t in np.linspace(-1, 1, 15):
            s(vol, [(1.5 * t, -2.0 + 0.35 * np.sin(3 * np.pi * t))], 5.6, 0.36, STROKE)
    elif name == "modecollapse":
        for sx in (-2.3, 2.3):
            sd = 1 if sx > 0 else -1
            s(vol, _line((sx - sd * 1.1, 2.9), (sx + sd * 1.1, 1.9)), 5.6, 0.4, STROKE)
            s(vol, _line((sx - sd * 1.1, 0.9), (sx + sd * 1.1, 1.9)), 5.6, 0.4, STROKE)
        # the mouth, open in dismay
        _dot(vol, 0, -2.1, 1.05, STROKE, 5.8)
        # ...and what comes out of it when the distribution collapses
        F = FEATURE_SCALE
        mx, my = 0.0, -2.1 * F
        for (fan, ln, curl) in ((-1.5, 7.0, 1.8), (-0.5, 6.0, -1.2), (0.5, 6.4, 1.4), (1.5, 5.4, -1.7)):
            pts = []
            for t in np.linspace(0, 1, 10):
                wig = 0.9 * np.sin(3.0 * t + fan)          # writhing, not draping
                px = mx + fan * 2.6 * t + wig * t
                py = my - 2.2 * t - 0.8 * t * t + 0.5 * wig * t
                pz = 9.0 + 4.5 * t - 2.8 * t * t + curl * 0.8 * t
                pts.append((C + px * K, fy + py * K, C + 2.0 * K + pz * K))
            _swept(vol, pts, 0.95 * K, 0.45 * K, GREEN_D, soft=0.5 * K)
            _sphere(vol, pts[-1][0], pts[-1][1], pts[-1][2], 0.72 * K, GREEN, soft=0.4 * K)


def draw_claudeguy(phase=0.0, blink=0.0, look=(0.0, 0.0), petal_flex=None,
                   expression="neutral", spin=0.0, wiggle=0.0, sheet=None):
    """One volumetric frame. Faces +z. All animation inputs optional (static default).
    petal_flex: optional per-petal radial flex array (N_PETALS,) in [-1, 1].
    expression: key into EXPRESSIONS (overrides look; composes with blink/petal_flex)."""
    vol = np.zeros((GRID3 * SS,) * 3 + (4,), dtype=np.float32)
    if petal_flex is None:
        petal_flex = np.zeros(N_PETALS)
    ex = EXPRESSIONS[expression]
    look = ex["look"] if expression != "neutral" else look
    droop = SHEET_DROOP.get(sheet, 0.0) if sheet else ex["droop"]

    fy = 17.0 * K          # face center height (32-units: y=17)
    fz = C + 2.0 * K       # face plane pushed toward the viewer

    # --- Petal ring (XY plane, radiating around the face) -------------------
    for k in range(N_PETALS):
        ang = k * 2 * np.pi / N_PETALS + 0.13 * np.sin(k * 2.7)   # handmade jitter
        ang += spin * phase                                # whole ring rotates, 1 rev/cycle
        ang += wiggle * 0.3 * np.sin(phase * 2 + k * 1.1)  # back-and-forth waggle
        # deterministic per-petal irregularity: length and plumpness vary
        wob = np.sin(k * 4.9) * 0.5 + np.sin(k * 1.3) * 0.5
        length = (11.2 + 1.4 * wob + 1.2 * petal_flex[k]) * K
        r0 = (2.35 + 0.2 * np.sin(k * 3.1)) * K   # slowly tapering cylinder...
        r1 = (1.95 + 0.15 * np.cos(k * 2.2)) * K  # ...with a dome cap (the last sweep sphere)
        dx, dy = np.cos(ang), np.sin(ang)
        pts = []
        n = 10
        for i in range(n):
            t = i / (n - 1)
            r = (6.0 * K) + length * t
            # nearly straight petals; just a whisper of backward cup for depth
            z = fz - (1.4 * t * t + 0.25 * wob * t) * K
            sag = droop * 4.5 * t * t * K  # melancholy: every petal tip sinks
            pts.append((C + dx * r, fy + dy * r - sag, z))
        _swept(vol, pts, r0, r1, PETAL, soft=1.3 * K)
        # a slightly darker, thinner back layer gives the petals depth
        back = [(p[0], p[1], p[2] - 1.6 * K) for p in pts]
        _swept(vol, back, r0 * 0.85, r1 * 0.85, PETAL_DEEP, soft=1.2 * K)

    # --- Face: one clean circular disk (round from the front, domed in depth) --
    _ellipsoid(vol, C, fy, fz + 2.6 * K, 8.9 * K, 8.9 * K, 3.4 * K, FACE, soft=1.0 * K)

    if sheet is not None:
        _paint_sheet(vol, sheet)
        small = vol.reshape(GRID3, SS, GRID3, SS, GRID3, SS, 4).mean(axis=(1, 3, 5))
        small[..., :3] *= small[..., 3:4]
        return np.ascontiguousarray(small.transpose(2, 1, 0, 3)).astype(np.float32)

    # --- Eyes, per expression -------------------------------------------------
    eo = 1.0 - blink
    lx, ly = look
    if ex["eyes"] == "balls":
        _sphere(vol, C - 3.9 * K, fy + 2.4 * K, fz + 6.4 * K, 4.0 * K * (0.5 + 0.5 * eo),
                EYE_WHITE, soft=0.6 * K)
        _sphere(vol, C + 3.9 * K, fy + 3.0 * K, fz + 6.8 * K, 4.6 * K * (0.5 + 0.5 * eo),
                EYE_WHITE, soft=0.6 * K)
        if blink < 0.85:
            _sphere(vol, C - 3.9 * K + lx * K, fy + (2.2 + ly) * K, fz + 9.8 * K,
                    1.8 * ex["pupil"] * K, PUPIL, soft=0.4 * K)
            _sphere(vol, C + 3.9 * K + lx * K, fy + (2.8 + ly) * K, fz + 10.8 * K,
                    2.0 * ex["pupil"] * K, PUPIL, soft=0.4 * K)
    else:
        for side, ecx, ecy in ((-1, -3.9, 2.4), (1, 3.9, 3.0)):
            if ex["eyes"] == "happy":       # ^ ^  (upward arcs)
                arc = [(ecx + 2.2 * np.cos(a), ecy - 1.2 + 2.4 * np.sin(a))
                       for a in np.linspace(0.35, np.pi - 0.35, 11)]
            elif ex["eyes"] == "sadclosed":  # gentle downward-curved closed lids
                arc = [(ecx + 2.2 * np.cos(a), ecy + 1.4 - 2.2 * np.sin(a))
                       for a in np.linspace(0.5, np.pi - 0.5, 11)]
            elif ex["eyes"] == "squeeze":    # > <  (two crossing strokes)
                arc = [(ecx + side * t * 1.9, ecy + t * 1.6) for t in np.linspace(-1, 1, 9)]
                arc += [(ecx + side * t * 1.9, ecy - t * 1.6) for t in np.linspace(-1, 1, 9)]
            else:                            # spiral (dizzy) — 1.5 turns
                arc = [((ecx + (0.5 + 1.6 * s) * np.cos(3 * np.pi * s)),
                        (ecy + (0.5 + 1.6 * s) * np.sin(3 * np.pi * s)))
                       for s in np.linspace(0, 1, 17)]
            _stroke_on_face(vol, arc, 5.4, 0.55, PUPIL)

    # --- Brows ---------------------------------------------------------------
    if ex["brows"] != "none":
        slope = -1.0 if ex["brows"] == "angry" else 0.9   # inner end down=angry, up=plead
        for side, ecx in ((-1, -3.9), (1, 3.9)):
            # s runs outer(-1) -> inner(+1); inner means toward the face center
            brow = [(ecx - side * s * 1.9, 6.6 + slope * s) for s in np.linspace(-1, 1, 9)]
            _stroke_on_face(vol, brow, 5.2, 0.5, PUPIL)

    # --- Mouth, per expression ------------------------------------------------
    for t in np.linspace(-1.0, 1.0, 15):
        mx = 2.6 * t
        if ex["mouth"] == "w":
            my = -2.9 - 1.05 * abs(np.sin(np.pi * t))
        elif ex["mouth"] == "frown":
            my = -3.0 - 1.15 * t * t
        elif ex["mouth"] == "wavy":
            my = -3.4 + 0.5 * np.sin(3 * np.pi * t)
        else:  # flat (small, worried)
            mx = 1.6 * t
            my = -3.3
        _sphere(vol, C + mx * K, fy + my * K, fz + 8.0 * K, 0.6 * K, MOUTH, soft=0.35 * K)

    small = vol.reshape(GRID3, SS, GRID3, SS, GRID3, SS, 4).mean(axis=(1, 3, 5))
    small[..., :3] *= small[..., 3:4]
    return np.ascontiguousarray(small.transpose(2, 1, 0, 3)).astype(np.float32)


def _preview(v, path):
    from PIL import Image

    a = np.clip(v[..., 3], 0, 1)
    w = a / (a.sum(axis=0, keepdims=True) + 1e-6)
    rgb = (np.clip(v[..., :3], 0, 1) * w[..., None]).sum(axis=0)
    al = 1 - np.prod(1 - a * 0.9, axis=0)
    img = np.concatenate([rgb, al[..., None]], axis=-1)[::-1]
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), "RGBA").resize(
        (512, 512), Image.NEAREST).save(path)


if __name__ == "__main__":
    _preview(draw_claudeguy(), "claudeguy_front.png")
    # side view: project along x instead (transpose z<->x of the (z,y,x) volume)
    v = draw_claudeguy()
    _preview(np.ascontiguousarray(v.transpose(2, 1, 0, 3)), "claudeguy_side.png")
    _preview(draw_claudeguy(blink=1.0), "claudeguy_blink.png")
    print("wrote claudeguy_{front,side,blink}.png")
