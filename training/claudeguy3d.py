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


def _stroke_on_face(vol, pts2d, fz_off, r, color):
    """Sweep small spheres along 2D face-plane points [(x_off, y_off), ...]."""
    fy = 17.0 * K
    for (ox, oy) in pts2d:
        _sphere(vol, C + ox * K, fy + oy * K, C + 2.0 * K + fz_off * K, r * K, color, soft=0.3 * K)


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


def draw_claudeguy(phase=0.0, blink=0.0, look=(0.0, 0.0), petal_flex=None,
                   expression="neutral"):
    """One volumetric frame. Faces +z. All animation inputs optional (static default).
    petal_flex: optional per-petal radial flex array (N_PETALS,) in [-1, 1].
    expression: key into EXPRESSIONS (overrides look; composes with blink/petal_flex)."""
    vol = np.zeros((GRID3 * SS,) * 3 + (4,), dtype=np.float32)
    if petal_flex is None:
        petal_flex = np.zeros(N_PETALS)
    ex = EXPRESSIONS[expression]
    look = ex["look"] if expression != "neutral" else look
    droop = ex["droop"]

    fy = 17.0 * K          # face center height (32-units: y=17)
    fz = C + 2.0 * K       # face plane pushed toward the viewer

    # --- Petal ring (XY plane, radiating around the face) -------------------
    for k in range(N_PETALS):
        ang = k * 2 * np.pi / N_PETALS + 0.13 * np.sin(k * 2.7)   # handmade jitter
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
