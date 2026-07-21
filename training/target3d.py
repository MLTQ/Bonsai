"""Procedural voxel bonsai — the 3D NCA's first growth target.

Drawn at 2x supersample (64^3) with soft-edged primitives, box-filtered to
GRID3^3, RGBA where alpha is volumetric density (premultiplied RGB).
Axes: x right, y up, z toward viewer. Pot sits at low y.

Preview: python3 target3d.py writes max-intensity projections along each axis.
"""

import os

import numpy as np

# Grid edge is env-overridable so the same art/trainers scale: BONSAI_GRID3=64.
GRID3 = int(os.environ.get("BONSAI_GRID3", "32"))
SS = 2
S = GRID3 * SS  # supersampled edge
SCALE = GRID3 / 32.0  # magnitudes in the drawings are authored for 32^3


def _coords():
    return np.mgrid[0:S, 0:S, 0:S].astype(np.float32) / SS  # (3, S, S, S) in grid units


def _sphere(vol, cx, cy, cz, r, color, soft=0.8):
    """Bounded: the mask is zero outside d < r + soft, so compute only that box.

    The old full-canvas version allocated a fresh 3xS^3 mgrid (~25 MB at 64^3
    SS=2) per call and swept the whole canvas — hundreds of times per mood
    frame once faces arrived, which turned a corpus build into a 13-hour job.
    Same formula on index-identical sub-coordinates: bit-identical output.
    """
    reach = r + soft
    i0 = max(0, int(np.floor((cx - reach) * SS)))
    j0 = max(0, int(np.floor((cy - reach) * SS)))
    k0 = max(0, int(np.floor((cz - reach) * SS)))
    i1 = min(S, int(np.ceil((cx + reach) * SS)) + 1)
    j1 = min(S, int(np.ceil((cy + reach) * SS)) + 1)
    k1 = min(S, int(np.ceil((cz + reach) * SS)) + 1)
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return
    x, y, z = np.mgrid[i0:i1, j0:j1, k0:k1].astype(np.float32) / SS
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    m = np.clip((r - d) / soft, 0, 1)
    sub = vol[i0:i1, j0:j1, k0:k1]
    a = m[..., None]
    rgb = np.array(color, np.float32)[None, None, None, :]
    sub[..., 3:4] = np.maximum(sub[..., 3:4], a)
    np.copyto(sub[..., 0:3], rgb, where=a > 0.15)


def _swept(vol, pts, r0, r1, color, soft=0.6):
    """Sweep spheres of tapering radius along a polyline of 3D points."""
    n = len(pts)
    for i, p in enumerate(pts):
        t = i / max(n - 1, 1)
        _sphere(vol, p[0], p[1], p[2], r0 + (r1 - r0) * t, color, soft)


def _cone_pot(vol, cx, cy, cz, r_top, r_bot, h, color):
    x, y, z = _coords()
    t = np.clip((y - (cy - h)) / h, 0, 1)          # 0 at bottom, 1 at top
    r = r_bot + (r_top - r_bot) * t
    d = np.sqrt((x - cx) ** 2 + (z - cz) ** 2)
    m = np.clip((r - d) / 0.7, 0, 1) * ((y >= cy - h) & (y <= cy)).astype(np.float32)
    _composite(vol, m, color)


def _composite(vol, m, color):
    """Painter's-order composite: density maxes, color overwrites where this
    primitive is substantially present (matches the 2D generators' semantics)."""
    a = m[..., None]
    rgb = np.array(color, np.float32)[None, None, None, :]
    vol[..., 3:4] = np.maximum(vol[..., 3:4], a)
    np.copyto(vol[..., 0:3], rgb, where=a > 0.15)


def _bezier3(p0, p1, p2, n=36):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.array(p, np.float32) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def make_target3d():
    """(GRID3, GRID3, GRID3, 4) float32, y-up, premultiplied alpha.
    All magnitudes are authored for 32^3 and multiplied by SCALE, so the same
    tree renders at any grid size (64^3 gains fine branch/leaf detail)."""
    pot = (0.78, 0.43, 0.27)
    soil = (0.30, 0.20, 0.13)
    bark = (0.45, 0.30, 0.18)
    leaf = (0.22, 0.47, 0.22)
    leaf_hi = (0.39, 0.66, 0.31)

    vol = np.zeros((S, S, S, 4), dtype=np.float32)
    c = GRID3 / 2
    k = SCALE

    # Pot (truncated cone) + soil disk
    _cone_pot(vol, c, 9.0 * k, c, 6.5 * k, 4.5 * k, 5.0 * k, pot)
    _cone_pot(vol, c, 9.6 * k, c, 5.6 * k, 5.4 * k, 0.9 * k, soil)

    # Trunk: S-curve up with a side branch
    trunk = _bezier3((c + 0.5 * k, 9.5 * k, c), (c + 3.0 * k, 15.0 * k, c - 1.0 * k),
                     (c - 2.0 * k, 21.0 * k, c + 0.5 * k))
    _swept(vol, trunk, 1.9 * k, 0.9 * k, bark, soft=0.5 * k)
    branch = _bezier3((c + 1.2 * k, 14.0 * k, c - 0.5 * k), (c + 4.0 * k, 16.5 * k, c + 1.5 * k),
                      (c + 6.0 * k, 19.0 * k, c + 2.5 * k))
    _swept(vol, branch, 1.1 * k, 0.6 * k, bark, soft=0.5 * k)

    # Foliage: overlapping soft spheres, dark base + light crowns.
    # At SCALE >= 2 add a second tier of small leaf clusters (the fine detail
    # 64^3 exists to show off).
    _sphere(vol, c - 2.5 * k, 23.5 * k, c + 0.5 * k, 4.8 * k, leaf, soft=1.4 * k)
    _sphere(vol, c + 6.0 * k, 20.5 * k, c + 2.5 * k, 3.4 * k, leaf, soft=1.2 * k)
    _sphere(vol, c - 6.0 * k, 21.0 * k, c - 1.5 * k, 2.8 * k, leaf, soft=1.1 * k)
    _sphere(vol, c - 1.0 * k, 25.5 * k, c, 3.2 * k, leaf_hi, soft=1.2 * k)
    _sphere(vol, c + 5.0 * k, 22.0 * k, c + 3.0 * k, 2.0 * k, leaf_hi, soft=1.0 * k)
    if SCALE >= 2.0:
        twig = _bezier3((c - 2.0 * k, 20.0 * k, c), (c - 5.0 * k, 22.0 * k, c - 2.0 * k),
                        (c - 7.5 * k, 23.5 * k, c - 3.0 * k))
        _swept(vol, twig, 0.9, 0.4, bark, soft=0.5)
        for (dx, dy, dz, r) in ((-7.5, 24.5, -3.0, 1.6), (-4.0, 26.5, 1.0, 1.4),
                                (2.5, 27.0, -1.5, 1.5), (7.5, 22.5, 3.5, 1.3),
                                (-8.5, 20.0, -2.0, 1.2)):
            _sphere(vol, c + dx * k, dy * k, c + dz * k, r * k * 0.55, leaf_hi, soft=1.0)

    # Downsample 2x (box filter), premultiply
    small = vol.reshape(GRID3, SS, GRID3, SS, GRID3, SS, 4).mean(axis=(1, 3, 5))
    small[..., :3] *= small[..., 3:4]
    # Index order: currently (x, y, z, c) from mgrid; standardize to (z, y, x, c)
    return np.ascontiguousarray(small.transpose(2, 1, 0, 3)).astype(np.float32)


if __name__ == "__main__":
    from PIL import Image

    t = make_target3d()
    for axis, name in [(0, "front_zy"), (1, "top"), (2, "side")]:
        # weighted projection along axis: emission-absorption-ish quick look
        alpha = t[..., 3]
        rgb = t[..., :3]
        w = alpha / (alpha.sum(axis=axis, keepdims=True) + 1e-6)
        proj_rgb = (rgb * w[..., None]).sum(axis=axis)
        proj_a = 1 - np.prod(1 - np.clip(alpha, 0, 1) * 0.9, axis=axis)
        img = np.concatenate([proj_rgb, proj_a[..., None]], axis=-1)
        if axis != 1:
            img = img[::-1]  # y-up for display
        png = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(png, "RGBA").resize((GRID3 * 8, GRID3 * 8), Image.NEAREST).save(
            f"target3d_{name}.png")
    print("wrote target3d_{front_zy,top,side}.png")
