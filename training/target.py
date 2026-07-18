"""Procedurally draws the pixel-art bonsai target the NCA is trained to grow."""

import numpy as np

GRID = 64  # final target resolution (matches the NCA grid)
SS = 4     # supersampling factor for soft anti-aliased edges


def _disk(mask_layers, cx, cy, r, color, s):
    """Stamp a filled disk onto the supersampled RGBA canvas."""
    canvas, = mask_layers
    yy, xx = np.mgrid[0:GRID * s, 0:GRID * s]
    d = (xx - cx * s) ** 2 + (yy - cy * s) ** 2 <= (r * s) ** 2
    canvas[d] = (*color, 1.0)


def _quad_bezier(p0, p1, p2, t):
    return ((1 - t) ** 2)[:, None] * p0 + (2 * (1 - t) * t)[:, None] * p1 + (t ** 2)[:, None] * p2


def _stroke(canvas, p0, p1, p2, w0, w1, color, s):
    """Stamp a tapered quadratic-bezier stroke (used for trunk and branches)."""
    t = np.linspace(0, 1, 200)
    pts = _quad_bezier(np.array(p0, float), np.array(p1, float), np.array(p2, float), t)
    widths = w0 + (w1 - w0) * t
    yy, xx = np.mgrid[0:GRID * s, 0:GRID * s]
    for (px, py), w in zip(pts, widths):
        d = (xx - px * s) ** 2 + (yy - py * s) ** 2 <= (w * s / 2) ** 2
        canvas[d] = (*color, 1.0)


def make_target():
    """Returns the bonsai as a (GRID, GRID, 4) float32 RGBA array, premultiplied alpha."""
    s = SS
    canvas = np.zeros((GRID * s, GRID * s, 4), dtype=np.float32)

    pot = (0.78, 0.43, 0.27)
    pot_dark = (0.58, 0.30, 0.19)
    soil = (0.30, 0.20, 0.13)
    bark = (0.45, 0.30, 0.18)
    leaf = (0.22, 0.47, 0.22)
    leaf_hi = (0.39, 0.66, 0.31)

    # Pot: tapering trapezoid with a wider rim
    yy, xx = np.mgrid[0:GRID * s, 0:GRID * s]
    for row in range(49, 57):
        half = 10 - (row - 49) * 0.55
        band = (yy >= row * s) & (yy < (row + 1) * s) & (np.abs(xx - 32 * s) <= half * s)
        canvas[band] = (*pot, 1.0)
    rim = (yy >= 47 * s) & (yy < 49 * s) & (np.abs(xx - 32 * s) <= 11.5 * s)
    canvas[rim] = (*pot_dark, 1.0)
    soil_band = (yy >= 46 * s) & (yy < 47 * s) & (np.abs(xx - 32 * s) <= 9.5 * s)
    canvas[soil_band] = (*soil, 1.0)

    # Trunk: S-curve up and to the left, plus one branch to the right
    _stroke(canvas, (33, 47), (37, 36), (27, 25), 4.0, 1.8, bark, s)
    _stroke(canvas, (31, 38), (36, 34), (41, 28), 2.2, 1.2, bark, s)

    # Foliage: overlapping clouds, dark base with lighter crowns
    _disk([canvas], 26, 18, 8.0, leaf, s)
    _disk([canvas], 41, 25, 5.5, leaf, s)
    _disk([canvas], 18, 25, 4.5, leaf, s)
    _disk([canvas], 24, 15, 5.5, leaf_hi, s)
    _disk([canvas], 31, 20, 4.0, leaf_hi, s)
    _disk([canvas], 40, 23, 3.0, leaf_hi, s)

    # Downsample (box filter) → anti-aliased edges, then premultiply RGB by alpha
    small = canvas.reshape(GRID, s, GRID, s, 4).mean(axis=(1, 3))
    small[..., :3] *= small[..., 3:4]
    return small.astype(np.float32)


if __name__ == "__main__":
    from PIL import Image

    t = make_target()
    rgba = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA").resize((GRID * 4, GRID * 4), Image.NEAREST)
    img.save("target_preview.png")
    print("wrote target_preview.png")
