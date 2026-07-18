"""Procedural animation frames for the Lain-esque face creature.

Two behaviors x 12 phases, 64x64 RGBA premultiplied:
  still   — vacant stare, slow 2-frame blink, one pupil saccade
  talking — mouth cycle over the 12 phases, same blink

The face: pale skin, grey-brown bob with side locks (asymmetric lengths),
the X hair clip on the viewer-left bang, slightly mismatched eyes with an
off-center gaze. Muted, desaturated, a little wrong on purpose.
"""

import numpy as np

GRID = 64
SS = 4
FRAMES = 12
BEHAVIORS = 2  # 0 = still, 1 = talking

SKIN = (0.86, 0.81, 0.77)
SKIN_SHADOW = (0.72, 0.63, 0.60)
HAIR = (0.32, 0.26, 0.24)
HAIR_HI = (0.43, 0.36, 0.33)
EYE_WHITE = (0.90, 0.90, 0.86)
IRIS = (0.30, 0.16, 0.15)
PUPIL = (0.09, 0.05, 0.05)
BROW = (0.27, 0.21, 0.19)
LIP = (0.60, 0.41, 0.39)
MOUTH_DARK = (0.20, 0.09, 0.10)
CLIP = (0.76, 0.79, 0.81)


def _grid(s):
    return np.mgrid[0:GRID * s, 0:GRID * s]  # yy, xx


def _ellipse(c, cx, cy, rx, ry, color, s):
    yy, xx = _grid(s)
    m = ((xx - cx * s) / (rx * s)) ** 2 + ((yy - cy * s) / (ry * s)) ** 2 <= 1.0
    c[m] = (*color, 1.0)


def _rect(c, x0, y0, x1, y1, color, s):
    yy, xx = _grid(s)
    m = (xx >= x0 * s) & (xx < x1 * s) & (yy >= y0 * s) & (yy < y1 * s)
    c[m] = (*color, 1.0)


def _tri_down(c, cx, ytop, w, h, color, s):
    """Downward-pointing triangle (bang fringe spikes)."""
    yy, xx = _grid(s)
    t = (yy - ytop * s) / (h * s)
    m = (t >= 0) & (t <= 1) & (np.abs(xx - cx * s) <= (1 - t) * (w / 2) * s)
    c[m] = (*color, 1.0)


def _line(c, x0, y0, x1, y1, width, color, s):
    yy, xx = _grid(s)
    n = 60
    for t in np.linspace(0, 1, n):
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        m = (xx - px * s) ** 2 + (yy - py * s) ** 2 <= (width * s / 2) ** 2
        c[m] = (*color, 1.0)


def _eye(c, cx, cy, w, h, pupil_dx, lid, s):
    """One eye: white, iris+pupil (gaze offset), then eyelid closing from the top."""
    _ellipse(c, cx, cy, w / 2, h / 2, EYE_WHITE, s)
    ix = cx + pupil_dx - 0.5  # resting gaze sits slightly off-center
    _ellipse(c, ix, cy, 0.95, h / 2 * 0.8, IRIS, s)
    _ellipse(c, ix, cy + 0.15, 0.45, 0.7, PUPIL, s)
    if lid > 0:
        cover = cy - h / 2 + h * lid
        _rect(c, cx - w / 2 - 0.5, cy - h / 2 - 0.5, cx + w / 2 + 0.5, cover, SKIN, s)
        _rect(c, cx - w / 2 - 0.5, cover - 0.6, cx + w / 2 + 0.5, cover, HAIR, s)  # lash line


def draw_face(mouth="closed", lid=0.0, pupil_dx=0.0):
    """One frame. mouth: closed|half|open|wide. lid: 0 open .. 1 closed."""
    s = SS
    c = np.zeros((GRID * s, GRID * s, 4), dtype=np.float32)

    # Back hair + side locks (viewer-left lock hangs lower: asymmetry)
    _ellipse(c, 32, 26, 16, 16.5, HAIR, s)
    _ellipse(c, 18.5, 36, 3.4, 12.0, HAIR, s)
    _ellipse(c, 45.5, 34, 3.4, 10.0, HAIR, s)
    # Face
    _ellipse(c, 32, 31, 10.0, 11.0, SKIN, s)
    # Bangs with ragged fringe
    _ellipse(c, 32, 21, 13.0, 8.5, HAIR, s)
    for fx, fh in ((23, 3.2), (27, 2.4), (31.5, 3.6), (36, 2.2), (40, 3.0)):
        _tri_down(c, fx, 26.2, 3.4, fh, HAIR, s)
    # Hair highlight
    _line(c, 22, 15, 30, 12.5, 1.1, HAIR_HI, s)
    # The clip: small X on the viewer-left bang
    _line(c, 21.4, 22.4, 24.6, 25.6, 0.9, CLIP, s)
    _line(c, 24.6, 22.4, 21.4, 25.6, 0.9, CLIP, s)
    # Brows: thin, flat, mismatched heights
    _rect(c, 22, 27.4, 27.5, 28.2, BROW, s)
    _rect(c, 36.5, 26.8, 42.5, 27.6, BROW, s)
    # Eyes: right one a touch wider (unsettling on purpose)
    _eye(c, 25, 31, 5.8, 4.1, pupil_dx, lid, s)
    _eye(c, 39, 31, 6.6, 4.3, pupil_dx, lid, s)
    # Under-eye shadows, nose
    _rect(c, 22.5, 33.4, 27.5, 34.0, SKIN_SHADOW, s)
    _rect(c, 36.5, 33.6, 41.5, 34.2, SKIN_SHADOW, s)
    _ellipse(c, 32.2, 35.8, 0.8, 0.6, SKIN_SHADOW, s)

    # Mouth
    if mouth == "closed":
        _rect(c, 29.5, 39.6, 34.5, 40.4, LIP, s)
    elif mouth == "half":
        _ellipse(c, 32, 40, 2.5, 1.1, LIP, s)
        _ellipse(c, 32, 40, 1.6, 0.6, MOUTH_DARK, s)
    elif mouth == "open":
        _ellipse(c, 32, 40.2, 2.3, 1.8, LIP, s)
        _ellipse(c, 32, 40.2, 1.6, 1.2, MOUTH_DARK, s)
    elif mouth == "wide":
        _ellipse(c, 32, 40.4, 3.0, 2.3, LIP, s)
        _ellipse(c, 32, 40.4, 2.2, 1.6, MOUTH_DARK, s)

    small = c.reshape(GRID, SS, GRID, SS, 4).mean(axis=(1, 3))
    small[..., :3] *= small[..., 3:4]
    return small.astype(np.float32)


# Per-phase specs: (mouth, lid, pupil_dx). Blink lives at phases 8-9 in both behaviors.
STILL = [
    ("closed", 0.0, 0.0), ("closed", 0.0, 0.0), ("closed", 0.0, 0.0),
    ("closed", 0.0, -1.2), ("closed", 0.0, -1.2), ("closed", 0.0, 0.0),
    ("closed", 0.0, 0.0), ("closed", 0.0, 0.0), ("closed", 1.0, 0.0),
    ("closed", 1.0, 0.0), ("closed", 0.0, 0.0), ("closed", 0.0, 0.0),
]
TALK = [
    ("closed", 0.0, 0.0), ("half", 0.0, 0.0), ("open", 0.0, 0.0),
    ("half", 0.0, 0.0), ("wide", 0.0, 0.0), ("open", 0.0, 0.0),
    ("closed", 0.0, 0.0), ("half", 0.0, 0.0), ("open", 1.0, 0.0),
    ("wide", 1.0, 0.0), ("half", 0.0, 0.0), ("closed", 0.0, 0.0),
]


def make_frames():
    """Returns (BEHAVIORS, FRAMES, GRID, GRID, 4) float32."""
    out = np.zeros((BEHAVIORS, FRAMES, GRID, GRID, 4), dtype=np.float32)
    for f, spec in enumerate(STILL):
        out[0, f] = draw_face(*spec)
    for f, spec in enumerate(TALK):
        out[1, f] = draw_face(*spec)
    return out


if __name__ == "__main__":
    from PIL import Image

    frames = make_frames()
    scale = 3
    sheet = np.zeros((BEHAVIORS * GRID, FRAMES * GRID, 4), dtype=np.float32)
    for b in range(BEHAVIORS):
        for f in range(FRAMES):
            sheet[b * GRID:(b + 1) * GRID, f * GRID:(f + 1) * GRID] = frames[b, f]
    img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").resize(
        (FRAMES * GRID * scale, BEHAVIORS * GRID * scale), Image.NEAREST
    ).save("lain_sheet.png")
    print("wrote lain_sheet.png")
