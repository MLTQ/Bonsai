"""Procedural animation frames for the latent-space shoggoth.

Two behaviors x 12 phases, 64x64 RGBA premultiplied:
  idle — body breathes, tentacles sway independently, eyes blink out of phase
  walk — a traveling wave runs through the bottom tentacles (locomotion gait),
         body bobs and leans into the direction of travel (rightward; the app
         mirrors the render when it walks left)

Design: dark iridescent mass, too many amber eyes, and a small rigid smiley
mask stuck on the upper right — the one part of it that never moves.
"""

import numpy as np

GRID = 64
SS = 4
FRAMES = 12
BEHAVIORS = 2  # 0 = idle, 1 = walk

BODY = (0.15, 0.11, 0.20)
BODY_HI = (0.27, 0.19, 0.38)
SHEEN = (0.16, 0.30, 0.34)
EYE_WHITE = (0.93, 0.90, 0.84)
IRIS = (0.85, 0.47, 0.30)
PUPIL = (0.12, 0.05, 0.04)
MASK = (0.96, 0.92, 0.83)
MASK_FACE = (0.24, 0.16, 0.12)


def _grid(s):
    return np.mgrid[0:GRID * s, 0:GRID * s]


def _disk(c, cx, cy, r, color, s):
    yy, xx = _grid(s)
    m = (xx - cx * s) ** 2 + (yy - cy * s) ** 2 <= (r * s) ** 2
    c[m] = (*color, 1.0)


def _blob(c, cx, cy, rx, ry, color, s, wobble=0.0, phase=0.0):
    """Ellipse with a sinusoidal edge wobble — the breathing, uneven body."""
    yy, xx = _grid(s)
    ang = np.arctan2(yy - cy * s, xx - cx * s)
    edge = 1.0 + wobble * np.sin(3 * ang + phase) + 0.5 * wobble * np.sin(5 * ang - phase)
    m = ((xx - cx * s) / (rx * s)) ** 2 + ((yy - cy * s) / (ry * s)) ** 2 <= edge
    c[m] = (*color, 1.0)


def _tentacle(c, x0, y0, length, base_angle, curl, width, color, s):
    """Tapered tentacle: polyline of disks whose direction curls along its length.
    base_angle in radians (0 = down), curl bends the tip sideways."""
    n = 26
    x, y = float(x0), float(y0)
    for i in range(n):
        t = i / (n - 1)
        ang = base_angle + curl * t
        x += (length / n) * np.sin(ang)
        y += (length / n) * np.cos(ang)
        _disk(c, x, y, width * (1.0 - 0.75 * t), color, s)


def _eye(c, cx, cy, r, blink, look_dx, s):
    """blink: 0 open .. 1 closed (drawn as body-colored lid shrinking the white)."""
    if blink >= 0.95:
        _disk(c, cx, cy, r, BODY_HI, s)
        return
    _disk(c, cx, cy, r, EYE_WHITE, s)
    open_r = r * (1.0 - 0.7 * blink)
    _disk(c, cx + look_dx, cy, open_r * 0.62, IRIS, s)
    _disk(c, cx + look_dx, cy, open_r * 0.30, PUPIL, s)


def draw_shoggoth(phase, walking):
    """One frame. phase in [0, 2pi)."""
    s = SS
    c = np.zeros((GRID * s, GRID * s, 4), dtype=np.float32)

    # NOTE: every frequency here must be an integer multiple of the base cycle,
    # or the loop doesn't close and the NCA learns a lurch at the frame-11→0 seam.
    bob = (1.6 * abs(np.sin(phase))) if walking else (0.7 * np.sin(phase))
    lean = 2.0 * np.sin(phase + 1.0) if walking else 0.0
    cy = 30.0 - bob
    cx = 32.0 + lean * 0.4

    # Bottom tentacles: the gait. Walking = traveling wave (phase offset per
    # tentacle); idle = slow independent sway.
    bases = [(-13, 1.0), (-8, 1.15), (-3, 1.3), (3, 1.3), (8, 1.15), (13, 1.0)]
    for k, (dx, lscale) in enumerate(bases):
        if walking:
            curl = 1.5 * np.sin(phase - k * (2 * np.pi / len(bases)))
            base_ang = 0.25 * np.sin(phase - k * (2 * np.pi / len(bases)) + 0.8)
        else:
            curl = 0.8 * np.sin(phase * (1 + k % 2) + k * 1.7)
            base_ang = 0.12 * np.sin(phase + k * 2.3)
        _tentacle(c, cx + dx, cy + 9, 14 * lscale, base_ang, curl, 2.6, BODY, s)

    # Two upper tentacles, always idly curling
    _tentacle(c, cx - 11, cy - 6, 11, -1.9, 1.1 * np.sin(phase + 0.5), 2.0, BODY, s)
    _tentacle(c, cx + 12, cy - 4, 9, 1.8, -1.0 * np.sin(phase * 2 + 2.0), 1.8, BODY, s)

    # Body mass: breathing wobble, iridescent patches, teal sheen
    breathe = 0.06 + (0.02 * np.sin(phase) if not walking else 0.0)
    _blob(c, cx, cy, 14.5, 12.0, BODY, s, wobble=breathe, phase=phase)
    _blob(c, cx - 4, cy - 3, 7.5, 5.5, BODY_HI, s, wobble=0.12, phase=phase * 2)
    _blob(c, cx + 6, cy + 4, 4.5, 3.2, SHEEN, s, wobble=0.15, phase=-phase)
    _blob(c, cx - 8, cy + 5, 3.2, 2.4, BODY_HI, s, wobble=0.1, phase=phase + 2)

    # Eyes: scattered, each blinking on its own schedule; look ahead when walking
    look = 1.0 if walking else 0.4 * np.sin(phase)
    eyes = [(-7, -5, 2.6, 0.0), (2, -7, 1.9, 2.1), (7, 0, 2.2, 4.2),
            (-2, 2, 1.5, 1.1), (-10, 1, 1.7, 3.4), (4, 5, 1.3, 5.3)]
    for ex, ey, er, boff in eyes:
        b = np.clip(3.5 * np.sin(phase + boff) - 2.6, 0, 1)  # brief staggered blinks
        _eye(c, cx + ex, cy + ey, er, b, look * 0.6, s)

    # The mask: rigid, slightly tilted, never animates. The joke must hold still.
    mx, my = cx + 10.5, cy - 7.5
    _disk(c, mx, my, 4.2, MASK, s)
    _disk(c, mx - 1.4, my - 1.0, 0.55, MASK_FACE, s)
    _disk(c, mx + 1.4, my - 1.2, 0.55, MASK_FACE, s)
    for t in np.linspace(-0.9, 0.9, 11):  # smile arc
        _disk(c, mx + t * 1.9, my + 0.9 + 0.9 * (1 - (t / 0.95) ** 2), 0.42, MASK_FACE, s)

    small = c.reshape(GRID, SS, GRID, SS, 4).mean(axis=(1, 3))
    small[..., :3] *= small[..., 3:4]
    return small.astype(np.float32)


def make_frames():
    """Returns (BEHAVIORS, FRAMES, GRID, GRID, 4) float32."""
    out = np.zeros((BEHAVIORS, FRAMES, GRID, GRID, 4), dtype=np.float32)
    for f in range(FRAMES):
        phase = 2 * np.pi * f / FRAMES
        out[0, f] = draw_shoggoth(phase, walking=False)
        out[1, f] = draw_shoggoth(phase, walking=True)
    return out


if __name__ == "__main__":
    from PIL import Image

    frames = make_frames()
    sheet = np.zeros((BEHAVIORS * GRID, FRAMES * GRID, 4), dtype=np.float32)
    for b in range(BEHAVIORS):
        for f in range(FRAMES):
            sheet[b * GRID:(b + 1) * GRID, f * GRID:(f + 1) * GRID] = frames[b, f]
    img = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").resize(
        (FRAMES * GRID * 2, BEHAVIORS * GRID * 2), Image.NEAREST
    ).save("shoggoth_sheet.png")
    print("wrote shoggoth_sheet.png")
