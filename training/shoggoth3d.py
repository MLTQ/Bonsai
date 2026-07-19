"""Shoggoth Mk. III — volumetric animation frames for the 3D cyclic NCA (rung 2).

12 phases x 2 behaviors at GRID3^3, y-up, (z, y, x, c) order:
  idle — the mass churns (four internal lobes orbit slowly), tentacles sway,
         eyes distributed over the whole surface blink on staggered schedules
  walk — an azimuthal traveling wave runs around the tentacle ring (it churns
         forward rather than striding) with a body bob

Same loop-closure discipline as every generator in this repo: all time
frequencies are integer multiples of the base cycle.
"""

import numpy as np

from target3d import GRID3, SCALE, SS, _composite, _coords, _sphere, _swept

FRAMES = 12
BEHAVIORS = 2
K = SCALE  # anatomy is authored in 32^3 units; K scales it to the active grid

BODY = (0.15, 0.11, 0.20)
BODY_HI = (0.27, 0.19, 0.38)
SHEEN = (0.16, 0.30, 0.34)
EYE_WHITE = (0.93, 0.90, 0.84)
IRIS = (0.85, 0.47, 0.30)
MASK = (0.96, 0.92, 0.83)
MASK_FACE = (0.24, 0.16, 0.12)

C = GRID3 / 2.0            # 16: lateral center (x and z)
BODY_Y = 19.0              # body center height
N_TENT = 8

# Eye placement: (azimuth, elevation, size, blink offset) on the body surface.
EYES = [(0.3, 0.5, 1.6, 0.0), (1.2, 0.2, 1.2, 2.1), (2.2, 0.6, 1.4, 4.2),
        (3.1, 0.1, 1.1, 1.1), (4.0, 0.45, 1.5, 3.4), (4.9, 0.15, 1.2, 5.3),
        (5.7, 0.55, 1.3, 2.8), (1.7, -0.25, 1.1, 0.9), (3.7, -0.2, 1.2, 4.8)]


def draw3d(phase, walking):
    vol = np.zeros((GRID3 * SS,) * 3 + (4,), dtype=np.float32)

    bob = (1.4 * abs(np.sin(phase)) if walking else 0.5 * np.sin(phase)) * K
    by = BODY_Y * K - bob

    # Body: central mass + four churning lobes orbiting at one cycle per loop
    _sphere(vol, C, by, C, 8.0 * K, BODY, soft=1.6 * K)
    for k in range(4):
        ang = phase + k * (np.pi / 2)
        lx = C + 4.5 * K * np.cos(ang)
        lz = C + 4.5 * K * np.sin(ang)
        ly = by + 2.0 * K * np.sin(phase * 2 + k * 1.9)
        _sphere(vol, lx, ly, lz, 4.2 * K, BODY, soft=1.5 * K)
    # Iridescent patches riding the churn
    _sphere(vol, C - 3.0 * K * np.cos(phase), by + 3.0 * K, C + 3.0 * K * np.sin(phase),
            3.0 * K, BODY_HI, soft=1.4 * K)
    _sphere(vol, C + 4.0 * K * np.cos(phase + 2.0), by - 1.0 * K,
            C - 4.0 * K * np.sin(phase + 2.0), 2.2 * K, SHEEN, soft=1.2 * K)

    # Tentacle ring: azimuthal traveling wave when walking, independent sway idle
    for k in range(N_TENT):
        az = k * 2 * np.pi / N_TENT
        if walking:
            curl = 1.3 * np.sin(phase - az)          # wave travels around the ring
            sway = 0.5 * np.cos(phase - az)
        else:
            curl = 0.7 * np.sin(phase * (1 + k % 2) + k * 1.7)
            sway = 0.3 * np.sin(phase + k * 2.3)
        pts = []
        length = 9.0 * K
        n = 8 if K < 2 else 12   # more sweep segments at higher res: smoother arms
        for i in range(n):
            t = i / (n - 1)
            # descend from body underside, curling radially and swaying tangentially
            rad = (6.0 + curl * 2.2 * t * t) * K
            a = az + sway * 0.35 * t
            pts.append((C + rad * np.cos(a), by - 5.0 * K - length * t, C + rad * np.sin(a)))
        _swept(vol, [(p[0], p[1], p[2]) for p in pts], 1.7 * K, 0.7 * K, BODY, soft=0.9 * K)

    # Eyes distributed over the surface, blinking staggered; look along +z when walking
    look = 0.6 if walking else 0.0
    for az, el, size, boff in EYES:
        blink = np.clip(3.5 * np.sin(phase + boff) - 2.6, 0, 1)
        r_surf = 8.9 * K  # bulge proud of the body surface so they survive projection
        ex = C + r_surf * np.cos(el) * np.cos(az)
        ey = by + r_surf * np.sin(el)
        ez = C + r_surf * np.cos(el) * np.sin(az)
        if blink >= 0.9:
            _sphere(vol, ex, ey, ez, size * 0.9 * K, BODY_HI, soft=0.6 * K)
        else:
            _sphere(vol, ex, ey, ez, size * K, EYE_WHITE, soft=0.5 * K)
            # iris pushed slightly outward along the surface normal (+ look bias in z)
            _sphere(vol, ex + 0.55 * K * np.cos(el) * np.cos(az),
                    ey + 0.55 * K * np.sin(el),
                    ez + 0.55 * K * np.cos(el) * np.sin(az) + look * 0.4 * K,
                    size * 0.55 * K, IRIS, soft=0.4 * K)

    # The mask: rigid pale disk on the +z face. It does not churn. It never churns.
    mz = C + 8.6 * K
    my = by + 2.5 * K
    _sphere(vol, C + 3.0 * K, my, mz, 2.9 * K, MASK, soft=0.5 * K)
    _sphere(vol, C + 2.0 * K, my + 0.8 * K, mz + 1.2 * K, 0.5 * K, MASK_FACE, soft=0.3 * K)
    _sphere(vol, C + 4.0 * K, my + 0.8 * K, mz + 1.2 * K, 0.5 * K, MASK_FACE, soft=0.3 * K)
    for t in np.linspace(-0.8, 0.8, 7 if K < 2 else 11):
        _sphere(vol, C + (3.0 + t * 1.4) * K, my + (-0.9 + 0.5 * (1 - (t / 0.85) ** 2)) * K,
                mz + 1.3 * K, 0.35 * K, MASK_FACE, soft=0.3 * K)

    small = vol.reshape(GRID3, SS, GRID3, SS, GRID3, SS, 4).mean(axis=(1, 3, 5))
    small[..., :3] *= small[..., 3:4]
    return np.ascontiguousarray(small.transpose(2, 1, 0, 3)).astype(np.float32)


def make_frames3d():
    """(BEHAVIORS, FRAMES, GRID3, GRID3, GRID3, 4) float16."""
    out = np.zeros((BEHAVIORS, FRAMES) + (GRID3,) * 3 + (4,), dtype=np.float16)
    for f in range(FRAMES):
        ph = 2 * np.pi * f / FRAMES
        out[0, f] = draw3d(ph, walking=False)
        out[1, f] = draw3d(ph, walking=True)
    return out


if __name__ == "__main__":
    from PIL import Image

    for name, (ph, walking) in {"idle0": (0, False), "walk3": (np.pi / 2, True),
                                "walk6": (np.pi, True)}.items():
        t = draw3d(ph, walking)
        alpha = np.clip(t[..., 3], 0, 1)
        w = alpha / (alpha.sum(axis=0, keepdims=True) + 1e-6)
        rgb = (np.clip(t[..., :3], 0, 1) * w[..., None]).sum(axis=0)
        a = 1 - np.prod(1 - alpha * 0.9, axis=0)
        img = np.concatenate([rgb, a[..., None]], axis=-1)[::-1]
        png = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(png, "RGBA").resize((GRID3 * 8,) * 2, Image.NEAREST).save(
            f"shog3d_{name}.png")
    print("wrote shog3d_{idle0,walk3,walk6}.png")
