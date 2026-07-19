"""Parametric volumetric shoggoth: the 3D manifold corpus (drama pass included).

draw3d(phase, z) renders a closed 12-frame volumetric cycle for ANY z in the
unit box. Compared to the fixed Mk. III art, motion ranges are 2-3x larger and
the body gets real squash-and-stretch — subtlety was the complaint; the `amp`
axis now goes to thrash.

Z_SPEC order is the format contract (Swift and the projector index by position).

Corpus: python3 manifold_shoggoth3d.py --n 1024 --out corpus_shoggoth3d.npz
Draws at SS=1 (soft-falloff primitives supply the anti-aliasing) so a 12k-volume
corpus generates in minutes, not hours.
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

import shoggoth3d as base
from target3d import GRID3

FRAMES = 12
ZDIM = 10

Z_SPEC = [
    ("walkness", 0.0, 1.0),   # idle churn .. traveling-wave gait
    ("amp",      0.3, 2.6),   # motion amplitude — top of range THRASHES
    ("droop",    0.0, 1.0),   # limp sagging tentacles, slumped body
    ("splay",    0.75, 1.35), # tentacle ring radius / body width
    ("bob",      0.0, 3.5),   # vertical bounce (voxels)
    ("churn",    0.3, 2.0),   # lobe orbit vigor (integer harmonics inside)
    ("eye_open", 0.1, 1.0),   # baseline eyelids
    ("bright",   0.65, 1.3),  # highlight/iris brightness
    ("jitter",   0.0, 1.0),   # 3rd-harmonic tremor
    ("squash",   0.0, 1.0),   # squash-and-stretch intensity
]

ANCHORS = {
    #             walk amp  droop splay bob  churn eye  brt  jit  sqsh
    "idle":      [0.0, 0.35, 0.15, 0.5, 0.15, 0.4, 0.8, 0.55, 0.05, 0.3],
    "walk":      [1.0, 0.5,  0.1,  0.5, 0.45, 0.5, 0.9, 0.6,  0.05, 0.5],
    "sleep":     [0.0, 0.08, 0.9,  0.4, 0.02, 0.15, 0.05, 0.3, 0.0, 0.1],
    "dread":     [0.0, 0.3,  0.75, 0.3, 0.1,  0.3, 0.95, 0.2, 0.5, 0.2],
    "manic":     [0.7, 1.0,  0.0,  1.0, 0.9,  1.0, 1.0, 1.0, 0.85, 1.0],
    "curious":   [0.25, 0.45, 0.1, 0.7, 0.3, 0.5, 1.0, 0.75, 0.1, 0.6],
    "content":   [0.0, 0.3,  0.25, 0.6, 0.2, 0.35, 0.6, 0.85, 0.0, 0.4],
    "agitated":  [0.8, 0.85, 0.05, 0.85, 0.75, 0.9, 1.0, 0.6, 0.7, 0.8],
}


def _p(z):
    return {name: lo + float(zi) * (hi - lo) for (name, lo, hi), zi in zip(Z_SPEC, z)}


# --- Direct-resolution primitives (SS=1; soft falloffs do the anti-aliasing) ---
_XYZ = np.mgrid[0:GRID3, 0:GRID3, 0:GRID3].astype(np.float32)  # (3: x, y, z)


def _sphere(vol, cx, cy, cz, r, color, soft=0.8):
    x, y, z = _XYZ
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    m = np.clip((r - d) / soft, 0, 1)[..., None]
    vol[..., 3:4] = np.maximum(vol[..., 3:4], m)
    np.copyto(vol[..., 0:3], np.array(color, np.float32), where=m > 0.15)


def _swept(vol, pts, r0, r1, color, soft=0.6):
    n = len(pts)
    for i, p in enumerate(pts):
        t = i / max(n - 1, 1)
        _sphere(vol, p[0], p[1], p[2], r0 + (r1 - r0) * t, color, soft)


def draw3d(phase, z):
    """One volumetric frame at `phase` for factor vector z, drawn at 32^3 directly.
    Axes here are (x, y, z) during drawing; transposed to (z, y, x, c) on return."""
    p = _p(z)
    w = p["walkness"]
    C = base.C

    vol = np.zeros((GRID3,) * 3 + (4,), dtype=np.float32)

    bob = p["bob"] * (w * abs(np.sin(phase)) + (1 - w) * 0.5 * np.sin(phase))
    # Squash-and-stretch: compress when low (bob min), stretch when rising
    sq = 1.0 + p["squash"] * 0.22 * np.sin(phase * 2)
    droop_dy = 3.0 * p["droop"]
    by = base.BODY_Y - bob + droop_dy * 0.6

    bright = p["bright"]
    def shade(c, k=bright):
        return tuple(min(1.0, ch * k) for ch in c)

    # Body: central mass + churning lobes, squashed/stretched
    _sphere(vol, C, by, C, 8.0 * p["splay"] / sq**0.5, base.BODY, soft=1.6)
    for k in range(4):
        ang = phase * max(1, round(p["churn"])) + k * (np.pi / 2)
        r_orbit = 4.5 * p["splay"] * (0.6 + 0.4 * p["churn"])
        lx = C + r_orbit * np.cos(ang)
        lz = C + r_orbit * np.sin(ang)
        ly = by + p["amp"] * 1.6 * np.sin(phase * 2 + k * 1.9)
        _sphere(vol, lx, ly * sq / sq, lz, 4.0, base.BODY, soft=1.5)
    _sphere(vol, C - 3.0 * np.cos(phase), by + 3.0 * sq, C + 3.0 * np.sin(phase),
                 3.0, shade(base.BODY_HI), soft=1.4)
    _sphere(vol, C + 4.0 * np.cos(phase + 2), by - 1.0, C - 4.0 * np.sin(phase + 2),
                 2.2, shade(base.SHEEN), soft=1.2)

    # Tentacle ring with amplified wave
    for k in range(base.N_TENT):
        az = k * 2 * np.pi / base.N_TENT
        walk_curl = p["amp"] * 1.3 * np.sin(phase - az)
        walk_sway = p["amp"] * 0.5 * np.cos(phase - az)
        idle_curl = p["amp"] * 0.7 * np.sin(phase * (1 + k % 2) + k * 1.7)
        idle_sway = p["amp"] * 0.3 * np.sin(phase + k * 2.3)
        curl = (1 - w) * idle_curl + w * walk_curl  # smooth in z: no cliffs in the manifold
        sway = (1 - w) * idle_sway + w * walk_sway
        curl += p["jitter"] * 0.5 * np.sin(3 * phase + k * 2.1)
        pts = []
        length = (9.0 + 3.0 * p["droop"]) * (2.0 - sq) ** 0.5
        n = 8
        for i in range(n):
            t = i / (n - 1)
            rad = 6.0 * p["splay"] + curl * 2.6 * t * t
            a = az + sway * 0.4 * t
            pts.append((C + rad * np.cos(a), by - 5.0 - length * t, C + rad * np.sin(a)))
        _swept(vol, pts, 1.7, 0.7, base.BODY, soft=0.9)

    # Eyes: staggered blinks scaled by eye_open; look forward when walking
    look = 0.6 * w
    for az, el, size, boff in base.EYES:
        blink = np.clip(3.5 * np.sin(phase + boff) - 2.6, 0, 1)
        blink = max(blink, (1.0 - p["eye_open"]) * 0.92)
        r_surf = 8.9 * p["splay"]
        ex = C + r_surf * np.cos(el) * np.cos(az)
        ey = by + r_surf * np.sin(el) * sq
        ez = C + r_surf * np.cos(el) * np.sin(az)
        if blink >= 0.9:
            _sphere(vol, ex, ey, ez, size * 0.9, shade(base.BODY_HI), soft=0.6)
        else:
            _sphere(vol, ex, ey, ez, size, base.EYE_WHITE, soft=0.5)
            _sphere(vol, ex + 0.55 * np.cos(el) * np.cos(az), ey + 0.55 * np.sin(el),
                         ez + 0.55 * np.cos(el) * np.sin(az) + look * 0.4,
                         size * 0.55, shade(base.IRIS), soft=0.4)

    # The mask abides. It does not squash. It has never squashed.
    mz = C + 8.6 * p["splay"]
    my = by + 2.5
    _sphere(vol, C + 3.0, my, mz, 2.9, base.MASK, soft=0.5)
    _sphere(vol, C + 2.0, my + 0.8, mz + 1.2, 0.5, base.MASK_FACE, soft=0.3)
    _sphere(vol, C + 4.0, my + 0.8, mz + 1.2, 0.5, base.MASK_FACE, soft=0.3)
    for t in np.linspace(-0.8, 0.8, 7):
        _sphere(vol, C + 3.0 + t * 1.4, my - 0.9 + 0.5 * (1 - (t / 0.85) ** 2),
                     mz + 1.3, 0.35, base.MASK_FACE, soft=0.3)

    vol[..., :3] *= vol[..., 3:4]
    return np.ascontiguousarray(vol.transpose(2, 1, 0, 3)).astype(np.float32)


def cycle_for(z):
    return np.stack([draw3d(2 * np.pi * f / FRAMES, z) for f in range(FRAMES)]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--out", default="corpus_shoggoth3d.npz")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    zs = rng.random((args.n, ZDIM), dtype=np.float32)
    anchor_z = np.array(list(ANCHORS.values()), dtype=np.float32)
    zs[: len(anchor_z) * 6] = np.repeat(anchor_z, 6, axis=0)

    with Pool(args.workers) as pool:
        cycles = pool.map(cycle_for, list(zs), chunksize=4)

    frames = np.stack(cycles)  # (N, F, G, G, G, 4) f16
    np.savez_compressed(args.out, z=zs, frames=frames)
    with open("anchors_shoggoth3d.json", "w") as f:
        json.dump({"z_spec": [s[0] for s in Z_SPEC], "anchors": ANCHORS}, f, indent=2)
    print(f"corpus: {frames.shape} -> {args.out}")


if __name__ == "__main__":
    main()
