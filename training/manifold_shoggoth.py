"""Parametric shoggoth: every animation cycle is a point in a 10-D factor space.

This is Phase 1 of the behavior-manifold plan: instead of two authored cycles,
draw(phase, z) generates a closed 12-frame loop for ANY z in the unit box, and
the corpus generator samples thousands of them as NCA training targets. The
factors are the latent axes an LLM (or a text->z projector) will steer later.

Loop-closure rule inherited from shoggoth.py: every time-frequency must be an
integer multiple of the base cycle or frame 11 doesn't flow back into frame 0.

Corpus: python3 manifold_shoggoth.py --n 2048 --out corpus_shoggoth.npz
Also writes anchors_shoggoth.json: named z vectors for semantic regions.
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

from shoggoth import (BODY, BODY_HI, EYE_WHITE, GRID, IRIS, MASK, MASK_FACE,
                      PUPIL, SHEEN, SS, _blob, _disk, _eye, _tentacle)

FRAMES = 12
ZDIM = 10

# (name, lo, hi): z is stored normalized in [0,1]; drawing maps to these ranges.
# ORDER IS THE FORMAT — the Swift runtime and the projector index z by position.
Z_SPEC = [
    ("walkness", 0.0, 1.0),   # 0 idle sway .. 1 traveling-wave gait
    ("amp",      0.2, 1.6),   # tentacle motion amplitude
    ("wave_k",   0.5, 2.0),   # spatial wavenumber of the gait wave
    ("droop",    0.0, 1.0),   # limpness: sagging tentacles, squashed body
    ("bob",      0.0, 2.2),   # body bob amplitude
    ("eye_open", 0.15, 1.0),  # baseline eyelid openness
    ("look",    -1.0, 1.0),   # gaze bias left/right
    ("bright",   0.7, 1.25),  # highlight/iris brightness
    ("jitter",   0.0, 1.0),   # high-frequency tremor (3rd harmonic)
    ("spread",   0.85, 1.2),  # tentacle splay / body width
]

ANCHORS = {
    #            walk amp  wavek droop bob  eye  look bright jit  spread
    "idle":     [0.0, 0.45, 0.5, 0.15, 0.3, 0.8, 0.5, 0.55, 0.05, 0.5],
    "walk":     [1.0, 0.55, 0.5, 0.10, 0.7, 0.9, 0.9, 0.60, 0.05, 0.5],
    "sleep":    [0.0, 0.10, 0.3, 0.85, 0.05, 0.1, 0.5, 0.30, 0.00, 0.4],
    "dread":    [0.0, 0.25, 0.4, 0.75, 0.15, 0.9, 0.2, 0.20, 0.45, 0.3],
    "manic":    [0.6, 1.00, 0.9, 0.00, 1.0, 1.0, 0.6, 1.00, 0.85, 0.9],
    "curious":  [0.2, 0.50, 0.6, 0.10, 0.4, 1.0, 0.95, 0.75, 0.10, 0.7],
    "content":  [0.0, 0.35, 0.5, 0.25, 0.3, 0.6, 0.5, 0.80, 0.00, 0.6],
    "agitated": [0.8, 0.85, 1.0, 0.05, 0.9, 1.0, 0.3, 0.65, 0.70, 0.8],
}


def _p(z):
    """Normalized z[0..1]^10 -> named physical parameters."""
    return {name: lo + float(zi) * (hi - lo) for (name, lo, hi), zi in zip(Z_SPEC, z)}


def _shade(color, k):
    return tuple(min(1.0, ch * k) for ch in color)


def draw(phase, z):
    """One frame of the cycle at `phase` for factor vector z (normalized)."""
    p = _p(z)
    s = SS
    c = np.zeros((GRID * s, GRID * s, 4), dtype=np.float32)
    w = p["walkness"]

    bob_amp = p["bob"] * (0.4 + 0.6 * w) + 0.25
    bob = bob_amp * (w * abs(np.sin(phase)) + (1 - w) * 0.5 * np.sin(phase))
    lean = 2.0 * w * np.sin(phase + 1.0)
    droop_dy = 2.5 * p["droop"]
    cy = 30.0 - bob + droop_dy * 0.6
    cx = 32.0 + lean * 0.4

    jit = p["jitter"]
    bases = [(-13, 1.0), (-8, 1.15), (-3, 1.3), (3, 1.3), (8, 1.15), (13, 1.0)]
    for k, (dx, lscale) in enumerate(bases):
        koff = k * p["wave_k"] * (2 * np.pi / len(bases))
        walk_curl = 1.5 * np.sin(phase - koff)
        idle_curl = 0.8 * np.sin(phase * (1 + k % 2) + k * 1.7)
        curl = p["amp"] * ((1 - w) * idle_curl + w * walk_curl)
        curl += jit * 0.35 * np.sin(3 * phase + k * 2.1)
        base_ang = 0.12 * np.sin(phase + k * 2.3) + 0.25 * w * np.sin(phase - koff + 0.8)
        length = 14 * lscale * (1.0 + 0.25 * p["droop"])
        width = 2.6 * (1.0 - 0.25 * p["droop"])
        _tentacle(c, cx + dx * p["spread"], cy + 9, length, base_ang, curl, width, BODY, s)

    up_curl = p["amp"] * (1.1 * np.sin(phase + 0.5)) + jit * 0.3 * np.sin(3 * phase)
    _tentacle(c, cx - 11 * p["spread"], cy - 6 + droop_dy, 11 * (1 - 0.3 * p["droop"]),
              -1.9, up_curl, 2.0, BODY, s)
    _tentacle(c, cx + 12 * p["spread"], cy - 4 + droop_dy, 9 * (1 - 0.3 * p["droop"]),
              1.8, -p["amp"] * np.sin(phase * 2 + 2.0) - jit * 0.25 * np.sin(3 * phase + 1),
              1.8, BODY, s)

    squash = 1.0 + 0.18 * p["droop"]
    breathe = 0.06 + 0.02 * (1 - w) * np.sin(phase) + 0.05 * jit * np.sin(3 * phase + 0.7)
    _blob(c, cx, cy, 14.5 * p["spread"], 12.0 / squash, BODY, s, wobble=breathe, phase=phase)
    hi = _shade(BODY_HI, p["bright"])
    _blob(c, cx - 4, cy - 3, 7.5, 5.5 / squash, hi, s, wobble=0.12, phase=phase * 2)
    _blob(c, cx + 6, cy + 4, 4.5, 3.2, _shade(SHEEN, p["bright"]), s, wobble=0.15, phase=-phase)
    _blob(c, cx - 8, cy + 5, 3.2, 2.4, hi, s, wobble=0.1, phase=phase + 2)

    look = p["look"] * (0.5 + 0.5 * w)
    eyes = [(-7, -5, 2.6, 0.0), (2, -7, 1.9, 2.1), (7, 0, 2.2, 4.2),
            (-2, 2, 1.5, 1.1), (-10, 1, 1.7, 3.4), (4, 5, 1.3, 5.3)]
    for ex, ey, er, boff in eyes:
        b = np.clip(3.5 * np.sin(phase + boff) - 2.6, 0, 1)
        b = max(b, (1.0 - p["eye_open"]) * 0.9)
        _eye(c, cx + ex, cy + ey + droop_dy * 0.3, er, b, look, s)

    mx, my = cx + 10.5, cy - 7.5 + droop_dy * 0.5
    _disk(c, mx, my, 4.2, MASK, s)
    _disk(c, mx - 1.4, my - 1.0, 0.55, MASK_FACE, s)
    _disk(c, mx + 1.4, my - 1.2, 0.55, MASK_FACE, s)
    for t in np.linspace(-0.9, 0.9, 11):
        _disk(c, mx + t * 1.9, my + 0.9 + 0.9 * (1 - (t / 0.95) ** 2), 0.42, MASK_FACE, s)

    small = c.reshape(GRID, SS, GRID, SS, 4).mean(axis=(1, 3))
    small[..., :3] *= small[..., 3:4]
    return small.astype(np.float32)


def cycle_for(z):
    """(FRAMES, GRID, GRID, 4) float16 — one full closed loop for factor vector z."""
    return np.stack([draw(2 * np.pi * f / FRAMES, z) for f in range(FRAMES)]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", default="corpus_shoggoth.npz")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    zs = rng.random((args.n, ZDIM), dtype=np.float32)
    # Seed the corpus with the named anchors (repeated so they're well-learned)
    anchor_z = np.array(list(ANCHORS.values()), dtype=np.float32)
    zs[: len(anchor_z) * 8] = np.repeat(anchor_z, 8, axis=0)

    with Pool(args.workers) as pool:
        cycles = pool.map(cycle_for, list(zs), chunksize=8)

    frames = np.stack(cycles)  # (N, FRAMES, GRID, GRID, 4) f16
    np.savez_compressed(args.out, z=zs, frames=frames)
    with open("anchors_shoggoth.json", "w") as f:
        json.dump({"z_spec": [s[0] for s in Z_SPEC], "anchors": ANCHORS}, f, indent=2)
    print(f"corpus: {frames.shape} -> {args.out}; anchors -> anchors_shoggoth.json")


if __name__ == "__main__":
    main()
