"""Claudeguy's mood manifold corpus — the capstone's soul, Mk. IV recipe.

Z_SPEC order is the format contract (Swift and the projector index by position).
Two factors are DISCRETE by rounding (dbltime, spin): they multiply the phase,
so any non-integer value would break loop closure. The expression factor picks
one of seven parametric faces; FiLM owns whatever the in-betweens look like,
and the morphing between faces is the charm, not a bug.

Corpus: BONSAI_GRID3=64 python3 manifold_claudeguy3d.py --n 128
Then:   BONSAI_GRID3=64 python3 train_manifold3d.py \\
            --corpus corpus_claudeguy3d.npz --init claudeguy_xshog.nca \\
            --out claudeguy_manifold.nca
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np

from claudeguy3d import N_PETALS, draw_claudeguy

FRAMES = 12
ZDIM = 10

EXPR_LIST = ["neutral", "serene", "wince", "pleading", "melancholy", "dizzy", "angry"]

Z_SPEC = [
    ("droop",   0.0, 0.65),   # petal sag — melancholy axis
    ("flutter", 0.0, 1.10),   # petal motion amplitude (breath .. flutter)
    ("splay",  -0.40, 0.80),  # constant petal reach (drawn-in .. spread wide)
    ("wiggle",  0.0, 1.00),   # ring waggle
    ("dbltime", 0.0, 1.00),   # DISCRETE 0/1: flutter at 2x (integer harmonic)
    ("spin",    0.0, 1.00),   # DISCRETE 0/1: one ring revolution per cycle
    ("blink",   0.0, 1.00),   # blink depth at mid-cycle
    ("look_y", -0.5, 1.00),   # gaze height (neutral face only)
    ("look_x", -1.0, 1.00),   # gaze side (neutral face only)
    ("expr",    0.0, 1.00),   # face variant: index into EXPR_LIST
]

ANCHORS = {
    #              droop flut  splay wig  dbl spin blnk lky  lkx  expr
    "idle":       [0.20, 0.35, 0.50, 0.15, 0,  0,  0.6, 0.45, 0.5, 0.00],
    "serene":     [0.15, 0.30, 0.55, 0.10, 0,  0,  0.5, 0.45, 0.5, 0.17],
    "delight":    [0.05, 0.90, 0.65, 0.85, 1,  0,  0.3, 0.55, 0.5, 0.17],
    "curious":    [0.05, 0.40, 0.60, 0.25, 0,  0,  0.4, 0.75, 0.75, 0.00],
    "wince":      [0.35, 0.45, 0.40, 0.35, 0,  0,  0.6, 0.45, 0.5, 0.33],
    "pleading":   [0.45, 0.25, 0.45, 0.10, 0,  0,  0.1, 0.85, 0.5, 0.50],
    "melancholy": [0.95, 0.10, 0.30, 0.05, 0,  0,  0.2, 0.30, 0.5, 0.67],
    "dizzy":      [0.50, 0.55, 0.50, 0.70, 1,  1,  0.2, 0.45, 0.5, 0.83],
    "angry":      [0.10, 0.70, 0.60, 0.55, 1,  0,  0.1, 0.35, 0.5, 1.00],
}


def _p(z):
    return {name: lo + float(zi) * (hi - lo) for (name, lo, hi), zi in zip(Z_SPEC, z)}


def draw_mood(phase, z):
    """One frame at phase under mood z. Every time term is an integer multiple
    of the base frequency, so frame FRAMES lands exactly on frame 0."""
    p = _p(z)
    k = np.arange(N_PETALS)
    freq = 2.0 if p["dbltime"] >= 0.5 else 1.0
    flex = p["splay"] + p["flutter"] * np.sin(freq * phase + k * 0.52)
    blink = p["blink"] * max(0.0, 1.0 - abs(np.sin((phase - np.pi) / 2.0)) * 7.0)
    expr = EXPR_LIST[min(len(EXPR_LIST) - 1, int(round(p["expr"] * (len(EXPR_LIST) - 1))))]
    return draw_claudeguy(
        phase=phase, blink=blink, look=(p["look_x"], p["look_y"]),
        petal_flex=np.clip(flex, -1, 1), expression=expr,
        spin=1.0 if p["spin"] >= 0.5 else 0.0,
        wiggle=p["wiggle"], droop=p["droop"])


def cycle_for(z):
    return np.stack([draw_mood(2 * np.pi * f / FRAMES, z)
                     for f in range(FRAMES)]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="corpus_claudeguy3d_manifold.npz")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    zs = rng.random((args.n, ZDIM), dtype=np.float32)
    # anchors get heavy representation so the named moods are exactly learnable
    anchor_z = np.array(list(ANCHORS.values()), dtype=np.float32)
    reps = max(1, (args.n // 4) // len(anchor_z))
    take = min(args.n, len(anchor_z) * reps)
    zs[:take] = np.repeat(anchor_z, reps, axis=0)[:take]

    with Pool(args.workers) as pool:
        cycles = pool.map(cycle_for, list(zs), chunksize=1)

    frames = np.stack(cycles)  # (N, F, G, G, G, 4) f16
    np.savez_compressed(args.out, z=zs, frames=frames)
    with open("anchors_claudeguy3d.json", "w") as f:
        json.dump({"z_spec": [s[0] for s in Z_SPEC], "anchors": ANCHORS}, f, indent=2)
    print(f"corpus: {frames.shape} -> {args.out}; anchors_claudeguy3d.json written")


if __name__ == "__main__":
    main()
