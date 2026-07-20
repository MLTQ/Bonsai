# train_cyclic3d.py

## Purpose
Trains the 3D phase-conditioned cyclic NCA (Shoggoth Mk. III). It combines volumetric perception with moving phase/behavior targets, checkpointed or fused BPTT, regeneration damage, preview rendering, and NC3C export.

## Components

### `CyclicNCA3D`
- **Does**: 16-channel 3D NCA with id/Sobel xyz perception, `(sin theta, cos theta, behavior)` conditioning, stochastic fire, life masking, and ±8 clamp.
- **Interacts with**: `make_frames3d` in `shoggoth3d.py`; NC3C readers consume `export`.

### `CyclicNCA3D.rollout`
- **Does**: Advances phase every step. Eager mode uses checkpoint chunks; fused mode passes the `(T,B,3)` condition sequence to one `fused_nca_rollout` node.
- **Rationale**: The single fused node preserves the exact per-step RNG/conditioning contract while removing repeated autograd and allocation overhead.

### `load_nc3c` / `export`
- **Does**: Reads and writes the NC3C weight layout for warm starts and runtime consumption.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `shoggoth3d.py` | `FRAMES`, `BEHAVIORS`, and `(B,F,D,H,W,4)` target frames | target ordering |
| NC3C runtime | magic, dimensions, fire rate, then w1/b1/w2/b2 | export layout |
| `fused_step.py` | condition order `(sin, cos, behavior)` and step-indexed RNG | condition/RNG semantics |
