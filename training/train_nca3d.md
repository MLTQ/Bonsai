# train_nca3d.py

## Purpose
Rung 1 of the 3D ladder: volumetric Growing NCA on a static voxel target. The 2D recipe lifted one dimension, plus the one genuinely new piece — gradient-checkpointed rollouts, without which BPTT at 32³ needs ~14 GB of activations.

## Components

### `NCA3D`
- **Does**: conv3d perception (identity + Sobel x/y/z per channel, 64 features, /32 normalization), 1×1×1 convs 64→128→16, sphere life mask, ±8 clamp
- **Interacts with**: To be mirrored by a Metal 3D kernel set (NCASimulation3D, future) — perception ordering [id, sx, sy, sz] interleaved per channel, volume index order (z, y, x)

### `NCA3D.rollout`
- **Does**: Runs steps in CHUNK=8 segments through `torch.utils.checkpoint` (non-reentrant) — peak activation memory ~ batch × chunk instead of batch × steps
- **Rationale**: ~35% recompute overhead buys 8× memory headroom; required for the shared 24 GB card, load-bearing for rung 3

### Training loop / `export`
- **Does**: Pool + sphere-damage training; NaN guards (discard batch, filter pool writeback) inherited from the manifold post-mortem; NC3D export (NCA1 layout family, w1 width ch*4)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `target3d.py` | `make_target3d()` (GRID3³, 4) premultiplied, (z,y,x,c) order | Shape/order |
| Future Swift 3D runtime | NC3D layout, kernel ordering, (z,y,x) indexing, ±8 clamp | Any math change |

## Notes
- Seed sits at (z=c, y=GRID3/3, x=c) — low center, where the soil line is.
- 32³ CPU smoke ≈ 0.12 it/s; 4090 estimate 1–4 it/s (validate at launch, consider CUDA graphs if <1).

### `--fused` (2026-07-19)
`rollout(..., seed=it)` bypasses checkpointing entirely: `fused_step.FusedNCAStep` saves only 16-ch input states per step and recomputes in backward (deterministic counter RNG). Gate-validated; eager stays the reference. See fused_step.md.
