# target3d.py

## Purpose
Procedural voxel bonsai — the 3D NCA's first growth target. Soft-edged volumetric primitives (spheres, swept tapered capsules, truncated-cone pot) drawn at 2× supersample (64³), box-filtered to 32³, premultiplied RGBA where alpha is density.

## Components

### `make_target3d`
- **Does**: Returns (32, 32, 32, 4) float32 in (z, y, x, c) order, y-up
- **Rationale**: Soft edges (per-primitive `soft` falloff) are the 3D analog of the 2D supersampled anti-aliasing that trains well

### `_sphere` / `_swept` / `_cone_pot` / `_composite`
- **Does**: Primitives compositing painter's-order (density maxes, color overwrites where present) — same semantics as the 2D generators

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_nca3d.py` | Shape (GRID3³,4), premultiplied, (z,y,x,c), GRID3 constant | Order/shape |

## Notes
- `python3 target3d.py` writes quick projection PNGs; the front (axis-0) view is the trustworthy one, side-view transpose is cosmetically unrotated.
