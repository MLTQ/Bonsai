# shoggoth3d.py

## Purpose
Generates the seeded 32³ Shoggoth Mk. III animation corpus used by the cyclic baseline and transport experiment. It renders closed idle/walk loops as premultiplied RGBA volumes in `(z,y,x,c)` order.

## Components

### `draw3d`
- **Does**: Render one procedural phase with a central body, moving lobes, tentacle ring, eyes, and rigid mask
- **Interacts with**: Volumetric primitives in `target3d.py`
- **Rationale**: Integer phase frequencies guarantee a closed loop; direct arbitrary-phase rendering supplies real transition geometry

### `make_frames3d`
- **Does**: Return the legacy 12-phase, two-behavior corpus consumed by `train_cyclic3d.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_cyclic3d.py` | `(2,12,G,G,G,4)` float16, idle then walk | Shape/order/behavior semantics |
| `transport_targets3d.py` | `draw3d(phase, walking=True)` accepts arbitrary phase and returns `(G,G,G,4)` | Function signature or axis order |
| 32³ lineage | Anatomy is authored at grid 32 and scales through `target3d.SCALE` | Scaling convention |
