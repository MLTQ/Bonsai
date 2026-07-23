# transport_targets3d.py

## Purpose
Owns dense procedural walking targets and the surface-aware visible objective for the 3D transport experiment. Dense rendering makes each target a real intermediate geometry instead of a crossfade between distant voxel poses.

## Components

### `dense_walk_frames` / `load_dense_walk_frames`
- **Does**: Render and cache a 48-frame closed walk cycle at the active 32³ grid
- **Interacts with**: `draw3d` in `shoggoth3d.py`

### `target_at_global`
- **Does**: Select the target from the internal ring token, interpolating only adjacent dense samples
- **Interacts with**: `phase_from_global` in `transport_nca3d.py`

### `visible_objective`
- **Does**: Combine premultiplied color, alpha, alpha-gradient, and premultiplication penalties
- **Rationale**: Matching surface gradients directly makes blur measurable during optimization rather than only in the final preview

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_transport3d.py` | Frames are `(F,D,H,W,4)` and converted to `(F,4,D,H,W)` on device | Shape/order |
| `eval_transport3d.py` | Four key poses occur every `F/4` samples | Cycle partition |
