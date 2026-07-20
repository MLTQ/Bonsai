# train_constellation.py

## Purpose
Trains a two-state 2D NCA to move continuously through graph- or rule-connected pose constellations. It selects waypoint chains from the nearest current pose, supervises intermediate arrivals, and trains cross-state metamorphosis.

## Components

### `main`
- **Does**: Loads a `2d_constellation` corpus, builds motion weighting and waypoint chains, runs pool/damage training, and exports through `train_states.export`.
- **Interacts with**: `StateNCA`, `damage`, `make_seed`, and `export` in `train_states.py`.

### Waypoint rollout
- **Does**: Runs one segment per supervised waypoint. Fused mode uses `fused_nca_rollout` per segment and carries `step_offset` forward so counter-RNG masks remain unique across the complete chain.
- **Rationale**: Segment-level nodes retain intermediate losses while avoiding one autograd node per NCA step.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| constellation `.npz` | poses, pose_state, transits; optional directed edges | keys/shape/order |
| `train_states.py` runtime/export | exactly two states represented by one condition scalar | condition semantics |
| `fused_step.py` | static `(B,1)` condition and globally increasing `step_offset` | seed/offset semantics |
