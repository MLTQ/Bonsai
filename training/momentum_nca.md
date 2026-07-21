# momentum_nca.py

## Purpose
Defines the eager PyTorch reference for explicit inertial NCA state and owns the `NCA4` binary format. It isolates experimental phase-space math from the legacy residual trainers.

## Components

### `MomentumNCA`
- **Does**: Perceives 32 channels (16 position + 16 velocity), predicts 16 forces, then applies `v = decay*v + fire*force; u = u+v`
- **Interacts with**: `train_autonomous.py`; mirrored by `nca_step` in `NCAShaders.swift`
- **Rationale**: A full phase-space lift gives every learned state channel a matched velocity without wasting update-head outputs

### `lift_state`
- **Does**: Converts a legacy 16-channel state and optional finite-difference velocity into the 32-channel NCA4 layout

### `transplant_residual`
- **Does**: Copies a residual donor into the momentum model, zeroing velocity perception and scaling the old delta head by `1-decay`
- **Rationale**: The scaled old delta is the steady-state velocity of the damped integrator under constant force

### `export_nca4`
- **Does**: Writes magic, dimensions, fire/decay values, and flat `w1,b1,w2,b2` arrays
- **Interacts with**: `NCAWeights.load(from:)` in Swift

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_autonomous.py` | First four position channels are RGBA; channels 16–31 are velocities | State ordering, integrator order |
| `NCAWeights.swift` | `NCA4`, i32 state/hidden/cond/position, f32 fire/decay, then weights | Header or flat-array order |
| `NCAShaders.swift` | Damping applies every step; stochastic fire gates force only; both halves clamp to ±8 | Update semantics |

## Notes
- NCA4 is currently 2D and eager-only during training; the Triton fused kernel still targets 16-channel residual formats.
