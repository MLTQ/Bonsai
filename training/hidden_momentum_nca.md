# hidden_momentum_nca.py

## Purpose
Defines the eager reference for the second inertia experiment and owns NCA5 serialization. It keeps visible RGBA residual while adding explicit velocity only to the twelve latent channels.

## Components

### `HiddenMomentumNCA`
- **Does**: Perceives 16 position plus 12 hidden-velocity channels; emits four visible deltas and twelve hidden forces
- **Rationale**: Visible inertia caused chromatic/alpha ringing in NCA4, while hidden velocity can still disambiguate direction of travel

### `lift_state`
- **Does**: Appends hidden-channel velocity to a legacy 16-channel donor state

### `transplant_residual`
- **Does**: Copies visible residual rows unchanged and scales hidden rows by `1-decay` for force initialization

### `export_nca5`
- **Does**: Writes layout metadata, fire/decay values, and `w1,b1,w2,b2`

### `load_nca5`
- **Does**: Validates the NCA5 header/payload and restores weights plus fire/decay dynamics into a compatible eager model
- **Interacts with**: `export_hidden_momentum_state.py` and contract tests

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_hidden_momentum.py` | State `[position16, velocity(hidden4:16)]`; output `[rgba_delta4, hidden_force12]` | Channel ordering |
| Swift runtime | NCA5 header is magic, six i32 layout fields, two f32 dynamics fields, then flat weights | Header or update semantics |

## Notes
- NCA5 remains 2D; the app now mirrors its hidden-only integration for visual evaluation.
