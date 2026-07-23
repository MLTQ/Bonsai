# export_hidden_momentum_state.py

## Purpose
Creates a deterministic mature NCS1 snapshot for the NCA5 clockless shoggoth. The experiment trained from grown conditioned-teacher states, so this artifact places the app on the evaluated state distribution instead of asking an untrained single seed to grow.

## Components

### `make_mature_state`
- **Does**: Grows the NCA2 teacher in walk mode and lifts its final hidden-channel finite difference into NCA5 velocity registers
- **Interacts with**: `CyclicNCA` in `train_cyclic.py`, `lift_state` in `hidden_momentum_nca.py`

### `write_ncs1`
- **Does**: Converts NCHW state into the runtime's cell-major NCS1 binary layout
- **Interacts with**: `NCASimulation.loadState(from:)`

### `main`
- **Does**: Validates teacher and student checkpoints, optionally advances the student, and writes the mature snapshot
- **Rationale**: Defaults are deterministic so app comparisons start from the same pose and internal velocity

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `Creature.swift` | Output is 64×64 with 28 channels and matches `shoggoth_auto_hidden_momentum.nca` | Grid or channel layout |
| `NCASimulation.swift` | Header is `NCS1` + width/height/channels i32, followed by cell-major float32 | Header or payload ordering |
