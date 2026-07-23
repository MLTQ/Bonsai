# export_mature_cycle_state.py

## Purpose

Exports one reviewed `2d_cycle` anchor as the mature NCS1 state required by an
NCA trained only on already-grown images. This keeps app startup on the same
state distribution as the experiment instead of invoking untrained growth.

## Components

### `make_state`
- **Does**: Loads one premultiplied RGBA anchor into channels 0–3 and copies
  alpha into hidden channels 4–15
- **Interacts with**: `make_mature_state` in `train_cyclic.py`
- **Rationale**: Matching the trainer's initialization is part of checkpoint
  semantics, not a display convenience

### `write_ncs1`
- **Does**: Writes `NCS1`, width/height/channels int32, then cell-major
  little-endian float32 state
- **Interacts with**: `NCASimulation.loadState(from:)`

### `main`
- **Does**: Exports frame zero of the Mega Man corpus by default and reports
  its live alpha-cell count

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `Creature.swift` | 128×128×16 state named `megaman_walk_mature.ncs` | Grid, channels, filename |
| `NCASimulation.swift` | Cell-major NCS1 byte layout | Header or ordering |
| `train_cyclic.py` | Visible target plus alpha-filled hidden channels | Initialization policy |
