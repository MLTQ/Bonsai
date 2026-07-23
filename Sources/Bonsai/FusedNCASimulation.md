# FusedNCASimulation.swift

## Purpose

Owns state and intermediate Metal buffers for one `FX2D` creature, advances the
hard global edge/pose schedule, and renders it through the fused shader suite.

## Components

### `FusedNCASimulation`

- Runs 24 configured edge steps followed by eight destination-attractor dwell
  steps, then commits to the next directed edge.
- Uses separate flow, slot, reaction, per-slot prediction, transported, and
  pre/post state buffers so the eager PyTorch operation order is retained.
- Loads and remembers a mature canonical `NCS1` snapshot for reset; single-cell
  growth was deliberately outside this experiment.
- Supports damage, raw RGBA readback, compatible hot reload, and crisp rendering.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `FusedPetView.swift` | `device`, grid, step, damage, reset, mirror | Public interface |
| `AppDelegate.swift` | Failable init, state loading, compatible hot reload | Lifecycle |
| `FusedNCAShaders.swift` | Buffer indices and uniform field order | Dispatch layout |
| `training/train_fused_expert2d.py` | Smooth progress and edge/handoff schedule | Controller timing |
