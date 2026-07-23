# FusedNCAWeights.swift

## Purpose

Parses the flat `FX2D` deployment format for the hard-routed fused pose/edge NCA.
This remains separate from `NCAWeights`: the expert-bank layout and transport
schedule are not compatible with the residual `NCA1`–`NCA5` family.

## Components

### `FusedNCAWeights`

- Validates the fixed four-expert, 16-channel model and bounded dynamic widths.
- Derives the exact payload size for pose, flow, slot, and repair affine banks.
- Exposes shape compatibility for safe live weight replacement.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `training/export_fused_expert2d.py` | 48-byte little-endian header and exact bank order | Header/layout changes |
| `FusedNCAShaders.swift` | Derived rule width and flat bank offsets | Tensor ordering/shape |
| `FusedNCASimulation.swift` | Grid, schedule, fire rate, and compatible hot reload | Field semantics |
