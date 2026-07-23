# export_fused_expert2d.py

## Purpose

Converts a trained `FEX2D2` PyTorch checkpoint into the flat little-endian
`FX2D` deployment format consumed by Bonsai and writes its matching canonical
pose-zero `NCS1` recurrent state.

## Format

The 48-byte `FX2D` header stores magic, grid, channel/expert/slot counts,
pose/flow widths, Fourier frequency count, maximum flow, fire rate, and the
transition/handoff schedule. Float32 tensors follow in `WEIGHT_ORDER`, each
contiguous and row-major. The companion JSON repeats the layout and provenance.

The `NCS1` snapshot is cell-major `(y, x, channel)` and contains the reviewed
pose-zero RGBA plus the deterministic canonical hidden-state encoding. This
experiment was trained from mature states, not from a single-cell seed.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `FusedNCAWeights.swift` | Exact header fields and tensor order | Header or `WEIGHT_ORDER` changes |
| `FusedNCASimulation.swift` | Hard slots, four experts, canonical 16-channel state | Routing or state layout |
| `fused_expert_nca2d.py` | Named bank tensors and row-major affine matrices | Parameter names/shapes |
