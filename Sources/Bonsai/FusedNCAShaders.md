# FusedNCAShaders.swift

## Purpose

Generates the Metal implementation of `FusedExpertNCA2D`. Pose experts use a
residual reaction step. Edge experts run hard slot selection, four backward
warps, MacCormack correction with a local monotonic limiter, repair, and the
same pre/post life mask as training.

## Pipeline

1. `fused_pose_step` runs one hard-selected pose bank.
2. `fused_edge_fields` computes flow fields, hard argmax slot, and repair.
3. `fused_edge_predict` performs the forward advection for every slot.
4. `fused_edge_correct` reverse-warps, applies MacCormack correction/limiting,
   selects one transported state per cell, and applies stochastic repair.
5. `fused_life` applies the pre/post alpha life mask and state clamp.
6. `fused_render` draws premultiplied RGBA with the standard optional crisp edge.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `FusedNCASimulation.swift` | Kernel names, buffer indices, and 56-byte uniform order | Signatures/layout |
| `fused_expert_nca2d.py` | Perception/Fourier ordering, hard argmax slots, advection, fire/life/clamp math | Numerical semantics |
| `FusedNCAWeights.swift` | Flat tensor bank order used to derive offsets | Export layout |
