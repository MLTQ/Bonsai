# test_fused_expert2d_contract.py

## Purpose

Provides CPU-fast interface tests for canonical recurrent states, hard expert
routing, differentiability, and fused checkpoint compatibility.

## Tests

- Canonical states preserve reviewed RGBA and expose the correct alpha-masked
  pose one-hot channels.
- Zero-output pose experts preserve canonical key states exactly.
- A pose or edge bias in one bank cannot affect another selected bank.
- Straight-through motion assignments are exactly one-hot in the forward pass.
- Motion assignments partition each cell and pose/edge banks receive finite,
  nonzero gradients.
- `FEX2D2` checkpoints round-trip parameters, widths, Fourier features, and
  architecture metadata; the loader remains compatible with F1 `FEX2D1` files.
- `FX2D` exports preserve the exact header, bank tensor order/values, schedule,
  and canonical cell-major `NCS1` deployment state.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `fused_state2d.py` | Hidden pose-code offsets 6:10 | Encoding order |
| `fused_expert_nca2d.py` | Hard bank/slot selection and exact zero-update dwell | Routing/life semantics |
| Trainer/runtime | Finite gradients and exact checkpoint round trip | Model/checkpoint format |
| `export_fused_expert2d.py` | Exact portable header, tensor payload, and state layout | Binary format |
