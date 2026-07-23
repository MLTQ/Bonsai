# test_layered_transport2d_contract.py

## Purpose

Provides CPU-fast contracts for the hard controller, corrected advection,
motion-slot partition, checkpoint format, and blur-rejection metrics.

## Tests

- Controller edge features are exactly one-hot and retain progress/transition.
- Zero flow is identity; positive X flow moves an impulse right.
- Dwell disables every flow and four assignments sum to one per cell.
- Edge-specific biases cannot leak into another edge.
- A transition has a finite nonzero gradient through corrected transport.
- `LT2D1` checkpoints round-trip metadata and parameters.
- Dice, boundary F1, alpha/RGB sharpness, and RGB gradient error reject a
  deliberately blurred target.
- The portable four-anchor solve separates adjacent from distant-pose leakage.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `layered_transport_nca2d.py` | XY sign, dwell identity, hard edge selection | Model semantics |
| `layered_transport_metrics2d.py` | Blur lowers all three acceptance metrics | Metric definitions |
| Checkpoint loader | Exact `LT2D1` round trip | Format fields |
