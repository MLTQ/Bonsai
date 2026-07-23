# transport_nca2d.py

## Purpose

Defines the higher-resolution 2D advection-reaction treatment for globally
coherent animation. It couples the cellular field to one shared internal ring,
hard-selects exactly one of four directed transition edges, transports the
entire recurrent state, and then applies donor-initialized local repair.

## Components

### `make_global` / `advance_global` / `phase_features`

- **Does**: Encodes and rotates `[sin, cos]`, returning edge progress and one
  hard active edge.
- **Rationale**: Every cell sees the same direction of travel and no head can
  mix nonadjacent gait anchors.

### `warp_state`

- **Does**: Backward-warps NCHW state with an XY displacement in cell units.
- **Rationale**: Coherent motion moves existing material before the local rule
  repairs detail, instead of asking reaction dynamics to erase/regrow limbs.

### `TransportNCA2D`

- **Does**: Runs Sobel perception, edge-selected flow, bilinear transport,
  stochastic local repair, life masking, and global-ring advancement.
- **Interacts with**: `train_transport2d.py`, `eval_transport2d.py`.

### `transplant_nca2`

- **Does**: Copies a phase-conditioned NCA2 donor into all repair heads.
- **Contract**: At zero flow, the treatment is step-equivalent to behavior 0 of
  the donor under a shared fire mask.

### Checkpoint helpers

- **Does**: Save/load the isolated `TN2D1` PyTorch experiment format.
- **Rationale**: Runtime integration remains gated on beating the donor.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_cyclic.CyclicNCA` | 16 channels, 128 hidden, Sobel ordering, `[sin, cos, behavior]` | Donor layout |
| Trainer/evaluator | Four edges over a 240-step cycle; global state `[sin, cos]` | Ring semantics |
| `grid_sample` | NCHW state, XY flow, base grid stores XY, `align_corners=True` | Axis/order |

## Notes

- This is a controlled global-controller/transport experiment, not a runtime
  format. Learned dwell/completion hysteresis is a later gate.

