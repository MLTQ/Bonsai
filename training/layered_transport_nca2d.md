# layered_transport_nca2d.py

## Purpose

Defines the multi-flow treatment for the 2D coherence experiment. A hard global
edge controller selects one adjacent transition while four soft motion slots
advect persistent state before a small stochastic local repair update.

## Components

### `controller_features`
- **Does**: Broadcasts one-hot edge, progress, and dwell/transition flag
- **Rationale**: Only one directed edge can be active; nonadjacent pose heads
  cannot blend

### `advect_state`
- **Does**: Applies forward/backward MacCormack correction and a local monotonic
  limiter around the shared bilinear warp
- **Rationale**: Repeated ordinary bilinear sampling destroys the exact
  high-frequency detail explicit transport is meant to preserve

### `LayeredTransportNCA2D`
- **Does**: Builds Sobel perception plus absolute coordinates, predicts four
  edge-specific flow fields and soft assignments, blends their transported
  states, then applies edge-specific local repair
- **Interacts with**: `train_layered_transport2d.py` and
  `eval_layered_transport2d.py`
- **Rationale**: Multiple flows can move distinct body regions differently;
  one flow cannot represent crossing limbs at one screen location

### `set_stage`
- **Does**: Separates noisy-anchor repair, flow-only motion, and joint tuning
- **Contract**: Repair controller/coordinate columns start at zero so unseen
  transition progress cannot perturb the frozen anchor stabilizer

### Checkpoint helpers
- **Does**: Save/load isolated `LT2D1` PyTorch checkpoints
- **Rationale**: Runtime integration remains gated on coherence metrics

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer/evaluator | 16-channel state; four edges and default four slots | State/controller shape |
| `transport_nca2d.warp_state` | XY flow in cell units, NCHW state | Flow axes/units |
| Checkpoint consumers | `LT2D1`, grid/slot/max-flow metadata | Format fields |

## Notes

- Deterministic global transport and stochastic local repair are intentionally
  separate. Stochastic per-cell flow produces tears rather than useful motion.
- This experiment starts from mature anchors; point growth and damage are later
  curricula.
