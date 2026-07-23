# fused_expert_nca2d.py

## Purpose

Implements one deployable NCA checkpoint containing four independently
parameterized pose attractors and four independently parameterized directed
transition experts. Routing is globally hard; raw expert weights are never
averaged.

## Components

### `FusedExpertNCA2D`
- **Does**: Perceives a shared 16-channel state and hard-selects either a full
  pose rule or a full flow/slot/repair edge rule.
- **Interacts with**: Canonical states from `fused_state2d.py` and corrected
  advection from `layered_transport_nca2d.py`.
- **Rationale**: Separate parameter banks isolate anchor stability from edge
  transport while retaining one runtime model and one recurrent state.
  F1b uses straight-through one-hot motion assignments: a cell is never the
  forward-pass average of four warped states, while gradients still reach the
  soft slot logits. `hard_slots=False` preserves the F1a ablation.
  F2 defaults to 384-wide pose/repair experts, 128-wide flow experts, and four
  Fourier coordinate bands in addition to raw XY. This raises detail capacity
  without duplicating runtime state.

### `step_with_aux`
- **Does**: Executes pose dwell or directed transport according to an oracle
  transition bit and expert index, returning flow/assignment diagnostics.
- **Rationale**: Hard routing prevents a 50/50 mixture of attractor vector
  fields. Mixed-mode batches are supported for contract completeness.

### `set_stage`
- **Does**: Freezes edge banks during anchor training, pose banks during edge
  training, and enables both for joint tuning.

### `save_fused_checkpoint` / `load_fused_checkpoint`
- **Does**: Writes `FEX2D2` width/position metadata and backward-compatibly
  loads `FEX2D1` F1 checkpoints by inferring their tensor widths.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer/evaluator | Expert indices 0:3 and Boolean transition mode | Routing semantics |
| Runtime | State `(B,16,H,W)`, progress `(B,)`, one selected bank and one flow slot per cell | Step signature |
| Checkpoint loader | Format magic `FEX2D1` or `FEX2D2` | Tensor names or metadata |

## Notes

- Progress is supplied by an oracle guide in F1. A learned hysteretic guide is
  deliberately postponed until expert transport passes independently.
- The parameter banks are full rules rather than post-hoc averages of
  independently permuted neural networks.
- `FEX2D1` checkpoints written before `hard_slots` are interpreted as soft-slot
  F1a checkpoints for reproducible evaluation.
