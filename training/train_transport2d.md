# train_transport2d.py

## Purpose

Trains the bounded higher-resolution 2D treatment that isolates explicit
transport from a phase-conditioned reaction-only donor. It preserves the NCA2
baseline and writes experimental `TN2D1` checkpoints only.

## Components

### `build_teacher_bank`

- **Does**: Grows reusable mature donor states paired with their internal ring
  tokens using behavior 0.
- **Rationale**: Refreshing from an on-manifold bank prevents the transport
  experiment from being dominated by seed growth.

### `make_optimizer`

- **Does**: Freezes copied repair during flow-only warmup, then enables joint
  flow/repair tuning at a lower learning rate.
- **Rationale**: Flow must first receive the motion gradient instead of letting
  the donor reaction rule absorb it immediately.

### `main`

- **Does**: Loads four reviewed anchors and an NCA2 donor, maintains a coupled
  state/ring pool, refreshes part of every batch, scores several rollout
  segments, regularizes smooth low-magnitude flow, and checkpoints treatment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU launch | Default 128x128 corpus, batch 8, pool 128, 32-step rollout, damage off | CLI/defaults |
| Evaluator | Full coupled rule and iteration in `TN2D1` | Checkpoint payload |
| Donor | NCA2 with behavior 0 trained on the same four-anchor corpus | Semantics |

## Notes

- The coherence A/B intentionally omits regeneration damage. Damage, learned
  dwell/completion triggers, and runtime export are later gates.

