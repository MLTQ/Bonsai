# train_transport3d.py

## Purpose
Trains the bounded 32³ treatment that tests whether global directed phase plus explicit transport improves a seeded cyclic NCA. It preserves the existing NC3C baseline and emits an experimental TN3D1 checkpoint only.

## Components

### `build_teacher_bank`
- **Does**: Grow a reusable bank of mature walking donor states paired with internal ring tokens
- **Rationale**: A fixed bank provides proportional on-manifold refresh without paying a 300-step teacher rollout every iteration

### `make_optimizer`
- **Does**: Freeze donor repair during flow-only warmup, then enable joint flow/repair tuning
- **Rationale**: Transport must first find useful gradients instead of allowing the reaction rule to absorb every error

### `main`
- **Does**: Load dense targets and NC3C donor or resume TN3D1, maintain a state/global pool, refresh 25% of each batch, score multiple rollout segments, apply surface-aware loss and flow regularization, and checkpoint TN3D1
- **Interacts with**: `transport_nca3d.py`, `transport_targets3d.py`, `train_cyclic3d.py`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU launch | Default grid 32³, batch 4, pool 64, 24-step rollout, 400 flow-only + 1600 joint iterations; damage off unless explicitly scheduled | CLI/defaults |
| Evaluator | Checkpoint stores model iteration and the full coupled rule | TN3D1 payload |
| Baseline | `shoggoth3d.nca` is NC3C with walking behavior at cond2=1 | Donor format/semantics |

## Notes
- Damage is deliberately off for the coherence A/B so the flow→joint stage transition changes only one variable. Regeneration, seed growth, part-attention, and 64³ scaling are later gates.
