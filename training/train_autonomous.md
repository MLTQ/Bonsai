# train_autonomous.py

## Purpose
Runs the clockless-cycle experiment by distilling a phase-conditioned NCA into a model that receives only the behavior flag. Supports a legacy residual baseline and the NCA4 momentum treatment for a direct A/B test.

## Components

### `AutoNCA`
- **Does**: Legacy 16-channel residual baseline with one conditioning input
- **Interacts with**: `CyclicNCA` from `train_cyclic.py`

### `teacher_states`
- **Does**: Grows conditioned donor states at random phases and returns a one-step finite-difference velocity estimate
- **Rationale**: Momentum pool slots start with both pose and direction of travel, avoiding an underdetermined cold start

### `main`
- **Does**: Builds the selected integrator, maintains phase-labelled pool slots, applies sync curriculum/damage, supervises visible trajectory and internal oscillator channels, and checkpoints weights
- **Interacts with**: `MomentumNCA` / `export_nca4` for `--integrator momentum`; NCA2 export for `--integrator residual`

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Autonomous experiment | `--integrator residual|momentum`; momentum is the default | CLI names/default |
| `NCAWeights.swift` | Residual emits NCA2 cond=1; momentum emits NCA4 cond=1 | Export layout |
| `train_cyclic.py` | Donor is NCA2 cond=3 and uses `OMEGA=2π/240` | Donor format / tempo |

## Notes
- Triton fused rollout is intentionally not used: this experimental trainer is eager PyTorch and NCA4 has a 32-channel phase-space state.
- Momentum donor states are lifted with `velocity = state_t - state_(t-1)`. The conditioned donor first drops its sin/cos columns into the cond=1 residual baseline, then that rule is lifted and its update head scaled by `1-momentum_decay`.
