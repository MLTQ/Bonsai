# train_hidden_momentum.py

## Purpose
Trains the second clockless inertia treatment: residual visible channels, explicit hidden velocity, and conditioned-teacher distillation. It is a separate experiment so the failed NCA4 baseline remains reproducible.

## Components

### `alive_masked_mse`
- **Does**: Measures hidden/velocity error only where the conditioned teacher is alive
- **Rationale**: Empty background would otherwise dominate twelve hidden channels with trivial zeros

### `initial_pool`
- **Does**: Creates synchronized NCA5 student and NCA2 teacher states with phase/behavior labels

### `main`
- **Does**: Advances student and teacher together, supervises authored RGBA, teacher hidden state, interval-averaged hidden velocity, and pooled oscillator channels, then exports NCA5
- **Rationale**: Interval averaging removes most one-step stochastic-fire noise from the velocity target
- **Rationale**: CUDA defaults to activation checkpointing, preserving stochastic RNG state while recomputing steps so batch 16 fits alongside long rollouts

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| RTX experiment launch | `--batch-size` defaults to 16; output defaults to `shoggoth_auto_hidden_momentum.nca` | CLI/defaults |
| `hidden_momentum_nca.py` | Position channels 4–15 align with teacher hidden channels; velocity starts at state channel 16 | Layout |
| `train_cyclic.py` | NCA2 teacher, target frames, phase schedule, damage helper | Teacher semantics |

## Notes
- Default auxiliary weights reflect measured teacher scales: hidden RMS ≈0.74 and interval velocity RMS ≈0.009.
- Runtime integration is intentionally gated on trajectory quality; NCA5 is initially a training artifact.
