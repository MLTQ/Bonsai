# train_heteroclinic.py

## Purpose
Max's unstable-attractor experiment: two pose targets under one flag, NO clock; the loss always aims at the pole OPPOSITE the nearest, making every pose unstable toward the other. Oscillation (and its period) should emerge from transit dynamics — heteroclinic cycling / winnerless competition.

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `--target` | 2d_states npz with exactly 2 states | pole count |
| verdict tooling | NCA2 cond=1; render with BONSAI_STATE=1, look for periodic recurrence | — |

## Notes
- Baseline for comparison: the failed oscillator-distillation clockless run (paper §Limitations).
