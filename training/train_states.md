# train_states.py

## Purpose
Multi-state attractor NCA (the werewolf pattern): dramatically different forms under a state flag; metamorphosis learned via mid-life switches. All motion is NCA-native — shimmer within states, transformation between.

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `tools/ingest.py states` | `kind=2d_states`, 2 states (cond=1 flag) | >2 states → manifold trainer |
| Swift `StateBehavior` | NCA2 cond=1; flag semantics 0=first state | state order in manifest |
