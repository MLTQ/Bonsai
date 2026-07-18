# train_cyclic.py

## Purpose
Trains a phase-conditioned *cyclic* NCA: the target is a moving point on a closed animation loop, so the automaton learns coherent animation (talking, blinking) as tracking dynamics — the experimental heart of this repo. Self-contained sibling of train_nca.py (shared recipe, plus conditioning).

## Components

### `CyclicNCA`
- **Does**: Same 16-channel CA as train_nca.py but w1 takes 48+COND inputs; cond = (sin θ, cos θ, behavior) broadcast to every cell and concatenated after perception
- **Interacts with**: Mirrored by `ncaMetalSource(cond:)` in Swift — cat order is load-bearing

### Training loop (`main`)
- **Does**: Pool training with per-slot (state, θ, behavior); θ advances OMEGA=2π/240 per step *during* rollouts; loss vs phase-lerped targets (`target_at`) at 3 rollout checkpoints (weighted toward the end); SWITCH_P=0.15 behavior re-roll mid-life (transition training); circular damage after iter 500
- **Rationale**: Advancing phase during rollout is what makes this *animation* — the network must track motion, not memorize snapshots. Mid-life behavior flips train still↔talk conversions of an existing body

### `export`
- **Does**: NCA2 format = NCA1 + int32 cond count in header, wider w1

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCAWeights.swift` | NCA2 layout | Header/order changes |
| `Creature.swift` (`LainBehavior.omega`) | OMEGA = 2π/240 | Cycle-length changes must be mirrored |
| `lain.py` | FRAMES=12, BEHAVIORS=2 | Cond semantics |

## Notes
- `cyclic_preview.png` at each checkpoint: 12 talk-cycle snapshots after growth — the fast visual check that tracking is converging.
