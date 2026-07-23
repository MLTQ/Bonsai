# train_cyclic.py

## Purpose
Trains a phase-conditioned *cyclic* NCA: the target is a moving point on a closed animation loop, so the automaton learns coherent animation (talking, blinking) as tracking dynamics — the experimental heart of this repo. Self-contained sibling of train_nca.py (shared recipe, plus conditioning).

## Components

### `CyclicNCA`
- **Does**: Same 16-channel CA as train_nca.py but w1 takes 48+COND inputs; cond = (sin θ, cos θ, behavior) broadcast to every cell and concatenated after perception
- **Interacts with**: Mirrored by `ncaMetalSource(cond:)` in Swift — cat order is load-bearing

### Training loop (`main`)
- **Does**: Pool training with per-slot (state, θ, behavior); θ advances OMEGA=2π/240 per step *during* rollouts; loss vs phase-lerped targets (`target_at`) at 3 rollout checkpoints (weighted toward the end); SWITCH_P=0.15 behavior re-roll mid-life only for two-behavior corpora; optionally scheduled circular damage. Ingested sparse RGBA corpora use the shared foreground/alpha/edge-aware `visible_objective`; built-in historical creatures retain their original canvas-wide MSE.
- **Rationale**: Advancing phase during rollout is what makes this *animation* — the network must track motion, not memorize snapshots. Mid-life behavior flips train still↔talk conversions of an existing body

### `sample_error`
- **Does**: Ranks pool samples with the same foreground/alpha emphasis used by
  ingested-corpus training, while preserving historical raw-MSE ranking for
  built-in creatures.
- **Rationale**: At 128², a sparse sprite occupies too little of the canvas for
  raw MSE; predicting transparent black becomes a low-loss absorbing solution.

### `make_mature_state` / `--init-mode target`
- **Does**: Embeds an interpolated target RGBA image in channels 0–3 and copies
  alpha into hidden channels 4–15, then initializes and refreshes pool slots
  from these live mature states. Preview rollout begins from anchor 0 instead
  of a point seed.
- **Rationale**: Mature-state initialization isolates the requested coherent
  pose dynamics from the separate problem of growing a ~100px character from
  one cell. A failed point-growth curriculum must not be mistaken for evidence
  about global animation.
- **Contract**: Target initialization is allowed only for ingested corpora.
  Point-seed mode remains the historical default and growth must be trained as
  a later curriculum before runtime export.

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
- Batch, pool size, rollout range, damage start, checkpoint interval, and preview
  path are CLI-configurable while preserving the historical defaults. This is
  required for higher-resolution ingested cycles and bounded GPU benchmarks.
- One-behavior corpora always use behavior condition 0 in both training and
  previews; behavior flipping is disabled rather than creating an invalid
  target index.
