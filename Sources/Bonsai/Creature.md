# Creature.swift

## Purpose
The creature registry (weights file + presentation + autonomy per creature) and the behavior layer that gives conditioned creatures a life of their own. Adding a creature = one registry entry + optionally one behavior class.

## Components

### `Creature`
- **Does**: Name, weights filename, render style, behavior factory, optional mature NCS1 initial state, and a dedicated fused-2D runtime flag; `registry` lists known creatures and `isAvailable` checks weights
- **Interacts with**: `NCAWeights.weightsDir()` for path resolution; consumed by `AppDelegate`

### `CreatureBehavior` (protocol)
- **Does**: `cond(step:)` supplies per-step conditioning values; `tick(sim:)` runs once per display tick for autonomous acts
- **Rationale**: Separates the *organism* (NCA dynamics) from the *personality* (when to talk, when to glitch)

### `LainBehavior`
- **Does**: Phase clock (`omega` = 2π/240, must match train_cyclic.py OMEGA); random murmur episodes (behavior flag → 1 for 5–15 s); occasional small self-damage "glitches" that heal
- **Interacts with**: `NCASimulation.damage`

### `ShoggothBehavior`
- **Does**: Same phase clock; alternates idle/walk episodes. Walking: settles the window to the Dock rail (visibleFrame.minY − footMargin), glides horizontally ~22 px/s, turns at screen edges, sets `sim.flipX` to face travel direction (art faces right)
- **Rationale**: Locomotion = trained gait in place + window translation; the NCA is the body, AppKit is the legs

### `ClocklessShoggothBehavior`
- **Does**: Holds the single behavior flag at walk (`cond0=1`) without supplying phase or moving the window
- **Interacts with**: The NCA4 `Shoggoth · Momentum` and NCA5 `Shoggoth · Hidden Momentum` preview entries
- **Rationale**: A fixed behavior isolates the experiment: any sustained gait timing must live in position/velocity state, not runner conditioning

### `PhaseOnlyCyclicBehavior`
- **Does**: Supplies the shared 240-step sin/cos clock with behavior flag zero
- **Interacts with**: `Mega Man · Mature Test`, whose one-behavior NCA2 corpus
  has no valid behavior-one target
- **Rationale**: Reusing `LainBehavior` would eventually toggle an untrained
  behavior flag; reusing locomotion behavior would move the desktop window

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `AppDelegate.swift` | `registry`, `isAvailable`, `path`, `makeBehavior` | Registry shape |
| `PetView.swift` | `tick(sim:window:)` cheap, main-thread | Protocol changes |
| `RenderTest.swift` | `LainBehavior.omega` constant | Cycle length changes must match training |
| `training/shoggoth.py` | footMargin matches art (tentacle tips ~row 56/64); rightward-facing walk art | Art layout changes |
| `training/train_autonomous.py` | NCA4 checkpoint uses one conditioning channel where `1` selects walk | Conditioning semantics |
| `training/train_hidden_momentum.py` | NCA5 checkpoint uses the same one-channel walk condition and a 28-channel mature state | Conditioning or state layout |
| `training/train_cyclic.py` | Mega Man uses a 240-step phase and behavior flag zero | Phase/condition semantics |
| `FusedNCASimulation.swift` | `Mega Man · Fused Experts` loads FX2D plus its canonical NCS1 state | Fused filename/runtime flag |

## Notes
- Both momentum previews load mature teacher-derived states because the clockless curricula did not train single-seed growth; this keeps each preview on its evaluated state distribution.
- `Mega Man · Mature Test` likewise requires its bundled 128² NCS1 snapshot;
  “Regrow from Seed” is expected to go blank because growth was not trained.
- `Mega Man · Fused Experts` uses the separate hard-routed transport runtime;
  resetting restores canonical pose zero rather than planting a single cell.
