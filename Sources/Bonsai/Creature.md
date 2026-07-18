# Creature.swift

## Purpose
The creature registry (weights file + presentation + autonomy per creature) and the behavior layer that gives conditioned creatures a life of their own. Adding a creature = one registry entry + optionally one behavior class.

## Components

### `Creature`
- **Does**: Name, weights filename, render style, behavior factory; `registry` lists all known creatures; `isAvailable` checks the weights file exists
- **Interacts with**: `NCAWeights.weightsDir()` for path resolution; consumed by `AppDelegate`

### `CreatureBehavior` (protocol)
- **Does**: `cond(step:)` supplies per-step conditioning values; `tick(sim:)` runs once per display tick for autonomous acts
- **Rationale**: Separates the *organism* (NCA dynamics) from the *personality* (when to talk, when to glitch)

### `LainBehavior`
- **Does**: Phase clock (`omega` = 2π/240, must match train_cyclic.py OMEGA); random murmur episodes (behavior flag → 1 for 5–15 s); occasional small self-damage "glitches" that heal
- **Interacts with**: `NCASimulation.damage`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `AppDelegate.swift` | `registry`, `isAvailable`, `path`, `makeBehavior` | Registry shape |
| `PetView.swift` | `tick(sim:)` cheap, main-thread | Protocol changes |
| `RenderTest.swift` | `LainBehavior.omega` constant | Cycle length changes must match training |
