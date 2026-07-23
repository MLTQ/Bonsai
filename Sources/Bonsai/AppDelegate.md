# AppDelegate.swift

## Purpose
Composition root: transparent floating window, status-bar item, creature loading/switching from the registry, and hot-reload of legacy or fused weights.

## Components

### `AppDelegate`
- **Does**: App lifecycle; owns window, status item, current sim + behavior
- **Interacts with**: `Creature.registry`, legacy/fused weights and simulations, `PetView`, `FusedPetView`

### `load(creature:)`
- **Does**: Loads legacy or FX2D weights, builds the matching shape-specialized sim, optionally replaces its seed with a mature NCS1 state, wires behavior conditioning, and swaps the window view
- **Rationale**: Sim rebuild (not weight swap) is required whenever cond width differs

### `makeWindow`
- **Does**: Borderless, clear, shadowless, `.floating`-level window joining all Spaces; frame autosaved ("BonsaiPetWindow") so the pet stays where you left it
- **Rationale**: This combination is the entire "desktop pet" illusion

### `watchWeights`
- **Does**: Polls the current creature's weights mtime every 2 s; hot-swaps legacy or fused banks on change, rebuilding the sim if `updateWeights` reports a shape mismatch
- **Rationale**: Lets a running training session visibly improve the pet live

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `AppDelegate()` + `applicationDidFinishLaunching` | Class name/protocol |
| `Creature.swift` | Registry consulted at launch and in menu rebuilds | Registry shape |
