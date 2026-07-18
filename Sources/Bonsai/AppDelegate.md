# AppDelegate.swift

## Purpose
Composition root: transparent floating window, status-bar item, creature loading/switching from the registry, and hot-reload of the current creature's weights file.

## Components

### `AppDelegate`
- **Does**: App lifecycle; owns window, status item, current sim + behavior
- **Interacts with**: `Creature.registry`, `NCAWeights`, `NCASimulation`, `PetView`

### `load(creature:)`
- **Does**: Loads weights, builds a sim (shader compiled for that creature's cond width), wires behavior → condProvider, swaps the window's PetView; persists choice in UserDefaults
- **Rationale**: Sim rebuild (not weight swap) is required whenever cond width differs

### `makeWindow`
- **Does**: Borderless, clear, shadowless, `.floating`-level window joining all Spaces; frame autosaved ("BonsaiPetWindow") so the pet stays where you left it
- **Rationale**: This combination is the entire "desktop pet" illusion

### `watchWeights`
- **Does**: Polls the current creature's weights mtime every 2 s; hot-swaps on change, rebuilding the sim if `updateWeights` reports a shape mismatch
- **Rationale**: Lets a running training session visibly improve the pet live

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `AppDelegate()` + `applicationDidFinishLaunching` | Class name/protocol |
| `Creature.swift` | Registry consulted at launch and in menu rebuilds | Registry shape |
