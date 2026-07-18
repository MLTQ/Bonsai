# AppDelegate.swift

## Purpose
Composition root: creates the transparent floating window, the status-bar item, loads weights, and hot-reloads the weights file when training writes a new checkpoint.

## Components

### `AppDelegate`
- **Does**: App lifecycle; owns window, status item, simulation
- **Interacts with**: `NCAWeights.defaultPath()/load`, `NCASimulation`, `PetView`

### `makeWindow`
- **Does**: Borderless, clear, shadowless, `.floating`-level window that joins all Spaces
- **Rationale**: This combination is the entire "desktop pet" illusion — the window is invisible except the pet's pixels

### `watchWeights`
- **Does**: Polls the weights file mtime every 2 s; reloads on change
- **Rationale**: Lets a running training session visibly improve the pet live; polling (not FSEvents) because 2 s latency is fine and it's 10× simpler

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `AppDelegate()` + `applicationDidFinishLaunching` | Class name/protocol |

## Notes
- Window spawns bottom-right of the main screen. If multiple pets are ever supported, window creation moves out of here.
