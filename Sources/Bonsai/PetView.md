# PetView.swift

## Purpose
The pet's visible body and touch surface: hosts the CAMetalLayer, runs the 30 fps tick loop, and maps mouse gestures to pet interactions.

## Components

### `PetView`
- **Does**: NSView backed by CAMetalLayer; timer tick = behavior tick + N sim steps + present; pauses while displays sleep (NSWorkspace notifications)
- **Interacts with**: `NCASimulation` (step/damage/reseed), `CreatureBehavior?` (autonomy)
- **Rationale**: Timer added to `.common` run-loop mode so the pet keeps animating while being dragged

### Mouse handling
- **Does**: Drag >4pt moves the window (`performDrag`); a clean click pokes a hole in the pet (it regrows); right-click shows a context menu (regrow / quit)
- **Rationale**: `isMovableByWindowBackground` is off — the view arbitrates click-vs-drag itself so both gestures can coexist

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `AppDelegate.swift` | `init(simulation:frame:)` | Init signature |

## Notes
- Grid coordinates are top-left origin; view coordinates bottom-left — `gridPoint(for:)` flips Y. Poke bugs are usually this.
