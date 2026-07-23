# FusedPetView.swift

## Purpose

Hosts `FusedNCASimulation` in a transparent `CAMetalLayer` without forcing the
specialized transport runtime through the legacy `NCASimulation` API.

## Components

- Advances two automaton steps per 30 Hz display tick.
- Preserves display sleep pausing, window dragging, click damage, reset, and
  transparent premultiplied rendering from the standard `PetView` experience.
- Reset restores the reviewed canonical pose-zero state because growth from a
  single cell was not part of the fused experiment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `AppDelegate.swift` | `init(simulation:frame:)` | Initializer |
| `FusedNCASimulation.swift` | Device/grid/step/damage/reset/mirror properties | Runtime API |
