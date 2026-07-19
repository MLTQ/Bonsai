# StateMapPanel.swift

## Purpose
The state-space explorer: per-creature 2D mood geography with a draggable cursor. Manifold creatures show their UMAP constellation (kNN-inverted 2D→z); flag creatures get synthesized two-island maps that send anchor commands.

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `AppDelegate` | `make(for: Creature?)`; panel invalidated on creature switch | init shape |
| control channel | writes {"z": ...} or {"anchor": ...} ≤4 Hz | payload |
| `tools/make_statemap.py` | json schema | keys |
