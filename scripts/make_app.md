# make_app.sh

## Purpose
Packages the release build as a double-clickable `dist/Bonsai.app`: binary, trained weights, optional mature-state snapshots, metadata, and an icon grown by the baseline NCA.

## Components
- Release build → bundle skeleton → `.nca`/`.fx2d`/`.ncs`/JSON resource copy → `--render-test` icon render → iconset/icns → ad-hoc codesign

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCAWeights.weightsDir()` | Weights land in `Contents/Resources` (its bundled fallback) | Moving weights elsewhere |
| `Creature.initialStateName` | Matching NCS1 snapshots land beside their weights | Omitting `.ncs` copy |
| `FusedNCAWeights` | Exported `.fx2d` banks land beside their canonical state | Omitting `.fx2d` copy |
| End users | Double-click launches the pet with bundled weights | Info.plist changes |

## Notes
- Ad-hoc signed only; fine locally, would need a real identity for distribution.
