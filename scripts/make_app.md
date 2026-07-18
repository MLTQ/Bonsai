# make_app.sh

## Purpose
Packages the release build as a double-clickable `dist/Bonsai.app`: binary + all trained weights in Resources + Info.plist (LSUIElement — status-bar app, no Dock icon) + an icon that is literally a render grown by the trained NCA.

## Components
- Release build → bundle skeleton → weights copy → `--render-test` icon render → iconset/icns → ad-hoc codesign

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCAWeights.weightsDir()` | Weights land in `Contents/Resources` (its bundled fallback) | Moving weights elsewhere |
| End users | Double-click launches the pet with bundled weights | Info.plist changes |

## Notes
- Ad-hoc signed only; fine locally, would need a real identity for distribution.
