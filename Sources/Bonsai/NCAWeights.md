# NCAWeights.swift

## Purpose
Loader for legacy residual, pooled, manifold, volumetric, NCA4 full-momentum, and NCA5 hidden-momentum `.nca` binaries plus weights-directory resolution. Keeps all file-format knowledge in one place on the Swift side.

## Components

### `NCAWeights`
- **Does**: Parses the format-specific header, dynamic state/output/momentum shapes, flat float32 weight block, and optional FiLM matrices
- **Rationale**: Weights remain one flat array because the GPU consumes them with offsets computed in its generated shader. NCA4 gives all 16 positions velocity; NCA5 gives only the 12 hidden positions velocity while RGBA remains residual

### `weightsDir` / `defaultPath`
- **Does**: Finds the weights directory ($BONSAI_WEIGHTS_DIR → ./weights → repo-relative → bundled Resources) / legacy bonsai.nca convenience

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | `flat` ordered w1,b1,w2,b2; dynamic state/position/momentum widths; integrator parameters | Order, shape fields |
| `Creature.swift` | `weightsDir()` | Search-order changes |
| `AppDelegate.swift`, `RenderTest.swift` | `load(from:)` throws descriptive errors | Signatures |
| `training/*.py` | Byte-level format agreement for all emitted magic values, including NCA4/NCA5 | Any header/layout change must be mirrored |

## Notes
- Header-length checks are format-specific so malformed files fail as `truncated` before any unaligned scalar read.
