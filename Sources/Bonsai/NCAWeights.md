# NCAWeights.swift

## Purpose
Loader for legacy residual, pooled, manifold, volumetric, and NCA4 momentum `.nca` binaries plus weights-directory resolution. Keeps all file-format knowledge in one place on the Swift side.

## Components

### `NCAWeights`
- **Does**: Parses the format-specific header, dynamic state/output shapes, flat float32 weight block, and optional FiLM matrices
- **Rationale**: Weights remain one flat array because the GPU consumes them with offsets computed in its generated shader. NCA4 records 32 state channels, 16 position/output channels, and momentum decay; legacy formats remain 16-channel residual systems

### `weightsDir` / `defaultPath`
- **Does**: Finds the weights directory ($BONSAI_WEIGHTS_DIR → ./weights → repo-relative → bundled Resources) / legacy bonsai.nca convenience

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | `flat` ordered w1,b1,w2,b2; dynamic `stateChannels` / `positionChannels`; integrator parameters | Order, shape fields |
| `Creature.swift` | `weightsDir()` | Search-order changes |
| `AppDelegate.swift`, `RenderTest.swift` | `load(from:)` throws descriptive errors | Signatures |
| `training/*.py` | Byte-level format agreement for all emitted magic values, including NCA4 | Any header/layout change must be mirrored |

## Notes
- Header-length checks are format-specific so malformed files fail as `truncated` before any unaligned scalar read.
