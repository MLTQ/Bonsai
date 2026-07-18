# NCAWeights.swift

## Purpose
Loader for `.nca` binary weight files (NCA1 from train_nca.py, NCA2 from train_cyclic.py) plus weights-directory resolution. Keeps all file-format knowledge in one place on the Swift side.

## Components

### `NCAWeights`
- **Does**: Parses header (magic NCA1/NCA2, shape, cond-channel count, fire rate) and the flat float32 weight block
- **Rationale**: Weights kept as one flat array because the GPU consumes them as a single buffer with offsets computed in the shader. `cond` (0 for NCA1) determines the w1 row width: 48 + cond

### `weightsDir` / `defaultPath`
- **Does**: Finds the weights directory ($BONSAI_WEIGHTS_DIR → ./weights → repo-relative → bundled Resources) / legacy bonsai.nca convenience

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | `flat` ordered w1,b1,w2,b2; `cond`; `channels=16`, `hidden=128` | Order, shape constants |
| `Creature.swift` | `weightsDir()` | Search-order changes |
| `AppDelegate.swift`, `RenderTest.swift` | `load(from:)` throws descriptive errors | Signatures |
| `training/*.py` | Byte-level format agreement (both NCA1 and NCA2) | Any header/layout change must be mirrored |
