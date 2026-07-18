# NCAWeights.swift

## Purpose
Loader for the `.nca` binary weights format produced by `training/train_nca.py::export()`. Keeps the file-format knowledge in one place on the Swift side.

## Components

### `NCAWeights`
- **Does**: Parses header (magic, shape, fire rate) and the flat float32 weight block
- **Rationale**: Weights kept as one flat array because the GPU consumes them as a single buffer with fixed offsets (shader computes w1/b1/w2/b2 pointers itself)

### `defaultPath`
- **Does**: Finds the weights file: `$BONSAI_WEIGHTS` → `./weights/bonsai.nca` → repo path relative to the built executable

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | `flat` ordered w1,b1,w2,b2; `channels=16`, `hidden=128` | Order, shape constants |
| `AppDelegate.swift`, `RenderTest.swift` | `load(from:)` throws descriptive errors; `defaultPath()` | Signatures |
| `training/train_nca.py` | Byte-level format agreement | Any header/layout change must be mirrored |
