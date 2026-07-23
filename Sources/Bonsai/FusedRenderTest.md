# FusedRenderTest.swift

## Purpose

Runs an exported `FX2D` model plus canonical `NCS1` state through the production
Metal fused runtime and writes raw nearest-neighbor RGBA evidence headlessly.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `main.swift` | `run(outputPath:steps:weightsPath:statePath:)` | CLI dispatch |
| Runtime verification | Premultiplied 4× PNG from raw state after an exact step count | Output semantics |
