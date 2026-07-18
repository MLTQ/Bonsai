# RenderTest.swift

## Purpose
Headless verification path: grows the NCA for N steps and writes a PNG, proving the Metal runtime reproduces the trained dynamics without needing a window or eyeballs on screen.

## Components

### `RenderTest.run(outputPath:steps:)`
- **Does**: Load weights → simulate → `readRGBA()` → 4× nearest-upscale → PNG via ImageIO
- **Interacts with**: `NCAWeights`, `NCASimulation.readRGBA`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `run` returns process exit code | Signature |

## Notes
- Used in CI-less verification: `bonsai --render-test out.png 300` then look at the image.
