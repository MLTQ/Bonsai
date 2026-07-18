# RenderTest.swift

## Purpose
Headless verification path: grows the NCA for N steps and writes a PNG, proving the Metal runtime reproduces the trained dynamics without needing a window or eyeballs on screen.

## Components

### `RenderTest.run(outputPath:steps:weightsPath:)`
- **Does**: Load weights → simulate → `readRGBA()` → 4× nearest-upscale → PNG via ImageIO
- **Interacts with**: `NCAWeights`, `NCASimulation.readRGBA`; conditioned weights (cond ≥ 3) get a phase-advancing condProvider with the behavior flag on (`LainBehavior.omega`)

### `RenderTest.runSequence(outDir:count:stride:weightsPath:)`
- **Does**: Warm-up growth, then dumps a PNG every `stride` steps — the frames become verification GIFs of the animation cycle

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `run`/`runSequence` return process exit codes | Signatures |

## Notes
- CI-less verification: `bonsai --render-test out.png 300 [weights]`, or `--render-seq dir 24 10 weights/lain.nca` for animation.
