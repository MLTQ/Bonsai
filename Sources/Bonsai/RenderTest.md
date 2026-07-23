# RenderTest.swift

## Purpose
Headless verification path: grows the NCA for N steps and writes a PNG, proving the Metal runtime reproduces the trained dynamics without needing a window or eyeballs on screen.

## Components

### `RenderTest.run(outputPath:steps:weightsPath:)`
- **Does**: Load weights and optional `$BONSAI_INITIAL_STATE` NCS1 snapshot → simulate → `readRGBA()` → PNG
- **Interacts with**: `NCAWeights`, `NCASimulation.readRGBA`; conditioned weights (cond ≥ 3) get a phase-advancing condProvider with the behavior flag on (`LainBehavior.omega`)
- **Override**: `$BONSAI_BEHAVIOR=0` selects a one-behavior cyclic corpus such
  as the Mega Man mature-state experiment; the historical default remains 1

### `RenderTest.runSequence(outDir:count:stride:weightsPath:)`
- **Does**: Warm-up growth, then dumps a PNG every `stride` steps — the frames become verification GIFs of the animation cycle

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.swift` | `run`/`runSequence` return process exit codes | Signatures |

## Notes
- CI-less verification: `bonsai --render-test out.png 300 [weights]`, or `--render-seq dir 24 10 weights/lain.nca` for animation.
