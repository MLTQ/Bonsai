# NCASimulation.swift

## Purpose
GPU runtime for the neural cellular automaton: owns Metal state buffers, weights, and compute pipelines. It dispatches both legacy 16-channel residual formats and the 32-channel NCA4 position/velocity format.

## Components

### `NCASimulation`
- **Does**: Holds grid state and advances the automaton; renders into drawable textures
- **Interacts with**: `NCAWeights` (weight data), `ncaMetalSource` in `NCAShaders.swift` (compiled at init)
- **Rationale**: Three-buffer rotation (`cur`/`tmp`/`next`) because the life mask reads pre- and post-update alpha from neighbors. State buffers use the loaded format's channel count; visible RGBA is always the first four position channels

### `step(count:renderInto:)`
- **Does**: Encodes N automaton steps (+optional render) in one command buffer
- **Rationale**: Encoders in one command buffer execute serially on the GPU, so multiple steps per frame are cheap

### `reseed` / `damage(atGridX:gridY:radius:)` / `updateWeights`
- **Does**: Plant a fresh seed cell / queue a circular wound (applied next step) / hot-swap weights (returns false on cond-shape mismatch → caller rebuilds the sim)
- **Interacts with**: Called by `PetView` (interactions), `AppDelegate` (menu, weights watcher), `LainBehavior` (glitches)

### `condProvider` / `renderStyle` / `condCount` / `flipX`
- **Does**: Per-step conditioning values (phase, behavior flag) fed into uniforms; render style flag; the cond width the shader was compiled with; horizontal render mirror
- **Rationale**: The shader is generated with conditioning, hidden width, FiLM use, and state/integrator shape baked in, so shape changes require a fresh simulation

### `zTarget` / `setZ` / `refreshFilm` (NCA3)
- **Does**: Mood steering. `zTarget` eases ~4%/tick (moods morph); `setZ` jumps immediately (headless tests). gamma/beta = filmW·z + filmB computed CPU-side into a small buffer the step kernel reads
- **Rationale**: z is uniform across cells, so FiLM params are a few thousand CPU MACs per tick — no need for the conditioning matrix on the GPU

### `readRGBA`
- **Does**: Sync copy of visible channels off the GPU
- **Interacts with**: `RenderTest` only

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `PetView.swift` | `step(count:renderInto:)`, `damage`, `reseed`, `device`, `gridWidth/Height` | Signatures; must stay main-thread-callable |
| `AppDelegate.swift` | failable `init(device:weights:...)`; `updateWeights` rejects cond/hidden/FiLM/pool/state-shape mismatches | Init and hot-reload behavior |
| `RenderTest.swift` | `readRGBA()` returns row-major RGBA floats | Channel order/layout |

## Notes
- `Uniforms` uses only 4-byte fields in declaration order so Swift matches Metal without manual padding. `momentumDecay` is ignored by residual shaders.
