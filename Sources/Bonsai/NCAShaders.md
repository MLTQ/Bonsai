# NCAShaders.swift

## Purpose
Metal source for the NCA kernels, generated per creature (`ncaMetalSource(cond:)`) and compiled at runtime — no `.metal` build step, and the cond-channel count is baked in as a shader constant. Must remain numerically equivalent to the PyTorch models in `training/train_nca.py` and `training/train_cyclic.py`.

## Components

### `nca_step` (kernel)
- **Does**: Perception (identity + sobelX + sobelY per channel, interleaved), then COND conditioning values from uniforms appended → 128-wide ReLU layer → 16-channel residual update with stochastic per-cell fire mask; applies queued damage circle
- **Rationale**: Perception ordering `[c*3+0..2]` matches PyTorch's `groups=CH` conv output ordering, and conditioning is appended after the 48 features exactly as `torch.cat` does in training — this equivalence is what lets Python-trained weights run unchanged

### `nca_life` (kernel)
- **Does**: Zeroes cells not alive (maxpool3x3 of alpha > 0.1) both before and after the update; writes to a third buffer
- **Rationale**: Separate pass because post-update aliveness depends on neighbors' post-update alpha — can't be fused with `nca_step` without a race

### `nca_render` (kernel)
- **Does**: Nearest-neighbor upscale of state RGBA into the drawable, clamped, premultiplied-alpha enforced; style 1 adds faint CRT scanlines (Lain)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | Kernel names, buffer indices, `Uniforms` field order (all 4-byte fields) | Any signature/layout change |
| `training/train_nca.py`, `training/train_cyclic.py` | Numerical equivalence (kernels, ordering, cond append, life rule, fire semantics) | Any math change must be mirrored there |

## Notes
- Zero-padding at borders matches PyTorch: conv pads zeros; maxpool pads -inf but the >0.1 threshold makes 0 vs -inf equivalent.
