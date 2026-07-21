# NCAShaders.swift

## Purpose
Metal source for the NCA kernels, generated per creature and compiled at runtime. Conditioning, hidden width, state shape, and integrator layout are baked in; uniforms carry runtime fire rate and momentum decay. Must remain numerically equivalent to the corresponding eager PyTorch model.

## Components

### `nca_step` (kernel)
- **Does**: Computes interleaved identity/Sobel perception, conditioning and optional pooled inputs, then the learned update. Legacy formats apply a stochastic residual; NCA4 predicts force, damps velocity, and advances position with symplectic Euler
- **Rationale**: Perception ordering `[c*3+0..2]` matches PyTorch grouped convolution. NCA4 gates force with the per-cell fire mask while integration continues every step, matching `MomentumNCA`

### `nca_life` (kernel)
- **Does**: Zeroes cells not alive (maxpool3x3 of alpha > 0.1) both before and after the update; writes to a third buffer
- **Rationale**: Separate pass because post-update aliveness depends on neighbors' post-update alpha — can't be fused with `nca_step` without a race

### `nca_render` (kernel)
- **Does**: Nearest-neighbor upscale of state RGBA into the drawable, clamped, premultiplied-alpha enforced; style 1 adds faint CRT scanlines (Lain)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCASimulation.swift` | Kernel names, buffer indices, `Uniforms` field order (all 4-byte fields) | Any signature/layout change |
| residual trainers + `training/momentum_nca.py` | Numerical equivalence (shape, kernels, ordering, integrator, life/fire/clamp semantics) | Any math change must be mirrored there |

## Notes
- Zero-padding at borders matches PyTorch: conv pads zeros; maxpool pads -inf but the >0.1 threshold makes 0 vs -inf equivalent.
