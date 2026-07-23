# test_transport_nca2d_contract.py

## Purpose

CPU contract tests for the 2D global-ring transport experiment. They protect
the spatial sign convention, ring closure, donor transplant, checkpoint
format, and shared-crop corpus preparation before GPU time is spent.

## Tests

- `test_global_ring_closes_after_240_steps` — fixed controller completes one
  cycle without phase drift.
- `test_positive_x_flow_moves_impulse_right` — XY/backward-warp sign is correct.
- `test_zero_flow_transplant_matches_behavior_zero_donor` — treatment begins as
  the exact reaction-only baseline under paired stochasticity.
- `test_tn2d1_checkpoint_roundtrip` — experimental format preserves parameters
  and iteration.
- `test_corpus_preparation_preserves_relative_motion` — union crop keeps
  per-frame translation and emits premultiplied RGBA.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| GPU experiment | All tests pass on CPU | Any failure |
| Evaluator | Positive X means visible motion right | Warp sign |
| A/B claim | Zero-flow treatment equals donor | Transplant layout |

