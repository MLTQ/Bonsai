# layered_transport_metrics2d.py

## Purpose

Defines differentiable transition constraints and independent acceptance
metrics for the layered transport experiment. Checkpoint selection must reject
blurred low-loss compromises rather than optimize one scalar blindly.

## Components

### `transition_shape_penalty`
- **Does**: Keeps intermediate squared alpha-gradient energy and support within adjacent
  anchor bands while enforcing premultiplied color and valid alpha
- **Rationale**: No RGBA crossfade target is used between anchors; these bounds
  keep the learned path sharp without dictating its pixels. Squared gradient
  energy is intentional: L1 total variation is conserved by a normalized blur
  and therefore cannot detect the failure this experiment targets.

### `alpha_dice` / `boundary_f1`
- **Does**: Measure filled-silhouette agreement and 2px-tolerant contour match

### `sharpness_ratio` / `support_ratio`
- **Does**: Expose lost squared-gradient edge energy and double-exposure area
  independently of endpoint loss

### `rgb_gradient_error` / `rgb_sharpness_ratio`
- **Does**: At exact destination anchors, measure RGB gradient mismatch and
  retained squared color-edge energy.
- **Rationale**: A crisp alpha silhouette can still contain a featureless blur
  where the face and armor details should be. These metrics are not used to
  prescribe ambiguous intermediate pixels.

### `nonadjacent_leakage`
- **Does**: Fits all four anchors and measures mass outside the active adjacent
  pair
- **Rationale**: Uses a regularized 4x4 normal-equation solve rather than the
  full least-squares operator so identical evaluation runs on CUDA, CPU, and
  Apple MPS.

### `dynamic_part_masks` / `part_phase_metrics`
- **Does**: Infer phase independently in four motion-weighted regions and
  report disagreement/error
- **Rationale**: Whole-image agreement cannot prove global pose coherence

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer | Shape penalty is differentiable and batch-reduced | Return types |
| Evaluator | Frames `(4,4,H,W)` and predictions `(B,4,H,W)` | Tensor order |
| Acceptance gate | Dice/boundary/sharpness/support have higher-is-better or ratio semantics as documented | Definitions |
