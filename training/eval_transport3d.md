# eval_transport3d.py

## Purpose
Runs paired stochastic multi-cycle evaluation of the reaction-only NC3C donor and the global transport treatment. It emits a three-row projection sheet and JSON metrics designed around sharpness, directed-pose purity, body-wide phase agreement, and cycle drift.

## Components

### `donor_step`
- **Does**: Evaluate NC3C with an injected fire mask shared by the treatment
- **Rationale**: Paired stochasticity removes fire-mask luck from the A/B comparison

### `softness` / `part_phase_metrics` / `nonadjacent_leakage`
- **Does**: Measure excess boundary voxels, independently inferred phase in four XZ body quadrants, and key-pose mixture mass outside the active directed edge
- **Rationale**: MSE alone rewards temporal averages and cannot establish coherent animation

### `main`
- **Does**: Start both models from one mature donor state, run repeated 240-step cycles, aggregate metrics, and save target/baseline/treatment projections

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment decision | Treatment should improve softness, nonadjacent leakage, part-phase dispersion/error, and drift without materially worsening MSE | Metric definitions |
| Preview reader | Rows are target, baseline, treatment; columns are consecutive stride captures | Sheet ordering |

## Notes
- Quadrant phase is a proxy suitable for the radial Shoggoth. A rigged humanoid successor should replace it with canonical part masks.
