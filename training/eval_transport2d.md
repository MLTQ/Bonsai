# eval_transport2d.py

## Purpose

Runs a paired stochastic multi-cycle comparison of the reaction-only NCA2
donor and the global-transport treatment. It saves target/baseline/treatment
rows and metrics that distinguish crisp directed motion from a low-MSE blend.

## Components

### `donor_step`

- **Does**: Evaluates behavior-0 NCA2 with the exact fire mask used by the
  treatment.
- **Rationale**: Paired stochasticity removes fire-mask luck from the A/B.

### `dynamic_part_masks` / `part_phase_metrics`

- **Does**: Builds four target-derived motion quadrants and independently
  estimates gait phase in each region.
- **Rationale**: Global coherence means moving regions agree on direction and
  phase; whole-image MSE cannot establish that.

### `softness` / `sharpness_deficit` / `nonadjacent_leakage`

- **Does**: Measures excess semi-transparent pixels, lost alpha edge energy,
  and fitted key-pose mass outside the active directed edge.
- **Rationale**: These are the characteristic signatures of fused-frame blur.

### `main`

- **Does**: Starts both rules from one mature donor state, runs repeated paired
  240-step cycles, aggregates metrics, and emits a three-row preview and JSON.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment decision | Treatment should reduce softness, sharpness deficit, phase dispersion/error, leakage, and drift without materially worsening MSE | Metric definitions |
| Preview reader | Rows are target, baseline, treatment; columns advance in time | Sheet ordering |
| Corpus | Exactly four reviewed anchors | Leakage/phase basis |

