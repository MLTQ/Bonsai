# eval_fused_expert2d.py

## Purpose

Runs deployment-like stochastic chained cycles and independently decides
whether a fused or layered checkpoint is sharp, coherent, bounded, and recurrently stable.
It is the acceptance authority; training-batch loss is not.

## Components

### `_append_endpoint`
- **Does**: Measures visible MSE, alpha Dice, boundary F1, squared-gradient
  sharpness, support, non-adjacent leakage, regional phase agreement, and state
  range before and after pose-expert handoff.
- **Rationale**: RGB gradient error and energy separately reject a sharp outer
  silhouette filled with blurred face/armor details.

### `_append_intermediate`
- **Does**: Measures sharpness against adjacent-anchor edge energy, support-band
  violation, distant-pose leakage, regional phase agreement, flow magnitude,
  and motion-slot usage at quarter-edge intervals.
- **Rationale**: Endpoint-only metrics cannot expose double exposure or a
  transition that destroys and redraws the creature.

### `_corrupt_anchors` / `_anchor_metrics`
- **Does**: Applies deterministic visible/hidden noise, sparse dropout, and a
  local erasure to every pose, then measures 12 pose-expert recovery steps.
- **Rationale**: Sharp reviewed endpoints do not by themselves prove the pose
  banks are attractors; recovery must reduce both visible and hidden error.

### `main`
- **Does**: Runs deterministic damaged-anchor recovery plus multiple stochastic
  trials through repeated four-edge cycles, records visible/hidden cycle drift,
  saves a four-row preview, and emits named numeric acceptance gates.
- **Interacts with**: `FEX2D1` checkpoints and the chosen canonical/alpha state
  interface. `--architecture layered --state-interface alpha` applies the same
  evidence standard to the `LT2D1` control.

## Preview contract

Rows are target endpoints, transition midpoints, states immediately before
handoff, and states after eight destination-attractor steps. Columns are the
four directed edges in cycle order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment decision | Named gates and `numeric_pass` in JSON | Threshold or key changes |
| Visual review | Four rows and four edge columns | Preview ordering |
| Ablation comparison | Same architecture flag, trials, seed, cycles, and metric definitions | Evaluation schedule |

## Notes

- Numeric pass is necessary but not sufficient; obvious double exposure,
  color explosion, or disappearing regions is a manual visual failure.
- The primary gate uses stochastic fire masks matching runtime behavior.
