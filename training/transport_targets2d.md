# transport_targets2d.py

## Purpose

Owns the reviewed four-anchor sprite targets and edge-aware visible objective
for the 2D transport experiment. Interpolation is strictly between the current
anchor and its directed successor; distant poses can never enter one target.

## Components

### `load_cycle_frames`

- **Does**: Loads a one-behavior, four-frame `2d_cycle` corpus into `(4,4,H,W)`.
- **Rationale**: Rejecting any other topology keeps the hard four-edge
  experiment honest.

### `target_at_global`

- **Does**: Crossfades only the two anchors on the active directed edge.
- **Rationale**: The user explicitly permits nearby-pose blending during a
  transition; hard adjacency forbids the damaging distant-pose average.

### `visible_objective`

- **Does**: Combines premultiplied color, alpha, alpha-gradient, and
  premultiplication penalties.
- **Rationale**: Edge loss gives transport a direct incentive to keep the
  silhouette crisp even where adjacent RGBA supervision is soft.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Trainer/evaluator | Frames `(4,4,H,W)` on device | Shape/order |
| Ring controller | Anchor `k` occupies phase `k*pi/2` | Edge partition |
| Corpus preparation | Exactly one behavior and four frames | Topology |

