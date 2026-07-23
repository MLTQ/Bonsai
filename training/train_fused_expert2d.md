# train_fused_expert2d.py

## Purpose

Trains four sharp pose attractors and four directed transition experts as one
hard-routed checkpoint. It treats the canonical hidden state and destination
handoff as explicit interfaces rather than hoping independently specialized
rules remain compatible.

## Components

### `anchor_episode` / `perturb_anchor`
- **Does**: Applies substantial visible/hidden noise, sparse live-cell dropout,
  and 6-12px erasure to exact canonical key states, then trains only the
  selected pose expert to return visible and hidden channels to its attractor.
- **Interacts with**: `canonical_key_state` in `fused_state2d.py`.
- **Rationale**: Tiny noise produced a near-identity pose rule that could not
  pull imperfect edge arrivals into a destination basin. Logs expose initial
  corruption and actual recovery gain separately.
  Exact RGB-gradient recovery prevents the destination basin from sharpening
  only alpha while smoothing face and armor details.

### `edge_episode`
- **Does**: Runs one or two predicted directed edges, scores every edge before
  handoff, then runs and scores the destination pose expert.
- **Rationale**: An edge cannot hide a broken arrival behind wholesale redraw
  by the destination attractor. Detachment between chained edges bounds BPTT
  while retaining deployment-like off-anchor inputs.
- **Rationale**: Hidden arrival errors and full-state range are logged; values
  outside `[-2,2]` receive a soft penalty before the hard `[-8,8]` safety clamp.
  Intermediate edge-energy bounds remain permissive between unlike adjacent
  anchors, but bridge and handoff endpoints receive an exact destination-only
  sharpness/support gate.
  Exact endpoints also receive RGB-gradient matching so a correct silhouette
  cannot conceal a featureless face/armor blur.

### `_key_state`
- **Does**: Selects canonical or alpha-repeated state initialization.
- **Rationale**: `--state-interface alpha` is the controlled hidden-interface
  ablation; all other training settings can remain fixed.

### `save_preview`
- **Does**: Saves target, midpoint, pre-handoff, and post-handoff rows from one
  stochastic but repeatable four-edge chained cycle.
- **Rationale**: Training-batch endpoint metrics previously concealed severe
  recurrent color and sharpness failure.

### `stage_at` / `make_optimizer` / `main`
- **Does**: Runs anchor-only, edge-only, and joint phases; joint training uses
  25% anchor replay and switches to two-edge scheduled sampling.
- **Interacts with**: backward-compatible `FEX2D1` and current `FEX2D2`
  checkpoints from `fused_expert_nca2d.py`.
- **Rationale**: Hard motion slots are the default. `--soft-slots` is retained
  solely for a controlled ablation against recurrent multi-warp averaging.
  Width and Fourier-coordinate CLI arguments make the F1/F2 capacity change
  explicit and checkpointed.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment ledger | Fixed transition/handoff schedule and named ablation | CLI semantics |
| Evaluator | Preview metadata identifies state interface and step counts | JSON fields |
| Runtime export | One fused checkpoint contains every conditional rule bank and width/coordinate metadata | Checkpoint format |

## Notes

- The oracle guide is intentional in F1. Learning autonomous switching before
  transition experts pass would confound routing and transport failures.
- No interpolated RGBA midpoint is used as a training target.
