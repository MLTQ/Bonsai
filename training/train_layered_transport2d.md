# train_layered_transport2d.py

## Purpose

Trains the hard-edge, four-flow treatment directly from reviewed mature
anchors. It separates anchor repair, flow-only motion, joint tuning, and
two-edge scheduled sampling so growth cannot confound coherence.

## Components

### `stabilize_episode`
- **Does**: Adds small live-region noise to exact anchors and trains local
  repair while transport is disabled
- **Rationale**: The blurred v3 reaction donor is not transplanted

### `transition_episode`
- **Does**: Starts from an exact source anchor, rolls one active edge with no
  RGBA midpoint target, constrains intermediate sharpness/support, and scores
  the adjacent destination endpoint
- **Rationale**: Linear frame crossfades explicitly teach double exposure

### `stage_at` / `make_optimizer`
- **Does**: Runs repair stabilization, flow-only attribution, then joint tuning

### Two-edge scheduled sampling
- **Does**: After `--chain-after`, trains the second edge from the first edge's
  predicted state while detaching between edges
- **Rationale**: Exposes off-anchor recurrent state without unbounded BPTT

### `save_preview`
- **Does**: Runs a deterministic chained cycle; target endpoints are row one
  and predictions are row two

### `main`
- **Does**: Logs endpoint Dice, boundary F1, sharpness ratio, flow magnitude,
  slot entropy/collapse, writes `LT2D1` checkpoints, saves previews, and reports
  peak allocated CUDA memory for batch-size selection

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus | One behavior, four premultiplied RGBA anchors | Topology/order |
| Model | `LayeredTransportNCA2D.step_with_aux` return order | Aux contract |
| Evaluator | `LT2D1` checkpoints and transition-step metadata | Format/schedule |

## Notes

- Default source states are 128² mature anchors. Point growth and damage remain
  out of scope until chained transport passes.
- Checkpoint activation recomputation preserves RNG state so stochastic repair
  masks match forward/backward execution.
