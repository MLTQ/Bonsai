# transport_nca3d.py

## Purpose
Defines the isolated 32³ advection–reaction treatment for globally coherent gait. It augments a local 3D NCA with one shared ring oscillator, hard selection among four directed transition edges, and learned spatial transport before donor-initialized repair.

## Components

### `make_global` / `advance_global` / `phase_features`
- **Does**: Encode and rotate the internal `[sin, cos]` controller; derive progress and exactly one active edge
- **Rationale**: Every voxel receives the same directed phase, and nonadjacent edge heads cannot mix

### `warp_state`
- **Does**: Backward-warps NCDHW state using an XYZ displacement field in voxel units
- **Rationale**: Motion transports material instead of requiring reaction dynamics to erase and regrow limbs

### `TransportNCA3D`
- **Does**: Runs shared 3D perception, edge-selected flow, trilinear transport, stochastic local repair, life masking, and global-state advancement
- **Interacts with**: `train_transport3d.py`, `eval_transport3d.py`

### `transplant_nc3c`
- **Does**: Copies the existing walking NC3C donor into all repair edges and folds behavior=1 into the repair bias
- **Rationale**: With zero flow, the treatment starts step-equivalent to the reaction-only donor

### `save_transport_checkpoint` / `load_transport_checkpoint`
- **Does**: Persist the experimental `TN3D1` PyTorch checkpoint
- **Rationale**: Runtime integration is deliberately deferred until the treatment beats the baseline

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_cyclic3d.CyclicNCA3D` | 16 channels, 128 hidden units, `[sin, cos, behavior]`, Sobel ordering | Donor layout |
| Trainer/evaluator | Global state is `[sin, cos]`; four edges partition one 240-step cycle | Phase order or cycle length |
| `grid_sample` | State is NCDHW; flow channels are XYZ; base grid stores XYZ | Axis order or `align_corners` |

## Notes
- The first treatment uses a fixed internal oscillator. Learned part-attention and completion-triggered hysteresis are gated on this core experiment succeeding.
