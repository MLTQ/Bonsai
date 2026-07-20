# train_manifold3d.py

## Purpose
The convergence trainer (Mk. IV): volumetric + cyclic + tanh-FiLM manifold. State space S¹ × [0,1]^10 — animation as traversal, in 3D. NC3M export (film matrices after base weights).

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `manifold_shoggoth3d.py` | corpus npz (N,F,G,G,G,4) f16 + z | Z_SPEC order |
| Swift NC3M path | tanh-bounded gamma (CPU-side in refreshFilm) | FiLM form |

### `--fused` (2026-07-19)
Fused rollout (Triton): gamma/beta computed once per iter in torch (autograd chains through the Function's dgamma/dbeta), sin/cos conds precomputed (T,B,2), per-step seed (it, i). Replaces checkpointing. Measured 1.3x vs eager on 4090 at 32³. See fused_step.md.
