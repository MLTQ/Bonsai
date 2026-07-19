# train_manifold3d.py

## Purpose
The convergence trainer (Mk. IV): volumetric + cyclic + tanh-FiLM manifold. State space S¹ × [0,1]^10 — animation as traversal, in 3D. NC3M export (film matrices after base weights).

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| `manifold_shoggoth3d.py` | corpus npz (N,F,G,G,G,4) f16 + z | Z_SPEC order |
| Swift NC3M path | tanh-bounded gamma (CPU-side in refreshFilm) | FiLM form |
