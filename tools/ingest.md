# ingest.py

## Purpose
Artist/asset ingestion: images, spritesheets, meshes, posed-mesh directories, and labeled state manifests → trainer-ready `.npz` targets. The escape hatch from primitive-based modeling.

## Components
- `image` / `sheet` — PNG(s) → 2D targets; `--key-white` converts near-white bg to alpha (threshold 228 on min-channel, soft band)
- `mesh` / `meshcycle` — trimesh voxelization: surface sampling with vertex/face/texture colors, filled interiors, soft shells; `(z,y,x,c)` y-up, grid via `BONSAI_GRID3`
- `states` — manifest json `{state: image}` → multi-state ATTRACTOR npz (`kind=2d_states`); no synthesized motion by design (all animation belongs to the NCA)

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| trainers' `--target` | kinds `2d`, `2d_cycle`, `2d_states`, `3d`, `3d_cycle`; premultiplied float | npz keys/shapes |
| Swift raymarcher previews | mesh output order (z,y,x,c) | axis order |

## Notes
- Rules for generated inputs: stylized models only, non-human subjects, human review before ingest (see guide §9).
