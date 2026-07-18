# target.py

## Purpose
Procedurally draws the pixel-art bonsai (pot, S-curve trunk, foliage clouds) that the NCA learns to grow. Procedural rather than a PNG asset so the tree is tweakable in code and the repo stays binary-free.

## Components

### `make_target`
- **Does**: Returns (64,64,4) float32 RGBA, premultiplied alpha, drawn 4× supersampled then box-downsampled for soft edges
- **Interacts with**: `train_nca.py` (training target); `GRID` constant shared as the NCA grid size
- **Rationale**: Soft AA edges train better than hard pixel edges (matches the emoji targets in the original paper)

### `_stroke`, `_disk`
- **Does**: Tapered quadratic-bezier strokes (trunk/branch) and filled disks (foliage, stamped dark-to-light)

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_nca.py` | `make_target()` shape (GRID,GRID,4), premultiplied, values 0..1 | Shape/range; changing GRID requires matching the Swift grid size |

## Notes
- `python3 target.py` writes `target_preview.png` for eyeballing. Redesigning the tree = editing shapes here + retraining.
