# shoggoth.py

## Purpose
Generates the authored 2D shoggoth animation used as the cyclic teacher target. It produces a closed 12-phase idle/walk loop on a 64×64 premultiplied-RGBA canvas.

## Components

### `draw_shoggoth`
- **Does**: Renders one supersampled animation frame from a phase and walking flag
- **Interacts with**: The private blob, tentacle, eye, and disk raster helpers
- **Rationale**: Integer phase frequencies keep the last-to-first frame transition continuous

### `make_frames`
- **Does**: Returns both behaviors across all phases as `(2, 12, 64, 64, 4)` float32 data
- **Interacts with**: `_load_creature` in `train_cyclic.py`, which installs this function as the active target source

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_cyclic.py` | `GRID=64`, `FRAMES=12`, `BEHAVIORS=2`, and premultiplied RGBA | Dimensions, behavior ordering, alpha convention |
| `train_hidden_momentum.py` | Behavior 0 is idle and behavior 1 is walk | Behavior semantics |

## Notes
- This is the 2D target generator; it is unrelated to the seeded 32³→64³ creature pipeline.
