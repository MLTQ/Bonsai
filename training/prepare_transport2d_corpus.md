# prepare_transport2d_corpus.py

## Purpose

Turns one reviewed 2x2 diffusion sheet into a higher-resolution four-anchor
`2d_cycle` corpus. It preserves relative motion inside the four cells by using
one union crop and one scale for every frame.

## Components

### `split_sheet`

- **Does**: Splits a sheet into four equal row-major cells.
- **Contract**: Pose order is contact-gold, passing, contact-teal, passing.

### `key_near_white`

- **Does**: Converts a near-white background into a soft alpha matte.
- **Rationale**: Diffusion backgrounds are not exact white; the soft band keeps
  anti-aliased line art while preventing a gray slab from entering training.

### `keep_largest_component`

- **Does**: Keeps the largest eight-connected foreground component plus a
  two-pixel antialias band.
- **Rationale**: Diffusion sometimes adds detached motion-ink specks; they are
  not creature state and would otherwise distort the shared crop and loss.

### `prepare_frames`

- **Does**: Computes one union subject box, crops every cell identically,
  resizes with one scale, centers on a shared grid, and premultiplies RGB.
- **Rationale**: Independent auto-crops would accidentally remove the gait's
  body translation and scale changes.
- **Contract**: Placement uses alpha compositing exactly once; soft matte alpha
  must not be squared by using the source as an additional paste mask.

### `main`

- **Does**: Writes a standard `(1,4,H,W,4)` `2d_cycle` NPZ plus extracted PNGs
  and a strip preview for mandatory visual review.
- **Naming**: `--frame-names F0 F1 F2 F3` records authored pose names instead
  of the historical gold/teal defaults; it does not change tensor order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `train_cyclic.py` | `kind=2d_cycle`, frames in premultiplied RGBA | NPZ layout |
| Transport trainer/evaluator | Exactly four row-major anchors | Frame count/order |
| Runtime experiment | Square grid; default 128, eight-cell margin | Scaling/canvas |

## Notes

- The script rejects sheets with foreground touching a cell edge; a clipped
  limb cannot be repaired by training.
- Largest-component filtering is enabled by default. Use
  `--keep-all-components` only for intentionally disconnected creature art.
- A visually wrong sheet must be discarded, even if all numeric checks pass.
