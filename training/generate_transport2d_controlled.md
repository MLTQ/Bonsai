# generate_transport2d_controlled.py

## Purpose

Generates a four-anchor walk cycle with explicit OpenPose conditioning and a
machine-checkable semantic leg swap. This replaces the rejected unconstrained
img2img corpus in which boot positions changed but the same leg remained in
front throughout.

## Components

### `POSES` / `make_pose_map`

- **Does**: Defines four OpenPose body skeletons in normalized coordinates:
  gold/right contact, passing, teal/left contact, passing.
- **Rationale**: Left/right limbs have distinct OpenPose colors and explicit
  ankle trajectories; text alone cannot reliably establish occlusion order.

### `load_pipeline`

- **Does**: Loads Illustrious SDXL plus the Xinsir SDXL OpenPose ControlNet and
  automatically selects whole-model offload on GPUs with at least 16 GB or
  sequential, layer-wise offload on smaller cards.
- **Interacts with**: The caller pins the GPU by UUID.
- **Rationale**: Whole-component offload still peaks above the card's 8 GB once
  the UNet and ControlNet execute together; sequential offload keeps the 2070S
  viable, while model offload makes use of the freed 4090 for rapid iteration.

### `leg_centroids` / `validate_leg_swap`

- **Does**: Segments gold and teal pixels in the lower body, measures their
  horizontal centroids, and requires contact frames 0 and 2 to have opposite
  ordering with a meaningful margin.
- **Rationale**: A visually plausible sheet is still invalid unless semantic
  leg order actually swaps. Segmentation begins below 60% image height so body
  accents cannot satisfy the test. Numeric validation runs before NCA
  ingestion; visual review remains mandatory afterward.

### `main`

- **Does**: Generates four pose-controlled frames from one approved identity
  image, saves pose maps and outputs separately, assembles a 2x2 sheet, writes
  validation JSON, and exits nonzero when the swap contract fails.
- **Dry run**: `--poses-only` renders `pose_controls_sheet.png` without loading
  either diffusion model, so gait geometry can be reviewed independently of
  image synthesis.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus preparation | Row-major contact-right, passing-left, contact-left, passing-right | Pose order |
| Leg validator | Right leg/boot is gold; left leg/boot is dark teal | Palette semantics |
| GPU launch | 2070S selected externally; CPU offload enabled | Device selection |
| Asset policy | Non-human, stylized, white background, reviewed before ingestion | Prompt/content |

## Notes

- ControlNet weights are `xinsir/controlnet-openpose-sdxl-1.0` (Apache-2.0).
- The rejected v3 shear-guided sheet and its aborted baseline are retained only
  as diagnostic artifacts; they are not valid experiment data.
