# generate_rigged_seedpod_walk.py

## Purpose

Creates the first trustworthy four-anchor 2D gait corpus as a deterministic
layered rig. Right and left are persistent semantic objects, not an inference
from four independently generated pictures. Diffusion may later restyle these
frames, but it is not allowed to define the experiment's pose truth.

## Components

### `POSES`

- **Does**: Defines hip, knee, ankle, and toe coordinates for contact-right,
  passing-left, contact-left, and passing-right.
- **Contract**: The right/gold leg has positive screen order in frame 0 and
  negative screen order in frame 2. Foreground order is right, left, left,
  right around the ring.

### `_leg_layer` / `_mask_leg`

- **Does**: Renders each named limb independently and emits its exact semantic
  mask. Right is gold; left is violet, distinct from the teal body.
- **Rationale**: Semantic masks remain valid even where the visible limbs
  overlap. Color inspection alone cannot recover the occluded leg.

### `_body_layer`

- **Does**: Renders one high-contrast, side-facing seedpod robot body shared
  pixel-for-pixel by all anchors.
- **Rationale**: The first NCA experiment should isolate coherent limb
  transport from diffusion-induced identity, lighting, and background drift.

### `render_pose`

- **Does**: Draws the far leg, the invariant body, then the declared foreground
  leg. Supersampling retains clean curves after downsampling.
- **Contract**: Layer order is an explicit part of each pose, not an accidental
  result of raster geometry.

### `validate`

- **Does**: Measures horizontal centroids from semantic masks, requires
  opposite signs at the two contacts, requires passing poses to be less
  extreme, and verifies the foreground sequence.
- **Rationale**: The rejected diffusion sheets passed pixel-distance checks
  while never changing semantic leg order. This is the acceptance gate those
  metrics lacked.

### `main`

- **Does**: Exports four 512px RGBA frames, per-leg masks, a 2x2 sheet, and
  `metadata.json`. It exits nonzero on any semantic contract failure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus preparation | 2x2 row-major sheet on white | Sheet order/background |
| NCA targets | contact-right, passing-left, contact-left, passing-right | Pose order |
| Semantic review | Gold always means right; violet always means left | Palette mapping |
| Occlusion audit | Foreground sequence right, left, left, right | Draw order |

## Notes

- Default source resolution is 512px per frame, then corpus preparation makes
  the shared 128px NCA grid.
- The invariant torso is intentional. Arm swing and body bob can be added only
  after the core four-state transport test succeeds.
- Generated files are experiment artifacts; this source and companion document
  define their reproducible provenance.
