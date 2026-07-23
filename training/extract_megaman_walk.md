# extract_megaman_walk.py

## Purpose

Extracts the authored four-frame run sequence from `sheets/megaman`, removes
the JPEG's baked checkerboard, and aligns every frame by a shared helmet anchor
without independently scaling or recentering the silhouette.

## Components

### `FRAMES`

- **Does**: Records reviewed source boxes and helmet anchors for the third-row
  flight-extension, support, compression, and flight-recovery sequence.
- **Rationale**: The sheet is not a regular grid; explicit provenance is safer
  than guessing cell boundaries from whitespace.
- **Naming**: Pose names describe visible silhouettes only. They deliberately
  do not claim persistent anatomical left/right labels that the symmetric art
  does not encode.

### `checker_distance`

- **Does**: Reconstructs the 15px two-color checker pattern and measures each
  pixel's distance from its expected background color.
- **Rationale**: A generic white key would preserve gray squares and erase
  white gloves.

### `extract_sprite`

- **Does**: Selects the largest authored connected component, restores enclosed
  neutral glove regions, and generates a soft alpha edge. Bright neutral JPEG
  seams are excluded even when compression makes them differ from the ideal
  checker template.
- **Contract**: The JPEG is a flattened reference, so recovered alpha is an
  approximation and requires visual inspection.

### `align_sprite`

- **Does**: Places every unscaled sprite on the same 144px canvas using its
  helmet center.
- **Rationale**: Independent centering would erase authored stride and body
  motion; independent scaling would create fake squash/stretch.

### `diagnostics`

- **Does**: Reports adjacent/opposite premultiplied RGBA differences and alpha
  support.
- **Limit**: Numeric uniqueness does not establish named-leg continuity. The
  extracted sheet must still be reviewed as an animation before NCA ingestion.

### `save_animation`

- **Does**: Writes a 4x nearest-neighbor looping GIF at 120 ms per anchor.
- **Rationale**: Motion order and loop closure are easier to reject in motion
  than in a static contact sheet.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus preparation | White 2x2 row-major `sheet.png` | Pose order/background |
| Visual audit | Unscaled sprites with shared helmet anchor | Alignment policy |
| Reproducibility | Exact 728x1279 source sheet | Source dimensions/crops |

## Notes

- The extraction preserves the user-supplied artwork; it does not generate or
  repaint missing poses.
- If the authored sequence itself repeats one leg phase, it must be rejected or
  augmented explicitly rather than described as a valid four-state gait.
