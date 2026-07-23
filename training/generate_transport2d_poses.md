# generate_transport2d_poses.py

## Purpose

Generates four identity-preserving gait anchors from one approved diffusion
hero. It alters only the lower-limb placement in temporary conditioning guides,
then uses SDXL img2img to repaint every final frame.

## Components

### `extract_hero`

- **Does**: Crops one approved model-sheet cell, estimates its border color,
  extracts the subject with a soft matte, and recenters it on white.
- **Rationale**: The candidate sheet may contain faint gray layout lines that
  must not become part of the identity reference.
- **Contract**: The selected crop must exclude neighboring model-sheet cells;
  the default border-distance rejects the sheet's faint grid.

### `make_pose_guide`

- **Does**: Continuously shears/scales the two visual lower-half regions in
  opposite contact/passing patterns while leaving the body reference fixed.
  Displacement is zero at the hip line and grows toward each boot. Exact-white
  pixels are transparent while the regions move.
- **Contract**: The default split is below the arms/torso; only the two visual
  leg regions may be displaced.
- **Rationale**: Low-strength img2img protects identity but resists pose change;
  moving the init pixels supplies the missing spatial instruction. Guides are
  conditioning only and are never ingested.

### `main`

- **Does**: Repaints four ordered anchors with one checkpoint and seed, saves
  guides separately, and assembles the four final diffusion outputs into a 2x2
  sheet for `prepare_transport2d_corpus.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus preparation | Final sheet row-major order: contact-gold, passing-gold, contact-teal, passing-teal | Pose order |
| Asset policy | Only `frames/`, never `guides/`, enters training | Directory semantics |
| GPU launch | Caller pins the 2070S; CPU offload is enabled | Device selection |

## Notes

- The base and per-pose prompts remain below the 77-token CLIP window.
- This implements the repository guide's approved transform-init/repaint
  technique; the final assets are diffusion outputs rather than cut-and-paste
  sprites.
- Two translation-guide drafts were rejected because one boot stayed detached,
  even at higher denoising strength. The current guide uses hip-anchored shear;
  continuity is structural rather than delegated to diffusion.
