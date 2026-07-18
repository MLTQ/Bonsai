# lain.py

## Purpose
Procedural animation frames for the Lain-esque face creature: 2 behaviors (still, talking) × 12 phases of a closed animation loop. These frames are the *points on the cycle* the phase-conditioned NCA learns to track — not a spritesheet the app plays back.

## Components

### `draw_face(mouth, lid, pupil_dx)`
- **Does**: One 64×64 RGBA frame: bob hair with asymmetric side locks, X hair clip, pale skin, mismatched wide eyes with off-center gaze, parameterized mouth/eyelids/pupils
- **Rationale**: Everything anti-symmetric on purpose — the unease is the aesthetic

### `STILL` / `TALK` (phase specs) and `make_frames`
- **Does**: 12-phase specs per behavior (blink at phases 8–9 in both; pupil saccade at 3–4 in still; mouth cycle in talk); `make_frames()` → (2, 12, 64, 64, 4) float32 premultiplied
- **Interacts with**: `train_cyclic.py` (targets), FRAMES/BEHAVIORS/GRID constants shared

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_cyclic.py` | `make_frames()` shape and premultiplied range | FRAMES/BEHAVIORS changes ripple into cond semantics |

## Notes
- `python3 lain.py` writes `lain_sheet.png` (full 2×12 sheet) for eyeballing.
