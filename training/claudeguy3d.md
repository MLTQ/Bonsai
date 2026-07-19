# claudeguy3d.py

## Purpose
The capstone creature's volumetric anatomy and 13-expression library (Max's emotion sheet + wideeyed + modecollapse). Petal ring with spin/wiggle params, circular face ellipsoid, canonical bulging eyes, :3 mouth; sheet-ratio feature painter with props (hat, monocle, cloud, nose, groucho kit, mouth-tentacles).

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| future capstone trainers | draw_claudeguy(phase, blink, look, petal_flex, expression, spin, wiggle, sheet) | param semantics |
| `design/claudeguy/DESIGN.md` | anatomy matches the blessed art direction | major redesigns need Max |

## Notes
- FEATURE_SCALE=1.45 after "features too small" direction; integer-harmonic rule applies to all motion params.
