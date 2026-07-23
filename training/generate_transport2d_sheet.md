# generate_transport2d_sheet.py

## Purpose

Generates candidate four-anchor gait sheets for the 2D global-transport NCA
experiment. All four poses are sampled in one 2x2 SDXL image so identity,
palette, line weight, and proportions have a stronger shared constraint than
four independent text-to-image calls.

## Components

### `PROMPT` / `NEGATIVE_PROMPT`

- **Does**: Requests exactly four views of one non-human seed-pod robot, with
  rigid color-coded legs and ordered contact/passing gait poses.
- **Rationale**: The simple silhouette makes gait phase and blur measurable;
  the flat near-white backdrop is compatible with Bonsai white-key ingestion.
- **Contract**: Both strings stay below the checkpoint's 77-token CLIP window;
  layout and background requirements must not be silently truncated.

### `main`

- **Does**: Loads an Illustrious SDXL single-file checkpoint with model CPU
  offload, renders one 1024px sheet per seed, and records the prompt metadata.
- **Interacts with**: `prepare_transport2d_corpus.py` after visual review.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Corpus preparation | Four equal cells in row-major gait order | Layout or pose order |
| GPU launch | The caller pins one GPU with `CUDA_VISIBLE_DEVICES`; the script never guesses an ordinal | Device selection |
| Asset policy | Stylized, non-human subject; every candidate reviewed before ingestion | Prompt subject |

## Notes

- Default seeds are deliberately few because the RTX 2070 SUPER uses CPU
  offload and each 1024px SDXL sample is relatively slow.
- The first draft used a 146-token prompt and was rejected after the model
  truncated every layout/background instruction and rendered irregular tan
  model sheets. Keep essential constraints concise and early.
- The checkpoint path is explicit rather than tied to the removable Gold
  drive. Pass `--checkpoint` when using another host.
