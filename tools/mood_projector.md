# mood_projector.py

## Purpose
Text → mood-space: the semantic steering layer. Powers `mood.sh --text`, feed watching, and the trace daemon (creature as ambient display of agent cognition).

## Components
- `Projector` — MiniLM embeddings of per-anchor phrase banks; softmax-similarity blend over anchors → z
- `--text` / `--watch FILE` / `--trace` — one-shot / feed daemon / transcript daemon (`~/.claude/projects/*/*.jsonl`, override `BONSAI_TRACE_GLOB`; EMA smoothing = temperament)

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| creature behaviors | writes `weights/control.json` `{"z": [...]}` | payload shape |
| anchors json | names shared with `PHRASES` banks | renaming anchors |

## Notes
- All local; opt-in only. Enrich phrase banks when a mood mishears (success vocabulary was once too thin — content read as dread).
