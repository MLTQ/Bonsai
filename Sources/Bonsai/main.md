# main.swift

## Purpose
Entry point. Dispatches between the desktop-pet app (default) and the headless render test (`--render-test`).

## Components

### top-level code
- **Does**: Arg parsing; sets `.accessory` activation policy (status-bar app, no Dock icon); runs NSApplication
- **Interacts with**: `AppDelegate`, `RenderTest`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CLI users / scripts | `--render-test [path] [steps] [weights]` and `--render-seq dir count stride [weights]` flag shapes | Flag renames |
| `scripts/make_app.sh` | `--render-test` used for icon generation | Flag renames |
