# main.swift

## Purpose
Entry point. Dispatches between the desktop-pet app (default), legacy headless
render tests, and the fused runtime test (`--render-fused out.png steps model.fx2d state.ncs`).

## Components

### top-level code
- **Does**: Arg parsing; sets `.accessory` activation policy (status-bar app, no Dock icon); runs NSApplication
- **Interacts with**: `AppDelegate`, `RenderTest`, `FusedRenderTest`

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CLI users / scripts | `--render-test [path] [steps] [weights]` and `--render-seq dir count stride [weights]` flag shapes | Flag renames |
| Fused runtime verification | `--render-fused out steps weights state` | Flag or argument order |
| `scripts/make_app.sh` | `--render-test` used for icon generation | Flag renames |
