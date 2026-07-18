# Bonsai 🌱

A desktop pet that is not a spritesheet.

Bonsai is a little tree that lives on your desktop in a transparent, borderless,
always-on-top window. Its body is a **neural cellular automaton** (Mordvintsev et
al., [*Growing Neural Cellular Automata*](https://distill.pub/2020/growing-ca/)):
every pixel is a cell running a tiny learned update rule on the GPU, 60 times a
second. Nothing is pre-drawn. The tree *grows* from a single seed cell, breathes,
shimmers — and if you poke a hole in it, it regrows, because regeneration was
part of its training.

## Quick start

```sh
# 1. Train the organism (~35 min on Apple Silicon; checkpoints every ~1 min)
cd training && python3 train_nca.py    # needs: torch, numpy, pillow

# 2. Run the pet (works as soon as the first checkpoint exists)
swift run -c release
```

The app hot-reloads `weights/bonsai.nca` every 2 seconds, so you can launch the
pet the moment the first checkpoint lands and watch it get better at being a
tree while training continues.

## Interactions

- **Drag** — pick the pet up and move it anywhere (it follows you across Spaces)
- **Click** — poke a hole in it; watch it heal
- **Right-click** — regrow from seed, or quit
- **🌱 menu bar item** — same, plus manual weights reload

## Layout

- `Sources/Bonsai/` — Swift app: transparent window shell + Metal NCA runtime
- `training/` — PyTorch training rig and the procedural target artwork
- `weights/` — trained `.nca` weight files (binary format: see `train_nca.py::export`)

Every code file has a companion `.md` documenting intent and contracts.

## Verifying without a window

```sh
swift run Bonsai --render-test out.png 300   # grow 300 steps headless, write PNG
```
