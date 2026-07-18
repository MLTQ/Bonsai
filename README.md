# Bonsai 🌱

Desktop pets that are not spritesheets.

Each creature lives on your desktop in a transparent, borderless, always-on-top
window. Its body is a **neural cellular automaton** (Mordvintsev et al.,
[*Growing Neural Cellular Automata*](https://distill.pub/2020/growing-ca/)):
every pixel is a cell running a tiny learned update rule on the GPU, 60 times a
second. Nothing is pre-drawn. Creatures *grow* from a single seed cell, breathe,
shimmer — and if you poke a hole in one, it regrows, because regeneration was
part of its training.

## Creatures

- **Bonsai** — a little tree. Static attractor: grows, persists, heals.
- **Lain** — an eerie face with a bob haircut. *Phase-conditioned cyclic NCA*:
  the automaton's conditioning channels carry `(sin θ, cos θ, behavior)` and the
  phase advances continuously, so the target is a moving point on a closed loop
  through sprite-space. She blinks slowly, sometimes murmurs (mouth cycle), and
  occasionally tears a small hole of static in herself, which heals. The
  animation is not frame playback — it is the automaton tracking a limit cycle,
  which is (as far as we know) lightly charted research territory: see
  `training/train_cyclic.py` for the recipe, including mid-life behavior
  switching so still↔talking transitions are trained rather than improvised.

## Quick start

```sh
# 1. Train an organism (~35–60 min on Apple Silicon; checkpoints every ~1 min)
cd training && python3 train_nca.py       # the bonsai   → weights/bonsai.nca
cd training && python3 train_cyclic.py    # lain         → weights/lain.nca
# needs: torch, numpy, pillow

# 2. Run the pet (works as soon as the first checkpoint exists)
swift run -c release

# or build a double-clickable app with bundled weights:
scripts/make_app.sh && open dist/Bonsai.app
```

The app hot-reloads the current creature's weights every 2 seconds, so you can
launch the pet the moment the first checkpoint lands and watch it get better at
being itself while training continues.

## Interactions

- **Drag** — pick the pet up and move it anywhere (it follows you across Spaces;
  its position is remembered)
- **Click** — poke a hole in it; watch it heal
- **Right-click** — regrow from seed, or quit
- **🌱 menu bar item** — switch creature, regrow, reload weights, quit

## Layout

- `Sources/Bonsai/` — Swift app: transparent window shell + Metal NCA runtime
- `training/` — PyTorch training rigs and the procedural target/frame artwork
- `weights/` — trained `.nca` weight files (formats: see `export()` in the trainers)
- `scripts/` — app bundling

Every code file has a companion `.md` documenting intent and contracts.

## Verifying without a window

```sh
swift run Bonsai --render-test out.png 300                    # grow, snapshot
swift run Bonsai --render-seq frames 24 10 weights/lain.nca   # animation frames
```
