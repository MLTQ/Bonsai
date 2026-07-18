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

## Make your own creature

The full guide — 2D, animated, mood-manifold, and volumetric, with real
training numbers and a troubleshooting table earned the hard way — is at
[docs/MAKE_YOUR_OWN_CREATURE.md](docs/MAKE_YOUR_OWN_CREATURE.md).

## Layout

- `Sources/Bonsai/` — Swift app: transparent window shell + Metal NCA runtime
- `training/` — PyTorch training rigs and the procedural target/frame artwork
- `weights/` — trained `.nca` weight files (formats: see `export()` in the trainers)
- `scripts/` — app bundling

Every code file has a companion `.md` documenting intent and contracts.

## Connecting an agent (optional)

The creature's entire external interface is one file: `weights/control.json`,
polled at 1 Hz. Anything that writes `{"anchor": "dread"}` or
`{"z": [10 floats in 0..1]}` steers the Manifold creature's mood. Three ways in:

```sh
scripts/mood.sh manic                          # named anchor
scripts/mood.sh 0.9 1 0.8 0 1 1 0.5 0.9 0.9 0.9   # raw z (the rich channel)
scripts/mood.sh --text "tests finally green"   # semantic projection
```

The projection layer (`tools/mood_projector.py`, needs Python +
sentence-transformers; ~90 MB local model on first run) maps arbitrary text
into mood-space. Its `--trace` mode tails your **local** Claude Code
transcripts (`~/.claude/projects/*/*.jsonl`) and drives the creature from
whatever your agent is currently doing — an ambient display of its cognition.

Privacy notes, plainly: there is no network connection to Claude or anyone
else. `--trace` reads transcript files that Claude Code already stores on
your disk, embeds them locally, and writes ten floats to a local file. It is
off unless you run it. Pin a specific project with
`BONSAI_TRACE_GLOB="~/.claude/projects/<your-project>/*.jsonl"`.

## Verifying without a window

```sh
swift run Bonsai --render-test out.png 300                    # grow, snapshot
swift run Bonsai --render-seq frames 24 10 weights/lain.nca   # animation frames
```
