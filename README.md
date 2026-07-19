# Bonsai 🌱

**Desktop creatures that are not spritesheets — they are organisms.**

Every creature in this project lives in a transparent, borderless, always-on-top
window and is, in a precise technical sense, *alive*: its body is a **neural
cellular automaton** — thousands of cells, each running the same tiny learned
rule, each seeing only its neighbors, sixty times a second on your GPU. Nothing
you see is drawn or played back. The creature **grows from a single seed cell**.
Poke a hole in it and it **heals**, because regeneration was part of its
training. Its idle shimmer is real stochastic dynamics, its animations are
**limit cycles of its own physics**, its moods are **coordinates in a learned
behavior manifold**, and — if you turn that part on — those moods can be driven
live by what your AI coding agent is doing in the next window.

A creature is a `.nca` weights file of 33–68 KB. The entire menagerie fits in
a quarter of a megabyte.

## What is actually going on here

The foundation is Mordvintsev et al.'s
[*Growing Neural Cellular Automata*](https://distill.pub/2020/growing-ca/)
(2020): train a per-cell update rule so that a target image is the *attractor*
of the dynamics — from a seed, and from any damage, the system flows back to
the target. We started there and kept going:

1. **Animation as limit cycles.** Feed every cell a phase clock
   `(sin θ, cos θ)` and let the training target *move along a closed loop of
   poses while the automaton runs*. The creature learns to track continuous
   motion — talking, blinking, walking — with no stored frames. Identity is
   carried by the persistent cell state, which is why a face holds still while
   its mouth moves. (`training/train_cyclic.py`)

2. **Moods as geometry.** A 10-dimensional latent `z` (droop, amplitude,
   tremor, gaze, brightness…) modulates the update rule via bounded FiLM.
   Trained over thousands of procedurally generated variations, the creature's
   entire manner of being becomes a *point in a continuous space*: sleep,
   dread, and mania are named regions; everything between them exists too, and
   transitions are learned metamorphosis, not crossfades.
   (`training/train_manifold.py`)

3. **Volumetric organisms.** The same machinery at 32³–64³ voxels, raymarched
   in real time: creatures with an interior and a back, whose wounds are
   craters that heal from the inside. A volumetric NCA performing a learned
   gait cycle appears to be a first. (`training/train_cyclic3d.py`,
   `docs/../paper/paper.tex`)

4. **The steering stack.** The creature's mood coordinate can be set by four
   hands, all writing one JSON file (`weights/control.json`): a wandering
   autopilot; a **state-space explorer panel** (menu bar → *State Space…*) —
   a 2D UMAP star-chart of the manifold you can drag a cursor through; a
   `mood` CLI any script or agent can call; and a **trace daemon**
   (`tools/mood_projector.py --trace`) that embeds your local Claude Code
   transcripts and projects them into mood-space — the creature becomes an
   ambient display of your agent's cognition. Frustrated agent, drooping
   creature. All local; nothing leaves your machine.

5. **Creatures from sentences.** The asset pipeline (`tools/ingest.py`) turns
   PNGs, spritesheets, or posed 3D model exports into training targets — and
   the diffusion front-end closes the loop: prompt an image model, review the
   output, key the background, train, and forty minutes later something no one
   drew is growing on your desktop. Multi-state creatures (the *werewolf
   pattern*: calm form ↔ beast form, transformation triggered by the agent's
   mood) come from one generated image per state; the metamorphosis between
   them is pure learned dynamics. (`training/train_states.py`)

The research artifacts live alongside the code: a working-draft paper
(`paper/`), including honest negative results (what happens when you remove
the phase clock naively), and ongoing experiments (currently: oscillation via
*heteroclinic cycling* — two unstable poses instead of any clock at all).

## The menagerie

| Creature | Kind | Born from |
|---|---|---|
| Bonsai | 2D static | procedural art |
| Lain | 2D cyclic (talks, blinks, glitches) | procedural art |
| Shoggoth Mk. I | 2D cyclic, walks the Dock | procedural art |
| Shoggoth Mk. II ("Manifold") | 2D, 10-D mood manifold, steerable | procedural corpus |
| Bonsai 3D | 32³ volumetric static | procedural voxels |
| Shoggoth Mk. III | 32³ volumetric cyclic (churn gait) | procedural voxels |
| Shoggoth Mk. IV | 32³ volumetric + mood manifold | procedural corpus |
| Shoggoth 64 | 64³ volumetric cyclic | procedural voxels |
| Moss Spirit | 2D two-state (calm ↔ beast) | **a diffusion prompt** |

Plus a capstone-in-waiting: `design/claudeguy/` and `training/claudeguy3d.py`
hold the full anatomy and 13-expression library of a Claude embodiment, to be
built on the most advanced version of the stack.

## Quick start

```sh
# Run the pets (zero dependencies beyond macOS 13+/Apple silicon)
scripts/make_app.sh && open dist/Bonsai.app
# ...or during development:
swift run -c release
```

Trained weights ship in `weights/`. Training your own needs Python with
torch/numpy/pillow; see the guide. The app hot-reloads the current creature's
weights every 2 s, so a training run visibly improves the creature live —
molting included.

## Interactions

- **Drag** to move (position is remembered); creatures follow you across Spaces
- **Click** to wound; watch regeneration (in 3D: scroll to orbit, click to crater)
- **Right-click / 🌱 menu** — switch creature, regrow, *State Space…* panel
- `scripts/mood.sh <anchor|z|--text "...">` — steer moods from anywhere
- `python3 tools/mood_projector.py --trace` — let your agent's work drive them

## Make your own creature

[docs/MAKE_YOUR_OWN_CREATURE.md](docs/MAKE_YOUR_OWN_CREATURE.md) — four tiers
(static, cyclic, manifold, volumetric) plus the asset and diffusion pipelines,
with real training numbers and a troubleshooting table earned the hard way.

## Layout

- `Sources/Bonsai/` — Swift/Metal runtime: transparent windows, 2D + volumetric
  simulation, raymarcher, state-space panel (every file has a companion `.md`)
- `training/` — PyTorch trainers (static / cyclic / states / manifold / 3D) and
  procedural creature art
- `tools/` — ingestion, mood projection, state-map builder
- `weights/` — trained creatures + anchors + control channel
- `paper/`, `docs/`, `design/` — the research and the plans

## Verifying without windows

```sh
swift run Bonsai --render-test out.png 300 weights/bonsai.nca
swift run Bonsai --render-seq frames 24 10 weights/lain.nca      # cycle GIF frames
BONSAI_GRID3=64 swift run Bonsai --render-test3d out.png 400 weights/shoggoth3d_64.nca 30
swift run Bonsai --render-vol authored.vol out.png               # preview art pre-training
```
