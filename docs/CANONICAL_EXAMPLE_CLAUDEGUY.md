# The Canonical Example: Claudeguy, end to end

`docs/MAKE_YOUR_OWN_CREATURE.md` is the reference manual — every option, every
rung of the ladder. This is the opposite: **one creature, start to finish, with
the reasoning shown**. If you are building your first creature, read this, then
go to the manual for the parts you want to vary.

The creature is Claudeguy: a clay flower-person with a ring of terracotta petals
around a cream face disk. He is the capstone of the project, and he is the
example because he exercises the whole stack — authored volumetric anatomy, a
phase-conditioned limit cycle, and a 64³ training run.

Files: `training/claudeguy3d.py` (anatomy + corpus), `training/train_cyclic3d.py`
(trainer), `design/claudeguy/` (the reference art this is built from).

---

## 1. Author the anatomy as a function, not a mesh

The single most important structural decision. `draw_claudeguy()` is a Python
function that rasterises one volumetric frame from parameters:

```python
draw_claudeguy(phase=0.0, blink=0.0, look=(0, 0), petal_flex=None,
               expression="neutral", spin=0.0, wiggle=0.0, sheet=None)
```

Everything that can move is an argument. That is what lets one function
generate an entire animation corpus later — the alternative, authoring frames
by hand or exporting a mesh per pose, gives you no way to sweep a parameter.

Geometry is authored in **32-unit space and scaled by `K = SCALE`**, so the same
code renders at 32³ or 64³ without edits. Follow this convention; the day you
want to retrain at a different resolution you will not have to touch the art.

### Budget the creature's reach against the grid wall

The first version of Claudeguy clipped. Petals reached 16 units from centre in
a 32-unit half-grid — exactly the wall — so every tip was sliced flat and the
bounding box hit all four sides:

```
extent [31 64 64] of 64      # clipped on x and y
```

Fixed by budgeting reach explicitly (petal ring now tops out ~12.5 of 16 units):

```
extent [27 52 53] of 64      # 5-7 voxels of margin on every side
```

Leave the margin. The NCA overshoots while it learns, and cells that need to
exist outside the target have nowhere to go if the target is already at the wall.
**Always measure the bounding box before training** — this is one cheap check
that catches a whole class of wasted runs:

```python
idx = np.argwhere(vol[..., 3] > 0.1)
print('extent', idx.max(0) - idx.min(0) + 1)
```

### Evaluate through the production renderer, not the preview

`claudeguy3d.py`'s `_preview()` is an alpha-weighted projection along z. The
shipped renderer is an emission-absorption raymarcher where the nearest surface
dominates. These disagree badly on exactly the features you care about: small
dark details in front of a large bright surface. Claudeguy's pupils looked like
faint grey smudges in the preview and like proper eyes through Metal.

```bash
cd training && BONSAI_GRID3=64 python3 -c "
from claudeguy3d import draw_claudeguy
draw_claudeguy().astype('<f4').tofile('claudeguy.vol')"
cd .. && BONSAI_GRID3=64 .build/release/bonsai --render-vol \
    training/claudeguy.vol /tmp/cg.png 0
```

Judge the art from `/tmp/cg.png`. If a feature has to be legible, it also has to
sit *forward* of whatever it is drawn on — Claudeguy's pupils and mouth are
pushed several units toward the viewer for this reason.

---

## 2. Turn the anatomy into a cyclic corpus

The trainer wants three names from a creature module:

| name | meaning |
|---|---|
| `FRAMES` | keyframes per cycle (12 is plenty; the NCA interpolates) |
| `BEHAVIORS` | how many distinct modes (idle, delight, …) |
| `make_frames3d()` | `-> (BEHAVIORS, FRAMES, G, G, G, 4)` float16 |

That is the entire contract. `train_cyclic3d.py --creature <module>` imports it
by name, so a new creature needs no trainer edits.

### Loop closure is non-negotiable

Frame `FRAMES` must land exactly on frame 0, or you are asking the automaton to
learn a discontinuity it can only smear. The rule that guarantees it: **every
time-varying term is an integer multiple of the base frequency.**

```python
flex  = 0.45 * np.sin(phase + k * 0.52)   # 1x — fine
flex  = 0.85 * np.sin(2 * phase + ...)    # 2x — fine
flex  = 0.45 * np.sin(1.3 * phase)        # WRONG: never returns to itself
```

Verify numerically rather than trusting the code. The wrap delta must sit inside
the range of ordinary frame-to-frame deltas:

```
beh 0 delta min 0.00042 max 0.00222  wrap(11->0) 0.00044   # closed
beh 1 delta min 0.01197 max 0.01499  wrap(11->0) 0.01474   # closed
```

### Watch where in the cycle your events land

Claudeguy's blink was originally `1 - |sin(phase)| * 7`. Two bugs in one line:
`|sin|` has period π, so it blinks **twice** per cycle, and it is zero at
phase 0 — meaning the eyes were **shut in frame 0**, the rest pose that rollouts
are scored against most often. Corrected to put a single blink at mid-cycle:

```python
blink = max(0.0, 1.0 - abs(np.sin((phase - np.pi) / 2.0)) * 7.0)
# blink per frame: [0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0]
```

Print the per-frame value of every event term. It takes ten seconds and the
failure is otherwise invisible until you are staring at a trained creature that
never opens its eyes.

---

## 3. Train

```bash
cd training
BONSAI_GRID3=64 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python3 train_cyclic3d.py --creature claudeguy3d \
    --iters 30000 --batch 4 --pool 256 --chunk 8 \
    --device cuda --out ../weights/claudeguy.nca
```

Notes that cost us time to learn:

- **`CUDA_DEVICE_ORDER=PCI_BUS_ID`.** Without it CUDA enumerates fastest-first,
  so `CUDA_VISIBLE_DEVICES=1` is *not* `nvidia-smi`'s device 1. We launched onto
  a 2070S instead of a 4090 and OOMed instantly.
- **Confirm the device at startup.** Every trainer prints it. A trainer that
  silently defaults to CPU is a 118× slowdown that looks like a slow run.
- **64³ needs the big card.** Gradient checkpointing (`--chunk 8`) trades ~35%
  recompute for a manageable activation footprint; without it, BPTT through
  48-72 steps at 64³ does not fit.
- **Checkpoints land every 500 iterations** plus a versioned `.itN` every 5000.
  Never point two experiments at the same `--out`; we lost a good creature that
  way and now every experiment gets a distinct filename, promoted only after it
  verifies.

## 4. Verify before shipping

```bash
BONSAI_GRID3=64 .build/release/bonsai --render-seq3d /tmp/cg_seq 24 20 \
    weights/claudeguy.nca 0
```

Twenty-four frames, twenty steps apart, while the camera orbits. What you are
looking for: the cycle returns to where it started, the body holds together for
the whole sequence rather than blurring after a few hundred steps, and the
motion is legible as *the motion you authored*.

Only then register it in `Sources/Bonsai/Creature.swift` and rebuild.

---

## 5. Why this creature is shaped the way it is

Claudeguy is a limit cycle, not a spritesheet. The NCA is not indexing 12 stored
frames; it learned a dynamical system whose trajectory passes near them, so it
fills in every intermediate state, recovers when damaged, and never lands on a
frame boundary. That is the whole thesis of this project, and it is the reason
the corpus is 12 keyframes rather than a rendered animation: **you are supplying
evidence about a trajectory, not the trajectory itself.**
