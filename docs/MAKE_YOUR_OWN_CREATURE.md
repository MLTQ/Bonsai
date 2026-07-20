# Make Your Own Creature

Every creature in this repo is a ~40KB weights file grown by the same pipeline:
**author a procedural target → train an NCA against it → drop the `.nca` in
`weights/` → register one line in Swift.** This guide walks all four tiers of
creature, with the real numbers and the real failure modes from building the
first six.

## 0. Choosing a subject (read this first)

NCAs render like soft volumetric clouds of paint. They are **kind to**:
blobs, plants, ghosts, faces with chunky features, anything organic or
slightly wrong. They **punish**: rigid geometry, thin lines, precise edges,
small text. Our shoggoth got *better-looking* as training fuzz became
protoplasm; a tachikoma would just look melted. At 64×64 (2D) or 32³ (3D),
features under ~2 px/voxels will not survive — draw chunky.

Palette advice: 3–6 flat colors with one highlight tone. Look at
`training/target.py` (bonsai) and `training/lain.py` (face) for the house
style.

## 1. A static 2D creature (~1 hour, trainable on a MacBook)

1. **Copy `training/target.py`** and redraw `make_target()` using the helpers
   (`_disk`, `_stroke` for tapered bezier strokes, plain masks for boxes).
   Rules that matter:
   - Draw at 4× supersample, box-downsample (already in the template). Soft
     edges train dramatically better than hard pixel edges.
   - Return premultiplied RGBA in `[0,1]`, shape `(64, 64, 4)`.
   - Keep the subject inside ~48px with margin — the automaton needs room to
     overshoot while growing.
2. **Preview**: `python3 yourcreature.py` → eyeball the PNG. Iterate here;
   this is the cheap place to have taste.
3. **Train**: point `train_nca.py` at your target function (or copy it), then
   `python3 train_nca.py --iters 8000`. On an M1 Pro expect ~4 it/s ≈ 35 min;
   checkpoints every 250 iters. Healthy loss trajectory (bonsai reference):
   ~0.01 @ 1k, ~0.005 @ 2k, ~5e-4 @ 8k.
4. **Verify headlessly**: `swift run Bonsai --render-test out.png 300 weights/you.nca`
   — the Metal runtime must grow the same creature the trainer previews.
5. **Register**: add one `Creature(...)` entry in
   `Sources/Bonsai/Creature.swift`, rebuild. If you launch the app during
   training, it hot-reloads every checkpoint — your creature visibly gets
   better at being itself, with occasional molting (a body grown by an older
   checkpoint is out-of-distribution for a newer rule; it heals).

## 2. An animated 2D creature (cyclic)

Animation = the target moves along a closed loop; the NCA learns to track it.

1. **Author 12 frames × N behaviors** as a module exporting
   `FRAMES, BEHAVIORS, GRID, make_frames()` returning
   `(BEHAVIORS, FRAMES, 64, 64, 4)`. See `training/lain.py` (face: still +
   talking) and `training/shoggoth.py` (blob: idle + walk).
2. **THE LOOP-CLOSURE RULE** (this will get you): every time-varying term in
   your frame generator must use an *integer multiple* of the base frequency.
   `sin(phase * 1.3)` does not return home at frame 12 → the NCA faithfully
   learns a once-per-cycle lurch. Verify numerically: the frame-11→frame-0
   pixel delta must be ≤ your typical adjacent-frame delta.
3. **Train**: add your module to the `--creature` choices in
   `train_cyclic.py`, run ~12–24k iters (M1 Pro: 50–100 min). The trainer
   advances phase *during* rollouts (that's what makes it animation, not
   frame lookup) and flips behaviors mid-life on 15% of samples, so
   transitions come free.
4. **Contract**: one cycle = 240 automaton steps (`OMEGA` in the trainer ↔
   `LainBehavior.omega` in Swift). Change one, change both.
5. **Verify**: `--render-seq dir 24 10 weights/you.nca` +
   `python3 scripts/frames_to_gif.py dir out.gif 160`. Identity should hold
   rock-steady while only the animated regions move.
6. **Behavior**: subclass `CreatureBehavior` if you want autonomy (murmur
   episodes, Dock-walking, self-glitches — see `LainBehavior`,
   `ShoggothBehavior`).

## 3. A mood-manifold creature (advanced)

Instead of N discrete behaviors: a factored latent `z ∈ [0,1]^k` where each
axis is a *motion parameter* (droop, amplitude, tremor, gaze...). See
`training/manifold_shoggoth.py` + `train_manifold.py`.

- Author `draw(phase, z)`; sample ~2k z-vectors as a corpus; name your
  anchors (they become the steering vocabulary and the projector's targets).
- z conditions via FiLM. **Keep the gamma bounded** (`tanh`) and give the
  FiLM layer a 10× lower learning rate — we learned this at iteration 19,500
  of a run that NaN'd and poisoned its own sample pool. The trainer now
  guards all of this; don't remove the guards.
- Ship `anchors_yourcreature.json` next to the weights; the control-file
  steering and the text projector (`tools/mood_projector.py`) work
  immediately once anchor names exist.

## 4. A 3D creature (volumetric)

Same ideas, one dimension up. Realities first: **don't train 3D on a laptop**
(M1 Pro: ~0.07 it/s; RTX 4090: ~3 it/s, so a 20k run ≈ 2 h). Rendering,
by contrast, is trivial on the Mac.

1. **Voxel targets**: `training/target3d.py` helpers (`_sphere`, `_swept`,
   `_cone_pot`) at 2× supersample → `(32, 32, 32, 4)` in `(z, y, x, c)`
   order, y-up. Chunkier is better: at 32³ your creature is 32 voxels wide.
2. **Static**: `train_nca3d.py`. **Cyclic**: author
   `make_frames3d()` (see `shoggoth3d.py` — note the azimuthal traveling
   wave, a gait with no 2D equivalent) and use `train_cyclic3d.py`. Rollouts
   are gradient-checkpointed in 8-step chunks; that's why 24 GB suffices.
3. **Seed position must match** between the trainer and the Swift registry
   entry (`seed3D:`) — the bonsai seeds low-center where the soil is, the
   shoggoth at center. Mismatch = your creature grows out of the wrong place
   or not at all.
4. **Verify**: `--render-test3d out.png 300 weights/you.nca 25` (still) and
   `--render-seq3d dir 36 8 weights/you.nca` (gait GIF). The raymarcher
   gives you orbit, gradient lighting, and click-to-crater for free.

## 5. Troubleshooting (all of these happened to us)

| Symptom | Cause | Fix |
|---|---|---|
| Once-per-cycle lurch | Non-integer harmonic in frame authoring | Loop-closure rule, §2.2 |
| Animated parts blur into a fused smear | Phase-averaging (usually clockless/underconditioned) | Keep the phase channels; if experimenting clockless, use a sync curriculum |
| Loss → `nan`, then everything is `nan` forever | Unbounded conditioning gain; pool poisoning | tanh-bound FiLM, discard non-finite batches, versioned checkpoints |
| Ghost haze around the creature mid-training | Normal transient dead matter | Wait for the LR decay; it cleans up |
| Doubled/molting creature during hot-reload | Old body + new rule (out-of-distribution) | Right-click → Regrow from Seed, or enjoy it |
| MPS out-of-memory | Everything shares unified memory | Smaller batch/pool, or train on a CUDA box |
| Creature invisible in app but fine in render-test | Wrong seed position, or weights filename not in registry | §4.3 / `Creature.registry` |

## 6. The contract that makes it all work

Python trains; Metal runs. They agree byte-for-byte on: perception ordering
(identity, sobelX, sobelY[, sobelZ] interleaved per channel), zero padding,
alive mask = pre AND post maxpool(alpha) > 0.1, per-cell stochastic fire,
state clamp ±8, and the flat weight formats (NCA1/2/3, NC3D/NC3C — see
`NCAWeights.swift`). If you change any of these in one place, change them in
both, and re-verify with a render-test before believing anything else.

Every code file has a companion `.md` with its contracts. Read them before
editing; update them after. That discipline is why six creatures got built in
a day without the pipeline ever rotting.


## 7. The asset pipeline (artists welcome, primitives optional)

`tools/ingest.py` converts normal art into training targets; every trainer
takes `--target creature.npz`:

- `image sprite.png --key-white` → 2D static target (near-white backgrounds
  become transparency; tune the threshold in `_key_white` if your background
  is dingy — 228 works for diffusion-model white)
- `sheet strip.png --frames 12 --behaviors 2` → 2D cycle frames
- `mesh model.glb` → colored solid voxel target (vertex colors, face colors,
  or textures; interiors filled; `BONSAI_GRID3` sets resolution)
- `meshcycle poses_dir/` → volumetric cycle from posed exports
  (Blender: pose 12 loop frames, export each as glTF — loop closure is on you)
- `states manifest.json --key-white` → multi-state attractors (see §8)

## 8. Multi-state creatures: the werewolf pattern

States are **attractors**, not animations: each state is one still image, and
all motion belongs to the NCA — shimmer within a state, learned metamorphosis
between states. `train_states.py` trains a state flag with mid-life switches,
so the transformation (calm moss ball → bramble beast) is a genuine conversion
of the living body. Wire the flag to the mood channel (`StateBehavior` maps
control anchors like *agitated* to the beast form) and your agent's frustration
becomes the full moon.

## 9. Creatures from prompts (diffusion front-end)

The pipeline: generate → **review** → `ingest image --key-white` → train.
Hard rules, written in a small amount of blood:

- **Stylized/anime checkpoints only. Never a realism model.** Check the
  filename; "realism" means realism.
- **Non-human subjects only** for generated creatures.
- **Review every image yourself before batching or ingesting.**

Practical tricks: prompt for `white background, flat color, sticker style`;
img2img from an approved canonical image preserves identity across states
(strength ~0.5–0.65); models resist *pose* changes at low strength — guide
them by transforming the **init image** (e.g. pre-rotate ±12° for a lean) and
let generation repaint it naturally. The final asset is still pure diffusion
output; nothing mechanical enters training.

## 10. Steering, maps, and the trace daemon

- `weights/control.json` is the universal steering socket:
  `{"anchor": "dread"}` or `{"z": [...]}` at up to 4 Hz.
- The **State Space… panel** shows each creature's actual space: UMAP
  constellations for manifold creatures (`tools/make_statemap.py`), two
  labeled islands for flag creatures. Dragging steers through the same socket.
- `tools/mood_projector.py` maps text → mood (`--text`, `--watch feed`,
  `--trace` for live Claude Code transcripts). Anchor phrase banks are in the
  file; add vocabulary when a mood mishears you.

## 11. Extra troubleshooting (continued from §5)

| Symptom | Cause | Fix |
|---|---|---|
| Creature grows on a gray slab | Background survived white-keying | Lower `_key_white` threshold below the bg's min-channel |
| Generated "sleeping" creature has open eyes | img2img protecting composition | Raise strength + weight the tag, or accept and re-roll |
| Generated poses won't lean/turn | Same | Transform the init image; let the model repaint |
| 64³ organism becomes a cube at long horizons | Growth trained, containment not | Longer fixed rollouts (`--fixed-t 96`); it also self-cures with pool age |
| torch.compile "CUDAGraphs overwritten" | reduce-overhead vs. chained steps + pools | `max-autotune-no-cudagraphs`, or don't compile — measure first; at 64³ you're compute-bound anyway |
| Diffusion job OOMs a training run | Forgot to pin the GPU | `CUDA_VISIBLE_DEVICES=<uuid>` always, by UUID not index |

## 12. Pose spacing vs. capacity: the rule that governs constellation creatures

Constellation/heteroclinic creatures only glide if the automaton can *tell its
poses apart*. Three quantities interact:

- **adjacent-pose distance** `d_adj` — MSE between consecutive poses in the graph
- **the model's error floor** `L∞` — where training loss plateaus (capacity-bound)
- **transit horizon** — rollout steps per edge

The rule, measured the hard way: **under single-hop training, traversal requires
`d_adj` comfortably above `L∞`** (aim for 3× or more). If `d_adj ≈ L∞`, "I am at
pose 3" and "I am at pose 4" are indistinguishable to the loss, the gradient
toward the successor drowns in reconstruction error, and the creature parks at a
compromise state. Observed directly: a 12-pose ring with `d_adj = 0.0101`
against a plateaued `L∞ = 0.0092` held position for 57 of 60 sampled frames —
one forward step, one back, in an entire free run.

> **Correction (2026-07-20): the 3× rule is a property of single-hop training,
> not of constellation creatures.** With waypoint chains (§below) the same ring
> traverses cleanly at **0.8×**. Three separate runs now confirm it; the best,
> `ringB_wp3` (4 waypoints, 90k iterations, hidden 256), laps all twelve poses
> with **23 forward steps and zero reversals** at a signal/floor ratio of 0.8×,
> which `verify_constellation.py` still flags as "TOO LOW".
>
> The reason the ratio stops mattering: single-hop training only ever supervises
> the *endpoint* of a rollout, so distinguishing pose 3 from pose 4 has to be
> done by the loss at that endpoint — and if the poses are closer together than
> the error floor, it cannot be. A waypoint chain supervises the whole
> trajectory, so the creature learns the *direction of travel* rather than
> having to resolve individual pose identity. Direction survives an error floor
> that swamps position.
>
> The verifier's threshold is left in place deliberately: it is still the right
> warning for a single-hop run. Read it as "this will not traverse unless you
> are training with waypoints."

A second correction while we are here: the earlier claim that *generation noise
dominates the pose signal* on diffusion-built rings was **wrong**. The
procedural ring (zero generation noise) shows the same 2.0× adjacent/opposite
ratio as the diffusion ring's 2.2×. Whatever limits these creatures, it is not
diffusion noise.

So denser poses are good *only until they cross the floor*. Levers, in the order
worth trying:

1. **Bolder poses** (bigger amplitude between waypoints) — raises `d_adj`, free.
2. **More capacity** (`--hidden 256`) — lowers `L∞`. Note the Swift runtime
   parses width from the header, but Metal keeps the hidden vector in registers,
   so measure render cost before shipping a wide creature.
3. **Fewer poses** — both raises `d_adj` and reduces what must be memorized;
   13 poses in an 8.6k-parameter rule is already a lot.
4. **Shorter horizons** — makes each transit easier, so the floor matters less.

Diagnostics that tell you which failure you have:

| Reading | Meaning |
|---|---|
| Loss plateaus early and flat over ~20k iters | capacity-bound: more iterations will not help |
| Position tape parks on one pose | `d_adj` ≲ `L∞` |
| Position tape jitters randomly among poses | horizon too long, or poses too similar to order |
| Motion smooth but details soft | generation noise between poses, or grid resolution |
