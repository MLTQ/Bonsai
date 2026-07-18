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
