# Mega Man authored run-cycle experiment

## Source and scope

- User-supplied sheet: `../sheets/megaman`, flattened JPEG, 728x1279.
- Selected source row: the four authored run silhouettes at x=282–625,
  y=258–364.
- Pose order: flight extension, support, compression, flight recovery.
- This is an authored silhouette-cycle test. The symmetric art does not expose
  persistent anatomical right/left labels, so the experiment must not be cited
  as a semantic named-leg tracking result.

## Extraction

- Reproducible script: `extract_megaman_walk.py`.
- Checker recovery uses the source's actual 15px JPEG checker period, colored
  or dark sprite seeds, connected-component selection, enclosed-neutral fill,
  and a soft alpha edge.
- Every frame remains at native scale and is aligned by a reviewed helmet
  anchor on a shared 144px canvas.
- Canonical review artifacts: `../experiments/megaman_walk_v1/sheet.png` and
  `loop.gif`.

## NCA corpus

- File: `corpus_megaman_walk_v1_128.npz`.
- Shape: `(1, 4, 128, 128, 4)`, premultiplied RGBA, float16.
- Pose names are embedded in the NPZ in authored order.
- The shared union crop and scale preserve relative silhouette motion.

## Training history

### Rejected baseline v1

- 4090, batch 20, pool 512, 128², damage disabled.
- Used historical canvas-wide MSE.
- Stopped at iteration 500 after the preview collapsed to transparent black.
- Cause: background pixels dominate sparse high-resolution RGBA; the empty
  state is a cheap objective solution. This is a loss failure, not a batch-size
  failure.

### Visible-loss baseline v2

- Relaunched from scratch with the shared foreground-weighted color, alpha,
  alpha-edge, and premultiplication objective in `transport_targets2d.py`.
- Persistent remote unit: `bonsai-megaman-visibleloss-v2.service`.
- Target: 8,000 iterations; checkpoints every 250; batch 20; pool 512; damage
  disabled.
- This baseline must become mature before it can seed the `TN2D1` transport A/B.
- Stopped after the iteration-500 preview remained transparent black. The loss
  no longer rewarded emptiness, but point-seed growth still confounded the
  animation experiment.

### Mature-state baseline v3

- Initializes the persistent pool from phase-matched visible targets with
  alpha copied into hidden live channels (`--init-mode target`).
- Tests only stability and coherent travel around the four-state ring.
- Point-seed growth is intentionally deferred to a later curriculum after the
  motion architecture passes.

## Acceptance gates

1. Preview grows a recognizable, non-black silhouette from a single seed.
2. Four phase samples remain globally coherent and visibly distinct.
3. No distant-pose double exposure in any phase sample.
4. Only then use the checkpoint as the same-corpus donor for transport training.

## Runtime preview

- The failed v3 checkpoint is intentionally available in Bonsai as
  `Mega Man · Mature Test` for direct inspection.
- Runtime artifacts are `weights/megaman_walk_mature.nca` plus the required
  `weights/megaman_walk_mature.ncs` phase-zero state.
- `PhaseOnlyCyclicBehavior` supplies sin/cos phase with behavior flag zero on a
  128² grid. “Regrow from Seed” is expected to go blank because point growth
  was never trained.
