# Fused attractor-and-edge NCA experiment ledger

## Research question

Can one deployable conditional NCA hold four sharp pose attractors and four
directed transition vector fields without distant-frame blending or recurrent
cycle decay?

## Decision criteria

The method is **reasonably working** only if a deployment-like chained rollout
meets all of the following and looks coherent in the saved preview:

- endpoint alpha Dice >= 0.93 and boundary F1 >= 0.85;
- endpoint squared-gradient sharpness ratio in `[0.80, 1.20]`;
- non-adjacent anchor leakage <= 0.05;
- regional phase dispersion <= 0.25;
- four-edge visible cycle drift MSE <= 0.005;
- no obvious color explosion, double exposure, or disappearing body region.

Training-batch endpoint metrics are diagnostics, never acceptance evidence.

## Fixed experimental controls

- Corpus: `training/corpus_megaman_walk_v1_128.npz`
- Resolution: 128x128 premultiplied RGBA
- Pose order: flight extension, support, compression, flight recovery
- Transition length: 24 recurrent steps per directed edge
- Primary GPU: RTX 4090 UUID `GPU-21d45575-7ece-a97c-35a0-294f7bce9c39`
- Secondary GPU: RTX 2070 SUPER UUID
  `GPU-4e207c93-ed93-c35e-f0f2-e37c8df2b047`

## Trial ledger

### Control L1: shared-trunk layered transport

- Unit: `bonsai-megaman-layered-v1`
- Architecture: shared repair/flow trunks, four selected heads, four soft flow
  slots, oracle hard edge/progress guide.
- Checkpoint 1250 training diagnostic: Dice about 0.92, boundary F1 about 0.85,
  sharpness ratio about 0.66.
- Chained preview verdict: **fail at checkpoint 1250**. First edges blur and
  later edges develop severe color/state drift, ending near-white. This proves
  randomly sampled one-edge endpoint scores do not predict recurrent quality.
- Final training diagnostic at iteration 3000: Dice `0.931`, boundary F1
  `0.911`, sharpness `0.647`.
- Independent four-cycle/four-trial evaluation: bridge Dice `0.571`, handoff
  Dice `0.563`, intermediate leakage `0.339`, phase dispersion `0.553`, mean
  cycle drift `0.369`, and recurrent state at the `8.0` clamp.
- Final verdict: **controlled failure**. Good sampled endpoint diagnostics did
  not survive chained deployment.

### Fused F1: canonical states plus full pose/edge expert banks

- Hypothesis: hard-routed full experts and an explicit canonical hidden-state
  interface will separate pose stabilization from transition transport and make
  destination handoff trainable.
- Required ablations: canonical hidden objective on/off; handoff loss on/off;
  one-edge versus chained training.
- Status: implementation in progress.

#### F1a pilot observation

- Mild anchor corruption produced clean anchor diagnostics but nearly identity
  pose fields; checkpoint 300 handoff barely improved bridge error.
- Two-cycle evaluation already reached the recurrent `8.0` clamp and failed all
  chained gates. This is not a final model verdict because edge experts had only
  100 updates, but it invalidates tiny-noise anchor training.
- F1b correction: strong noise/dropout/local erasure, recovery-gain logging,
  hidden arrival diagnostics, hidden weight `0.20`, and soft state-range penalty.
- F1a checkpoint 400 reached one-edge training Dice about 0.92, but two-cycle
  evaluation Dice was only 0.56 with sharpness 0.35. The forward path still
  averaged four warped recurrent states per step. F1b uses straight-through
  hard flow-slot assignment; `--soft-slots` remains as the exact ablation.
- After chained training, F1a checkpoint 600 improved two-cycle Dice to `0.806`
  and cycle drift to `0.037`, already an order of magnitude below the completed
  layered control. It remains a visual/metric failure: handoff sharpness `0.403`,
  leakage `0.205`, and phase dispersion `0.369`.
- F1a final iteration 1000, four cycles/four trials: handoff Dice `0.840`,
  boundary F1 `0.712`, alpha sharpness `0.346`, RGB sharpness `0.131`, leakage
  `0.186`, phase dispersion `0.318`, and mean cycle drift `0.028`. This is a
  substantial recurrent improvement over L1 but a decisive visual/detail fail.

### Fused F1b: strong basins plus hard motion slots

- Changes from F1a: substantial anchor corruption, recovery-gain logging,
  hidden weight `0.20`, recurrent range penalty, and straight-through one-hot
  slot assignment.
- Unit: `bonsai-megaman-fused-f1b-main`
- Checkpoint 600 (100 edge updates), two cycles/two trials: handoff Dice
  `0.763`, sharpness `0.575`, cycle drift `0.064`, recurrent maximum `2.65`.
  At equal edge-training age F1a scored Dice `0.498`, sharpness `0.381`, drift
  `0.107`, and hit `8.0`. The F1b correction improves every intended metric.
- Handoff increases sharpness from bridge `0.522` to `0.575`, whereas F1a's
  weak handoff reduced it. Strong pose basins are functioning as designed.
- Visual verdict at 600: still blurred and not acceptable; continue main run.
- Checkpoint 1400 after 200 chained updates: bridge Dice `0.892`, handoff Dice
  `0.900`, boundary F1 `0.859`, phase dispersion `0.152`, and cycle drift
  `0.022`. Global pose structure and synchronization are now recognizable.
  Alpha sharpness `0.497`, RGB sharpness `0.133`, and leakage `0.135` remain
  failures: this is structural success but still a blurred sprite.
- Stopped at durable checkpoint 2200 after about 675 plateaued joint updates.
  Four-cycle/four-trial result: handoff Dice `0.921`, boundary `0.906`, phase
  dispersion `0.124`, intermediate sharpness `0.713`, and drift `0.018`.
  Handoff alpha sharpness remained `0.505`, RGB sharpness `0.162`, and leakage
  `0.145`. Longer optimization is not addressing the missing detail objective;
  4090 reallocated to F2.

### Fused F1c: exact endpoint sharpness branch

- Observation: F1b one-edge Dice reached about `0.94` while sharpness stalled
  near `0.54`. The intermediate constraint uses the minimum adjacent-anchor
  edge energy, appropriate mid-transition but underconstrained at arrival.
- Change: add destination-only sharpness/support penalties immediately before
  and after pose-expert handoff, plus exact endpoint RGB-gradient matching for
  face/armor detail. Branch from the preserved F1b checkpoint 800; leave F1b
  running unchanged as the control.
- Status: implementation validated; training pending 2070S availability.
- Checkpoint 1000 before chained training: two-cycle handoff alpha sharpness
  `0.652` versus F1b `0.497` at the same iteration. RGB sharpness remains about
  `0.124` and gradient error `0.079`; the alpha gate generalizes, but internal
  texture does not. Continue through chained/joint phases while preparing the
  anchor-detail/capacity correction.
- Stopped at checkpoint 1400 after 200 chained updates because F2 dominated it.
  Handoff Dice `0.904`, alpha sharpness `0.785`, RGB sharpness `0.130`, leakage
  `0.158`, phase dispersion `0.171`, and drift `0.023`. Endpoint alpha gating
  works, but F1 capacity cannot restore internal detail.

### Planned F1d/F2 anchor-detail correction

- Observation: F1b pose experts were trained with color MSE and alpha-edge loss
  but no RGB-gradient recovery. Handoff can therefore improve silhouettes while
  softening internal sprite detail.
- Change for the next clean branch: apply exact RGB-gradient loss during
  corrupted anchor recovery as well as edge arrival. Do not relabel the already
  running F1c branch.
- F2 capacity hypothesis: F1 has only about 114k learned parameters and raw XY
  coordinates. Use 384-wide pose/repair banks, 128-wide flow banks, and four
  Fourier XY bands so pose attractors can represent high-frequency sprite
  texture. Checkpoint as backward-compatible `FEX2D2`.

### Fused F2: wide Fourier experts plus anchor detail

- Unit: `bonsai-megaman-fused-f2-wide`
- Parameter count: `334,000` versus F1 `114,352`.
- Early anchor result through iteration 275: corrupted-anchor RGB gradient error
  fell from `0.052` to `0.027` while Dice remained about `0.962` and boundary F1
  `0.996`. Continue into transition training.
- Checkpoint 600 after only 100 one-edge updates, evaluated for two stochastic
  cycles: handoff Dice `0.916`, alpha sharpness `0.796`, RGB sharpness `0.213`,
  leakage `0.091`, phase dispersion `0.049`, and cycle drift `0.0116`.
- Visual verdict: first **promising/rough success**. Post-handoff frames visibly
  restore faces, armor divisions, and four distinct coherent poses rather than
  featureless silhouettes. Artifacts remain and numeric acceptance is not met;
  continue through chained/joint training.
- Checkpoint 800, still before chained training: handoff Dice `0.933` (pass),
  alpha sharpness `0.852` (pass), intermediate sharpness `0.810` (pass), phase
  dispersion `0.059` (pass), leakage `0.061`, and drift `0.0075`. RGB sharpness
  improved to `0.435`. Visual verdict: **reasonably working**—recognizable faces,
  armor, and four intended poses after handoff—with detached color specks,
  imperfect contours, and missing high-frequency detail still to correct.
- Checkpoint 1000: RGB sharpness rose to `0.618` and post-handoff sprites are
  visually close to reviewed anchors. Remaining failures localize to detached
  debris/recurrence: boundary F1 `0.741`, leakage `0.070`, and drift `0.0083`.
  Chained training begins at 1200 and directly targets those accumulations.
- Checkpoint 1400, four cycles/four stochastic trials: Dice `0.964`, boundary
  F1 `0.977`, alpha sharpness `0.892`, leakage `0.036`, phase dispersion `0.057`,
  cycle drift `0.00126`, recurrent max `1.543`, and RGB sharpness `0.735`.
- Verdict: **successful fused-NCA candidate**. It passes every structural and
  recurrent gate; post-handoff sprites are nearly the reviewed anchors. Only
  strict RGB-detail error (`0.030` versus `0.020`) / energy (`0.735` versus
  `0.750`) and a 0.043 state-range margin remain. Preserve checkpoint 1400 and
  continue low-rate joint tuning.
- Checkpoint 1600, the end of edge/chained training, four cycles/four stochastic
  trials: handoff Dice `0.964`, boundary F1 `0.977`, alpha sharpness `0.904`,
  RGB sharpness `0.779`, RGB-gradient error `0.0265`, leakage `0.037`, phase
  dispersion `0.060`, and visible cycle drift `0.00094`. This improves RGB
  detail and recurrence over checkpoint 1400. The recurrent handoff state stays
  at mean maximum magnitude `1.387` but has one worst-case sample at `1.581`,
  narrowly exceeding the deliberately strict `1.5` acceptance bound.
- Visual verdict at 1600: destination states are sharp, distinct, and globally
  coherent across four repeated cycles. Transition midpoints retain adjacent-
  pose ghosting and color debris, so this establishes the value of hard-routed
  pose/edge experts but does not yet establish clean learned in-between frames.
  Preserve checkpoint 1600 as the current baseline before joint fine-tuning.
- Joint checkpoints 1800–2400 monotonically improved destination detail while
  preserving the recurrent result. At checkpoint 2400 (four cycles/four
  stochastic trials), handoff Dice is `0.966`, boundary F1 `0.983`, alpha
  sharpness `0.949`, RGB sharpness `0.905`, leakage `0.035`, phase dispersion
  `0.059`, visible cycle drift `0.00074`, and worst handoff state magnitude
  `1.442`. It passes every structural, synchronization, recurrence, and bounded-
  state gate. Its RGB-gradient error is `0.02016`, just `0.00016` above the
  deliberately strict `0.02000` detail threshold; the two-trial quick screen
  passed all gates at `0.01970`. This is stochastic threshold noise, not a
  qualitative failure. Preserve checkpoint 2400 as the current visual best.
- **Selected checkpoint 2600**, four cycles/four stochastic trials plus all-pose
  damaged-attractor recovery: handoff Dice `0.968`, boundary F1 `0.996`, alpha
  sharpness `0.952`, RGB sharpness `0.902`, RGB-gradient error `0.01856`,
  leakage `0.0350`, phase dispersion `0.0589`, visible cycle drift `0.000396`,
  and worst handoff state magnitude `1.425`. Corrupted-pose recovery improves
  Dice to `0.967`, boundary F1 to `0.997`, and RGB-gradient error to `0.02296`
  while reducing both visible and hidden error. **Every predefined acceptance
  gate passes.** Stop the main run at this durable checkpoint; further training
  is unnecessary and risks trading away the selected basin.

### F2 alpha-interface ablation

- Exact F2 architecture/training with canonical hidden features replaced by
  repeated alpha (`--state-interface alpha`), on the 2070S.
- Unit: `bonsai-megaman-fused-f2-alpha`
- Purpose: test whether the explicit shared hidden-state language contributes
  beyond hard expert routing and added capacity.
- Exact-stage checkpoint 800 comparison strongly favors the canonical state:
  alpha-only handoff Dice `0.936`, boundary `0.832`, RGB sharpness `0.404`,
  leakage `0.067`, cycle drift `0.00727`, and recurrent max `3.536`, versus
  canonical checkpoint 800 Dice `0.933`, RGB sharpness `0.435`, leakage `0.061`,
  drift `0.0075`, with the alpha run visibly blurrier and less bounded. More
  importantly, the alpha hidden-state errors were already several times larger
  at the start of edge training. Complete its scheduled run for the controlled
  final comparison; do not mistake capacity alone for the successful change.
- Final alpha-only checkpoint 1000, evaluated identically: damaged-anchor Dice
  `0.965` / detail error `0.0288`, handoff Dice `0.961`, boundary `0.968`, alpha
  sharpness `0.850`, RGB sharpness `0.489`, RGB-gradient error `0.0488`, leakage
  `0.0494`, visible drift `0.00354`, and recurrent max `1.979`. It fails RGB
  detail and bounded-state gates. The canonical interface is causally useful,
  not decorative: at comparable capacity it roughly halves detail error,
  reduces drift by almost 9×, and keeps the recurrent state bounded.

### Bonsai FX2D deployment

- Added a portable `FX2D` exporter, canonical `NCS1` state, dedicated Swift
  loader, multi-pass Metal pose/flow/slot/MacCormack/repair runtime, fused pet
  view, headless render command, app registry entry, and bundle packaging.
- Production Metal verification with selected checkpoint 2600 reaches the
  correct next pose after 32 steps at Dice `0.966` / boundary F1 `1.000`,
  returns to pose zero after 128 steps at Dice `0.969` / boundary F1 `1.000`,
  and remains at Dice `0.969` / boundary F1 `1.000` through 512 steps (16 full
  cycles). A checkpoint-1600 512-step debug rollout took `2.48 s` including
  process startup and runtime Metal compilation, leaving ample room for two
  steps per 30 Hz app tick once the pipeline is resident.
- The runtime intentionally uses the same fixed hard global schedule as the
  controlled experiment. A learned hysteretic guide is a separate follow-up;
  it should not be conflated with validating the expert decomposition.

## Stop conditions

Reject this parameterization after at least two controlled corrective trials if
chained sharpness remains below 0.70 or cycle drift remains above 0.02 despite
good anchor stabilization. A failure under those conditions points to missing
object-layer correspondence or occlusion memory, not insufficient optimization.
