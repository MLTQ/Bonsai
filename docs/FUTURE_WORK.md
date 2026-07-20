# Future Work (parked, not scoped)

## Negative result: the blur is not undertraining (2026-07-20)

Max's read on the H100 sweep was that the arms "look simply undertrained". Arm A
was resumed from its 22.8k checkpoint and run a further 67,000 iterations at
lr 5e-4 with growth rollouts re-enabled, until its improvement rate fell below
2% per 10k window twice running (a real plateau, not a timeout).

| | A @ 22.8k | A @ 89.8k | wp3 @ 90k |
|---|---|---|---|
| forward / reverse | 16 / 0 | 20 / 0 | 23 / 0 |
| period (steps/lap) | 270 | 216 | 188 |
| **sharpness (vs 0.0368 target)** | 0.0239 | **0.0233** | 0.0275 |
| render error | 0.0317 | 0.0369 | 0.0224 |
| pose coverage | skips 0,1 | skips 0,1 | all 12 |

**4x the training bought better traversal and zero sharpness.** Edge energy sat
at ~63% of the targets before and after; reconstruction error got worse, the
run having traded accuracy for persistence under growth rollouts. Duration is
not the lever.

What this rules in: the remaining hypotheses are architectural — perception
receptive field (see "Breaking one-cell-per-step"), and representational
capacity for high-frequency detail. What it rules out: "just train it longer",
which was the cheapest explanation and is now dead. Effort should go to the
dilated-perception and pooled arms, not to bigger iteration counts.

## J-space creature (interpretability interface)
Anthropic's Jacobian-lens work (transformer-circuits.pub/2026/workspace) finds a
sparse "global workspace" subspace: mid-layers, <10% of activation variance,
~10–25 concurrently active concepts, causally load-bearing for flexible
reasoning, and — critically — containing content the model does *not* verbalize.

Proposal: replace the text-embedding trace daemon with a J-lens readout from a
local model (gemma3:1b is hookable). Compute J-lens directions for anchor-ish
tokens (confused / stuck / aha / wrong / careful), project the live mid-layer
residual stream onto them, map to creature z. The creature becomes an ambient
display of a model's workspace, including the unverbalized parts. Dimensionality
match to our 10-D manifold is nearly exact. This is the version an
interpretability researcher would want on a second monitor.

## J-space LoRA (research, handle with care)
Max's idea: an adapter that accumulates from J-space activity, reshaping what the
model holds in mind. The paper's counterfactual-reflection training is evidence
the write direction works ("representations used for verbal report are the same
ones that govern how the model silently reasons"). Genuinely interesting; also
genuinely a model whose dispositions drift from whatever it dwells on. Wants
explicit hypotheses and evals, not a weekend integration. Keep separate from the
menagerie.

## Emergent representation (from the "how big can memory get" thread)
- Autoencoder NCA: one network, all creatures, small *learned* latent (not hand-
  factored) → does it invent axes nobody named?
- Self-prediction NCA: hidden channels trained to predict own next state under
  perturbation; fully self-supervised.
- Diversity objective (DIAYN-style): mutual information between latent and
  resulting dynamics → un-authored behaviors.
Note the coordination-latency ceiling: under our 3x3 perception, information
travels one cell per step, so large grids take O(diameter) steps to agree. See
"Breaking one-cell-per-step" below — this is a property of the kernel we chose,
not of cellular automata.

## Breaking one-cell-per-step (propagation speed)

Filed 2026-07-20, from Max's question: *why is "information travels one cell per
step" a hard constraint? why not broadcast information in a kernel around the
cell?* It is not a hard constraint. It is a consequence of our fixed 3x3
perception (identity + Sobel x/y), which reads exactly one ring of neighbours.
Change the neighbourhood and the propagation speed changes with it.

Why it matters: a 96^2 creature is ~72px across, so a signal needs ~72 steps to
cross it while training rollouts are 48-72 steps. The body cannot agree with
itself within one rollout. Sweep arm B (96^2) diverged after 14k iterations;
the 64^3 shoggoth has the same problem in three dimensions.

Four options, cheapest first:

1. **Dilated taps** — a 3x3 sampled at spacing d costs the same nine taps but
   propagates d cells per step. A dense 3x3 (detail) plus a dilated 3x3 at
   spacing 3 (reach) doubles perception cost for ~3x the propagation speed:
   72 steps to cross a 96^2 body becomes 24. Best ratio of the four, and it
   fits our architecture because perception is *fixed* filters, not learned —
   the change is choosing to include identity + Sobel@1 + Sobel@3.
2. **Larger dense kernels** — 5x5 gives 2 cells/step, 7x7 gives 3. Simple, but
   taps go 9 -> 25 -> 49 in 2D and 27 -> 125 -> 343 in 3D. Wrong direction on
   the axis that already dominates the step cost.
3. **Pooled / global channels** — broadcast a spatial reduction of a few
   channels to every cell each step. Instantaneous global coordination for one
   reduction per step. Max is most interested in this one; see below, it is now
   an active experiment rather than a parked idea.
4. **Hierarchical multi-scale** — pool to a coarse grid, propagate, upsample.
   Log-time global agreement, but it adds real machinery and moves furthest from
   "a grown thing" toward "a convnet with extra steps".

**Cost that is not compute**: perception ordering is a three-way numerical
contract across the PyTorch trainers, the Metal shaders, and the Triton kernel.
Any change means updating all three plus the parity gates, and it invalidates
every existing creature's weights — a new format version, not a patch. Prove it
on one cheap 64^2 run before touching the volumetric path.

## Pooled channels: giving the creature a nervous system

**Status: active experiment**, `training/pooled_nca.py`, launched 2026-07-20.

Option 3 above, but the reason to want it turned out not to be propagation
speed. Max, on why he is not married to NCA purity: *"I want something that
isn't just 'it changes place on a spritesheet'... I want to not necessarily know
what it is going to do."*

Strict locality means every cell is a local reflex and the creature's state is
exactly the sum of its pixels. A pooled channel changes the kind of system it
is. Each step the alive-masked spatial mean of NPOOL hidden channels is
broadcast back to every cell; because the update rule both reads that mean and
writes the channels it is pooled from, the loop closes and the creature gets a
global variable with its own dynamics that no individual cell owns. A slow field
coupled to fast local physics is the standard recipe for slow-fast systems —
bursting, spontaneous transitions, behaviour that is not a lookup into a pose
set. The creature can hold a mood that lives nowhere in particular.

This is a real trade, stated honestly: locality is what makes damage heal from
the edges inward and what makes these things feel grown rather than drawn. A
pooled creature is no longer a cellular automaton in the strict sense. The bet
is that one global variable buys emergence worth more than the purity it costs.

### First measurement (arm F, 4.6k iterations, 2026-07-20)

`tools/trace_pooled.py` on the live checkpoint, 4000-step free rollout from
seed, post-transient window:

| channel | mean | swing | lag-1 AC | lag-100 AC | decorrelation |
|---|---|---|---|---|---|
| g[0] | -0.115 | 0.054 | +0.998 | +0.044 | ~56 steps |
| g[1] | -0.033 | 0.095 | +0.999 | +0.398 | ~108 steps |
| g[2] | -0.135 | 0.088 | +0.999 | +0.588 | ~159 steps |
| g[3] | -0.151 | 0.059 | +0.997 | +0.352 | ~98 steps |

Amplitude alone would have proved nothing — g is a mean over ~2000
stochastically-firing cells, so it jitters even if the update rule ignores it.
The autocorrelation is what settles it: sampling noise decorrelates in one step
(lag-1 AC ~ 0), and g's lag-1 AC is 0.998 with a 56-159 step memory. **g is a
genuine slow state variable, two orders of magnitude slower than the one-step
local dynamics.** The amplitude holds steady from step 1333 to 4000, so this is
persistent dynamics rather than a long relaxation.

Nothing in the objective asked for this. The targets are static poses; the slow
field is emergent. Caveat: 4.6k iterations is early, and "slow variable exists"
is not yet "slow variable does something legible" — the next question is whether
g's excursions correlate with visible behaviour changes.

### Loss: pooled beats local on a properly matched control

Arm F vs arm D — identical hidden width (128), batch (16), pool (2048), target,
horizon, waypoints and motion weighting. The **only** difference is 4 pooled
channels. Median loss per 4k-iteration window:

| iters | D local | F pooled | ratio |
|---|---|---|---|
| 0-4k | 0.0355 | 0.0290 | 0.82 |
| 4k-8k | 0.0289 | 0.0289 | 1.00 |
| 8k-12k | 0.0274 | 0.0229 | 0.84 |
| 12k-16k | 0.0268 | 0.0198 | 0.74 |
| 16k-20k | 0.0236 | 0.0177 | 0.75 |
| 20k-24k | 0.0191 | 0.0169 | 0.88 |

Consistent direction across six windows with a widening gap; F continued to
0.0140 by 49k, below D's final 0.0191. This supersedes the earlier four-window
read, which was too noisy to lean on.

### Sharpness: pooling does NOT fix the blur

Lower loss is not a better-looking creature, so this was measured separately.
`tools/sharpness_compare.py` runs local (NCA2) and pooled (NCAP) families
through one PyTorch harness — the Swift verifier cannot parse NCAP, and
rendering the two families in different renderers would make the renderer the
confound. Sharpness is mean Sobel gradient magnitude of the rendered RGB;
the target set's own sharpness is the ceiling.

| creature | config | sharpness | % of target | poses visited |
|---|---|---|---|---|
| D local | h128, 23k | 0.01352 | 60% | 11.3 / 12 |
| F pooled | h128, 50k | 0.01558 | **70%** | 12 / 12 |
| wp3 local (shipped) | h256, 90k, waypoints | 0.01576 | **70%** | 12 / 12 |

**Pooled matches the best local creature rather than beating it** — same 70%
ceiling, reached with half the hidden width and half the training. That is an
efficiency result, not a visual one, and it should not be reported as "pooling
makes them look better".

Taken with the undertraining negative result above, two candidate explanations
for the blur are now dead:

- more training (67k extra iterations: no sharpness change)
- global coordination (pooling: same 70% ceiling)

Both of those were about *coordination* — getting the body to agree with itself.
The blur survives fixing coordination, which points instead at the update rule's
capacity to represent high-frequency detail at all: 16 channels through a 1x1
convolution, with an alive mask and stochastic firing that both average. The
next thing to test is perception receptive field (dilated taps) and channel
count, not anything that makes the creature better coordinated.

### 96^2: suggestive, but NOT a clean control

Arm G (pooled, 96^2) against arm B (local, 96^2), the resolution where strict
locality demonstrably failed — B bottomed out near 0.0331 around 8-12k and then
*rose* (0.0339, 0.0351) rather than converging.

| iters | B local | G pooled | ratio |
|---|---|---|---|
| 0.5k-2k | 0.0620 | 0.0348 | 0.56 |
| 2k-4k | 0.0469 | 0.0318 | 0.68 |
| 4k-6k | 0.0396 | 0.0334 | 0.84 |

G reaches B's lifetime-best loss in roughly 8k iterations instead of ~10k.
**Caveat, and it is a real one:** G runs batch 8 / pool 1024 because it is on
the Mac's MPS backend, against B's batch 16 / pool 2048. That is a confound, so
treat this as suggestive only — the 64^2 F-vs-D pair above is the controlled
result. The question G actually settles is whether it *diverges* the way B did
past 12k, which needs it to get there.

### Second measurement: g is proprioceptive, not a driver

`tools/pooled_behavior.py`, arm F at 14k iterations, 5 independent rollouts
(the timing statistic flips sign between single runs — one rollout gave
+0.051 at lag +16 and the next +0.033 at lag -32, pure noise crossing a
threshold, so everything below is averaged over trials):

- **Level coupling**: eta^2 of g[0] against nearest-pose identity is
  **0.864 +/- 0.002**. Extremely tight. 86% of g's variance is explained by
  which pose the body currently occupies.
- **Transition timing**: peak cross-correlation +0.041 +/- 0.014, at the edge of
  the lag window, with a shallow U-shape (negative near lag 0, weakly positive
  at both extremes). That shape is what quasi-periodic transitions produce, not
  a causal lead.
- The creature freely traverses all 12 poses of the ring, transitioning about
  once per 15 steps, dwelling 22-356 steps.

So g reliably *encodes* what the body is doing and does not *drive* what it does
next. It is a sense organ, not a will.

**Why, and what to do about it.** The creature is handed its state flag `st`
every single step. It has no reason to use g to remember anything, because the
answer is supplied for free — g is redundant with an external signal, so
gradient descent settles for making it a readout. To make g load-bearing, take
the answer away: withhold the flag for stretches of the rollout and force the
creature to maintain its own state across the gap. The only place that state can
live is the global variable.

That is `--flag-dropout` in `train_constellation.py` (implemented, not yet run).
If it works, g stops being proprioception and becomes memory — and a creature
whose mood persists in a variable no cell owns is much closer to "I don't
necessarily know what it's going to do" than anything driven by a flag we set.

### Arm F completed (60k): g becomes an oscillator, and a near-perfect mirror

Re-measured on the finished checkpoint. Everything got *more* pronounced:

| | F @ 14k | F @ 60k |
|---|---|---|
| swing | 0.095 | **0.758** (8x) |
| lag-100 autocorrelation | +0.04 .. +0.59 | **-0.80 .. -0.89** |
| decorrelation | 56-159 steps | ~34 steps |
| eta^2 vs pose | 0.86 / 0.40 / 0.46 / 0.58 | **0.96 / 0.96 / 0.96 / 0.96** |
| timing lead | none | none (r 0.016 +/- 0.005) |

Positive autocorrelation at short lags with strongly negative correlation at
lag 100 is an **oscillation** with period ~200 steps — close to the pose ring's
~188-step lap. g is not drifting; it is cycling with the body.

Two things follow, and both sharpen the next experiment:

1. **Training made g a better mirror, not a driver.** eta^2 went 0.86 -> 0.96
   with no timing lead appearing. Given the state flag is handed over every
   step, the most useful thing g can possibly be is an encoding of body state,
   and gradient descent refined it toward exactly that. The proprioception
   finding is not an artifact of early training; it is where this objective
   converges.
2. **The four pooled channels collapsed into one.** At 14k they were
   differentiated (eta^2 0.86 / 0.40 / 0.46 / 0.58); at 60k all four sit at
   0.96, carrying the same signal. **npool=4 is effectively npool=1.** Paying
   for four global channels and getting one is worth knowing before scaling
   this up.

**Prediction for arm H (`--flag-dropout 0.5`, running):** if withholding the
flag works, eta^2 should *fall* — g stops being a pure pose mirror because it
has to carry state the flag no longer supplies — and/or the four channels
should differentiate again, and/or a timing lead should appear. If eta^2 stays
at 0.96 and nothing else moves, the flag was not what was holding g back and
the hypothesis is wrong.

### CLOSED (2026-07-20): removed from the product

Max called it: *"the blur is nearly identical across them, the pooling doesn't
seem to have helped in any way."* Confirmed on every axis that matters to the
product, so the pooled creatures are out of the app menu. Final ledger:

**Arm H (flag dropout) falsified the driver hypothesis.** eta^2 stayed at
0.92-0.95, no channel differentiation, no timing lead. And the experiment had a
design flaw worth recording: the state flag is 0/1 and "withholding" masked it
to 0 — which is indistinguishable from the state-0 signal. So dropout was a
no-op for calm samples and mislabeling for rage ones. A clean rerun would encode
state as +/-1 with 0 = absent. Not run: even a success would make g a memory,
not a driver, and the product case was already dead.

**What flag-dropout did buy: mood inertia.** Hysteresis test — settle each
creature in rage, then cut the flag:

| creature | state at +60 steps | +200 | +600 |
|---|---|---|---|
| H (dropout) | **RAGE** | calm | calm |
| F (pooled) | calm | calm | calm |
| wp3 (local) | calm | calm | calm |

H lingers in rage for 60-200 steps (2-7 s at display rate) where the others snap
back instantly. Real, measurable, and not worth the machinery: the LLM listener
can produce the same lingering by easing the flag, for free.

**What stays in the repo**: pooled_nca.py, the three analysis tools, and the
NCAP runtime path (zero-cost when npool = 0; archived checkpoints still load).
The scientific results stand — pooled trains to lower loss at equal width
(0.74-0.88 ratio), g self-organizes into a body-locked oscillator, four
channels collapse into one — they just do not serve this product.

Open questions:
- ~~Does g carry dynamics?~~ Slow variable; an oscillator at full training.
- ~~Does g correspond to behaviour?~~ Readout only (eta^2 0.96, no lead).
- ~~Does flag-dropout make g a driver?~~ **No — inertia only, and the flag
  encoding made the test weaker than designed (see above).**
- Does it fix 96^2 coherence, where strict locality demonstrably failed?
- Does damage recovery survive, or does the global variable make wounds global?
- If it works, it needs Metal + Triton support and an NCAP parser before it can
  reach the desktop app.
