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

Loss at matched iterations vs the local control (arm D, same width/batch/target,
differing only in pooling): ratios 0.74 / 0.95 / 0.87 / 0.94 over the first four
1k windows. Consistent sign, but four noisy medians is weak evidence; revisit
with the full curves.

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

Open questions the current arms should answer:
- ~~Does g carry dynamics, or settle to a constant?~~ **Answered: slow variable,
  tau ~100 steps.**
- ~~Does g correspond to visible behaviour?~~ **Answered: yes, but as a readout
  (eta^2 0.86, no timing lead).**
- Does `--flag-dropout` convert g from proprioception into memory?
- Does it fix 96^2 coherence, where strict locality demonstrably failed?
- Does damage recovery survive, or does the global variable make wounds global?
- If it works, it needs Metal + Triton support and an NCAP parser before it can
  reach the desktop app.
