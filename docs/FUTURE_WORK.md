# Future Work (parked, not scoped)

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

Open questions the current arms should answer:
- ~~Does g carry dynamics, or settle to a constant?~~ **Answered: slow variable,
  tau ~100 steps.** Next: does g's motion correspond to visible behaviour?
- Does it fix 96^2 coherence, where strict locality demonstrably failed?
- Does damage recovery survive, or does the global variable make wounds global?
- If it works, it needs Metal + Triton support and an NCAP parser before it can
  reach the desktop app.
