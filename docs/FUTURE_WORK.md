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
Note the hard structural limit: information travels one cell per step, so large
grids have a coordination-latency ceiling regardless of channel count.
