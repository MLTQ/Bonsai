# Claudeguy — the capstone creature

Max's gift: a Claude embodiment built from the most advanced version of the
stack. Built LAST, from the answered versions of tonight's experiments.

## Reference art (from Max, 2026-07-18)
- `emotion_anchor_sheet.jpeg` — **this is the manifold anchor spec**: twelve
  named moods of one identity (serene, wince, indignant, mind-blown, monocle,
  pleading, suspicious, cowboy, melancholy, incognito, dizzy,
  shoggoth-attacked). Same petals, different posture/face.
- `reference_illustration.jpeg` — canonical 2D style: ~12 terracotta petals,
  white face disk, dot eyes, wobble mouth.
- `reference_claymation_3d.jpeg` — the volumetric target aesthetic: soft,
  rounded, slightly translucent, big glossy eyes.
- `reference_meme.jpeg` — two Claudeguys reviewing a dev environment. Morale.

## Architecture (the convergence creature)
- **Body**: volumetric (NC3C-family) at 32³, petal ring = Mk. III's azimuthal
  ring topology; traveling-wave machinery ports directly (sway/flutter/droop).
- **Moods**: FiLM manifold (Mk. II recipe, tanh-bounded), factors: petal droop,
  splay, amplitude, tremor, brightness, face-variant behavior index. Anchors
  named from the emotion sheet.
- **Clock**: internal oscillator if train_autonomous.py verdict is YES;
  otherwise phase-conditioned with the internalization retried per-creature.
- **Mood source**: the trace projector — embed recent Claude-session activity
  → project to z. Wince while debugging, ^^ on green tests, mushroom-cloud
  head reserved for genuine NaN events.

## Sequencing
Blocked on: (1) oscillator verdict, (2) 3D manifold (FiLM in the NC3C
trainer), (3) trace-projector daemon. Then: petal-ring frame generator →
corpus over mood factors → train on Aine → ship as the seventh creature.
