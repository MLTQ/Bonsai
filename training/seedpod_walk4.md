# Rejected seedpod shear-guided pose sheet

## Purpose

Diagnostic record for a rejected higher-resolution 2D sheet. The subject is a
non-human chibi seed robot, but the four images do **not** form a valid walk
cycle: apparent boot displacement and foreshortening changed while the same
semantic leg remained in front of the other leg.

This sheet and every derivative below are excluded from the
global-controller/transport A/B. A baseline accidentally started from this
corpus was stopped at iteration 1,100 as soon as the semantic error was found.

## Files

- `seedpod_walk4_source.png` — rejected 2x2 diffusion sheet.
- `corpus_seedpod_walk4_128.npz` — premultiplied `(1,4,128,128,4)`
  diagnostic derivative; **not a training target**.
- `../experiments/seedpod_walk4_128_targets.png` — rejected transparent target
  strip retained for forensic comparison.

## Intended pose order (not achieved)

1. Gold-side contact: legs separated, leading boot extended.
2. Gold-side passing: planted leading boot, trailing leg passing.
3. Opposite contact: opposite leg extended.
4. Opposite passing: opposite planted/passing configuration.

The intended directed ring was `0 -> 1 -> 2 -> 3 -> 0`, with only endpoints on
one directed edge allowed to blend. Visual review showed that frames 0 and 2
did not exchange named-leg screen order or foreground occlusion, so this
sequence cannot test that contract.

## Generation provenance

- Model: `waiIllustriousSDXL_v170`, SHA-256
  `f116b0c78ff441467b0cdc8f1936e1ed18ea31e9997c7b132b1b8db533f0bd04`.
- Hardware: RTX 2070 SUPER selected by GPU UUID.
- Identity source: seed 37 from the concise model-sheet pass.
- Final anchors: seed 37 img2img, strength 0.58, 32 steps, guidance 6.0,
  using the continuous hip-anchored pose guides from
  `generate_transport2d_poses.py`.
- Ingestion: near-white soft key, largest connected component, shared union
  crop, one shared scale, 128x128 output.

Two earlier translated-guide passes were rejected because they produced a
detached boot. The continuous shear guides fixed limb attachment but still did
not control semantic leg identity or depth order.

## Diagnostics (insufficient acceptance criteria)

- Adjacent-pose MSE: `0.00624, 0.00846, 0.00479, 0.00912`.
- Opposite-pose MSE: `0.01216, 0.00567`.
- Visible alpha support: 3,334–3,478 cells per frame at threshold 0.05.

These pixel statistics failed to detect the semantic pose error. Replacement
generation must use explicit pose conditioning plus a palette-bound named-leg
swap test, followed by visual review, before any NCA training starts.
