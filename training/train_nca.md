# train_nca.py

## Purpose
Trains the Growing NCA (Mordvintsev et al., distill.pub 2020) to grow, persist as, and regrow the bonsai target. Exports weights in the flat binary format the Swift/Metal runtime consumes.

## Components

### `NCA`
- **Does**: 16-channel CA: fixed perception (identity/sobelX/sobelY per channel via `groups=16` conv) → 1×1 conv 48→128 ReLU → 1×1 conv 128→16, zero-init final layer; stochastic fire mask; pre&post life masking
- **Interacts with**: Mirrored exactly by `nca_step`/`nca_life` in `Sources/Bonsai/NCAShaders.swift`
- **Rationale**: The `groups=16` output ordering (identity, sx, sy interleaved per channel) is load-bearing — the Metal shader assumes it

### Pool training loop (`main`)
- **Does**: 1024-sample pool, batch 8; worst sample reset to seed (persistence), 2 samples circular-damaged after iter 500 (regeneration); 64–96 steps/iter; per-parameter grad normalization; lr 2e-3 → 2e-4 at iter 2000
- **Rationale**: Pool + damage is what makes the pet stable on screen and regrow when poked

### `export`
- **Does**: Writes `NCA1` header + w1,b1,w2,b2 float32 LE
- **Interacts with**: Parsed by `Sources/Bonsai/NCAWeights.swift` — formats must stay in lockstep

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `NCAWeights.swift` | Exact binary layout, ch=16 hidden=128 | Header/shape/order changes |
| `NCAShaders.swift` | Identical math (kernels, fire semantics, life rule) | Any model change |

## Notes
- ~4 it/s on M1 Pro MPS; full 8000-iter run ≈ 35 min. Checkpoints + `train_preview.png` every 250 iters; the app hot-reloads checkpoints live.
