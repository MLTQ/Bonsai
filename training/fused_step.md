# fused_step.py

## Purpose
The Triton port of the Metal fused formulation (docs/TRITON_KERNEL_PLAN.md): the whole NCA step as two CUDA kernels (`_nca_step_fwd`, `_nca_life_fwd`) instead of ~15 eager launches. Perception is folded into w1 on the host (W1eff = taps ⊗ w1), so perception + first MLP layer run as one K-chunked tensor matmul over the raw 9/27-cell neighborhood. Fire mask is counter-based (`tl.rand(seed, cell_index)`) — deterministic given (seed, step), so backward and checkpoint replays recompute it exactly, and `fire_mask()` reproduces it bit-for-bit for eager references.

Backward (`FusedNCAStep`) saves only each step's 16-ch input state, then: replays the forward kernel once with SAVE=True (one launch materializes percept, h_lin flat + raw pre-clamp x_mid), runs `_nca_bwd_gates` (life+clamp+fire in one launch), cuBLAS mms for the MLP/FiLM grads, and `_nca_bwd_percept` (gather-form perception transpose + residual). Handles dims 2/3, cond 0–4, FiLM on/off, clamp on/off — every trainer variant.

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| all trainers' `--fused` | `fused_nca_step(x, w1(H,PIN), b1, w2(CH,H), b2, cond, gamma, beta, seed, step, fire_rate, clamp)`; step derived from loop indices (checkpoint replay) | signature, RNG scheme |
| eager models + Metal shaders | three-way numerical contract: perception ordering [id,sx,sy(,sz)], sobel /8 (2D) /32 (3D), life = alive(pre)&alive(post) via maxpool alpha > 0.1, ±8 clamp | any math change updates all three |
| `test_fused_parity.py` | fwd < 1e-5 vs eager; grads at eager-f32 noise vs f64 truth | numerics |

## Hard-won constraints (do not regress)
- `tl.dot(..., input_precision="ieee")` everywhere: no tf32, or parity dies on Ampere+.
- The fold costs ~4.75x eager FLOPs but the FLOP-exact alternative (explicit taps + outer-product FMA accumulation into (BLOCK,HIDDEN)) is **30x slower** — register spills defeat Triton codegen. Keep the K-chunked `tl.dot`.
- NO cuDNN grouped convs (groups=16) anywhere: grouped conv_transpose2d was 74% of 2D backward, grouped conv3d 60% of 3D (genericTranspose engine). Perception transpose lives in `_nca_bwd_percept`.
- Backward replay passes clamp=None: the clamp gate needs the RAW pre-clamp x_mid — on already-clamped values the gate `|x_mid*life| <= 8` is vacuously true and gradient leaks at the boundary (bit us at chained steps ≥3: states park at exactly ±8, eager passes grad there, strict comparisons don't).
- Fold cache keyed (w1.data_ptr, dims, cond_n), invalidated by `_version` — trainers pass fresh `.reshape` views every call; identity must survive that.
- No Triton → `HAS_TRITON=False`, import still succeeds (MPS/CPU boxes use eager).
