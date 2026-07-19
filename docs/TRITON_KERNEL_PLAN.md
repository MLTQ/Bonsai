# Triton Fused NCA Kernel — Plan

The one optimization that actually matters, written down before we build it.
Status: designed, not started. Owner: next session with a free day and a GPU.

## 1. Why (measured, not assumed)

Every trainer steps the automaton with ~15 small CUDA ops per step from a
Python loop, ×48–96 steps per rollout, ×2–3 with checkpoint recompute:

- 2D/32³: **launch-latency-bound.** GPU util ~20% while "100% busy" queueing.
  The 4090 trained 32³ at 2.7 it/s; the H100 managed 0.75 it/s at 64³ —
  roughly a 4090's projected pace, ~25% of the silicon's paper advantage.
- 64³: compute is heavier but fusion still pays; torch-checkpoint recompute
  adds ~35% on top.
- What we already tried (2026-07-19, all committed):
  - `torch.compile(reduce-overhead)` → **CUDAGraphs output-recycling crashes**:
    chained step outputs + pool writeback + checkpoint recompute all hold
    graph-owned tensors. `cudagraph_mark_step_begin` did not save us.
  - `max-autotune-no-cudagraphs` → **net zero** (41 vs 45 steps/s at 64³).
    Inductor fuses elementwise chains, but the convs were already cuDNN and
    the launch count barely moved.

Conclusion: we need what the Metal runtime already has — the whole step as
one (well, two) kernels. The Metal shaders ARE the proven fused formulation;
this is a port, not an invention.

## 2. Kernel design (mirror NCAShaders.swift / NCAShaders3D.swift)

Two kernels per step, exactly like Metal:

**`nca_step_fwd`** — one program per cell tile:
- Load state tile + 1-cell halo (16 ch) into registers/SRAM.
- Perception inline: identity + Sobel x/y (+z), `/8` (2D) or `/32` (3D);
  append cond values (scalars) → 48/50/51 or 64/66/67 inputs.
- MLP inline: w1 (≤128×67×4B ≈ 34 KB) + w2 (8 KB) fit in shared memory,
  loaded once per program. ReLU. Optional FiLM gamma/beta (2×H scalars).
- Fire mask via **counter-based RNG**: `tl.rand(seed, cell_offset + step_ctr)`
  — deterministic given (seed, step), which makes recompute exact (§3) and
  matches the Metal hash philosophy. Do NOT use torch's stateful RNG.
- Residual add, clamp ±8 → write `x_mid`.

**`nca_life_fwd`** — 3×3(×3) alpha maxpool over pre and mid, zero dead cells
→ `x_out`. (Two kernels because post-life needs neighbors' post-alpha;
same reasoning as Metal's three-buffer rotation.)

Launch count per step: 15 → 2. Python loop overhead stays (µs; irrelevant
once kernels are chunky). If it ever matters: CUDA-graph the fused pair —
far easier to make safe with 2 kernels and preallocated ping-pong buffers
than with torch.compile's ownership rules.

## 3. Backward: per-step recompute inside a custom autograd.Function

The memory insight: don't save activations at all — save only each step's
**input state** (16 ch = 8× smaller than hidden). Backward per step:

1. Recompute forward internals from saved x_in (fire mask is free to
   recompute thanks to counter-based RNG; life mask likewise from x_in/x_mid).
2. Backprop analytically: through clamp gate → life gate (0/1) → fire gate →
   w2ᵀ → ReLU gate → w1ᵀ → perception transpose (correlation with flipped
   kernels; cond grads reduce-sum; FiLM grads reduce over cells).

This subsumes torch gradient checkpointing (which recomputes whole CHUNKS);
per-step save/recompute is the optimal point on that curve and removes
`torch.utils.checkpoint` from the hot path entirely — along with its
CUDAGraphs/RNG entanglements.

v0 shortcut if hand-backward stalls: fused forward + `torch.autograd` over a
recomputed eager step in backward (still saves all the launch overhead in
forward, which dominates).

## 4. Validation gates (in order, no skipping)

1. **Forward parity**: fused vs eager on 16²/16³ grids, fixed seed, max
   abs diff < 1e-5 (fp32). Fire masks must match bit-exact via shared hash.
2. **Backward parity**: `gradcheck`-style comparison vs eager autograd on
   tiny dims (float64 eager reference).
3. **Training A/B**: 1k iters bonsai (2D) fused vs eager — loss curves must
   overlay within noise.
4. **Throughput report**: it/s on M-series (Triton CPU? no — CUDA only;
   MPS stays eager), 4090, and rental H100, table into this doc + the paper.

## 5. Expected wins (honest ranges)

| Workload | Now | Fused (est.) |
|---|---|---|
| 2D 64² (M1 eager stays) | 4 it/s | — (MPS unaffected) |
| 32³ on 4090 | ~2.7 it/s | 8–20 it/s |
| 64³ on 4090 | ~0.2–0.5 it/s | 1.5–4 it/s |
| 64³ on H100 | 0.75 it/s | 4–10 it/s (silicon finally visible) |

Economics: the aborted $25 H100 creature becomes ~$4; 128³ becomes thinkable.

## 6. Integration

- `training/fused_step.py`: `FusedNCAStep(w1, b1, w2, b2, film?, dims=2|3)`
  as an `autograd.Function` + thin module; feature-flag `--fused` in every
  trainer, default off until gates 1–3 pass.
- The eager path is the permanent reference implementation — it, the Metal
  shaders, and the fused kernel form a three-way contract; any math change
  updates all three (see guide §6).
- Requirements on the box: triton ≥3.x (ships with torch cu12x wheels ✓ on
  both Aine and Vast images).

## 7. Sequencing (≈ one focused day)

1. `nca_step_fwd` 2D + parity gate (≈3 h)
2. Backward-by-recompute + parity gate (≈3 h)
3. Trainer flag + A/B + throughput table (≈2 h)
4. 3D variant (same structure, 27-halo, 64-feature perception) (≈2 h)
5. Only then: rent the H100 again and spend the banked credits at real speed.
