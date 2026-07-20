# Triton Fused NCA Kernel — Plan

The one optimization that actually matters, written down before we build it.
Status: **BUILT + OPTIMIZED — all gates pass** (2026-07-19). See §8 for measured results
and §9 for the H100 runbook. Implementation: training/fused_step.py (+ .md),
gates in training/test_fused_parity.py, bench in training/bench_fused.py.
`--fused` flag live in train_nca, train_nca3d, train_cyclic3d, train_manifold3d.

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

## 8. Measured results (2026-07-19, torch 2.11/cu130, triton 3.6)

All validation gates pass: forward < 1e-6 vs eager with bit-exact fire masks
(6 configs: 2D/3D x cond 0-3 x FiLM x clamp); backward grads at eager-fp32's
own rounding noise vs a float64 eager reference (incl. dgamma/dbeta); 1k-iter
bonsai A/B loss curves overlay within iteration noise (0.0103 vs 0.0090 at
iter 1000, noise band ±0.0015). On Ada+, gates pin the eager reference to
fp32 (`cudnn.allow_tf32 = False`) — cuDNN otherwise runs convs in TF32
(~1e-3), which is eager's imprecision, not the kernel's.

train = full training iter (rollout + loss + backward + opt); fwd = no-grad
rollout. Eager 3D uses CHUNK=8 checkpointing (what the trainers do).

| RTX 4090       | eager train | fused train | eager fwd | fused fwd | train mem  |
|----------------|-------------|-------------|-----------|-----------|------------|
| 2D 64² B8 T80  | 26.43 it/s  | 33.77 (1.28x)| 52.71    | 142.25 (2.70x)| 1.9→0.3 GiB|
| 3D 32³ B8 T64  | 2.19        | 2.99 (1.37x)| 7.26      | 25.41 (3.50x)| 2.2→1.9   |
| 3D 64³ B4 T48  | 0.59        | 0.98 (1.66x)| 1.99      | 8.14 (4.09x)| 8.7→6.5    |

| RTX 2070S (Turing) | eager train | fused train | eager fwd | fused fwd |
|--------------------|-------------|-------------|-----------|-----------|
| 2D 64² B8 T80      | 10.9 it/s   | 8.7 (0.8x)  | 27.3      | 44.0 (1.6x)|
| 3D 32³ B8 T64      | 0.68        | 0.66 (1.0x) | 2.2       | 2.4 (1.1x)|

Real trainer validation on the 4090: 2D bonsai reached 31.53 it/s over 1k
iterations and converged to loss 0.00960 (inside the prior A/B noise band).
The 2D and static-3D trainers also passed end-to-end CUDA smoke runs.

Honest notes vs §5's estimates:
- Forward fusion now delivers 2.7–4.1x on Ada; whole-train is 1.3–1.7x.
  Production follows eager cuDNN's TF32 policy, so Triton forward dots and
  flattened backward GEMMs use tensor cores; strict parity runs remain IEEE.
  A whole trajectory is one autograd node with contiguous state history and
  reusable scratch, removing per-step engine/allocation overhead. Backward
  still dominates (replay + gates + GEMMs + perception transpose).
- The speedup GROWS with grid size and GPU speed (the 2070 row predates the
  rollout/TF32 optimization; 4090 is now 1.3–1.7x) — the launch-bound pathology barely exists on
  Turing. Expect the H100 gap to be wider than the 4090's, especially at
  64³ where eager's checkpoint recompute and launch count hurt most.
- Memory: fused training saves only 16-ch input states per step (no
  checkpoint chunks, no stored activations): 0.3 vs 1.9 GiB at 2D-64,
  6.5 vs 8.7 GiB at 64³ B4. 128³ becomes thinkable on 80 GB.
- Next lever if more train speed is needed: fuse backward's FiLM/relu
  elementwise chain and the dw-mms into the replay kernel family. The
  failed experiments are documented in fused_step.md — do NOT retry
  outer-product FMA accumulation or cuDNN grouped convs.

## 9. H100 runbook (after renting)

```
scp training/*.py <box>:bonsai/training/
python3 test_fused_parity.py        # must print ALL GATES PASS — no skipping
python3 bench_fused.py --iters 25   # fill the H100 row into §8's table
BONSAI_GRID3=64 python3 train_manifold3d.py --fused --corpus <corpus> ...
```
Requirements already on Vast images: triton >= 3.x ships with torch cu12x+
wheels. If gates fail on a new torch/triton: check tf32 defaults first (§8).
