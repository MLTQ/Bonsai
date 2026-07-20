# bench_fused.py

## Purpose
Gate 4 instrument: eager-vs-fused throughput (full train iter and no-grad rollout) at 2D-64/3D-32/3D-64, plus peak memory. Eager 3D uses CHUNK=8 checkpointing (what the trainers do); the 3D-64 config auto-skips under 20 GiB. Results table lives in docs/TRITON_KERNEL_PLAN.md §8 — add a row per new GPU (H100 next).

## Contracts
Imports `EagerNCA`/`_aux` from test_fused_parity.py — bench configs stay in lockstep with the parity configs by construction. Fused training uses the single-node `fused_nca_rollout`; forward-only uses its no-history inference branch.
