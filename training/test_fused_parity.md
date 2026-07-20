# test_fused_parity.py

## Purpose
Gates 1–2 of docs/TRITON_KERNEL_PLAN.md §4. Gate 1: fused forward vs eager (shared bit-exact fire mask), max abs diff < 1e-5 fp32, across all six trainer configs (2D/3D × cond 0–3 × FiLM × clamp). Gate 2: 4-step chained backward — analytic fused grads vs eager-f32 autograd vs eager-f64 ground truth; fused error must be < max(3× eager-f32 error, 1e-6) and < 1e-4, for dx, dw1, db1, dw2, db2, dgamma, dbeta.

`EagerNCA` here is the parameterized reference copy of the trainer forwards with the fire mask injectable — w2 deliberately non-zero (trainers zero-init it, which would mask bugs).

## Contracts
| Dependent | Expects | Breaking changes |
|---|---|---|
| CI-by-hand before any fused training run | exit 0, "ALL GATES PASS" | any fused_step.py or trainer-math change |

Run on CUDA: `python3 test_fused_parity.py`. Last verified: 2026-07-19 on RTX 2070 SUPER and RTX 4090 (Aine), torch 2.11/triton 3.6 — all pass. On Ada+ the test pins cudnn.allow_tf32=False (TF32 convs are eager imprecision, not kernel error).
