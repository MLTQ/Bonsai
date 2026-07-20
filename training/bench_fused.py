"""Throughput benchmark: eager vs fused NCA training steps (plan §4 gate 4).

Measures full training iterations (rollout fwd + loss + backward) and
forward-only rollouts, eager vs fused, on the current CUDA device:

  2D 64^2  B=8 T=80  (bonsai-class,   no cond/film, no clamp)
  3D 32^3  B=8 T=64  (manifold-class, cond=2 + FiLM, clamp 8)
  3D 64^3  B=4 T=48  (H100-class; auto-skipped if it doesn't fit)

Eager runs use CHUNK=8 gradient checkpointing for 3D (what the trainers do);
fused runs use the per-step recompute built into FusedNCAStep.

Usage: python3 bench_fused.py [--iters 30] [--skip-64]
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fused_step import fused_nca_rollout
from test_fused_parity import CH, HIDDEN, FIRE_RATE, EagerNCA, _aux

CHUNK = 8


def eager_rollout(model, x, aux, steps, use_ckpt):
    def run_chunk(x0, n):
        for s in range(int(n)):
            fm = (torch.rand(x0.shape[0], 1, *x0.shape[2:], device=x0.device)
                  <= FIRE_RATE).float()
            x0 = model(x0, fm, *aux)
        return x0

    if not use_ckpt:
        return run_chunk(x, steps)
    done = 0
    while done < steps:
        n = min(CHUNK, steps - done)
        if x.requires_grad:
            x = checkpoint(run_chunk, x, torch.tensor(n), use_reentrant=False)
        else:
            x = run_chunk(x, n)
        done += n
    return x


def fused_rollout(model, x, aux, steps, seed):
    return fused_nca_rollout(
        x, model.w1, model.b1, model.w2, model.b2, steps,
        cond=aux[0], gamma=aux[1], beta=aux[2], seed=seed,
        fire_rate=FIRE_RATE, clamp=model.clamp,
    )


def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return iters / (time.time() - t0)


def run_config(name, dims, grid, batch, steps, cond_n, film, clamp,
               device, iters, use_ckpt):
    model = EagerNCA(dims, cond_n, film, clamp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand((batch, CH) + (grid,) * dims, device=device) * 0.4
    x0[:, 3] += 0.3
    aux = _aux(model, x0)
    tgt = torch.rand_like(x0[:, :4])
    it_counter = [0]

    def train_step(fused):
        it_counter[0] += 1
        x = x0.clone().requires_grad_(True)
        if fused:
            out = fused_rollout(model, x, aux, steps, seed=it_counter[0])
        else:
            out = eager_rollout(model, x, aux, steps, use_ckpt)
        loss = ((out[:, :4] - tgt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def fwd_only(fused):
        with torch.no_grad():
            if fused:
                fused_rollout(model, x0, aux, steps, seed=1)
            else:
                eager_rollout(model, x0, aux, steps, use_ckpt=False)

    torch.cuda.reset_peak_memory_stats()
    r_train_eager = bench(lambda: train_step(False), iters)
    m_eager = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    r_train_fused = bench(lambda: train_step(True), iters)
    m_fused = torch.cuda.max_memory_allocated() / 2**30
    r_fwd_eager = bench(lambda: fwd_only(False), iters)
    r_fwd_fused = bench(lambda: fwd_only(True), iters)
    print(f"{name:14s} train: eager {r_train_eager:6.2f} it/s ({m_eager:.1f} GiB) | "
          f"fused {r_train_fused:6.2f} it/s ({m_fused:.1f} GiB) | x{r_train_fused/r_train_eager:.1f}")
    print(f"{'':14s} fwd:   eager {r_fwd_eager:6.2f} it/s          | "
          f"fused {r_fwd_fused:6.2f} it/s          | x{r_fwd_fused/r_fwd_eager:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--skip-64", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    print(f"device: {name} ({total:.0f} GiB)")
    torch.manual_seed(0)

    run_config("2d-64  B8 T80", 2, 64, 8, 80, 0, False, None,
               device, args.iters, use_ckpt=False)
    run_config("3d-32  B8 T64", 3, 32, 8, 64, 2, True, 8.0,
               device, args.iters, use_ckpt=True)
    if not args.skip_64 and total > 20:
        run_config("3d-64  B4 T48", 3, 64, 4, 48, 2, True, 8.0,
                   device, max(args.iters // 3, 5), use_ckpt=True)


if __name__ == "__main__":
    main()
