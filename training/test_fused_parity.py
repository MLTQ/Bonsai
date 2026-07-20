"""Parity gates for the fused Triton NCA step (docs/TRITON_KERNEL_PLAN.md §4).

Gate 1 — forward parity: fused vs eager, fixed seed, max abs diff < 1e-5 fp32,
         fire masks bit-exact (eager consumes the kernel's own mask).
Gate 2 — backward parity: analytic grads vs eager autograd (fp32) with a
         float64 eager reference as ground truth; the fused error must be
         within 3x the eager-fp32 rounding error (and < 1e-4 relative).
Gate 3 — rollout parity: one trajectory node vs chained step nodes, including
         time-varying 3D conditioning and FiLM gradients.

The eager model here is a parameterized copy of the trainer forward()s
(train_nca / train_cyclic / train_manifold / train_nca3d / train_manifold3d)
with the fire mask injectable — the same math, the permanent reference.

Run on a CUDA box:  python3 test_fused_parity.py
Picks the least-busy GPU unless CUDA_VISIBLE_DEVICES is set.
"""

import os
import sys

import torch
import torch.nn.functional as F

import fused_step
from fused_step import (CH, fire_mask, fused_nca_rollout, fused_nca_step,
                        perception_conv_weight)

HIDDEN = 128
FIRE_RATE = 0.5


class EagerNCA(torch.nn.Module):
    """Trainer-forward replica: dims 2/3, cond 0-3, FiLM on/off, clamp on/off."""

    def __init__(self, dims, cond_n, film, clamp, dtype=torch.float32):
        super().__init__()
        nk = 3 if dims == 2 else 4
        self.dims, self.cond_n, self.film_on, self.clamp = dims, cond_n, film, clamp
        g = torch.Generator().manual_seed(7)
        self.w1 = torch.nn.Parameter(
            torch.randn(HIDDEN, CH * nk + cond_n, generator=g, dtype=dtype) * 0.15)
        self.b1 = torch.nn.Parameter(torch.randn(HIDDEN, generator=g, dtype=dtype) * 0.05)
        # w2 nonzero (trainers init zero, which would hide most bugs)
        self.w2 = torch.nn.Parameter(torch.randn(CH, HIDDEN, generator=g, dtype=dtype) * 0.05)
        self.b2 = torch.nn.Parameter(torch.randn(CH, generator=g, dtype=dtype) * 0.02)
        self.register_buffer("percept_w", perception_conv_weight(dims, "cpu").to(dtype))

    def forward(self, x, fmask, cond=None, gamma=None, beta=None):
        conv = F.conv2d if self.dims == 2 else F.conv3d
        pool = F.max_pool2d if self.dims == 2 else F.max_pool3d
        bc = (slice(None), slice(None)) + (None,) * self.dims
        pre_life = pool(x[:, 3:4], 3, stride=1, padding=1) > 0.1
        p = conv(x, self.percept_w, padding=1, groups=CH)
        if self.cond_n:
            cmap = cond[bc].expand(-1, -1, *x.shape[2:])
            p = torch.cat([p, cmap], dim=1)
        nk = 3 if self.dims == 2 else 4
        h = conv(p, self.w1.reshape(HIDDEN, CH * nk + self.cond_n, *(1,) * self.dims), self.b1)
        if self.film_on:
            h = h * (1 + gamma[bc]) + beta[bc]
        h = F.relu(h)
        dx = conv(h, self.w2.reshape(CH, HIDDEN, *(1,) * self.dims), self.b2)
        x = x + dx * fmask
        life = (pre_life & (pool(x[:, 3:4], 3, stride=1, padding=1) > 0.1)).to(x.dtype)
        out = x * life
        return out.clamp(-self.clamp, self.clamp) if self.clamp else out


def grown_state(model, x0, device, steps, seed):
    """Run eager steps (kernel-RNG masks) to reach a realistic mixed state."""
    x = x0
    with torch.no_grad():
        for s in range(steps):
            fm = fire_mask(x, seed, 1000 + s, FIRE_RATE)
            x = model(x, fm, *_aux(model, x))
    return x


def _aux(model, x, requires_grad=False):
    """Deterministic cond/gamma/beta for a config (grad-enabled on request)."""
    b = x.shape[0]
    g = torch.Generator().manual_seed(11)
    cond = gamma = beta = None
    if model.cond_n:
        cond = torch.randn(b, model.cond_n, generator=g).to(x.device).to(x.dtype)
        cond.requires_grad_(requires_grad)
    if model.film_on:
        gamma = torch.tanh(torch.randn(b, HIDDEN, generator=g) * 0.3).to(x.device).to(x.dtype)
        beta = (torch.randn(b, HIDDEN, generator=g) * 0.1).to(x.device).to(x.dtype)
        gamma.requires_grad_(requires_grad)
        beta.requires_grad_(requires_grad)
    return cond, gamma, beta


def seed_state(dims, batch, grid, device):
    shape = (batch, CH) + (grid,) * dims
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(shape)
    # random blob: alive patch in the middle, so life mask has structure
    center = (slice(None), slice(None)) + tuple(
        slice(grid // 4, grid - grid // 4) for _ in range(dims))
    x[center] = torch.randn(x[center].shape, generator=g) * 0.6
    x[:, 3] = x[:, 3].abs()  # some alpha above threshold, some below
    return x.to(device)


CONFIGS = [  # (name, dims, cond_n, film, clamp)
    ("2d-bonsai      dims=2 cond=0 film=0 clamp=None", 2, 0, False, None),
    ("2d-states      dims=2 cond=1 film=0 clamp=8", 2, 1, False, 8.0),
    ("2d-cyclic      dims=2 cond=3 film=0 clamp=None", 2, 3, False, None),
    ("2d-manifold    dims=2 cond=2 film=1 clamp=8", 2, 2, True, 8.0),
    ("3d-nca3d       dims=3 cond=0 film=0 clamp=8", 3, 0, False, 8.0),
    ("3d-manifold    dims=3 cond=2 film=1 clamp=8", 3, 2, True, 8.0),
]


def gate1_forward(device):
    print("=== Gate 1: forward parity (fused vs eager, shared fire mask)")
    ok = True
    for name, dims, cond_n, film, clamp in CONFIGS:
        grid = 16
        model = EagerNCA(dims, cond_n, film, clamp).to(device)
        x = seed_state(dims, 2, grid, device)
        x = grown_state(model, x, device, 12, seed=5)
        aux = _aux(model, x)
        for step in (0, 1, 77):
            fm = fire_mask(x, 42, step, FIRE_RATE)
            with torch.no_grad():
                ref = model(x, fm, *aux)
            out = fused_nca_step(
                x, model.w1, model.b1, model.w2, model.b2,
                cond=aux[0], gamma=aux[1], beta=aux[2],
                seed=42, step=step, fire_rate=FIRE_RATE, clamp=clamp)
            diff = (out - ref).abs().max().item()
            alive_frac = (ref.abs().sum(1) > 0).float().mean().item()
            status = "PASS" if diff < 1e-5 else "FAIL"
            ok &= diff < 1e-5
            print(f"  [{status}] {name}  step={step:3d}  maxdiff={diff:.3e}  alive={alive_frac:.2f}")
    return ok


def gate2_backward(device):
    print("=== Gate 2: backward parity (analytic vs eager autograd vs f64 truth)")
    ok = True
    for name, dims, cond_n, film, clamp in CONFIGS:
        grid = 10 if dims == 3 else 12
        steps = 4
        model32 = EagerNCA(dims, cond_n, film, clamp).to(device)
        model64 = EagerNCA(dims, cond_n, film, clamp, dtype=torch.float64).to(device)
        with torch.no_grad():
            for p64, p32 in zip(model64.parameters(), model32.parameters()):
                p64.copy_(p32.double())
        x0 = seed_state(dims, 2, grid, device)
        x0 = grown_state(model32, x0, device, 8, seed=9)
        masks = [fire_mask(x0, 13, s, FIRE_RATE) for s in range(steps)]
        gproj = torch.randn(x0.shape, generator=torch.Generator().manual_seed(21)).to(device)

        def eager_loss(model, dtype):
            aux = _aux(model32, x0.to(dtype), requires_grad=False)
            aux = tuple(a.detach().to(dtype) if a is not None else None for a in aux)
            film_leaves = [a.requires_grad_(True) for a in aux[1:] if a is not None]
            x = x0.detach().clone().to(dtype).requires_grad_(True)
            y = x
            for s in range(steps):
                y = model(y, masks[s].to(dtype), aux[0], *aux[1:])
            loss = (y * gproj.to(dtype)).sum()
            grads = torch.autograd.grad(
                loss, [x] + list(model.parameters()) + film_leaves)
            return grads

        def fused_loss():
            aux = _aux(model32, x0, requires_grad=False)
            film_leaves = [a.requires_grad_(True) for a in aux[1:] if a is not None]
            x = x0.detach().clone().requires_grad_(True)
            y = x
            for s in range(steps):
                y = fused_nca_step(
                    y, model32.w1, model32.b1, model32.w2, model32.b2,
                    cond=aux[0], gamma=aux[1], beta=aux[2],
                    seed=13, step=s, fire_rate=FIRE_RATE, clamp=clamp)
            loss = (y * gproj).sum()
            return torch.autograd.grad(
                loss, [x] + list(model32.parameters()) + film_leaves)

        g64 = eager_loss(model64, torch.float64)
        g32 = eager_loss(model32, torch.float32)
        gfu = fused_loss()
        names = ["x"] + [n for n, _ in model32.named_parameters()]
        if film:
            names += ["gamma", "beta"]
        for nm, a64, a32, afu in zip(names, g64, g32, gfu):
            denom = a64.norm().item() + 1e-12
            e_eager = (a32.double() - a64).norm().item() / denom
            e_fused = (afu.double() - a64).norm().item() / denom
            passed = e_fused < max(3 * e_eager, 1e-6) and e_fused < 1e-4
            ok &= passed
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}  d{nm:<12} rel_err fused={e_fused:.2e}  eager32={e_eager:.2e}")
    return ok


def gate3_rollout(device):
    """Whole-rollout node must match the already-gated per-step recurrence."""
    print("=== Gate 3: rollout parity (single node vs per-step nodes)")
    ok = True
    cases = [
        ("2d-static", 2, 0, False, None),
        ("3d-sequence-film", 3, 2, True, 8.0),
    ]
    for name, dims, cond_n, film, clamp in cases:
        model = EagerNCA(dims, cond_n, film, clamp).to(device)
        grid = 8 if dims == 3 else 10
        steps = 4
        x0 = seed_state(dims, 2, grid, device)
        x0 = grown_state(model, x0, device, 5, seed=31)
        aux = _aux(model, x0)
        if cond_n:
            offsets = torch.arange(steps, device=device)[:, None, None] * 0.03
            cond = aux[0][None] + offsets
        else:
            cond = None

        def run(as_rollout):
            x = x0.detach().clone().requires_grad_(True)
            gamma = aux[1].detach().clone().requires_grad_(True) if film else None
            beta = aux[2].detach().clone().requires_grad_(True) if film else None
            if as_rollout:
                y = fused_nca_rollout(
                    x, model.w1, model.b1, model.w2, model.b2, steps,
                    cond=cond, gamma=gamma, beta=beta, seed=23, step_offset=7,
                    fire_rate=FIRE_RATE, clamp=clamp, fast_math=False)
            else:
                y = x
                for step in range(steps):
                    cond_step = cond[step] if cond is not None else None
                    y = fused_nca_step(
                        y, model.w1, model.b1, model.w2, model.b2,
                        cond=cond_step, gamma=gamma, beta=beta,
                        seed=23, step=7 + step, fire_rate=FIRE_RATE,
                        clamp=clamp, fast_math=False)
            leaves = [x] + list(model.parameters())
            if film:
                leaves += [gamma, beta]
            loss = (y * torch.linspace(-0.2, 0.3, y.numel(), device=device)
                    .reshape_as(y)).sum()
            return y.detach(), torch.autograd.grad(loss, leaves)

        ref_y, ref_grads = run(False)
        out_y, out_grads = run(True)
        fwd_err = (out_y - ref_y).abs().max().item()
        grad_err = max(
            (got - ref).norm().item() / (ref.norm().item() + 1e-12)
            for got, ref in zip(out_grads, ref_grads)
        )
        passed = fwd_err < 1e-6 and grad_err < 2e-6
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<18} "
              f"maxdiff={fwd_err:.2e} max_grad_rel={grad_err:.2e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        sys.exit(2)
    # the eager reference must be true fp32: on Ampere+/Ada, cuDNN silently
    # runs convs in TF32 by default (~1e-3 error), which is eager's inaccuracy,
    # not the kernel's — the fused kernels always use input_precision="ieee"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}")
    torch.manual_seed(0)
    ok = gate1_forward(device)
    ok &= gate2_backward(device)
    ok &= gate3_rollout(device)
    print("ALL GATES PASS" if ok else "GATE FAILURES — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
