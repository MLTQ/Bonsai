"""The internal-clock experiment: can the automaton keep its own beat?

Takes the trained phase-conditioned shoggoth (Mk. I) and REMOVES the sin/cos
phase inputs (keeping only the behavior flag), then fine-tunes so the cycle
continues at the trained tempo with no external clock. The current body pose is
the only phase information available — the hidden channels must become a
distributed oscillator that advances it.

Method: pool slots are initialized by rolling the ORIGINAL conditioned model to
diverse phases (state carries pose ~= phase); fine-tuning rolls the clockless
model T steps and penalizes deviation from the target at (slot theta + T*OMEGA).
Matching the schedule enforces tempo; fresh conditioned states are re-injected
each iteration to anchor phase truth.

Usage: python3 train_autonomous.py [--iters 8000] [--out ../weights/shoggoth_auto.nca]
"""

import argparse
import struct
import time

import numpy as np
import torch

from train_cyclic import (BATCH, CH, DAMAGE_N, FIRE_RATE, HIDDEN, OMEGA,
                          POOL_SIZE, CyclicNCA, _load_creature, cond_for,
                          damage, make_seed, target_at)

ACOND = 1  # behavior flag only — the clock is gone


class AutoNCA(CyclicNCA):
    """CyclicNCA with COND=1 (behavior only). Reuses forward via cond width."""

    def __init__(self):
        # Parent builds w1 for COND=3; rebuild for ACOND after init.
        super().__init__()
        import torch.nn as nn
        self.w1 = nn.Conv2d(CH * 3 + ACOND, HIDDEN, 1)


def load_conditioned(path, model):
    """Load NCA2 cond=3 weights into a CyclicNCA (the teacher/donor)."""
    with open(path, "rb") as f:
        assert f.read(4) == b"NCA2"
        ch, hid, cond = struct.unpack("<3i", f.read(12))
        assert (ch, hid, cond) == (CH, HIDDEN, 3)
        f.read(4)
        def arr(n):
            return torch.from_numpy(np.frombuffer(f.read(n * 4), dtype="<f4").copy())
        with torch.no_grad():
            model.w1.weight.copy_(arr(HIDDEN * (CH * 3 + 3)).view(HIDDEN, CH * 3 + 3, 1, 1))
            model.w1.bias.copy_(arr(HIDDEN))
            model.w2.weight.copy_(arr(CH * HIDDEN).view(CH, HIDDEN, 1, 1))
            model.w2.bias.copy_(arr(CH))


def transplant(donor, auto):
    """Copy weights, dropping the sin/cos input columns (48, 49); keep behavior (50)."""
    with torch.no_grad():
        auto.w1.weight[:, : CH * 3].copy_(donor.w1.weight[:, : CH * 3])
        auto.w1.weight[:, CH * 3].copy_(donor.w1.weight[:, CH * 3 + 2])  # behavior column
        auto.w1.bias.copy_(donor.w1.bias)
        auto.w2.weight.copy_(donor.w2.weight)
        auto.w2.bias.copy_(donor.w2.bias)


def export(model, path):
    with open(path, "wb") as f:
        f.write(b"NCA2")
        np.array([CH, HIDDEN, ACOND], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3 + ACOND).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def teacher_states(teacher, n, device):
    """Grow n organisms with the conditioned teacher to random phases/behaviors."""
    with torch.no_grad():
        x = make_seed(n, device)
        theta = torch.rand(n, device=device) * 2 * np.pi
        beh = torch.randint(0, 2, (n,), device=device)
        for _ in range(300 + np.random.randint(0, 240)):
            x = teacher(x, cond_for(theta, beh))
            theta = theta + OMEGA
        return x, theta % (2 * np.pi), beh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--init", default="../weights/shoggoth.nca")
    ap.add_argument("--out", default="../weights/shoggoth_auto.nca")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    _load_creature("shoggoth")
    from train_cyclic import make_frames  # populated by _load_creature

    device = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    frames_t = torch.from_numpy(make_frames()).permute(0, 1, 4, 2, 3).to(device)

    teacher = CyclicNCA().to(device)
    load_conditioned(args.init, teacher)
    teacher.eval()

    model = AutoNCA().to(device)
    transplant(teacher, model)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # Pool seeded entirely from teacher states with known phases
    pool, pool_theta, pool_beh = teacher_states(teacher, POOL_SIZE, device)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        beh = pool_beh[idx].clone()

        with torch.no_grad():
            # Anchor phase truth: one slot per batch refreshed from the teacher
            fx, ft, fb = teacher_states(teacher, 1, device)
            x[0], theta[0], beh[0] = fx[0], ft[0], fb[0]
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        T = np.random.randint(48, 97)
        loss = torch.zeros((), device=device)
        checkpoints = {T - 24: 0.5, T - 12: 0.75, T - 1: 1.0}
        bcond = beh.float()[:, None]
        for t in range(T):
            x = model(x, bcond)          # no phase input — only the behavior flag
            theta = theta + OMEGA        # the *schedule* still advances
            if t in checkpoints:
                loss = loss + checkpoints[t] * ((x[:, :4] - target_at(frames_t, beh, theta)) ** 2).mean()

        if not torch.isfinite(loss):
            print(f"iter {it}: non-finite loss, discarded", flush=True)
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= p.grad.norm() + 1e-8
        opt.step()

        with torch.no_grad():
            finite = torch.isfinite(x).flatten(1).all(dim=1)
            pool[idx[finite.cpu()]] = x.detach()[finite]
            pool_theta[idx[finite.cpu()]] = (theta % (2 * np.pi))[finite]
            pool_beh[idx[finite.cpu()]] = beh[finite]

        if it % 50 == 0:
            print(f"iter {it:5d}  loss {loss.item():.5f}  {it/(time.time()-t0):.2f} it/s", flush=True)
        if it % 250 == 0 or it == args.iters:
            export(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
