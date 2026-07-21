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

The default treatment uses NCA4 explicit momentum; pass ``--integrator residual``
to reproduce the original NCA2 baseline.

Usage: python3 train_autonomous.py [--integrator momentum] [--iters 8000]
"""

import argparse
import struct
import time

import numpy as np
import torch

from train_cyclic import (BATCH, CH, DAMAGE_N, FIRE_RATE, HIDDEN, OMEGA,
                          POOL_SIZE, CyclicNCA, _load_creature, cond_for,
                          damage, make_seed, target_at)
from momentum_nca import (DEFAULT_DECAY, MomentumNCA, export_nca4,
                          lift_state, transplant_residual)

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


def export_residual(model, path):
    with open(path, "wb") as f:
        f.write(b"NCA2")
        np.array([CH, HIDDEN, ACOND], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(HIDDEN, CH * 3 + ACOND).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, HIDDEN).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def teacher_states(teacher, n, device):
    """Grow donor states and estimate their current velocity by finite difference."""
    with torch.no_grad():
        x = make_seed(n, device)
        theta = torch.rand(n, device=device) * 2 * np.pi
        beh = torch.randint(0, 2, (n,), device=device)
        previous = x
        for _ in range(300 + np.random.randint(0, 240)):
            previous = x
            x = teacher(x, cond_for(theta, beh))
            theta = theta + OMEGA
        return x, x - previous, theta % (2 * np.pi), beh


def experiment_states(teacher, n, device, integrator):
    """Return phase-labelled states in the selected experiment's layout."""
    position, velocity, theta, beh = teacher_states(teacher, n, device)
    if integrator == "momentum":
        return lift_state(position, velocity), theta, beh
    return position, theta, beh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--init", default="../weights/shoggoth.nca")
    ap.add_argument("--out", default="../weights/shoggoth_auto.nca")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--integrator", choices=["momentum", "residual"], default="momentum",
                    help="NCA4 explicit phase-space state or the original NCA2 baseline")
    ap.add_argument("--momentum-decay", type=float, default=DEFAULT_DECAY,
                    help="velocity retained per step by the momentum treatment")
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

    if args.integrator == "momentum":
        # First perform the experiment's clock-removal transplant (cond 3 -> 1),
        # then lift that like-shaped residual rule into phase space.
        residual_donor = AutoNCA().to(device)
        transplant(teacher, residual_donor)
        model = MomentumNCA(cond=ACOND, hidden=HIDDEN, fire_rate=FIRE_RATE,
                            momentum_decay=args.momentum_decay).to(device)
        transplant_residual(residual_donor, model)
    else:
        model = AutoNCA().to(device)
        transplant(teacher, model)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # The momentum treatment lifts donor states with a one-step direction estimate,
    # so identical poses travelling in opposite directions begin distinguishable.
    pool, pool_theta, pool_beh = experiment_states(
        teacher, POOL_SIZE, device, args.integrator
    )
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (BATCH,))
        x = pool[idx].clone()
        theta = pool_theta[idx].clone()
        beh = pool_beh[idx].clone()

        with torch.no_grad():
            # Anchor phase truth: one slot per batch refreshed from the teacher
            fx, ft, fb = experiment_states(teacher, 1, device, args.integrator)
            x[0], theta[0], beh[0] = fx[0], ft[0], fb[0]
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        # Sync curriculum: short horizons first (staying in phase for 1/10 cycle is
        # learnable; free-running half a cycle from scratch collapses to the temporal
        # mean of the gait — observed as a fused "fog skirt" at iteration 3.1k).
        tmax = min(96, 12 + it // 60)
        T = np.random.randint(12, tmax + 1)
        loss = torch.zeros((), device=device)
        check_every = max(6, T // 4)
        bcond = beh.float()[:, None]
        for t in range(T):
            x = model(x, bcond)          # no phase input — only the behavior flag
            theta = theta + OMEGA        # the *schedule* still advances
            if (t + 1) % check_every == 0 or t == T - 1:
                loss = loss + ((x[:, :4] - target_at(frames_t, beh, theta)) ** 2).mean()
                # Oscillator distillation: alive-population mean of hidden channels
                # 14/15 should encode (cos, sin) of the schedule phase — a learned
                # wristwatch, fully internal at runtime.
                alive = (x[:, 3:4] > 0.1).float()
                denom = alive.sum(dim=(2, 3)).clamp(min=1.0)
                osc_c = (x[:, 14:15] * alive).sum(dim=(2, 3)) / denom
                osc_s = (x[:, 15:16] * alive).sum(dim=(2, 3)) / denom
                loss = loss + 0.05 * (
                    (osc_c.squeeze(1) - torch.cos(theta)) ** 2
                    + (osc_s.squeeze(1) - torch.sin(theta)) ** 2
                ).mean()

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
            if args.integrator == "momentum":
                export_nca4(model, args.out)
            else:
                export_residual(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
