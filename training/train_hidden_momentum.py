"""Clockless cycle treatment with residual RGBA and distilled hidden velocity.

This is the follow-up to the failed all-channel NCA4 experiment.  A conditioned
NCA2 teacher advances beside every student pool slot.  The student is supervised
on authored RGBA, teacher hidden state, and the teacher's interval-averaged hidden
velocity while receiving only the behavior flag at runtime.
"""

import argparse
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from hidden_momentum_nca import (DEFAULT_DECAY, POSITION_CH, VISIBLE_CH,
                                 HiddenMomentumNCA, export_nca5, lift_state,
                                 transplant_residual)
from train_autonomous import (ACOND, AutoNCA, load_conditioned, teacher_states,
                              transplant)
from train_cyclic import (DAMAGE_N, FIRE_RATE, HIDDEN, OMEGA, POOL_SIZE,
                          CyclicNCA, _load_creature, cond_for, damage, target_at)


def alive_masked_mse(pred, target, alive):
    """Channel-mean MSE over teacher-alive cells, excluding empty background."""
    weighted = (pred - target).square() * alive
    return weighted.sum() / (alive.sum().clamp(min=1.0) * pred.shape[1])


def initial_pool(teacher, count, device):
    """Create synchronized student/teacher pool slots at known cycle phases."""
    position, one_step_velocity, theta, behavior = teacher_states(teacher, count, device)
    student = lift_state(position, one_step_velocity[:, VISIBLE_CH:])
    return student, position, theta, behavior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--init", default="../weights/shoggoth.nca")
    ap.add_argument("--out", default="../weights/shoggoth_auto_hidden_momentum.nca")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--momentum-decay", type=float, default=DEFAULT_DECAY)
    ap.add_argument("--hidden-weight", type=float, default=0.05)
    ap.add_argument("--velocity-weight", type=float, default=10.0)
    ap.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=None,
                    help="recompute student steps during backward (default: on for CUDA)")
    args = ap.parse_args()
    if not 1 <= args.batch_size <= POOL_SIZE:
        ap.error(f"--batch-size must be in [1, {POOL_SIZE}]")

    _load_creature("shoggoth")
    from train_cyclic import make_frames

    device = torch.device(args.device)
    use_checkpoint = args.checkpoint if args.checkpoint is not None else device.type == "cuda"
    torch.manual_seed(0)
    np.random.seed(0)
    frames_t = torch.from_numpy(make_frames()).permute(0, 1, 4, 2, 3).to(device)

    teacher = CyclicNCA().to(device)
    load_conditioned(args.init, teacher)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    residual_donor = AutoNCA().to(device)
    transplant(teacher, residual_donor)
    model = HiddenMomentumNCA(
        cond=ACOND,
        hidden=HIDDEN,
        fire_rate=FIRE_RATE,
        momentum_decay=args.momentum_decay,
    ).to(device)
    transplant_residual(residual_donor, model)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    pool, teacher_pool, pool_theta, pool_beh = initial_pool(teacher, POOL_SIZE, device)
    print(f"batch {args.batch_size}, activation checkpointing {use_checkpoint}", flush=True)
    t0 = time.time()

    for it in range(1, args.iters + 1):
        idx = torch.randint(0, POOL_SIZE, (args.batch_size,))
        x = pool[idx].clone()
        teacher_x = teacher_pool[idx].clone()
        theta = pool_theta[idx].clone()
        behavior = pool_beh[idx].clone()

        with torch.no_grad():
            fresh, fresh_teacher, fresh_theta, fresh_behavior = initial_pool(teacher, 1, device)
            x[0], teacher_x[0] = fresh[0], fresh_teacher[0]
            theta[0], behavior[0] = fresh_theta[0], fresh_behavior[0]
            if it > 500:
                x[-DAMAGE_N:] = damage(x[-DAMAGE_N:])

        tmax = min(96, 12 + it // 60)
        steps = np.random.randint(12, tmax + 1)
        check_every = max(6, steps // 4)
        behavior_cond = behavior.float()[:, None]
        teacher_anchor = teacher_x[:, VISIBLE_CH:].clone()
        anchor_steps = 0
        visible_loss = torch.zeros((), device=device)
        hidden_loss = torch.zeros((), device=device)
        velocity_loss = torch.zeros((), device=device)
        oscillator_loss = torch.zeros((), device=device)

        for step in range(steps):
            if use_checkpoint:
                # preserve_rng_state (the default) replays the same stochastic
                # fire mask during backward recomputation.
                x = checkpoint(model, x, behavior_cond, use_reentrant=False)
            else:
                x = model(x, behavior_cond)
            with torch.no_grad():
                teacher_x = teacher(teacher_x, cond_for(theta, behavior))
            theta = theta + OMEGA
            anchor_steps += 1

            if (step + 1) % check_every == 0 or step == steps - 1:
                alive = (teacher_x[:, 3:4] > 0.1).to(x.dtype)
                visible_loss = visible_loss + (
                    x[:, :VISIBLE_CH] - target_at(frames_t, behavior, theta)
                ).square().mean()
                hidden_loss = hidden_loss + alive_masked_mse(
                    x[:, VISIBLE_CH:POSITION_CH],
                    teacher_x[:, VISIBLE_CH:POSITION_CH],
                    alive,
                )
                target_velocity = (
                    teacher_x[:, VISIBLE_CH:POSITION_CH] - teacher_anchor
                ) / anchor_steps
                velocity_loss = velocity_loss + alive_masked_mse(
                    x[:, POSITION_CH:], target_velocity, alive
                )
                teacher_anchor = teacher_x[:, VISIBLE_CH:POSITION_CH].clone()
                anchor_steps = 0

                student_alive = (x[:, 3:4] > 0.1).to(x.dtype)
                denom = student_alive.sum(dim=(2, 3)).clamp(min=1.0)
                osc_cos = (x[:, 14:15] * student_alive).sum(dim=(2, 3)) / denom
                osc_sin = (x[:, 15:16] * student_alive).sum(dim=(2, 3)) / denom
                oscillator_loss = oscillator_loss + (
                    (osc_cos.squeeze(1) - torch.cos(theta)).square()
                    + (osc_sin.squeeze(1) - torch.sin(theta)).square()
                ).mean()

        loss = (visible_loss
                + args.hidden_weight * hidden_loss
                + args.velocity_weight * velocity_loss
                + 0.05 * oscillator_loss)
        if not torch.isfinite(loss):
            print(f"iter {it}: non-finite loss, discarded", flush=True)
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad /= parameter.grad.norm() + 1e-8
        opt.step()

        with torch.no_grad():
            finite = torch.isfinite(x).flatten(1).all(dim=1)
            valid_idx = idx[finite.cpu()]
            pool[valid_idx] = x.detach()[finite]
            teacher_pool[valid_idx] = teacher_x[finite]
            pool_theta[valid_idx] = (theta % (2 * np.pi))[finite]
            pool_beh[valid_idx] = behavior[finite]

        if it % 50 == 0:
            elapsed_rate = it / (time.time() - t0)
            print(
                f"iter {it:5d} loss {loss.item():.5f} "
                f"vis {visible_loss.item():.5f} hid {hidden_loss.item():.5f} "
                f"vel {velocity_loss.item():.6f} {elapsed_rate:.2f} it/s",
                flush=True,
            )
        if it % 250 == 0 or it == args.iters:
            export_nca5(model, args.out)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
