"""Train the bounded global-controller transport experiment in 2D."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train_cyclic import (OMEGA, CyclicNCA, cond_for, damage, load_nca2)
from transport_nca2d import (TransportNCA2D, load_transport_checkpoint,
                             make_global, save_transport_checkpoint,
                             transplant_nca2)
from transport_targets2d import load_cycle_frames, target_at_global, visible_objective


ROOT = Path(__file__).resolve().parent.parent
CH = 16


def make_seed(count, grid, device):
    state = torch.zeros(count, CH, grid, grid, device=device)
    state[:, 3:, grid // 2, grid // 2] = 1.0
    return state


def build_teacher_bank(donor, count, grid, device, grow_steps=360):
    """Grow mature donor states paired with their final internal ring state."""
    phase = torch.rand(count, device=device) * 2 * np.pi
    behavior = torch.zeros(count, dtype=torch.long, device=device)
    state = make_seed(count, grid, device)
    donor.eval()
    with torch.no_grad():
        for _ in range(grow_steps):
            state = donor(state, cond_for(phase, behavior))
            phase = phase + OMEGA
    return state, make_global(torch.remainder(phase, 2 * np.pi))


def _flow_smoothness(flow):
    return (
        (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
        + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
    )


def make_optimizer(model, stage):
    model.set_stage(stage)
    learning_rate = 1e-3 if stage == "flow" else 4e-4
    return torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--init", required=True, help="NCA2 donor")
    parser.add_argument("--out", default=str(ROOT / "weights" / "transport2d.pt"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--iters", type=int, default=2400)
    parser.add_argument("--flow-iters", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pool", type=int, default=128)
    parser.add_argument("--teacher-bank", type=int, default=48)
    parser.add_argument("--teacher-grow", type=int, default=360)
    parser.add_argument("--refresh-fraction", type=float, default=0.25)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument("--segment", type=int, default=8)
    parser.add_argument("--damage-after", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = parser.parse_args()
    if args.rollout <= 0 or args.segment <= 0 or args.rollout % args.segment:
        parser.error("--rollout must be a positive multiple of --segment")
    if args.batch <= 0 or args.pool < args.batch or args.teacher_bank < args.batch:
        parser.error("pool and teacher bank must each be at least batch size")
    if not 0 < args.refresh_fraction <= 1:
        parser.error("--refresh-fraction must be in (0, 1]")
    if args.checkpoint_every <= 0:
        parser.error("--checkpoint-every must be positive")

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    frames = load_cycle_frames(args.target, device)
    grid = frames.shape[-1]
    print(f"four-anchor corpus {tuple(frames.shape)} <- {args.target}", flush=True)

    donor = CyclicNCA().to(device)
    load_nca2(donor, args.init)
    for parameter in donor.parameters():
        parameter.requires_grad_(False)

    if args.resume:
        model, start_iteration = load_transport_checkpoint(args.resume, device)
        if model.grid != grid:
            raise ValueError("resume grid does not match target corpus")
        print(f"resumed iteration {start_iteration} <- {args.resume}", flush=True)
    else:
        model = TransportNCA2D(grid=grid).to(device)
        transplant_nca2(donor, model)
        start_iteration = 0
    if start_iteration >= args.iters:
        parser.error("resume iteration must be less than --iters")
    stage = "flow" if start_iteration < args.flow_iters else "joint"
    optimizer = make_optimizer(model, stage)

    print(f"building {args.teacher_bank}-state mature teacher bank...", flush=True)
    teacher_state, teacher_global = build_teacher_bank(
        donor, args.teacher_bank, grid, device, args.teacher_grow
    )
    pool_choice = torch.randint(0, args.teacher_bank, (args.pool,), device=device)
    pool_state = teacher_state[pool_choice].clone()
    pool_global = teacher_global[pool_choice].clone()
    refresh_count = max(1, int(round(args.batch * args.refresh_fraction)))
    print(
        f"grid {grid} batch {args.batch} pool {args.pool} refresh "
        f"{refresh_count}/{args.batch} rollout {args.rollout} flow-only through "
        f"{args.flow_iters}, damage-after {args.damage_after or 'off'}",
        flush=True,
    )

    started = time.time()
    for iteration in range(start_iteration + 1, args.iters + 1):
        if iteration == args.flow_iters + 1 and stage == "flow":
            stage = "joint"
            optimizer = make_optimizer(model, stage)
            print("stage -> joint flow + repair", flush=True)

        indices = torch.randint(0, args.pool, (args.batch,), device=device)
        state = pool_state[indices].clone()
        global_state = pool_global[indices].clone()
        with torch.no_grad():
            bank_indices = torch.randint(
                0, args.teacher_bank, (refresh_count,), device=device
            )
            state[:refresh_count] = teacher_state[bank_indices]
            global_state[:refresh_count] = teacher_global[bank_indices]
            if args.damage_after > 0 and iteration >= args.damage_after:
                state[-1:] = damage(state[-1:])

        state.requires_grad_(True)
        total_visible = state.new_zeros(())
        flow_penalty = state.new_zeros(())
        flow_magnitude = state.new_zeros(())
        diagnostic = {}
        segment_count = args.rollout // args.segment
        weight_sum = 0.0
        for segment_index in range(segment_count):
            state, global_state, flow, _ = model.step_with_aux(state, global_state)
            if args.segment > 1:
                state, global_state = model.rollout(
                    state, global_state, args.segment - 1
                )
            target = target_at_global(frames, global_state)
            visible, diagnostic = visible_objective(state[:, :4], target)
            weight = 2.0 if segment_index == segment_count - 1 else 1.0
            total_visible = total_visible + weight * visible
            weight_sum += weight
            flow_penalty = flow_penalty + 0.001 * flow.square().mean()
            flow_penalty = flow_penalty + 0.008 * _flow_smoothness(flow)
            flow_magnitude = flow_magnitude + flow.square().mean().sqrt().detach()
        total_visible = total_visible / weight_sum
        flow_penalty = flow_penalty / segment_count
        loss = total_visible + flow_penalty

        if not torch.isfinite(loss):
            print(f"iter {iteration}: non-finite loss; batch discarded", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

        with torch.no_grad():
            finite = torch.isfinite(state).flatten(1).all(dim=1)
            pool_state[indices[finite]] = state.detach()[finite]
            pool_global[indices[finite]] = global_state.detach()[finite]

        if iteration % 25 == 0:
            rate = (iteration - start_iteration) / (time.time() - started)
            diagnostic_text = " ".join(
                f"{name} {value.item():.5f}" for name, value in diagnostic.items()
            )
            print(
                f"iter {iteration:5d} {stage:5s} loss {loss.item():.5f} "
                f"vis {total_visible.item():.5f} flow "
                f"{(flow_magnitude / segment_count).item():.4f} "
                f"{diagnostic_text} {rate:.2f} it/s",
                flush=True,
            )
        if iteration % args.checkpoint_every == 0 or iteration == args.iters:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_transport_checkpoint(model, path, iteration)
            numbered = path.with_name(f"{path.stem}_it{iteration}{path.suffix}")
            save_transport_checkpoint(model, numbered, iteration)
            print(f"  checkpoint -> {path} ({numbered.name})", flush=True)


if __name__ == "__main__":
    main()
