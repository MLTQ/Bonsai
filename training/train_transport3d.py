"""Train the bounded 32^3 global-controller transport experiment."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from train_cyclic3d import (OMEGA, CyclicNCA3D, damage, load_nc3c,
                            make_seed)
from transport_nca3d import (TransportNCA3D, load_transport_checkpoint, make_global,
                             save_transport_checkpoint, transplant_nc3c)
from transport_targets3d import (DENSE_FRAMES, load_dense_walk_frames,
                                 target_at_global, visible_objective)


ROOT = Path(__file__).resolve().parent.parent


def build_teacher_bank(donor, count, device, grow_steps=300):
    """Grow mature walking donor states paired with their internal ring phase."""
    phase = torch.rand(count, device=device) * 2 * np.pi
    behavior = torch.ones(count, dtype=torch.long, device=device)
    state = make_seed(count, device)
    donor.eval()
    donor.use_checkpoint = False
    with torch.no_grad():
        state = donor.rollout(state, phase, behavior, grow_steps)
    final_phase = torch.remainder(phase + grow_steps * OMEGA, 2 * np.pi)
    return state, make_global(final_phase)


def _flow_smoothness(flow):
    return (
        (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
        + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
        + (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    )


def make_optimizer(model, stage):
    model.set_stage(stage)
    lr = 1e-3 if stage == "flow" else 5e-4
    return torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--flow-iters", type=int, default=400)
    ap.add_argument("--resume", default=None,
                    help="resume model/iteration from a TN3D1 checkpoint")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--teacher-bank", type=int, default=32)
    ap.add_argument("--refresh-fraction", type=float, default=0.25)
    ap.add_argument("--rollout", type=int, default=24)
    ap.add_argument("--segment", type=int, default=8)
    ap.add_argument("--damage-after", type=int, default=0,
                    help="first damage iteration; 0 disables damage for coherence A/B")
    ap.add_argument("--init", default=str(ROOT / "weights" / "shoggoth3d.nca"))
    ap.add_argument("--out", default=str(ROOT / "weights" / "shoggoth3d_transport.pt"))
    ap.add_argument("--cache", default=str(
        ROOT / "training" / "corpus_shoggoth3d_walk_dense48_32.npz"))
    ap.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = ap.parse_args()
    if args.rollout <= 0 or args.segment <= 0 or args.rollout % args.segment:
        ap.error("--rollout must be a positive multiple of --segment")
    if not 0 < args.refresh_fraction <= 1:
        ap.error("--refresh-fraction must be in (0, 1]")
    if args.pool < args.batch or args.teacher_bank < args.batch:
        ap.error("pool and teacher bank must each be at least batch size")

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    frames_np = load_dense_walk_frames(args.cache, DENSE_FRAMES)
    frames = torch.from_numpy(frames_np).permute(0, 4, 1, 2, 3).float().to(device)
    print(f"dense corpus {tuple(frames.shape)} <- {args.cache}", flush=True)

    donor = CyclicNCA3D().to(device)
    load_nc3c(donor, args.init)
    for parameter in donor.parameters():
        parameter.requires_grad_(False)

    if args.resume:
        model, start_iteration = load_transport_checkpoint(args.resume, device)
        if model.grid != frames.shape[-1]:
            raise ValueError("resume grid does not match dense corpus")
        print(f"resumed iteration {start_iteration} <- {args.resume}", flush=True)
    else:
        model = TransportNCA3D(grid=frames.shape[-1]).to(device)
        transplant_nc3c(donor, model)
        start_iteration = 0
    if start_iteration >= args.iters:
        ap.error("resume iteration must be less than --iters")
    stage = "flow" if start_iteration < args.flow_iters else "joint"
    optimizer = make_optimizer(model, stage)

    print(f"building {args.teacher_bank}-state mature teacher bank...", flush=True)
    teacher_state, teacher_global = build_teacher_bank(
        donor, args.teacher_bank, device
    )
    pool_choice = torch.randint(0, args.teacher_bank, (args.pool,), device=device)
    pool_state = teacher_state[pool_choice].clone()
    pool_global = teacher_global[pool_choice].clone()
    refresh_count = max(1, int(round(args.batch * args.refresh_fraction)))
    print(
        f"grid {frames.shape[-1]}^3 batch {args.batch} pool {args.pool} "
        f"refresh {refresh_count}/{args.batch} rollout {args.rollout} "
        f"flow-only through {args.flow_iters}, damage-after {args.damage_after or 'off'}",
        flush=True,
    )

    start = time.time()
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
        total_loss = state.new_zeros(())
        visible_sum = state.new_zeros(())
        flow_penalty = state.new_zeros(())
        flow_magnitude = state.new_zeros(())
        diagnostic = {}
        segments = args.rollout // args.segment
        for segment_index in range(segments):
            state, global_state, flow, _ = model.step_with_aux(state, global_state)
            if args.segment > 1:
                state, global_state = model.rollout(
                    state, global_state, args.segment - 1
                )
            target = target_at_global(frames, global_state)
            visible, diagnostic = visible_objective(state[:, :4], target)
            weight = 2.0 if segment_index == segments - 1 else 1.0
            total_loss = total_loss + weight * visible
            visible_sum = visible_sum + weight * visible.detach()
            flow_penalty = flow_penalty + 0.002 * flow.square().mean()
            flow_penalty = flow_penalty + 0.010 * _flow_smoothness(flow)
            flow_magnitude = flow_magnitude + flow.square().mean().sqrt().detach()
        total_loss = total_loss / (segments + 1)
        flow_penalty = flow_penalty / segments
        loss = total_loss + flow_penalty

        if not torch.isfinite(loss):
            print(f"iter {iteration}: non-finite loss; batch discarded", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()

        with torch.no_grad():
            finite = torch.isfinite(state).flatten(1).all(dim=1)
            pool_state[indices[finite]] = state.detach()[finite]
            pool_global[indices[finite]] = global_state.detach()[finite]

        if iteration % 25 == 0:
            rate = (iteration - start_iteration) / (time.time() - start)
            diag_text = " ".join(
                f"{name} {value.item():.5f}" for name, value in diagnostic.items()
            )
            print(
                f"iter {iteration:5d} {stage:5s} loss {loss.item():.5f} "
                f"vis {(visible_sum / (segments + 1)).item():.5f} "
                f"flow {(flow_magnitude / segments).item():.4f} "
                f"{diag_text} {rate:.2f} it/s",
                flush=True,
            )
        if iteration % 250 == 0 or iteration == args.iters:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_transport_checkpoint(model, args.out, iteration)
            print(f"  checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
