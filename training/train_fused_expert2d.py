"""Train canonical pose attractors and directed transitions as one fused NCA."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from fused_expert_nca2d import (
    EDGE_COUNT, FusedExpertNCA2D, load_fused_checkpoint, save_fused_checkpoint,
)
from fused_state2d import canonical_key_state
from layered_transport_metrics2d import (
    alpha_dice, boundary_f1, rgb_gradient_error, sharpness_ratio,
    transition_shape_penalty,
)
from train_cyclic import FIRE_RATE, make_mature_state
from transport_targets2d import load_cycle_frames, visible_objective


ROOT = Path(__file__).resolve().parent.parent


def _key_state(frames, indices, interface):
    if interface == "canonical":
        return canonical_key_state(frames, indices)
    if interface == "alpha":
        return make_mature_state(frames[indices])
    raise ValueError(f"unknown state interface {interface}")


def _smooth_progress(value):
    return value * value * (3.0 - 2.0 * value)


def _flow_smoothness(flows):
    return (
        (flows[..., 1:, :] - flows[..., :-1, :]).square().mean()
        + (flows[..., 1:] - flows[..., :-1]).square().mean()
    )


def _slot_regularizers(assignments, alpha):
    support = (alpha > 0.05).to(assignments.dtype)
    entropy_map = -(assignments.clamp(min=1e-6).log() * assignments).sum(
        dim=1, keepdim=True
    )
    entropy = (entropy_map * support).sum() / support.sum().clamp(min=1)
    usage = (assignments * support).sum(dim=(2, 3)) / support.sum(
        dim=(2, 3)
    ).clamp(min=1)
    collapse = torch.relu(usage.max(dim=1).values - 0.85).square().mean()
    return entropy, collapse


def model_step(model, state, expert, progress, transition, use_checkpoint):
    if use_checkpoint and state.requires_grad:
        return checkpoint(
            model.step_with_aux, state, expert, progress, transition,
            use_reentrant=False, preserve_rng_state=True,
        )
    return model.step_with_aux(state, expert, progress, transition)


def perturb_anchor(state):
    """Apply noise, sparse dropout, and local erasure around a canonical key."""
    support = torch.nn.functional.max_pool2d(
        state[:, 3:4].clamp(0, 1), 5, stride=1, padding=2
    )
    visible_noise = torch.randn_like(state[:, :4]) * 0.075 * support
    hidden_noise = torch.randn_like(state[:, 4:]) * 0.15 * support
    damaged = torch.cat([
        state[:, :4] + visible_noise,
        state[:, 4:] + hidden_noise,
    ], dim=1)
    sparse_keep = (
        torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device) > 0.06
    ).to(state.dtype)
    damaged = damaged * (1 - support + support * sparse_keep)
    for sample in range(state.shape[0]):
        if torch.rand((), device=state.device) >= 0.5:
            continue
        coordinates = torch.nonzero(state[sample, 3] > 0.1)
        if not len(coordinates):
            continue
        selected = coordinates[
            torch.randint(len(coordinates), (), device=state.device)
        ]
        size = int(torch.randint(6, 13, (), device=state.device).item())
        center_y, center_x = (int(value.item()) for value in selected)
        y0, x0 = max(0, center_y - size // 2), max(0, center_x - size // 2)
        damaged[sample, :, y0:y0 + size, x0:x0 + size] = 0
    return damaged


def anchor_episode(model, frames, batch, steps, interface, hidden_weight,
                   use_checkpoint):
    expert = torch.randint(0, EDGE_COUNT, (batch,), device=frames.device)
    target_state = _key_state(frames, expert, interface)
    state = perturb_anchor(target_state).requires_grad_(True)
    initial_visible, _ = visible_objective(state[:, :4], target_state[:, :4])
    progress = torch.zeros(batch, device=frames.device, dtype=frames.dtype)
    transition = torch.zeros(batch, device=frames.device, dtype=torch.bool)
    total = state.new_zeros(())
    samples = 0
    diagnostic = {}
    for step in range(steps):
        state, _, _, _ = model_step(
            model, state, expert, progress, transition, use_checkpoint
        )
        if step == steps - 1 or (step + 1) % 4 == 0:
            visible, diagnostic = visible_objective(
                state[:, :4], target_state[:, :4]
            )
            hidden = (state[:, 4:] - target_state[:, 4:]).square().mean()
            detail = rgb_gradient_error(state[:, :4], target_state[:, :4])
            total = total + visible + hidden_weight * hidden + 0.75 * detail
            samples += 1
    auxiliary = {
        "anchor_initial": initial_visible.detach(),
        "anchor_gain": (initial_visible - visible.detach()),
        "anchor_hidden": (
            state[:, 4:] - target_state[:, 4:]
        ).square().mean().detach(),
        "anchor_detail": detail.detach(),
    }
    return total / max(1, samples), state, target_state[:, :4], diagnostic, auxiliary


def edge_episode(model, frames, batch, steps, handoff_steps, chain_edges,
                 interface, hidden_weight, use_checkpoint):
    expert = torch.randint(0, EDGE_COUNT, (batch,), device=frames.device)
    state = _key_state(frames, expert, interface).requires_grad_(True)
    endpoint_total = state.new_zeros(())
    shape_total = state.new_zeros(())
    flow_total = state.new_zeros(())
    entropy_total = state.new_zeros(())
    collapse_total = state.new_zeros(())
    range_total = state.new_zeros(())
    endpoint_gate_total = state.new_zeros(())
    endpoint_detail_total = state.new_zeros(())
    shape_samples = 0
    bridge_diagnostic = {}
    final_target = frames[expert]
    final_flows = None
    bridge_visible_value = state.new_zeros(())
    handoff_visible_value = state.new_zeros(())
    bridge_hidden_value = state.new_zeros(())
    handoff_hidden_value = state.new_zeros(())

    for chain_index in range(chain_edges):
        source_visible = frames[expert]
        destination = (expert + 1).remainder(EDGE_COUNT)
        destination_state = _key_state(frames, destination, interface)
        transition = torch.ones(batch, device=frames.device, dtype=torch.bool)
        for step in range(steps):
            scalar = _smooth_progress((step + 1) / steps)
            progress = torch.full(
                (batch,), scalar, device=frames.device, dtype=frames.dtype
            )
            state, flows, assignments, _ = model_step(
                model, state, expert, progress, transition, use_checkpoint
            )
            if step in {
                steps // 4 - 1, steps // 2 - 1,
                3 * steps // 4 - 1, steps - 1,
            }:
                shape, _ = transition_shape_penalty(
                    state[:, :4], source_visible, destination_state[:, :4]
                )
                entropy, collapse = _slot_regularizers(
                    assignments, state[:, 3:4].detach()
                )
                shape_total = shape_total + shape
                flow_total = flow_total + (
                    0.0005 * flows.square().mean()
                    + 0.002 * _flow_smoothness(flows)
                )
                entropy_total = entropy_total + entropy
                collapse_total = collapse_total + collapse
                range_total = range_total + torch.relu(
                    state.abs() - 2.0
                ).square().mean()
                shape_samples += 1
                final_flows = flows

        bridge_visible, bridge_diagnostic = visible_objective(
            state[:, :4], destination_state[:, :4]
        )
        bridge_hidden = (
            state[:, 4:] - destination_state[:, 4:]
        ).square().mean()
        bridge_gate, _ = transition_shape_penalty(
            state[:, :4], destination_state[:, :4], destination_state[:, :4]
        )
        bridge_detail = rgb_gradient_error(
            state[:, :4], destination_state[:, :4]
        )
        bridge_visible_value = bridge_visible.detach()
        bridge_hidden_value = bridge_hidden.detach()

        dwell_progress = torch.zeros(
            batch, device=frames.device, dtype=frames.dtype
        )
        dwell_mode = torch.zeros(batch, device=frames.device, dtype=torch.bool)
        for _ in range(handoff_steps):
            state, _, _, _ = model_step(
                model, state, destination, dwell_progress, dwell_mode,
                use_checkpoint,
            )
        handoff_visible, _ = visible_objective(
            state[:, :4], destination_state[:, :4]
        )
        handoff_hidden = (
            state[:, 4:] - destination_state[:, 4:]
        ).square().mean()
        handoff_gate, _ = transition_shape_penalty(
            state[:, :4], destination_state[:, :4], destination_state[:, :4]
        )
        handoff_detail = rgb_gradient_error(
            state[:, :4], destination_state[:, :4]
        )
        handoff_visible_value = handoff_visible.detach()
        handoff_hidden_value = handoff_hidden.detach()
        endpoint_total = endpoint_total + (
            bridge_visible + 0.5 * handoff_visible
            + hidden_weight * (bridge_hidden + 0.5 * handoff_hidden)
            + 0.5 * bridge_gate + 0.25 * handoff_gate
            + 0.75 * bridge_detail + 0.35 * handoff_detail
        )
        endpoint_gate_total = endpoint_gate_total + bridge_gate + handoff_gate
        endpoint_detail_total = endpoint_detail_total + bridge_detail + handoff_detail
        final_target = destination_state[:, :4]
        expert = destination
        if chain_index + 1 < chain_edges:
            state = state.detach().requires_grad_(True)

    denominator = max(1, shape_samples)
    regularizer = (
        0.20 * shape_total / denominator
        + flow_total / denominator
        + 0.001 * entropy_total / denominator
        + 0.01 * collapse_total / denominator
        + 0.01 * range_total / denominator
    )
    loss = endpoint_total / chain_edges + regularizer
    auxiliary = {
        "bridge": bridge_visible_value,
        "handoff": handoff_visible_value,
        "bridge_hidden": bridge_hidden_value,
        "handoff_hidden": handoff_hidden_value,
        "shape": (shape_total / denominator).detach(),
        "endpoint_gate": (endpoint_gate_total / (2 * chain_edges)).detach(),
        "endpoint_detail": (endpoint_detail_total / (2 * chain_edges)).detach(),
        "slot_entropy": (entropy_total / denominator).detach(),
        "slot_collapse": (collapse_total / denominator).detach(),
        "flow_rms": (
            final_flows.square().mean().sqrt().detach()
            if final_flows is not None else state.new_zeros(())
        ),
        "state_abs_max": state.abs().max().detach(),
    }
    return loss, state, final_target, bridge_diagnostic, auxiliary


def stage_at(iteration, anchor_iters, edge_iters):
    if iteration <= anchor_iters:
        return "anchor"
    if iteration <= edge_iters:
        return "edge"
    return "joint"


def make_optimizer(model, stage):
    model.set_stage(stage)
    learning_rate = {"anchor": 1e-3, "edge": 1.5e-3, "joint": 3e-4}[stage]
    return torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )


def _composite(frame, background=0.08):
    rgba = frame.detach().permute(1, 2, 0).cpu().numpy()
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    return np.clip(rgba[..., :3], 0, 1) + background * (1 - alpha)


def _preview_fire(state, generator):
    return (
        torch.rand(
            state.shape[0], 1, *state.shape[2:], device=state.device,
            generator=generator,
        ) <= FIRE_RATE
    ).to(state.dtype)


def save_preview(model, frames, path, steps, handoff_steps, interface):
    """Save one deployment-like chained cycle with bridge/handoff separation."""
    model.eval()
    device = frames.device
    generator = torch.Generator(device=device)
    generator.manual_seed(20260722)
    state = _key_state(
        frames, torch.zeros(1, device=device, dtype=torch.long), interface
    )
    midpoints, bridges, handoffs = [], [], []
    with torch.no_grad():
        for edge_index in range(EDGE_COUNT):
            expert = torch.tensor([edge_index], device=device)
            transition = torch.ones(1, device=device, dtype=torch.bool)
            for step in range(steps):
                progress = torch.tensor(
                    [_smooth_progress((step + 1) / steps)],
                    device=device, dtype=frames.dtype,
                )
                state = model(
                    state, expert, progress, transition,
                    _preview_fire(state, generator),
                )
                if step == steps // 2 - 1:
                    midpoints.append(state[0, :4].cpu())
            bridges.append(state[0, :4].cpu())
            destination = torch.tensor([(edge_index + 1) % EDGE_COUNT], device=device)
            progress = torch.zeros(1, device=device, dtype=frames.dtype)
            transition = torch.zeros(1, device=device, dtype=torch.bool)
            for _ in range(handoff_steps):
                state = model(
                    state, destination, progress, transition,
                    _preview_fire(state, generator),
                )
            handoffs.append(state[0, :4].cpu())
    targets = [frames[(index + 1) % EDGE_COUNT].cpu() for index in range(EDGE_COUNT)]
    rows = [
        np.concatenate([_composite(frame) for frame in row], axis=1)
        for row in (targets, midpoints, bridges, handoffs)
    ]
    sheet = np.concatenate(rows, axis=0)
    image = Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8), "RGB")
    image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(path)
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", default=str(ROOT / "weights" / "fused_expert2d.pt"))
    parser.add_argument("--preview", default=str(
        ROOT / "experiments" / "fused_expert2d_preview.png"))
    parser.add_argument("--resume")
    parser.add_argument("--iters", type=int, default=3200)
    parser.add_argument("--anchor-iters", type=int, default=400)
    parser.add_argument("--edge-iters", type=int, default=1400)
    parser.add_argument("--chain-after", type=int, default=1100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--transition-steps", type=int, default=24)
    parser.add_argument("--anchor-steps", type=int, default=12)
    parser.add_argument("--handoff-steps", type=int, default=8)
    parser.add_argument("--hidden-weight", type=float, default=0.20)
    parser.add_argument("--state-interface", choices=("canonical", "alpha"),
                        default="canonical")
    parser.add_argument("--soft-slots", action="store_true")
    parser.add_argument("--expert-hidden", type=int, default=384)
    parser.add_argument("--flow-hidden", type=int, default=128)
    parser.add_argument("--position-frequencies", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--no-activation-checkpoint", action="store_true")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = parser.parse_args()
    if not (0 <= args.anchor_iters < args.edge_iters < args.iters):
        parser.error("require 0 <= anchor-iters < edge-iters < iters")
    if not (args.anchor_iters <= args.chain_after <= args.iters):
        parser.error("chain-after must be between anchor-iters and iters")
    if min(args.batch, args.anchor_steps, args.handoff_steps) <= 0:
        parser.error("batch, anchor-steps, and handoff-steps must be positive")
    if args.transition_steps < 4 or args.hidden_weight < 0:
        parser.error("transition-steps must be >= 4 and hidden-weight nonnegative")
    if min(args.expert_hidden, args.flow_hidden) <= 0 or args.position_frequencies < 0:
        parser.error("hidden widths must be positive and position frequencies nonnegative")

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    frames = load_cycle_frames(args.target, device)
    grid = frames.shape[-1]
    if args.resume:
        model, start_iteration, _ = load_fused_checkpoint(args.resume, device)
        if model.grid != grid:
            parser.error("resume grid differs from target")
    else:
        model = FusedExpertNCA2D(
            grid=grid, hard_slots=not args.soft_slots,
            expert_hidden=args.expert_hidden, flow_hidden=args.flow_hidden,
            position_frequencies=args.position_frequencies,
        ).to(device)
        start_iteration = 0

    current_stage = stage_at(start_iteration + 1, args.anchor_iters, args.edge_iters)
    optimizer = make_optimizer(model, current_stage)
    print(
        f"grid {grid} batch {args.batch} interface {args.state_interface} "
        f"slots {'hard' if model.hard_slots else 'soft'} "
        f"width {model.expert_hidden}/{model.flow_hidden} "
        f"fourier {model.position_frequencies} "
        f"transition {args.transition_steps} handoff {args.handoff_steps} "
        f"stages anchor<={args.anchor_iters} edge<={args.edge_iters} "
        f"chain-after {args.chain_after}", flush=True,
    )
    started = time.time()
    for iteration in range(start_iteration + 1, args.iters + 1):
        stage = stage_at(iteration, args.anchor_iters, args.edge_iters)
        if stage != current_stage:
            current_stage = stage
            optimizer = make_optimizer(model, stage)
            print(f"stage -> {stage}", flush=True)

        anchor_replay = stage == "joint" and iteration % 4 == 0
        if stage == "anchor" or anchor_replay:
            result = anchor_episode(
                model, frames, args.batch, args.anchor_steps,
                args.state_interface, args.hidden_weight,
                not args.no_activation_checkpoint,
            )
            episode_name = "anchor" if stage == "anchor" else "replay"
        else:
            chain_edges = 2 if iteration >= args.chain_after else 1
            result = edge_episode(
                model, frames, args.batch, args.transition_steps,
                args.handoff_steps, chain_edges, args.state_interface,
                args.hidden_weight, not args.no_activation_checkpoint,
            )
            episode_name = f"edge{chain_edges}"
        loss, state, target, diagnostic, auxiliary = result
        if not torch.isfinite(loss):
            print(f"iter {iteration}: non-finite loss; discarded", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()

        if iteration % 25 == 0:
            with torch.no_grad():
                prediction = state[:, :4]
                gate = {
                    "dice": alpha_dice(prediction, target).item(),
                    "boundary": boundary_f1(prediction, target).item(),
                    "sharp": sharpness_ratio(prediction, target).item(),
                }
            rate = (iteration - start_iteration) / (time.time() - started)
            detail = " ".join(
                f"{name} {value.item():.4f}" for name, value in auxiliary.items()
            )
            print(
                f"iter {iteration:5d} {stage:6s}/{episode_name:6s} "
                f"loss {loss.item():.5f} dice {gate['dice']:.3f} "
                f"edge {gate['boundary']:.3f} sharp {gate['sharp']:.3f} "
                f"{detail} {rate:.2f} it/s", flush=True,
            )

        if iteration % args.checkpoint_every == 0 or iteration == args.iters:
            path = Path(args.out)
            save_fused_checkpoint(model, path, iteration, current_stage)
            numbered = path.with_name(f"{path.stem}_it{iteration}{path.suffix}")
            save_fused_checkpoint(model, numbered, iteration, current_stage)
            preview = Path(args.preview)
            preview.parent.mkdir(parents=True, exist_ok=True)
            save_preview(
                model, frames, preview, args.transition_steps,
                args.handoff_steps, args.state_interface,
            )
            preview.with_suffix(".json").write_text(json.dumps({
                "iteration": iteration,
                "stage": current_stage,
                "target": str(Path(args.target).resolve()),
                "transition_steps": args.transition_steps,
                "handoff_steps": args.handoff_steps,
                "state_interface": args.state_interface,
                "hidden_weight": args.hidden_weight,
                "hard_slots": model.hard_slots,
                "expert_hidden": model.expert_hidden,
                "flow_hidden": model.flow_hidden,
                "position_frequencies": model.position_frequencies,
                "batch": args.batch,
            }, indent=2) + "\n")
            print(f"  checkpoint -> {path}; preview -> {preview}", flush=True)

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak allocated {peak_gib:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
