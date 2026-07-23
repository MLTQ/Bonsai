"""Train hard-edge multi-flow transport from reviewed mature 2D anchors."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from layered_transport_metrics2d import (
    alpha_dice, boundary_f1, sharpness_ratio, transition_shape_penalty,
)
from layered_transport_nca2d import (
    EDGE_COUNT, LayeredTransportNCA2D, load_layered_checkpoint,
    save_layered_checkpoint,
)
from train_cyclic import make_mature_state
from transport_targets2d import load_cycle_frames, visible_objective


ROOT = Path(__file__).resolve().parent.parent


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


def _smooth_progress(value):
    return value * value * (3.0 - 2.0 * value)


def make_optimizer(model, stage):
    model.set_stage(stage)
    learning_rate = {
        "stabilize": 1e-3,
        "flow": 1.5e-3,
        "joint": 4e-4,
    }[stage]
    return torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )


def stage_at(iteration, stabilize_iters, flow_iters):
    if iteration <= stabilize_iters:
        return "stabilize"
    if iteration <= flow_iters:
        return "flow"
    return "joint"


def model_step(model, state, edge, progress, transition, use_checkpoint):
    if use_checkpoint and state.requires_grad:
        return checkpoint(
            model.step_with_aux, state, edge, progress, transition,
            use_reentrant=False, preserve_rng_state=True,
        )
    return model.step_with_aux(state, edge, progress, transition)


def stabilize_episode(model, frames, batch, steps, use_checkpoint):
    indices = torch.randint(0, EDGE_COUNT, (batch,), device=frames.device)
    target = frames[indices]
    state = make_mature_state(target)
    live = state[:, 3:4].clamp(0, 1)
    state = state + torch.randn_like(state) * 0.025 * live
    state.requires_grad_(True)
    progress = torch.zeros(batch, device=frames.device)
    transition = torch.zeros(batch, device=frames.device, dtype=torch.bool)
    loss = state.new_zeros(())
    samples = 0
    diagnostic = {}
    for step in range(steps):
        state, _, _, _ = model_step(
            model, state, indices, progress, transition, use_checkpoint
        )
        if step == steps - 1 or (step + 1) % 4 == 0:
            visible, diagnostic = visible_objective(state[:, :4], target)
            loss = loss + visible
            samples += 1
    return loss / max(1, samples), state, target, diagnostic, {}


def transition_episode(
        model, frames, batch, steps, chain_edges, use_checkpoint):
    edge = torch.randint(0, EDGE_COUNT, (batch,), device=frames.device)
    state = make_mature_state(frames[edge]).requires_grad_(True)
    total_endpoint = state.new_zeros(())
    total_shape = state.new_zeros(())
    total_flow = state.new_zeros(())
    total_entropy = state.new_zeros(())
    total_collapse = state.new_zeros(())
    shape_samples = 0
    endpoint_diagnostic = {}
    last_target = frames[edge]
    last_flows = None

    for chain_index in range(chain_edges):
        source = frames[edge]
        next_edge = (edge + 1).remainder(EDGE_COUNT)
        destination = frames[next_edge]
        for step in range(steps):
            scalar = _smooth_progress((step + 1) / steps)
            progress = torch.full(
                (batch,), scalar, device=frames.device, dtype=frames.dtype
            )
            transition = torch.ones(
                batch, device=frames.device, dtype=torch.bool
            )
            state, flows, assignments, _ = model_step(
                model, state, edge, progress, transition, use_checkpoint
            )
            if step in {steps // 4 - 1, steps // 2 - 1,
                        3 * steps // 4 - 1, steps - 1}:
                shape, _ = transition_shape_penalty(
                    state[:, :4], source, destination
                )
                entropy, collapse = _slot_regularizers(
                    assignments, state[:, 3:4].detach()
                )
                total_shape = total_shape + shape
                total_flow = total_flow + (
                    0.0005 * flows.square().mean()
                    + 0.002 * _flow_smoothness(flows)
                )
                total_entropy = total_entropy + entropy
                total_collapse = total_collapse + collapse
                shape_samples += 1
                last_flows = flows

        endpoint, endpoint_diagnostic = visible_objective(
            state[:, :4], destination
        )
        hidden_target = destination[:, 3:4].expand(-1, 12, -1, -1)
        hidden = (state[:, 4:] - hidden_target).square().mean()
        total_endpoint = total_endpoint + endpoint + 0.01 * hidden
        last_target = destination
        edge = next_edge
        if chain_index + 1 < chain_edges:
            state = state.detach().requires_grad_(True)

    denominator = max(1, shape_samples)
    regularizer = (
        0.15 * total_shape / denominator
        + total_flow / denominator
        + 0.001 * total_entropy / denominator
        + 0.01 * total_collapse / denominator
    )
    loss = total_endpoint / chain_edges + regularizer
    auxiliary = {
        "endpoint": (total_endpoint / chain_edges).detach(),
        "shape": (total_shape / denominator).detach(),
        "slot_entropy": (total_entropy / denominator).detach(),
        "slot_collapse": (total_collapse / denominator).detach(),
        "flow_rms": (
            last_flows.square().mean().sqrt().detach()
            if last_flows is not None else state.new_zeros(())
        ),
    }
    return loss, state, last_target, endpoint_diagnostic, auxiliary


def _composite(frame, background=0.08):
    rgba = frame.detach().permute(1, 2, 0).cpu().numpy()
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    return np.clip(rgba[..., :3], 0, 1) + background * (1 - alpha)


def save_preview(model, frames, path, transition_steps):
    model.eval()
    predictions = []
    state = make_mature_state(frames[0:1])
    fire = torch.ones(1, 1, frames.shape[-2], frames.shape[-1], device=frames.device)
    with torch.no_grad():
        for edge_index in range(EDGE_COUNT):
            edge = torch.tensor([edge_index], device=frames.device)
            transition = torch.ones(1, device=frames.device, dtype=torch.bool)
            for step in range(transition_steps):
                scalar = _smooth_progress((step + 1) / transition_steps)
                progress = torch.tensor(
                    [scalar], device=frames.device, dtype=frames.dtype
                )
                state = model(
                    state, edge, progress, transition, fire_mask=fire
                )
            predictions.append(state[0, :4].cpu())
    targets = [frames[(index + 1) % EDGE_COUNT].cpu() for index in range(EDGE_COUNT)]
    rows = [
        np.concatenate([_composite(frame) for frame in row], axis=1)
        for row in (targets, predictions)
    ]
    sheet = np.concatenate(rows, axis=0)
    image = Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8), "RGB")
    image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(path)
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", default=str(ROOT / "weights" / "layered_transport2d.pt"))
    parser.add_argument("--preview", default=str(
        ROOT / "experiments" / "layered_transport2d_preview.png"))
    parser.add_argument("--resume")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--stabilize-iters", type=int, default=300)
    parser.add_argument("--flow-iters", type=int, default=1200)
    parser.add_argument("--chain-after", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--transition-steps", type=int, default=24)
    parser.add_argument("--stabilize-steps", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--no-activation-checkpoint", action="store_true")
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")))
    args = parser.parse_args()
    if not (0 <= args.stabilize_iters < args.flow_iters < args.iters):
        parser.error("require 0 <= stabilize-iters < flow-iters < iters")
    if args.batch <= 0 or args.transition_steps < 4 or args.stabilize_steps < 1:
        parser.error("batch/steps must be positive and transition-steps >= 4")
    if args.chain_after < args.flow_iters:
        parser.error("chain-after must be at least flow-iters")

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
        model, start_iteration, _ = load_layered_checkpoint(args.resume, device)
        if model.grid != grid:
            parser.error("resume grid differs from target")
    else:
        model = LayeredTransportNCA2D(grid=grid).to(device)
        start_iteration = 0
    current_stage = stage_at(
        start_iteration + 1, args.stabilize_iters, args.flow_iters
    )
    optimizer = make_optimizer(model, current_stage)
    print(
        f"grid {grid} batch {args.batch} transition {args.transition_steps} "
        f"stages stabilize<={args.stabilize_iters} flow<={args.flow_iters} "
        f"chain-after {args.chain_after}", flush=True,
    )

    started = time.time()
    for iteration in range(start_iteration + 1, args.iters + 1):
        stage = stage_at(iteration, args.stabilize_iters, args.flow_iters)
        if stage != current_stage:
            current_stage = stage
            optimizer = make_optimizer(model, stage)
            print(f"stage -> {stage}", flush=True)

        if stage == "stabilize":
            result = stabilize_episode(
                model, frames, args.batch, args.stabilize_steps,
                not args.no_activation_checkpoint,
            )
        else:
            chain_edges = 2 if iteration >= args.chain_after else 1
            result = transition_episode(
                model, frames, args.batch, args.transition_steps, chain_edges,
                not args.no_activation_checkpoint,
            )
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
                f"iter {iteration:5d} {stage:9s} loss {loss.item():.5f} "
                f"dice {gate['dice']:.3f} edge {gate['boundary']:.3f} "
                f"sharp {gate['sharp']:.3f} {detail} {rate:.2f} it/s",
                flush=True,
            )

        if iteration % args.checkpoint_every == 0 or iteration == args.iters:
            path = Path(args.out)
            save_layered_checkpoint(model, path, iteration, current_stage)
            numbered = path.with_name(f"{path.stem}_it{iteration}{path.suffix}")
            save_layered_checkpoint(model, numbered, iteration, current_stage)
            preview = Path(args.preview)
            preview.parent.mkdir(parents=True, exist_ok=True)
            save_preview(model, frames, preview, args.transition_steps)
            metadata = {
                "iteration": iteration,
                "stage": current_stage,
                "target": str(Path(args.target).resolve()),
                "transition_steps": args.transition_steps,
                "batch": args.batch,
            }
            preview.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
            print(f"  checkpoint -> {path}; preview -> {preview}", flush=True)

    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak allocated {peak_gib:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
