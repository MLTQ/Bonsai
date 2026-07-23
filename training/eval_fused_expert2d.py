"""Evaluate fused NCA experts under stochastic chained deployment rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fused_expert_nca2d import EDGE_COUNT, load_fused_checkpoint
from fused_state2d import canonical_key_state
from layered_transport_metrics2d import (
    alpha_dice, boundary_f1, dynamic_part_masks, edge_energy_batch,
    nonadjacent_leakage, part_phase_metrics, sharpness_ratio, support_ratio,
    rgb_gradient_error, rgb_sharpness_ratio,
)
from layered_transport_nca2d import load_layered_checkpoint
from train_cyclic import FIRE_RATE, make_mature_state
from transport_targets2d import load_cycle_frames


ROOT = Path(__file__).resolve().parent.parent


def _key_state(frames, indices, interface):
    if interface == "canonical":
        return canonical_key_state(frames, indices)
    if interface == "alpha":
        return make_mature_state(frames[indices])
    raise ValueError(f"unknown state interface {interface}")


def _smooth_progress(value):
    return value * value * (3.0 - 2.0 * value)


def _fire(state, generator):
    return (
        torch.rand(
            state.shape[0], 1, *state.shape[2:], device=state.device,
            generator=generator,
        ) <= FIRE_RATE
    ).to(state.dtype)


def _corrupt_anchors(state, generator):
    """Apply the same noise/dropout/erasure class used to train pose basins."""
    support = F.max_pool2d(state[:, 3:4].clamp(0, 1), 5, stride=1, padding=2)
    visible_noise = torch.randn(
        state[:, :4].shape, device=state.device, dtype=state.dtype,
        generator=generator,
    ) * 0.075 * support
    hidden_noise = torch.randn(
        state[:, 4:].shape, device=state.device, dtype=state.dtype,
        generator=generator,
    ) * 0.15 * support
    damaged = torch.cat([
        state[:, :4] + visible_noise, state[:, 4:] + hidden_noise,
    ], dim=1)
    sparse_keep = (
        torch.rand(
            state.shape[0], 1, *state.shape[2:], device=state.device,
            generator=generator,
        ) > 0.06
    ).to(state.dtype)
    damaged = damaged * (1 - support + support * sparse_keep)
    for sample in range(state.shape[0]):
        coordinates = torch.nonzero(state[sample, 3] > 0.1)
        if not len(coordinates):
            continue
        selected = coordinates[torch.randint(
            len(coordinates), (), device=state.device, generator=generator,
        )]
        size = int(torch.randint(
            6, 13, (), device=state.device, generator=generator,
        ).item())
        center_y, center_x = (int(value.item()) for value in selected)
        y0, x0 = max(0, center_y - size // 2), max(0, center_x - size // 2)
        damaged[sample, :, y0:y0 + size, x0:x0 + size] = 0
    return damaged


def _anchor_metrics(state, target):
    prediction, visible_target = state[:, :4], target[:, :4]
    return {
        "visible_mse": (prediction - visible_target).square().mean().item(),
        "dice": alpha_dice(prediction, visible_target).item(),
        "boundary_f1": boundary_f1(prediction, visible_target).item(),
        "rgb_gradient_error": rgb_gradient_error(
            prediction, visible_target
        ).item(),
        "hidden_mse": (state[:, 4:] - target[:, 4:]).square().mean().item(),
        "state_abs_max": state.abs().max().item(),
    }


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _summary(values):
    if not values:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _append_endpoint(group, state, target, frames, expert, masks):
    prediction = state[:, :4]
    group["visible_mse"].append((prediction - target).square().mean().item())
    group["dice"].append(alpha_dice(prediction, target).item())
    group["boundary_f1"].append(boundary_f1(prediction, target).item())
    group["sharpness_ratio"].append(sharpness_ratio(prediction, target).item())
    group["rgb_gradient_error"].append(
        rgb_gradient_error(prediction, target).item()
    )
    group["rgb_sharpness_ratio"].append(
        rgb_sharpness_ratio(prediction, target).item()
    )
    group["support_ratio"].append(support_ratio(prediction, target).item())
    group["nonadjacent_leakage"].append(
        nonadjacent_leakage(prediction, frames, expert).item()
    )
    dispersions, errors = [], []
    expected = float((int(expert[0].item()) + 1) % EDGE_COUNT)
    for sample in prediction:
        dispersion, error = part_phase_metrics(sample, frames, expected, masks)
        dispersions.append(float(dispersion.item()))
        errors.append(float(error.item()))
    group["part_phase_dispersion"].append(_mean(dispersions))
    group["part_phase_error"].append(_mean(errors))
    group["state_abs_max"].append(state.abs().max().item())


def _append_intermediate(group, prediction, source, destination, frames,
                         expert, expected_index, assignments, flows, masks):
    alpha = prediction[:, 3:4].clamp(0, 1)
    source_energy = edge_energy_batch(source[:, 3:4].clamp(0, 1))
    destination_energy = edge_energy_batch(destination[:, 3:4].clamp(0, 1))
    reference_energy = 0.5 * (source_energy + destination_energy)
    ratio = edge_energy_batch(alpha) / reference_energy.clamp(min=1e-8)
    group["sharpness_ratio"].append(ratio.mean().item())

    mass = alpha.sum(dim=(1, 2, 3))
    source_mass = source[:, 3:4].clamp(0, 1).sum(dim=(1, 2, 3))
    destination_mass = destination[:, 3:4].clamp(0, 1).sum(dim=(1, 2, 3))
    low = 0.85 * torch.minimum(source_mass, destination_mass)
    high = 1.15 * torch.maximum(source_mass, destination_mass)
    violation = (
        torch.relu(low - mass) / low.clamp(min=1)
        + torch.relu(mass - high) / high.clamp(min=1)
    )
    group["support_band_violation"].append(violation.mean().item())
    group["nonadjacent_leakage"].append(
        nonadjacent_leakage(prediction, frames, expert).item()
    )
    dispersions, errors = [], []
    for sample in prediction:
        dispersion, error = part_phase_metrics(
            sample, frames, expected_index, masks
        )
        dispersions.append(float(dispersion.item()))
        errors.append(float(error.item()))
    group["part_phase_dispersion"].append(_mean(dispersions))
    group["part_phase_error"].append(_mean(errors))
    support = (alpha > 0.05).to(assignments.dtype)
    usage = (assignments * support).sum(dim=(0, 2, 3)) / support.sum().clamp(min=1)
    entropy = -(
        assignments.clamp(min=1e-6).log() * assignments * support
    ).sum() / support.sum().clamp(min=1)
    group["max_slot_usage"].append(usage.max().item())
    group["slot_entropy"].append(entropy.item())
    group["flow_rms"].append(flows.square().mean().sqrt().item())


def _composite(frame, background=0.08):
    rgba = frame.detach().permute(1, 2, 0).cpu().numpy()
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    return np.clip(rgba[..., :3], 0, 1) + background * (1 - alpha)


def _save_preview(captures, frames, path):
    targets = [frames[(index + 1) % EDGE_COUNT].cpu() for index in range(EDGE_COUNT)]
    rows = []
    for row in (targets, captures["midpoint"], captures["bridge"], captures["handoff"]):
        rows.append(np.concatenate([_composite(frame) for frame in row], axis=1))
    sheet = np.concatenate(rows, axis=0)
    image = Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8), "RGB")
    image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(path)


def _metric_group(names):
    return {name: [] for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--architecture", choices=("fused", "layered"),
                        default="fused")
    parser.add_argument("--state-interface", choices=("canonical", "alpha"),
                        default="canonical")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--transition-steps", type=int, default=24)
    parser.add_argument("--handoff-steps", type=int, default=8)
    parser.add_argument("--anchor-steps", type=int, default=12)
    parser.add_argument("--preview", default=str(
        ROOT / "experiments" / "fused_expert2d_eval.png"))
    parser.add_argument("--json", default=str(
        ROOT / "experiments" / "fused_expert2d_eval.json"))
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if min(args.cycles, args.trials, args.handoff_steps, args.anchor_steps) <= 0:
        parser.error("cycles, trials, handoff-steps, and anchor-steps must be positive")
    if args.transition_steps < 4:
        parser.error("transition-steps must be at least 4")

    device = torch.device(args.device)
    frames = load_cycle_frames(args.target, device)
    masks = dynamic_part_masks(frames)
    if args.architecture == "fused":
        model, iteration, stage = load_fused_checkpoint(args.checkpoint, device)
    else:
        model, iteration, stage = load_layered_checkpoint(args.checkpoint, device)
    model.eval()
    if model.grid != frames.shape[-1]:
        parser.error("checkpoint grid differs from target")
    generator = torch.Generator(device=device)
    generator.manual_seed(8675309)

    anchor_generator = torch.Generator(device=device)
    anchor_generator.manual_seed(314159)
    anchor_indices = torch.arange(EDGE_COUNT, device=device).repeat_interleave(
        args.trials
    )
    anchor_target = _key_state(frames, anchor_indices, args.state_interface)
    anchor_state = _corrupt_anchors(anchor_target, anchor_generator)
    anchor_before = _anchor_metrics(anchor_state, anchor_target)
    anchor_progress = torch.zeros(
        anchor_state.shape[0], device=device, dtype=frames.dtype
    )
    anchor_mode = torch.zeros(
        anchor_state.shape[0], device=device, dtype=torch.bool
    )
    with torch.no_grad():
        for _ in range(args.anchor_steps):
            anchor_state = model(
                anchor_state, anchor_indices, anchor_progress, anchor_mode,
                _fire(anchor_state, anchor_generator),
            )
    anchor_after = _anchor_metrics(anchor_state, anchor_target)
    initial_indices = torch.zeros(args.trials, device=device, dtype=torch.long)
    initial = _key_state(frames, initial_indices, args.state_interface)
    state = initial.clone()
    endpoint_names = (
        "visible_mse", "dice", "boundary_f1", "sharpness_ratio",
        "rgb_gradient_error", "rgb_sharpness_ratio",
        "support_ratio", "nonadjacent_leakage", "part_phase_dispersion",
        "part_phase_error", "state_abs_max",
    )
    intermediate_names = (
        "sharpness_ratio", "support_band_violation", "nonadjacent_leakage",
        "part_phase_dispersion", "part_phase_error", "max_slot_usage",
        "slot_entropy", "flow_rms",
    )
    bridge = _metric_group(endpoint_names)
    handoff = _metric_group(endpoint_names)
    intermediate = _metric_group(intermediate_names)
    cycle_visible_drift, cycle_hidden_drift = [], []
    captures = {name: [] for name in ("midpoint", "bridge", "handoff")}

    with torch.no_grad():
        for cycle in range(args.cycles):
            for edge_index in range(EDGE_COUNT):
                expert = torch.full(
                    (args.trials,), edge_index, device=device, dtype=torch.long
                )
                destination_index = (edge_index + 1) % EDGE_COUNT
                destination_indices = torch.full_like(expert, destination_index)
                source = frames[expert]
                destination_state = _key_state(
                    frames, destination_indices, args.state_interface
                )
                transition = torch.ones(
                    args.trials, device=device, dtype=torch.bool
                )
                for step in range(args.transition_steps):
                    raw_progress = (step + 1) / args.transition_steps
                    progress = torch.full(
                        (args.trials,), _smooth_progress(raw_progress),
                        device=device, dtype=frames.dtype,
                    )
                    state, flows, assignments, _ = model.step_with_aux(
                        state, expert, progress, transition,
                        _fire(state, generator),
                    )
                    if step in {
                        args.transition_steps // 4 - 1,
                        args.transition_steps // 2 - 1,
                        3 * args.transition_steps // 4 - 1,
                        args.transition_steps - 1,
                    }:
                        expected = edge_index + raw_progress
                        _append_intermediate(
                            intermediate, state[:, :4], source,
                            destination_state[:, :4], frames, expert,
                            expected, assignments, flows, masks,
                        )
                    if cycle == 0 and step == args.transition_steps // 2 - 1:
                        captures["midpoint"].append(state[0, :4].cpu())

                _append_endpoint(
                    bridge, state, destination_state[:, :4],
                    frames, expert, masks,
                )
                if cycle == 0:
                    captures["bridge"].append(state[0, :4].cpu())
                dwell_progress = torch.zeros(
                    args.trials, device=device, dtype=frames.dtype
                )
                dwell_mode = torch.zeros(
                    args.trials, device=device, dtype=torch.bool
                )
                for _ in range(args.handoff_steps):
                    state = model(
                        state, destination_indices, dwell_progress, dwell_mode,
                        _fire(state, generator),
                    )
                _append_endpoint(
                    handoff, state, destination_state[:, :4],
                    frames, expert, masks,
                )
                if cycle == 0:
                    captures["handoff"].append(state[0, :4].cpu())

            cycle_visible_drift.append(
                (state[:, :4] - initial[:, :4]).square().mean().item()
            )
            cycle_hidden_drift.append(
                (state[:, 4:] - initial[:, 4:]).square().mean().item()
            )

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "iteration": iteration,
        "stage": stage,
        "architecture": args.architecture,
        "state_interface": args.state_interface,
        "cycles": args.cycles,
        "trials": args.trials,
        "anchor_recovery": {
            "steps": args.anchor_steps,
            "before": anchor_before,
            "after": anchor_after,
            "visible_gain": anchor_before["visible_mse"] - anchor_after["visible_mse"],
            "hidden_gain": anchor_before["hidden_mse"] - anchor_after["hidden_mse"],
        },
        "bridge": {name: _summary(values) for name, values in bridge.items()},
        "handoff": {name: _summary(values) for name, values in handoff.items()},
        "intermediate": {
            name: _summary(values) for name, values in intermediate.items()
        },
        "cycle_visible_drift": _summary(cycle_visible_drift),
        "cycle_hidden_drift": _summary(cycle_hidden_drift),
    }
    gates = {
        "anchor_recovery": (
            report["anchor_recovery"]["after"]["dice"] >= 0.95
            and report["anchor_recovery"]["after"]["boundary_f1"] >= 0.90
            and report["anchor_recovery"]["after"]["rgb_gradient_error"] <= 0.03
            and report["anchor_recovery"]["visible_gain"] > 0
            and report["anchor_recovery"]["hidden_gain"] > 0
        ),
        "bridge_dice": report["bridge"]["dice"]["mean"] >= 0.88,
        "handoff_dice": report["handoff"]["dice"]["mean"] >= 0.93,
        "handoff_boundary": report["handoff"]["boundary_f1"]["mean"] >= 0.85,
        "handoff_sharpness": (
            0.80 <= report["handoff"]["sharpness_ratio"]["mean"] <= 1.20
        ),
        "handoff_rgb_detail": (
            report["handoff"]["rgb_gradient_error"]["mean"] <= 0.02
            and report["handoff"]["rgb_sharpness_ratio"]["mean"] >= 0.75
        ),
        "intermediate_sharpness": (
            report["intermediate"]["sharpness_ratio"]["mean"] >= 0.70
        ),
        "nonadjacent_leakage": (
            report["intermediate"]["nonadjacent_leakage"]["mean"] <= 0.05
        ),
        "phase_dispersion": (
            report["intermediate"]["part_phase_dispersion"]["mean"] <= 0.25
        ),
        "cycle_drift": report["cycle_visible_drift"]["max"] <= 0.005,
        "bounded_state": report["handoff"]["state_abs_max"]["max"] <= 1.5,
    }
    report["acceptance"] = {**gates, "numeric_pass": all(gates.values())}
    preview_path, json_path = Path(args.preview), Path(args.json)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _save_preview(captures, frames, preview_path)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"preview -> {preview_path}")
    print(f"metrics -> {json_path}")


if __name__ == "__main__":
    main()
