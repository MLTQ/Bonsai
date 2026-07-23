"""Compare reaction-only NCA2 and global transport across repeated 2D cycles."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_cyclic import CH, FIRE_RATE, OMEGA, CyclicNCA, load_nca2
from train_transport2d import ROOT, build_teacher_bank
from transport_nca2d import (TransportNCA2D, load_transport_checkpoint,
                             phase_from_global, transplant_nca2)
from transport_targets2d import load_cycle_frames, target_at_global


def donor_step(model, state, phase, fire_mask):
    """Evaluate behavior-0 NCA2 with an injected fire mask."""
    pre_life = model.alive(state)
    perception = F.conv2d(state, model.percept_w, padding=1, groups=CH)
    condition = torch.stack([phase.sin(), phase.cos(), torch.zeros_like(phase)], dim=1)
    condition_map = condition[:, :, None, None].expand(-1, -1, *state.shape[2:])
    delta = model.w2(F.relu(model.w1(torch.cat([perception, condition_map], dim=1))))
    candidate = state + delta * fire_mask
    life = (pre_life & model.alive(candidate)).to(state.dtype)
    return (candidate * life).clamp(-8.0, 8.0)


def softness(alpha, target_alpha):
    alpha = alpha.clamp(0, 1)
    target_alpha = target_alpha.clamp(0, 1)
    support = (alpha > 0.02) | (target_alpha > 0.02)
    denominator = support.sum().clamp(min=1)
    pred_soft = (((alpha > 0.1) & (alpha < 0.9)) & support).sum() / denominator
    target_soft = (
        ((target_alpha > 0.1) & (target_alpha < 0.9)) & support
    ).sum() / denominator
    return pred_soft.item(), target_soft.item()


def _total_variation(alpha):
    return (
        (alpha[1:] - alpha[:-1]).abs().mean()
        + (alpha[:, 1:] - alpha[:, :-1]).abs().mean()
    )


def sharpness_deficit(alpha, target_alpha):
    pred = _total_variation(alpha.clamp(0, 1))
    target = _total_variation(target_alpha.clamp(0, 1))
    return ((target - pred) / target.clamp(min=1e-8)).item()


def dynamic_part_masks(frames):
    """Return four motion-weighted quadrants derived only from target variation."""
    variance = frames.var(dim=0).sum(dim=0)
    support = frames[:, 3].amax(dim=0) > 0.02
    energy = variance * support
    coordinates = torch.nonzero(support)
    if not len(coordinates):
        raise ValueError("target corpus has no visible support")
    y_center = coordinates[:, 0].float().mean()
    x_center = coordinates[:, 1].float().mean()
    height, width = support.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=frames.device),
        torch.arange(width, device=frames.device),
        indexing="ij",
    )
    regions = (
        (y < y_center) & (x < x_center),
        (y < y_center) & (x >= x_center),
        (y >= y_center) & (x < x_center),
        (y >= y_center) & (x >= x_center),
    )
    masks = []
    floor = energy[support].mean() * 0.05
    for region in regions:
        mask = energy * region
        if mask.sum() <= floor:
            mask = support.to(frames.dtype) * region
        masks.append(mask / mask.sum().clamp(min=1e-8))
    return masks


def part_phase_metrics(prediction, frames, expected_phase, masks):
    """Infer anchor phase independently in four motion-weighted regions."""
    inferred = []
    for mask in masks:
        errors = ((frames - prediction).square() * mask[None, None]).sum(
            dim=(1, 2, 3)
        )
        inferred.append(int(errors.argmin().item()))
    angles = torch.tensor(inferred, device=prediction.device) * (2 * math.pi / 4)
    resultant = torch.stack([angles.cos().mean(), angles.sin().mean()]).norm().item()
    expected_index = expected_phase * (4 / (2 * math.pi))
    indices = torch.tensor(inferred, device=prediction.device, dtype=torch.float32)
    delta = torch.remainder(indices - expected_index + 2, 4) - 2
    error = (delta.abs() / 2).mean().item()
    return 1.0 - resultant, error


def nonadjacent_leakage(prediction, frames, expected_phase):
    """Fit four anchors and report nonnegative mass outside the active edge."""
    design = frames.flatten(1).T
    solution = torch.linalg.lstsq(design, prediction.flatten()).solution
    weights = solution.clamp(min=0)
    weights = weights / weights.sum().clamp(min=1e-8)
    cycle_position = expected_phase * (4 / (2 * math.pi))
    edge = int(math.floor(cycle_position)) % 4
    allowed = {edge, (edge + 1) % 4}
    return sum(weights[index].item() for index in range(4) if index not in allowed)


def composite_rgba(frame, background=0.10):
    rgba = frame.detach().permute(1, 2, 0).cpu().numpy()
    alpha = np.clip(rgba[..., 3:4], 0, 1)
    rgb = np.clip(rgba[..., :3], 0, 1) + background * (1 - alpha)
    return np.clip(rgb, 0, 1)


def save_preview(captures, path):
    rows = []
    for name in ("target", "baseline", "treatment"):
        rows.append(np.concatenate([composite_rgba(frame) for frame in captures[name]], axis=1))
    sheet = np.concatenate(rows, axis=0)
    image = Image.fromarray((sheet * 255).astype(np.uint8), "RGB")
    image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST).save(path)


def summarize(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", default=None)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--teacher-grow", type=int, default=360)
    parser.add_argument("--preview", default=str(ROOT / "experiments" / "transport2d_preview.png"))
    parser.add_argument("--json", default=str(ROOT / "experiments" / "transport2d_metrics.json"))
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(123)
    np.random.seed(123)

    frames = load_cycle_frames(args.target, device)
    grid = frames.shape[-1]
    part_masks = dynamic_part_masks(frames)
    baseline = CyclicNCA().to(device).eval()
    load_nca2(baseline, args.baseline)
    if args.treatment:
        treatment, iteration = load_transport_checkpoint(args.treatment, device)
    else:
        treatment = TransportNCA2D(grid=grid).to(device)
        transplant_nca2(baseline, treatment)
        iteration = 0
    treatment.eval()
    treatment.use_checkpoint = False

    initial, global_state = build_teacher_bank(
        baseline, 1, grid, device, args.teacher_grow
    )
    baseline_state = initial.clone()
    treatment_state = initial.clone()
    phase = phase_from_global(global_state)
    total_steps = args.cycles * 240
    captures = {name: [] for name in ("target", "baseline", "treatment")}
    metric_names = (
        "mse", "excess_softness", "sharpness_deficit",
        "part_phase_dispersion", "part_phase_error",
        "nonadjacent_leakage", "cycle_drift",
    )
    metrics = {
        name: {key: [] for key in metric_names}
        for name in ("baseline", "treatment")
    }
    cycle_reference = {
        "baseline": baseline_state[:, :4].clone(),
        "treatment": treatment_state[:, :4].clone(),
    }

    with torch.no_grad():
        for step in range(1, total_steps + 1):
            fire = (
                torch.rand(1, 1, grid, grid, device=device) <= FIRE_RATE
            ).float()
            baseline_state = donor_step(baseline, baseline_state, phase, fire)
            treatment_state, global_state, _, _ = treatment.step_with_aux(
                treatment_state, global_state, fire
            )
            phase = torch.remainder(phase + OMEGA, 2 * math.pi)

            if step % 240 == 0:
                for name, state in (("baseline", baseline_state),
                                    ("treatment", treatment_state)):
                    drift = (state[:, :4] - cycle_reference[name]).square().mean().item()
                    metrics[name]["cycle_drift"].append(drift)
                    cycle_reference[name] = state[:, :4].clone()
            if step % args.stride:
                continue

            target = target_at_global(frames, global_state)[0]
            expected_phase = phase_from_global(global_state)[0].item()
            if len(captures["target"]) < 8:
                captures["target"].append(target.cpu())
                captures["baseline"].append(baseline_state[0, :4].cpu())
                captures["treatment"].append(treatment_state[0, :4].cpu())
            target_leakage = nonadjacent_leakage(target, frames, expected_phase)
            for name, state in (("baseline", baseline_state[0, :4]),
                                ("treatment", treatment_state[0, :4])):
                metrics[name]["mse"].append((state - target).square().mean().item())
                pred_soft, target_soft = softness(state[3], target[3])
                metrics[name]["excess_softness"].append(pred_soft - target_soft)
                metrics[name]["sharpness_deficit"].append(
                    sharpness_deficit(state[3], target[3])
                )
                dispersion, phase_error = part_phase_metrics(
                    state, frames, expected_phase, part_masks
                )
                metrics[name]["part_phase_dispersion"].append(dispersion)
                metrics[name]["part_phase_error"].append(phase_error)
                leakage = nonadjacent_leakage(state, frames, expected_phase)
                metrics[name]["nonadjacent_leakage"].append(leakage - target_leakage)

    summary = {
        "treatment_iteration": iteration,
        "cycles": args.cycles,
        "stride": args.stride,
        **{
            name: {key: summarize(values) for key, values in group.items()}
            for name, group in metrics.items()
        },
    }
    preview_path, json_path = Path(args.preview), Path(args.json)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_preview(captures, preview_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"preview -> {preview_path}")
    print(f"metrics -> {json_path}")


if __name__ == "__main__":
    main()
