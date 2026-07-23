"""Compare reaction-only NC3C and global transport across repeated 3D cycles."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_cyclic3d import (CH, FIRE_RATE, OMEGA, CyclicNCA3D,
                            load_nc3c)
from train_transport3d import ROOT, build_teacher_bank
from transport_nca3d import (TransportNCA3D, load_transport_checkpoint,
                             phase_from_global, transplant_nc3c)
from transport_targets3d import (DENSE_FRAMES, load_dense_walk_frames,
                                 target_at_global)


def donor_step(model, state, phase, fire_mask):
    """NC3C step with an injected fire mask for paired stochastic evaluation."""
    pre_life = model.alive(state)
    perception = F.conv3d(state, model.percept_w, padding=1, groups=CH)
    cond = torch.stack([phase.sin(), phase.cos(), torch.ones_like(phase)], dim=1)
    cond_map = cond[:, :, None, None, None].expand(-1, -1, *state.shape[2:])
    delta = model.w2(F.relu(model.w1(torch.cat([perception, cond_map], dim=1))))
    candidate = state + delta * fire_mask
    life = (pre_life & model.alive(candidate)).to(state.dtype)
    return (candidate * life).clamp(-8.0, 8.0)


def softness(alpha, target_alpha):
    alpha = alpha.clamp(0, 1)
    target_alpha = target_alpha.clamp(0, 1)
    support = (alpha > 0.02) | (target_alpha > 0.02)
    denominator = support.sum().clamp(min=1)
    pred_soft = (((alpha > 0.1) & (alpha < 0.9)) & support).sum() / denominator
    target_soft = (((target_alpha > 0.1) & (target_alpha < 0.9)) & support).sum() / denominator
    return pred_soft.item(), target_soft.item()


def part_phase_metrics(prediction, frames, expected_phase):
    """Estimate phase independently in four XZ body quadrants."""
    grid = prediction.shape[-1]
    z, _, x = torch.meshgrid(
        torch.arange(grid, device=prediction.device),
        torch.arange(grid, device=prediction.device),
        torch.arange(grid, device=prediction.device),
        indexing="ij",
    )
    masks = [
        (x < grid // 2) & (z < grid // 2),
        (x >= grid // 2) & (z < grid // 2),
        (x < grid // 2) & (z >= grid // 2),
        (x >= grid // 2) & (z >= grid // 2),
    ]
    best = []
    target_alpha = frames[:, 3]
    for mask in masks:
        errors = ((target_alpha - prediction[3]).square() * mask).sum(dim=(1, 2, 3))
        errors = errors / mask.sum().clamp(min=1)
        best.append(int(errors.argmin().item()))
    angles = torch.tensor(best, device=prediction.device) * (2 * math.pi / frames.shape[0])
    resultant = torch.stack([angles.cos().mean(), angles.sin().mean()]).norm().item()
    expected_index = expected_phase * (frames.shape[0] / (2 * math.pi))
    indices = torch.tensor(best, device=prediction.device, dtype=torch.float32)
    delta = torch.remainder(
        indices - expected_index + frames.shape[0] / 2, frames.shape[0]
    ) - frames.shape[0] / 2
    expected_error = (delta.abs() / (frames.shape[0] / 2)).mean().item()
    return 1.0 - resultant, expected_error


def nonadjacent_leakage(prediction, frames, expected_phase):
    """Project alpha onto four key poses and report mass outside the active edge."""
    key_indices = torch.arange(4, device=frames.device) * (frames.shape[0] // 4)
    design = frames[key_indices, 3].flatten(1).T
    solution = torch.linalg.lstsq(design, prediction[3].flatten()).solution
    weights = solution.clamp(min=0)
    weights = weights / weights.sum().clamp(min=1e-8)
    cycle_position = expected_phase * (4 / (2 * math.pi))
    edge = int(math.floor(cycle_position)) % 4
    allowed = {edge, (edge + 1) % 4}
    return sum(weights[index].item() for index in range(4) if index not in allowed)


def project_rgba(volume):
    """Front alpha-weighted projection used by existing 3D previews."""
    value = volume.detach().permute(1, 2, 3, 0).cpu().numpy()
    alpha = np.clip(value[..., 3], 0, 1)
    weight = alpha / (alpha.sum(axis=0, keepdims=True) + 1e-6)
    rgb = (np.clip(value[..., :3], 0, 1) * weight[..., None]).sum(axis=0)
    out_alpha = 1 - np.prod(1 - alpha * 0.9, axis=0)
    return np.concatenate([rgb, out_alpha[..., None]], axis=-1)[::-1]


def save_preview(captures, path):
    rows = []
    for name in ("target", "baseline", "treatment"):
        row = np.concatenate([project_rgba(frame) for frame in captures[name]], axis=1)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    image = (np.clip(sheet, 0, 1) * 255).astype(np.uint8)
    scale = 4
    Image.fromarray(image, "RGBA").resize(
        (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST
    ).save(path)


def summarize(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ROOT / "weights" / "shoggoth3d.nca"))
    ap.add_argument("--treatment", default=None)
    ap.add_argument("--cache", default=str(
        ROOT / "training" / "corpus_shoggoth3d_walk_dense48_32.npz"))
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--preview", default=str(ROOT / "experiments" / "transport3d_preview.png"))
    ap.add_argument("--json", default=str(ROOT / "experiments" / "transport3d_metrics.json"))
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(123)
    np.random.seed(123)

    frames_np = load_dense_walk_frames(args.cache, DENSE_FRAMES)
    frames = torch.from_numpy(frames_np).permute(0, 4, 1, 2, 3).float().to(device)
    baseline = CyclicNCA3D().to(device).eval()
    load_nc3c(baseline, args.baseline)
    if args.treatment:
        treatment, iteration = load_transport_checkpoint(args.treatment, device)
    else:
        treatment = TransportNCA3D(grid=frames.shape[-1]).to(device)
        transplant_nc3c(baseline, treatment)
        iteration = 0
    treatment.eval()
    treatment.use_checkpoint = False

    initial, global_state = build_teacher_bank(baseline, 1, device)
    baseline_state = initial.clone()
    treatment_state = initial.clone()
    phase = phase_from_global(global_state)
    total_steps = args.cycles * 240
    captures = {"target": [], "baseline": [], "treatment": []}
    metrics = {
        name: {key: [] for key in (
            "mse", "excess_softness", "part_phase_dispersion",
            "part_phase_error", "nonadjacent_leakage", "cycle_drift")}
        for name in ("baseline", "treatment")
    }
    cycle_reference = {
        "baseline": baseline_state[:, :4].clone(),
        "treatment": treatment_state[:, :4].clone(),
    }

    with torch.no_grad():
        for step in range(1, total_steps + 1):
            fire = (
                torch.rand(1, 1, *baseline_state.shape[2:], device=device) <= FIRE_RATE
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
                dispersion, phase_error = part_phase_metrics(
                    state, frames, expected_phase
                )
                metrics[name]["part_phase_dispersion"].append(dispersion)
                metrics[name]["part_phase_error"].append(phase_error)
                leakage = nonadjacent_leakage(state, frames, expected_phase)
                metrics[name]["nonadjacent_leakage"].append(
                    leakage - target_leakage
                )

    summary = {
        "treatment_iteration": iteration,
        "cycles": args.cycles,
        "stride": args.stride,
        **{
            name: {key: summarize(values) for key, values in group.items()}
            for name, group in metrics.items()
        },
    }
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    save_preview(captures, args.preview)
    Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"preview -> {args.preview}")
    print(f"metrics -> {args.json}")


if __name__ == "__main__":
    main()
