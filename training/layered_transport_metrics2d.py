"""Differentiable gates and evaluation metrics for layered 2D transport."""

import math

import torch
import torch.nn.functional as F


def total_variation_batch(alpha):
    """Return alpha total variation independently for each batch sample."""
    dy = (alpha[:, :, 1:] - alpha[:, :, :-1]).abs().mean(dim=(1, 2, 3))
    dx = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    return dx + dy


def edge_energy_batch(alpha):
    """Return squared alpha-gradient energy for each batch sample.

    Unlike L1 total variation, this score falls when a fixed contrast edge is
    spread across several pixels, so it distinguishes blur from translation.
    """
    dy = (alpha[:, :, 1:] - alpha[:, :, :-1]).square().mean(dim=(1, 2, 3))
    dx = (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).square().mean(
        dim=(1, 2, 3)
    )
    return dx + dy


def transition_shape_penalty(prediction, source, destination):
    """Penalize blur, support inflation, and invalid premultiplied color."""
    alpha = prediction[:, 3:4].clamp(0, 1)
    source_alpha = source[:, 3:4].clamp(0, 1)
    destination_alpha = destination[:, 3:4].clamp(0, 1)

    variation = edge_energy_batch(alpha)
    source_variation = edge_energy_batch(source_alpha)
    destination_variation = edge_energy_batch(destination_alpha)
    variation_low = 0.85 * torch.minimum(source_variation, destination_variation)
    variation_high = 1.15 * torch.maximum(source_variation, destination_variation)
    sharpness = (
        (F.relu(variation_low - variation) / variation_low.clamp(min=1e-6)).square()
        + (F.relu(variation - variation_high) / variation_high.clamp(min=1e-6)).square()
    ).mean()

    mass = alpha.sum(dim=(1, 2, 3))
    source_mass = source_alpha.sum(dim=(1, 2, 3))
    destination_mass = destination_alpha.sum(dim=(1, 2, 3))
    mass_low = 0.85 * torch.minimum(source_mass, destination_mass)
    mass_high = 1.15 * torch.maximum(source_mass, destination_mass)
    support = (
        (F.relu(mass_low - mass) / mass_low.clamp(min=1)).square()
        + (F.relu(mass - mass_high) / mass_high.clamp(min=1)).square()
    ).mean()

    premult = F.relu(prediction[:, :3] - alpha).square().mean()
    alpha_range = (
        F.relu(-prediction[:, 3:4]).square().mean()
        + F.relu(prediction[:, 3:4] - 1).square().mean()
    )
    return sharpness + support + 0.25 * premult + 0.25 * alpha_range, {
        "shape_sharp": sharpness.detach(),
        "shape_support": support.detach(),
        "shape_premult": premult.detach(),
    }


def alpha_dice(prediction, target):
    pred = prediction[:, 3:4].clamp(0, 1)
    truth = target[:, 3:4].clamp(0, 1)
    intersection = (pred * truth).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def _boundary(alpha):
    support = (alpha > 0.1).float()
    dilated = F.max_pool2d(support, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-support, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def boundary_f1(prediction, target, tolerance=2):
    pred = _boundary(prediction[:, 3:4].clamp(0, 1))
    truth = _boundary(target[:, 3:4].clamp(0, 1))
    kernel = tolerance * 2 + 1
    pred_near = F.max_pool2d(pred, kernel, stride=1, padding=tolerance)
    truth_near = F.max_pool2d(truth, kernel, stride=1, padding=tolerance)
    precision = (pred * truth_near).sum(dim=(1, 2, 3)) / pred.sum(
        dim=(1, 2, 3)
    ).clamp(min=1)
    recall = (truth * pred_near).sum(dim=(1, 2, 3)) / truth.sum(
        dim=(1, 2, 3)
    ).clamp(min=1)
    return (2 * precision * recall / (precision + recall).clamp(min=1e-6)).mean()


def sharpness_ratio(prediction, target):
    pred = edge_energy_batch(prediction[:, 3:4].clamp(0, 1))
    truth = edge_energy_batch(target[:, 3:4].clamp(0, 1))
    return (pred / truth.clamp(min=1e-8)).mean()


def rgb_gradient_error(prediction, target):
    """Return exact endpoint RGB-gradient L1 error."""
    pred = prediction[:, :3]
    truth = target[:, :3]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    truth_dx = truth[:, :, :, 1:] - truth[:, :, :, :-1]
    pred_dy = pred[:, :, 1:] - pred[:, :, :-1]
    truth_dy = truth[:, :, 1:] - truth[:, :, :-1]
    return (pred_dx - truth_dx).abs().mean() + (pred_dy - truth_dy).abs().mean()


def rgb_sharpness_ratio(prediction, target):
    """Compare squared RGB-gradient energy at an exact endpoint."""
    pred = prediction[:, :3]
    truth = target[:, :3]
    pred_energy = (
        (pred[:, :, :, 1:] - pred[:, :, :, :-1]).square().mean(dim=(1, 2, 3))
        + (pred[:, :, 1:] - pred[:, :, :-1]).square().mean(dim=(1, 2, 3))
    )
    truth_energy = (
        (truth[:, :, :, 1:] - truth[:, :, :, :-1]).square().mean(dim=(1, 2, 3))
        + (truth[:, :, 1:] - truth[:, :, :-1]).square().mean(dim=(1, 2, 3))
    )
    return (pred_energy / truth_energy.clamp(min=1e-8)).mean()


def support_ratio(prediction, target):
    pred = prediction[:, 3:4].clamp(0, 1).sum(dim=(1, 2, 3))
    truth = target[:, 3:4].clamp(0, 1).sum(dim=(1, 2, 3))
    return (pred / truth.clamp(min=1)).mean()


def nonadjacent_leakage(prediction, frames, edge):
    """Fit four anchors and return mass outside the declared directed edge."""
    values = []
    design = frames.flatten(1).T
    gram = design.T @ design
    scale = gram.diagonal().mean().clamp(min=1e-8)
    gram = gram + torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    ) * (scale * 1e-6)
    for sample, active_edge in zip(prediction, edge):
        right_hand_side = design.T @ sample.flatten()
        solution = torch.linalg.solve(gram, right_hand_side).clamp(min=0)
        weights = solution / solution.sum().clamp(min=1e-8)
        first = int(active_edge.item()) % frames.shape[0]
        allowed = {first, (first + 1) % frames.shape[0]}
        values.append(sum(
            weights[index] for index in range(frames.shape[0])
            if index not in allowed
        ))
    return torch.stack(values).mean()


def dynamic_part_masks(frames):
    """Build four target-derived motion regions for independent phase checks."""
    variance = frames.var(dim=0).sum(dim=0)
    support = frames[:, 3].amax(dim=0) > 0.02
    coordinates = torch.nonzero(support)
    if not len(coordinates):
        raise ValueError("target corpus has no visible support")
    y_center = coordinates[:, 0].float().mean()
    x_center = coordinates[:, 1].float().mean()
    height, width = support.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=frames.device),
        torch.arange(width, device=frames.device), indexing="ij",
    )
    regions = (
        (y < y_center) & (x < x_center),
        (y < y_center) & (x >= x_center),
        (y >= y_center) & (x < x_center),
        (y >= y_center) & (x >= x_center),
    )
    masks = []
    energy = variance * support
    for region in regions:
        mask = energy * region
        if mask.sum() <= 1e-8:
            mask = support.to(frames.dtype) * region
        masks.append(mask / mask.sum().clamp(min=1e-8))
    return masks


def part_phase_metrics(prediction, frames, expected_index, masks):
    """Return circular regional disagreement and normalized phase error."""
    inferred = []
    for mask in masks:
        errors = ((frames - prediction).square() * mask[None, None]).sum(
            dim=(1, 2, 3)
        )
        inferred.append(int(errors.argmin().item()))
    angles = torch.tensor(inferred, device=prediction.device) * (2 * math.pi / 4)
    resultant = torch.stack([angles.cos().mean(), angles.sin().mean()]).norm()
    indices = torch.tensor(inferred, device=prediction.device, dtype=torch.float32)
    delta = torch.remainder(indices - expected_index + 2, 4) - 2
    return 1 - resultant, (delta.abs() / 2).mean()
