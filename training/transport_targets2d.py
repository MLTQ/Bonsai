"""Four-anchor targets and visible losses for the 2D transport experiment."""

import math

import numpy as np
import torch

from transport_nca2d import EDGE_COUNT, phase_from_global


def load_cycle_frames(path, device=None):
    """Load a one-behavior, four-anchor corpus as `(4,4,H,W)` float32."""
    payload = np.load(path, allow_pickle=True)
    if str(payload["kind"]) != "2d_cycle":
        raise ValueError("expected kind=2d_cycle")
    source = payload["frames"]
    if source.shape[:2] != (1, EDGE_COUNT) or source.shape[-1] != 4:
        raise ValueError("transport2d requires frames shaped (1,4,H,W,4)")
    frames = torch.from_numpy(source[0].astype(np.float32)).permute(0, 3, 1, 2)
    return frames.to(device) if device is not None else frames


def target_at_global(frames, global_state):
    """Blend only the current anchor and directed successor on the active edge."""
    phase = phase_from_global(global_state)
    cycle_position = phase * (EDGE_COUNT / (2 * math.pi))
    first = cycle_position.floor().long().remainder(EDGE_COUNT)
    second = (first + 1).remainder(EDGE_COUNT)
    weight = (cycle_position - cycle_position.floor())[:, None, None, None]
    return frames[first] * (1 - weight) + frames[second] * weight


def _spatial_gradients(image):
    return (
        image[:, :, 1:] - image[:, :, :-1],
        image[:, :, :, 1:] - image[:, :, :, :-1],
    )


def visible_objective(prediction, target):
    """Edge-aware premultiplied RGBA objective with named diagnostics."""
    pred_rgb, pred_alpha = prediction[:, :3], prediction[:, 3:4]
    target_rgb, target_alpha = target[:, :3], target[:, 3:4]
    color_weight = 0.25 + target_alpha
    color = ((pred_rgb - target_rgb).square() * color_weight).mean()
    alpha = (pred_alpha - target_alpha).square().mean()
    edge = prediction.new_zeros(())
    for pred_gradient, target_gradient in zip(
            _spatial_gradients(pred_alpha), _spatial_gradients(target_alpha)):
        edge = edge + (pred_gradient - target_gradient).abs().mean()
    premult = torch.relu(pred_rgb - pred_alpha).square().mean()
    total = color + 2.0 * alpha + 0.35 * edge + 0.05 * premult
    return total, {
        "color": color.detach(),
        "alpha": alpha.detach(),
        "edge": edge.detach(),
        "premult": premult.detach(),
    }
