"""Dense walking targets and surface-aware losses for the transport experiment."""

import math
from pathlib import Path

import numpy as np
import torch

from shoggoth3d import draw3d
from transport_nca3d import phase_from_global


DENSE_FRAMES = 48


def dense_walk_frames(count=DENSE_FRAMES):
    """Render a closed dense walk cycle without crossfading distant key poses."""
    return np.stack([
        draw3d(2 * math.pi * frame / count, walking=True)
        for frame in range(count)
    ]).astype(np.float16)


def load_dense_walk_frames(cache_path, count=DENSE_FRAMES):
    """Load or create `(F,D,H,W,4)` procedural walking targets."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = np.load(cache_path)
        frames = payload["frames"]
        if frames.shape[0] == count:
            return frames
    frames = dense_walk_frames(count)
    np.savez_compressed(cache_path, frames=frames)
    return frames


def target_at_global(frames, global_state):
    """Interpolate only neighboring dense samples at the global ring phase."""
    phase = phase_from_global(global_state)
    position = phase * (frames.shape[0] / (2 * math.pi))
    first = position.floor().long().remainder(frames.shape[0])
    second = (first + 1).remainder(frames.shape[0])
    weight = (position - position.floor())[:, None, None, None, None]
    return frames[first] * (1 - weight) + frames[second] * weight


def _spatial_gradients(volume):
    return (
        volume[:, :, 1:] - volume[:, :, :-1],
        volume[:, :, :, 1:] - volume[:, :, :, :-1],
        volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1],
    )


def visible_objective(prediction, target):
    """Surface-aware RGBA objective; returns total plus named diagnostics."""
    pred_rgb, pred_alpha = prediction[:, :3], prediction[:, 3:4]
    target_rgb, target_alpha = target[:, :3], target[:, 3:4]
    color_weight = 0.25 + target_alpha
    color = ((pred_rgb - target_rgb).square() * color_weight).mean()
    alpha = (pred_alpha - target_alpha).square().mean()
    edge = prediction.new_zeros(())
    for pred_grad, target_grad in zip(
            _spatial_gradients(pred_alpha), _spatial_gradients(target_alpha)):
        edge = edge + (pred_grad - target_grad).abs().mean()
    premult = torch.relu(pred_rgb - pred_alpha).square().mean()
    total = color + 2.0 * alpha + 0.20 * edge + 0.05 * premult
    return total, {
        "color": color.detach(),
        "alpha": alpha.detach(),
        "edge": edge.detach(),
        "premult": premult.detach(),
    }
