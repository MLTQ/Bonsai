"""Canonical recurrent-state interfaces for fused 2D NCA experts."""

import torch
import torch.nn.functional as F

from train_cyclic import CH


def canonical_key_state(frames, indices):
    """Encode reviewed RGBA anchors as compatible 16-channel NCA states."""
    visible = frames[indices]
    alpha = visible[:, 3:4].clamp(0, 1)
    batch, _, height, width = visible.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=visible.device,
                       dtype=visible.dtype),
        torch.linspace(-1.0, 1.0, width, device=visible.device,
                       dtype=visible.dtype),
        indexing="ij",
    )
    x = x[None, None]
    y = y[None, None]
    sobel_x = visible.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    gradient_x = F.conv2d(alpha, sobel_x, padding=1)
    gradient_y = F.conv2d(alpha, sobel_x.transpose(2, 3), padding=1)
    # Recurrent life masking preserves at most the immediate alpha
    # neighborhood, so the canonical interface must not place state farther
    # away than that neighborhood.
    soft_support = F.avg_pool2d(alpha, 3, stride=1, padding=1)
    pose_code = F.one_hot(indices, frames.shape[0]).to(visible.dtype)
    pose_code = pose_code[:, :, None, None].expand(-1, -1, height, width) * alpha
    luminance = visible[:, :3].mean(dim=1, keepdim=True)
    saturation = (
        visible[:, :3].amax(dim=1, keepdim=True)
        - visible[:, :3].amin(dim=1, keepdim=True)
    )
    hidden = torch.cat([
        alpha, alpha * x, alpha * y, gradient_x, gradient_y, soft_support,
        pose_code, luminance, saturation,
    ], dim=1)
    if hidden.shape[1] != CH - 4:
        raise RuntimeError("canonical state encoder must fill exactly 12 hidden channels")
    return torch.cat([visible, hidden], dim=1)
