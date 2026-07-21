"""Phase-space NCA update and NCA4 serialization.

NCA4 lifts the familiar 16-channel NCA state into explicit position and
velocity halves.  Perception reads both halves, the learned rule emits forces,
and a damped symplectic-Euler step advances the state::

    velocity = decay * velocity + fire * force
    position = position + velocity

The first four position channels remain premultiplied RGBA.  This module is the
eager PyTorch reference for the matching Metal implementation.
"""

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


POSITION_CH = 16
STATE_CH = POSITION_CH * 2
DEFAULT_DECAY = 0.95
STATE_CLAMP = 8.0


class MomentumNCA(nn.Module):
    """A 2D NCA whose second state half is explicit per-channel velocity."""

    def __init__(self, cond: int, hidden: int = 128, fire_rate: float = 0.5,
                 momentum_decay: float = DEFAULT_DECAY):
        super().__init__()
        if cond < 0:
            raise ValueError("cond must be non-negative")
        if not 0.0 <= momentum_decay <= 1.0:
            raise ValueError("momentum_decay must be in [0, 1]")
        self.cond = cond
        self.hidden = hidden
        self.fire_rate = fire_rate
        self.momentum_decay = momentum_decay
        self.w1 = nn.Conv2d(STATE_CH * 3 + cond, hidden, 1)
        self.w2 = nn.Conv2d(hidden, POSITION_CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        kernels = torch.stack([ident, sx, sy]).repeat(STATE_CH, 1, 1).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

    @staticmethod
    def alive(x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def forward(self, x, cond, fire_mask=None):
        """Advance one step; ``fire_mask`` is injectable for contract tests."""
        if x.shape[1] != STATE_CH:
            raise ValueError(f"expected {STATE_CH} state channels, got {x.shape[1]}")
        if cond.shape != (x.shape[0], self.cond):
            raise ValueError(f"expected cond shape {(x.shape[0], self.cond)}, got {tuple(cond.shape)}")

        pre_life = self.alive(x)
        percept = F.conv2d(x, self.percept_w, padding=1, groups=STATE_CH)
        cmap = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        force = self.w2(F.relu(self.w1(torch.cat([percept, cmap], dim=1))))
        if fire_mask is None:
            fire_mask = (
                torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= self.fire_rate
            ).to(x.dtype)

        position, velocity = x[:, :POSITION_CH], x[:, POSITION_CH:]
        velocity = self.momentum_decay * velocity + force * fire_mask
        position = position + velocity
        out = torch.cat([position, velocity], dim=1).clamp(-STATE_CLAMP, STATE_CLAMP)
        life = (pre_life & self.alive(out)).to(out.dtype)
        return out * life


def lift_state(position, velocity=None):
    """Lift a legacy 16-channel state into NCA4 position/velocity layout."""
    if position.shape[1] != POSITION_CH:
        raise ValueError(f"expected {POSITION_CH} position channels")
    if velocity is None:
        velocity = torch.zeros_like(position)
    if velocity.shape != position.shape:
        raise ValueError("velocity must have the same shape as position")
    return torch.cat([position, velocity], dim=1)


def transplant_residual(donor, model):
    """Warm-start NCA4 from a residual donor with the same hidden/cond shape.

    Position perception and conditioning weights are copied directly; velocity
    perception starts at zero.  The donor's displacement head is scaled by
    ``1-decay`` so its old per-step delta is the new integrator's steady-state
    velocity under a constant force.
    """
    donor_cond = donor.w1.weight.shape[1] - POSITION_CH * 3
    if donor_cond != model.cond or donor.w1.weight.shape[0] != model.hidden:
        raise ValueError("donor and momentum model shapes do not match")
    with torch.no_grad():
        model.w1.weight.zero_()
        model.w1.weight[:, : POSITION_CH * 3].copy_(
            donor.w1.weight[:, : POSITION_CH * 3]
        )
        if model.cond:
            model.w1.weight[:, STATE_CH * 3 :].copy_(
                donor.w1.weight[:, POSITION_CH * 3 :]
            )
        model.w1.bias.copy_(donor.w1.bias)
        scale = 1.0 - model.momentum_decay
        model.w2.weight.copy_(donor.w2.weight * scale)
        model.w2.bias.copy_(donor.w2.bias * scale)


def export_nca4(model, path):
    """Write the NCA4 binary contract consumed by ``NCAWeights.swift``."""
    header = struct.pack(
        "<4s4iff",
        b"NCA4",
        STATE_CH,
        model.hidden,
        model.cond,
        POSITION_CH,
        model.fire_rate,
        model.momentum_decay,
    )
    with open(path, "wb") as f:
        f.write(header)
        model.w1.weight.detach().cpu().numpy().reshape(
            model.hidden, STATE_CH * 3 + model.cond
        ).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(
            POSITION_CH, model.hidden
        ).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)
