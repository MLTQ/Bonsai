"""Hidden-only phase-space NCA and NCA5 serialization.

The visible RGBA channels retain the proven residual update.  Only the twelve
latent channels receive matched velocity registers, preventing inertial color
and alpha overshoot while still giving the automaton explicit internal motion.
"""

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


POSITION_CH = 16
VISIBLE_CH = 4
MOMENTUM_CH = POSITION_CH - VISIBLE_CH
STATE_CH = POSITION_CH + MOMENTUM_CH
DEFAULT_DECAY = 0.95
STATE_CLAMP = 8.0


class HiddenMomentumNCA(nn.Module):
    """Residual RGBA plus damped phase-space dynamics for hidden channels."""

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
        # Four visible residual deltas followed by twelve hidden forces.
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
        if x.shape[1] != STATE_CH:
            raise ValueError(f"expected {STATE_CH} state channels, got {x.shape[1]}")
        if cond.shape != (x.shape[0], self.cond):
            raise ValueError(f"expected cond shape {(x.shape[0], self.cond)}, got {tuple(cond.shape)}")

        pre_life = self.alive(x)
        percept = F.conv2d(x, self.percept_w, padding=1, groups=STATE_CH)
        cmap = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        update = self.w2(F.relu(self.w1(torch.cat([percept, cmap], dim=1))))
        if fire_mask is None:
            fire_mask = (
                torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= self.fire_rate
            ).to(x.dtype)

        position = x[:, :POSITION_CH]
        velocity = x[:, POSITION_CH:]
        visible = position[:, :VISIBLE_CH] + update[:, :VISIBLE_CH] * fire_mask
        velocity = self.momentum_decay * velocity + update[:, VISIBLE_CH:] * fire_mask
        hidden = position[:, VISIBLE_CH:] + velocity
        out = torch.cat([visible, hidden, velocity], dim=1).clamp(-STATE_CLAMP, STATE_CLAMP)
        life = (pre_life & self.alive(out)).to(out.dtype)
        return out * life


def lift_state(position, hidden_velocity=None):
    """Append twelve hidden-channel velocities to a legacy 16-channel state."""
    if position.shape[1] != POSITION_CH:
        raise ValueError(f"expected {POSITION_CH} position channels")
    if hidden_velocity is None:
        hidden_velocity = torch.zeros_like(position[:, VISIBLE_CH:])
    expected = position[:, VISIBLE_CH:].shape
    if hidden_velocity.shape != expected:
        raise ValueError(f"expected hidden velocity shape {tuple(expected)}")
    return torch.cat([position, hidden_velocity], dim=1)


def transplant_residual(donor, model):
    """Lift a like-conditioned residual donor into hidden-only phase space."""
    donor_cond = donor.w1.weight.shape[1] - POSITION_CH * 3
    if donor_cond != model.cond or donor.w1.weight.shape[0] != model.hidden:
        raise ValueError("donor and hidden-momentum model shapes do not match")
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
        model.w2.weight[:VISIBLE_CH].copy_(donor.w2.weight[:VISIBLE_CH])
        model.w2.bias[:VISIBLE_CH].copy_(donor.w2.bias[:VISIBLE_CH])
        scale = 1.0 - model.momentum_decay
        model.w2.weight[VISIBLE_CH:].copy_(donor.w2.weight[VISIBLE_CH:] * scale)
        model.w2.bias[VISIBLE_CH:].copy_(donor.w2.bias[VISIBLE_CH:] * scale)


def export_nca5(model, path):
    """Write hidden-only momentum weights in the experimental NCA5 format."""
    header = struct.pack(
        "<4s6iff",
        b"NCA5",
        STATE_CH,
        model.hidden,
        model.cond,
        POSITION_CH,
        VISIBLE_CH,
        MOMENTUM_CH,
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


def load_nca5(model, path):
    """Load an NCA5 checkpoint into a shape-compatible hidden-momentum model."""
    with open(path, "rb") as f:
        header = f.read(36)
        if len(header) != 36:
            raise ValueError("truncated NCA5 header")
        (magic, state_ch, hidden, cond, position_ch, visible_ch, momentum_ch,
         fire_rate, momentum_decay) = struct.unpack("<4s6iff", header)
        expected_layout = (STATE_CH, model.hidden, model.cond, POSITION_CH,
                           VISIBLE_CH, MOMENTUM_CH)
        if magic != b"NCA5" or (
                state_ch, hidden, cond, position_ch, visible_ch, momentum_ch
        ) != expected_layout:
            raise ValueError(
                "incompatible NCA5 layout: "
                f"{(state_ch, hidden, cond, position_ch, visible_ch, momentum_ch)}"
            )
        payload = f.read()

    w1_count = model.hidden * (STATE_CH * 3 + model.cond)
    w2_count = POSITION_CH * model.hidden
    expected_count = w1_count + model.hidden + w2_count + POSITION_CH
    if len(payload) != expected_count * 4:
        raise ValueError("truncated or oversized NCA5 payload")
    values = torch.from_numpy(np.frombuffer(payload, dtype="<f4").copy()).to(
        device=model.w1.weight.device, dtype=model.w1.weight.dtype
    )
    offset = 0
    with torch.no_grad():
        model.w1.weight.copy_(values[offset:offset + w1_count].view_as(model.w1.weight))
        offset += w1_count
        model.w1.bias.copy_(values[offset:offset + model.hidden])
        offset += model.hidden
        model.w2.weight.copy_(values[offset:offset + w2_count].view_as(model.w2.weight))
        offset += w2_count
        model.w2.bias.copy_(values[offset:offset + POSITION_CH])
    model.fire_rate = float(fire_rate)
    model.momentum_decay = float(momentum_decay)
    return model
