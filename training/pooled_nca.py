"""Pooled NCA: a cellular automaton with a nervous system.

Every creature in this repo so far obeys strict locality — a cell sees its eight
neighbours and nothing else, so influence crosses the body at exactly one cell
per step. That is what makes damage heal from the edges inward, and it is also
the coordination-latency ceiling that keeps 96^2 and 64^3 creatures muddy: a
signal needs ~72 steps to cross a 72px body, and training rollouts are shorter
than that.

This module breaks locality on purpose, in the cheapest possible way. Each step
we take the alive-masked spatial mean of NPOOL hidden channels and broadcast it
back to every cell as extra perception input:

    g_t = mean over living cells of x_t[4 : 4+NPOOL]        (B, NPOOL)
    x_{t+1} = cell_update(perception, state_flag, g_t)

Two things follow, and the second is the interesting one.

1. Coordination becomes instantaneous. The whole body shares a value every step,
   so global agreement no longer costs O(diameter) steps.

2. The loop closes. The update rule both *reads* g and *writes* the channels g
   is pooled from, so g is a genuine global variable with its own dynamics that
   no individual cell owns — a slow field coupled to fast local physics. That is
   the standard recipe for slow-fast systems, which is to say: bursting,
   spontaneous transitions, and behaviour that is not a lookup into a pose set.
   The creature can hold a mood that lives nowhere in particular.

Cost is one reduction per step, negligible next to the 1x1 convs.

Exports NCAP (= NCA2 + an i32 npool field after cond). Not yet parsed by the
Swift runtime — this is an experiment, and the Metal/Triton/PyTorch numerical
contract only gets updated if it earns it.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_states
from train_states import CH, COND, FIRE_RATE


class PooledNCA(nn.Module):
    """StateNCA with NPOOL globally-broadcast feedback channels.

    Deliberately mirrors StateNCA line for line everywhere else, so a pooled run
    and a local run of the same width differ in exactly one mechanism.
    """

    def __init__(self, npool=4):
        super().__init__()
        self.npool = npool
        hidden = train_states.HIDDEN
        self.w1 = nn.Conv2d(CH * 3 + COND + npool, hidden, 1)
        self.w2 = nn.Conv2d(hidden, CH, 1)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sy = sx.T.contiguous()
        self.register_buffer("percept_w",
                             torch.stack([ident, sx, sy]).repeat(CH, 1, 1).unsqueeze(1))

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > 0.1

    def pooled(self, x, alive_f):
        """Alive-masked spatial mean of the feedback channels, per sample.

        Masking matters: an unmasked mean would dilute as the body grows, so the
        global signal would encode size instead of state.
        """
        n = alive_f.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        chans = x[:, 4:4 + self.npool]
        return (chans * alive_f).sum(dim=(2, 3), keepdim=True) / n   # (B,npool,1,1)

    def forward(self, x, state, flag_mask=None):
        """flag_mask: optional (B,) in {0,1}. Zeroed entries have their state
        flag withheld for this step, so the creature must hold its own state.
        The only place that state can live is the global variable — this is what
        turns g from a readout of the body into a memory the body depends on."""
        pre = self.alive(x)
        p = F.conv2d(x, self.percept_w, padding=1, groups=CH)
        s = state.float()
        if flag_mask is not None:
            s = s * flag_mask.float()
        smap = s[:, None, None, None].expand(-1, 1, *x.shape[2:])
        g = self.pooled(x, pre.float()).expand(-1, -1, *x.shape[2:])
        dx = self.w2(F.relu(self.w1(torch.cat([p, smap, g], dim=1))))
        fire = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= FIRE_RATE).float()
        x = x + dx * fire
        return (x * (pre & self.alive(x)).float()).clamp(-8.0, 8.0)


def export_pooled(model, path):
    """NCAP: magic, i32 ch/hidden/cond/npool, f32 fire, then w1, b1, w2, b2."""
    hidden = train_states.HIDDEN
    pin = CH * 3 + COND + model.npool
    with open(path, "wb") as f:
        f.write(b"NCAP")
        np.array([CH, hidden, COND, model.npool], dtype="<i4").tofile(f)
        np.array([FIRE_RATE], dtype="<f4").tofile(f)
        model.w1.weight.detach().cpu().numpy().reshape(hidden, pin).astype("<f4").tofile(f)
        model.w1.bias.detach().cpu().numpy().astype("<f4").tofile(f)
        model.w2.weight.detach().cpu().numpy().reshape(CH, hidden).astype("<f4").tofile(f)
        model.w2.bias.detach().cpu().numpy().astype("<f4").tofile(f)


def trace_global(model, x, state, steps):
    """Record the global variable over a free rollout — the readout that says
    whether g actually does anything or just sits at a constant."""
    g = []
    with torch.no_grad():
        for _ in range(steps):
            g.append(model.pooled(x, model.alive(x).float()).flatten(1).cpu().numpy())
            x = model(x, state)
    return np.stack(g, axis=0)   # (steps, B, npool)
