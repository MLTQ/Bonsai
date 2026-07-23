"""Global-ring, hard-edge, advection-reaction NCA for the 2D experiment."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from train_cyclic import CH, FIRE_RATE, HIDDEN, OMEGA


EDGE_COUNT = 4
PHASE_FEATURES = 2 + 1 + EDGE_COUNT
FLOW_HIDDEN = 64
MAX_FLOW_CELLS = 0.55
CHECKPOINT_CHUNK = 8


def make_global(phase):
    """Encode phase as the shared internal ring token `[sin, cos]`."""
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def phase_from_global(global_state):
    """Decode `[sin, cos]` to phase in `[0, 2pi)`."""
    return torch.remainder(
        torch.atan2(global_state[:, 0], global_state[:, 1]), 2 * math.pi
    )


def advance_global(global_state, steps=1):
    """Advance the shared ring by a fixed rotation and renormalize it."""
    angle = OMEGA * steps
    sin_angle, cos_angle = math.sin(angle), math.cos(angle)
    sin_phase, cos_phase = global_state[:, 0], global_state[:, 1]
    advanced = torch.stack(
        [sin_phase * cos_angle + cos_phase * sin_angle,
         cos_phase * cos_angle - sin_phase * sin_angle],
        dim=1,
    )
    return advanced / advanced.norm(dim=1, keepdim=True).clamp(min=1e-8)


def phase_features(global_state):
    """Return ring/progress features and exactly one active edge per sample."""
    phase = phase_from_global(global_state)
    cycle_position = phase * (EDGE_COUNT / (2 * math.pi))
    edge = cycle_position.floor().long().remainder(EDGE_COUNT)
    progress = (cycle_position - cycle_position.floor()).unsqueeze(1)
    one_hot = F.one_hot(edge, EDGE_COUNT).to(global_state.dtype)
    return torch.cat([global_state, progress, one_hot], dim=1), edge


def warp_state(state, flow_cells, base_grid):
    """Backward-warp NCHW state by an XY flow expressed in cell units."""
    height, width = state.shape[2:]
    scale = state.new_tensor([
        2.0 / max(width - 1, 1),
        2.0 / max(height - 1, 1),
    ])
    normalized_flow = flow_cells.permute(0, 2, 3, 1) * scale
    sample_grid = base_grid.expand(state.shape[0], -1, -1, -1) - normalized_flow
    return F.grid_sample(
        state, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )


class TransportNCA2D(nn.Module):
    """One global oscillator plus edge-gated flow and local reaction/repair."""

    def __init__(self, grid=128, max_flow=MAX_FLOW_CELLS):
        super().__init__()
        self.grid = grid
        self.max_flow = max_flow
        perception_channels = CH * 3
        rule_inputs = perception_channels + PHASE_FEATURES

        self.repair_w1 = nn.Conv2d(rule_inputs, HIDDEN, 1)
        self.flow_w1 = nn.Conv2d(rule_inputs, FLOW_HIDDEN, 1)
        self.repair_w = nn.Parameter(torch.zeros(EDGE_COUNT, CH, HIDDEN))
        self.repair_b = nn.Parameter(torch.zeros(EDGE_COUNT, CH))
        self.flow_w = nn.Parameter(torch.zeros(EDGE_COUNT, 2, FLOW_HIDDEN))
        self.flow_b = nn.Parameter(torch.zeros(EDGE_COUNT, 2))

        identity = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.T.contiguous()
        kernels = torch.stack([identity, sobel_x, sobel_y]).repeat(
            CH, 1, 1
        ).unsqueeze(1)
        self.register_buffer("percept_w", kernels)

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid),
            torch.linspace(-1.0, 1.0, grid),
            indexing="ij",
        )
        self.register_buffer("base_grid", torch.stack([x, y], dim=-1).unsqueeze(0))
        self.use_checkpoint = True

    @staticmethod
    def alive(state):
        return F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1

    @staticmethod
    def _selected_head(hidden, weights, bias, edge):
        batch, _, height, width = hidden.shape
        selected_weights = weights[edge]
        selected_bias = bias[edge]
        flat = torch.bmm(selected_weights, hidden.flatten(2))
        flat = flat + selected_bias.unsqueeze(2)
        return flat.view(batch, selected_weights.shape[1], height, width)

    def _step(self, state, global_state, fire_mask=None):
        pre_life = self.alive(state)
        perception = F.conv2d(state, self.percept_w, padding=1, groups=CH)
        features, edge = phase_features(global_state)
        feature_map = features[:, :, None, None].expand(-1, -1, *state.shape[2:])
        rule_input = torch.cat([perception, feature_map], dim=1)

        repair_hidden = F.relu(self.repair_w1(rule_input))
        flow_hidden = F.relu(self.flow_w1(rule_input))
        reaction = self._selected_head(
            repair_hidden, self.repair_w, self.repair_b, edge
        )
        raw_flow = self._selected_head(flow_hidden, self.flow_w, self.flow_b, edge)
        flow = torch.tanh(raw_flow) * self.max_flow

        transported = warp_state(state, flow, self.base_grid)
        if fire_mask is None:
            fire_mask = (
                torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device)
                <= FIRE_RATE
            ).to(state.dtype)
        candidate = transported + reaction * fire_mask
        life = (pre_life & self.alive(candidate)).to(state.dtype)
        output = (candidate * life).clamp(-8.0, 8.0)
        return output, advance_global(global_state), flow, reaction

    def forward(self, state, global_state, fire_mask=None):
        output, next_global, _, _ = self._step(state, global_state, fire_mask)
        return output, next_global

    def step_with_aux(self, state, global_state, fire_mask=None):
        return self._step(state, global_state, fire_mask)

    def rollout(self, state, global_state, steps):
        """Advance local/global recurrent state with bounded BPTT memory."""
        def run_chunk(local_state, ring_state, count):
            for _ in range(int(count)):
                local_state, ring_state = self(local_state, ring_state)
            return local_state, ring_state

        done = 0
        while done < steps:
            count = min(CHECKPOINT_CHUNK, steps - done)
            if self.training and self.use_checkpoint and state.requires_grad:
                state, global_state = checkpoint(
                    run_chunk, state, global_state, torch.tensor(count),
                    use_reentrant=False,
                )
            else:
                state, global_state = run_chunk(state, global_state, count)
            done += count
        return state, global_state

    def set_stage(self, stage):
        if stage not in {"flow", "joint"}:
            raise ValueError("stage must be 'flow' or 'joint'")
        repair_trainable = stage == "joint"
        for parameter in self.repair_w1.parameters():
            parameter.requires_grad_(repair_trainable)
        self.repair_w.requires_grad_(repair_trainable)
        self.repair_b.requires_grad_(repair_trainable)
        for parameter in self.flow_w1.parameters():
            parameter.requires_grad_(True)
        self.flow_w.requires_grad_(True)
        self.flow_b.requires_grad_(True)


def transplant_nca2(donor, treatment):
    """Copy behavior-0 NCA2 so zero-flow treatment is step-equivalent."""
    with torch.no_grad():
        treatment.repair_w1.weight.zero_()
        treatment.repair_w1.weight[:, :CH * 3].copy_(
            donor.w1.weight[:, :CH * 3]
        )
        treatment.repair_w1.weight[:, CH * 3:CH * 3 + 2].copy_(
            donor.w1.weight[:, CH * 3:CH * 3 + 2]
        )
        treatment.repair_w1.bias.copy_(donor.w1.bias)
        donor_w2 = donor.w2.weight[:, :, 0, 0]
        for edge in range(EDGE_COUNT):
            treatment.repair_w[edge].copy_(donor_w2)
            treatment.repair_b[edge].copy_(donor.w2.bias)
        treatment.flow_w.zero_()
        treatment.flow_b.zero_()


def save_transport_checkpoint(model, path, iteration=0):
    torch.save({
        "format": "TN2D1",
        "iteration": int(iteration),
        "grid": model.grid,
        "max_flow": model.max_flow,
        "state_dict": model.state_dict(),
    }, path)


def load_transport_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "TN2D1":
        raise ValueError("not a TN2D1 transport checkpoint")
    model = TransportNCA2D(
        grid=int(payload["grid"]), max_flow=float(payload["max_flow"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, int(payload.get("iteration", 0))
