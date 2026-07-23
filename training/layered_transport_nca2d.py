"""Hard-edge multi-flow advection and local repair for coherent 2D NCA motion."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_cyclic import CH, FIRE_RATE, HIDDEN
from transport_nca2d import warp_state


EDGE_COUNT = 4
MOTION_SLOTS = 4
CONTROLLER_FEATURES = EDGE_COUNT + 2  # one-hot edge, progress, transition flag
FLOW_HIDDEN = 96
MAX_FLOW_CELLS = 1.5


def controller_features(edge, progress, transition, dtype):
    """Encode exactly one active edge plus continuous transition progress."""
    one_hot = F.one_hot(edge, EDGE_COUNT).to(dtype)
    return torch.cat(
        [one_hot, progress[:, None], transition[:, None].to(dtype)], dim=1
    )


def advect_state(state, flow, base_grid):
    """MacCormack-corrected recurrent advection with a monotonic limiter."""
    predicted = warp_state(state, flow, base_grid)
    reversed_state = warp_state(predicted, -flow, base_grid)
    corrected = predicted + 0.5 * (state - reversed_state)
    low = -F.max_pool2d(-state, 3, stride=1, padding=1)
    high = F.max_pool2d(state, 3, stride=1, padding=1)
    return torch.maximum(torch.minimum(corrected, high), low)


class LayeredTransportNCA2D(nn.Module):
    """Four soft motion slots under one hard global edge controller."""

    def __init__(self, grid=128, slots=MOTION_SLOTS, max_flow=MAX_FLOW_CELLS):
        super().__init__()
        self.grid = int(grid)
        self.slots = int(slots)
        self.max_flow = float(max_flow)
        rule_inputs = CH * 3 + CONTROLLER_FEATURES + 2

        self.flow_w1 = nn.Conv2d(rule_inputs, FLOW_HIDDEN, 1)
        self.slot_w1 = nn.Conv2d(rule_inputs, FLOW_HIDDEN, 1)
        self.repair_w1 = nn.Conv2d(rule_inputs, HIDDEN, 1)
        self.flow_w = nn.Parameter(torch.empty(EDGE_COUNT, self.slots * 2, FLOW_HIDDEN))
        self.flow_b = nn.Parameter(torch.zeros(EDGE_COUNT, self.slots * 2))
        self.slot_w = nn.Parameter(torch.empty(EDGE_COUNT, self.slots, FLOW_HIDDEN))
        self.slot_b = nn.Parameter(torch.zeros(EDGE_COUNT, self.slots))
        self.repair_w = nn.Parameter(torch.zeros(EDGE_COUNT, CH, HIDDEN))
        self.repair_b = nn.Parameter(torch.zeros(EDGE_COUNT, CH))

        nn.init.normal_(self.flow_w, std=1e-3)
        nn.init.normal_(self.slot_w, std=1e-3)
        with torch.no_grad():
            # Anchor stabilization first sees progress=transition=0. Keeping
            # unseen controller columns neutral prevents a frozen repair rule
            # from changing arbitrarily when flow training begins.
            self.repair_w1.weight[:, CH * 3:].zero_()

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
            torch.linspace(-1.0, 1.0, self.grid),
            torch.linspace(-1.0, 1.0, self.grid),
            indexing="ij",
        )
        coordinate_grid = torch.stack([x, y], dim=0).unsqueeze(0)
        self.register_buffer("coordinate_grid", coordinate_grid)
        self.register_buffer(
            "base_grid", torch.stack([x, y], dim=-1).unsqueeze(0)
        )

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

    def _rule_input(self, state, edge, progress, transition):
        perception = F.conv2d(state, self.percept_w, padding=1, groups=CH)
        control = controller_features(edge, progress, transition, state.dtype)
        control_map = control[:, :, None, None].expand(
            -1, -1, state.shape[2], state.shape[3]
        )
        coordinates = self.coordinate_grid.expand(state.shape[0], -1, -1, -1)
        return torch.cat([perception, control_map, coordinates], dim=1)

    def step_with_aux(self, state, edge, progress, transition, fire_mask=None):
        rule_input = self._rule_input(state, edge, progress, transition)
        flow_hidden = F.relu(self.flow_w1(rule_input))
        slot_hidden = F.relu(self.slot_w1(rule_input))
        repair_hidden = F.relu(self.repair_w1(rule_input))

        raw_flow = self._selected_head(flow_hidden, self.flow_w, self.flow_b, edge)
        flows = torch.tanh(raw_flow).view(
            state.shape[0], self.slots, 2, *state.shape[2:]
        ) * self.max_flow
        flows = flows * transition[:, None, None, None, None].to(state.dtype)
        slot_logits = self._selected_head(
            slot_hidden, self.slot_w, self.slot_b, edge
        )
        assignments = F.softmax(slot_logits / 0.5, dim=1)

        transported = state.new_zeros(state.shape)
        for slot in range(self.slots):
            moved = advect_state(state, flows[:, slot], self.base_grid)
            transported = transported + moved * assignments[:, slot:slot + 1]

        reaction = self._selected_head(
            repair_hidden, self.repair_w, self.repair_b, edge
        )
        if fire_mask is None:
            fire_mask = (
                torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device)
                <= FIRE_RATE
            ).to(state.dtype)
        candidate = transported + reaction * fire_mask
        life = (self.alive(transported) & self.alive(candidate)).to(state.dtype)
        output = (candidate * life).clamp(-8.0, 8.0)
        return output, flows, assignments, reaction

    def forward(self, state, edge, progress, transition, fire_mask=None):
        return self.step_with_aux(
            state, edge, progress, transition, fire_mask
        )[0]

    def set_stage(self, stage):
        if stage not in {"stabilize", "flow", "joint"}:
            raise ValueError("stage must be stabilize, flow, or joint")
        repair_trainable = stage in {"stabilize", "joint"}
        flow_trainable = stage in {"flow", "joint"}
        for parameter in self.repair_w1.parameters():
            parameter.requires_grad_(repair_trainable)
        self.repair_w.requires_grad_(repair_trainable)
        self.repair_b.requires_grad_(repair_trainable)
        for module in (self.flow_w1, self.slot_w1):
            for parameter in module.parameters():
                parameter.requires_grad_(flow_trainable)
        for parameter in (self.flow_w, self.flow_b, self.slot_w, self.slot_b):
            parameter.requires_grad_(flow_trainable)


def save_layered_checkpoint(model, path, iteration=0, stage="joint"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "LT2D1",
        "iteration": int(iteration),
        "stage": stage,
        "grid": model.grid,
        "slots": model.slots,
        "max_flow": model.max_flow,
        "state_dict": model.state_dict(),
    }, path)


def load_layered_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "LT2D1":
        raise ValueError("not an LT2D1 layered transport checkpoint")
    model = LayeredTransportNCA2D(
        grid=int(payload["grid"]), slots=int(payload["slots"]),
        max_flow=float(payload["max_flow"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    return model, int(payload.get("iteration", 0)), str(payload.get("stage", "joint"))
