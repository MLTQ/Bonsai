"""Hard-routed pose and directed-edge expert banks for a fused 2D NCA."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from layered_transport_nca2d import EDGE_COUNT, MAX_FLOW_CELLS, MOTION_SLOTS, advect_state
from train_cyclic import CH, FIRE_RATE


DEFAULT_EXPERT_HIDDEN = 384
DEFAULT_FLOW_HIDDEN = 128
DEFAULT_POSITION_FREQUENCIES = 4


def _bank_parameter(experts, outputs, inputs, scale=None):
    value = torch.empty(experts, outputs, inputs)
    if scale is None:
        for expert in range(experts):
            nn.init.kaiming_uniform_(value[expert], a=5 ** 0.5)
    else:
        nn.init.normal_(value, std=scale)
    return nn.Parameter(value)


class FusedExpertNCA2D(nn.Module):
    """One runtime rule containing four pose and four directed-edge experts."""

    def __init__(self, grid=128, slots=MOTION_SLOTS, max_flow=MAX_FLOW_CELLS,
                 hard_slots=True, expert_hidden=DEFAULT_EXPERT_HIDDEN,
                 flow_hidden=DEFAULT_FLOW_HIDDEN,
                 position_frequencies=DEFAULT_POSITION_FREQUENCIES):
        super().__init__()
        self.grid = int(grid)
        self.slots = int(slots)
        self.max_flow = float(max_flow)
        self.hard_slots = bool(hard_slots)
        self.expert_hidden = int(expert_hidden)
        self.flow_hidden = int(flow_hidden)
        self.position_frequencies = int(position_frequencies)
        if min(self.expert_hidden, self.flow_hidden) <= 0:
            raise ValueError("expert and flow hidden widths must be positive")
        if self.position_frequencies < 0:
            raise ValueError("position frequencies must be nonnegative")
        coordinate_channels = 2 + 4 * self.position_frequencies
        rule_inputs = CH * 3 + coordinate_channels + 1

        self.pose_w1 = _bank_parameter(EDGE_COUNT, self.expert_hidden, rule_inputs)
        self.pose_b1 = nn.Parameter(torch.zeros(EDGE_COUNT, self.expert_hidden))
        self.pose_w2 = nn.Parameter(torch.zeros(EDGE_COUNT, CH, self.expert_hidden))
        self.pose_b2 = nn.Parameter(torch.zeros(EDGE_COUNT, CH))

        self.edge_flow_w1 = _bank_parameter(
            EDGE_COUNT, self.flow_hidden, rule_inputs
        )
        self.edge_flow_b1 = nn.Parameter(torch.zeros(EDGE_COUNT, self.flow_hidden))
        self.edge_flow_w2 = _bank_parameter(
            EDGE_COUNT, self.slots * 2, self.flow_hidden, scale=1e-3
        )
        self.edge_flow_b2 = nn.Parameter(torch.zeros(EDGE_COUNT, self.slots * 2))
        self.edge_slot_w1 = _bank_parameter(
            EDGE_COUNT, self.flow_hidden, rule_inputs
        )
        self.edge_slot_b1 = nn.Parameter(torch.zeros(EDGE_COUNT, self.flow_hidden))
        self.edge_slot_w2 = _bank_parameter(
            EDGE_COUNT, self.slots, self.flow_hidden, scale=1e-3
        )
        self.edge_slot_b2 = nn.Parameter(torch.zeros(EDGE_COUNT, self.slots))
        self.edge_repair_w1 = _bank_parameter(
            EDGE_COUNT, self.expert_hidden, rule_inputs
        )
        self.edge_repair_b1 = nn.Parameter(torch.zeros(EDGE_COUNT, self.expert_hidden))
        self.edge_repair_w2 = nn.Parameter(
            torch.zeros(EDGE_COUNT, CH, self.expert_hidden)
        )
        self.edge_repair_b2 = nn.Parameter(torch.zeros(EDGE_COUNT, CH))

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
        coordinate_features = [x, y]
        for frequency in range(self.position_frequencies):
            scale = torch.pi * (2 ** frequency)
            coordinate_features.extend([
                torch.sin(scale * x), torch.cos(scale * x),
                torch.sin(scale * y), torch.cos(scale * y),
            ])
        self.register_buffer(
            "coordinate_grid", torch.stack(coordinate_features)[None]
        )
        self.register_buffer("base_grid", torch.stack([x, y], dim=-1)[None])

    @staticmethod
    def alive(state):
        return F.max_pool2d(state[:, 3:4], 3, stride=1, padding=1) > 0.1

    @staticmethod
    def _selected_affine(field, weights, bias, expert):
        batch, _, height, width = field.shape
        output = torch.bmm(weights[expert], field.flatten(2))
        output = output + bias[expert].unsqueeze(2)
        return output.view(batch, weights.shape[1], height, width)

    def _rule_input(self, state, progress):
        perception = F.conv2d(state, self.percept_w, padding=1, groups=CH)
        coordinates = self.coordinate_grid.expand(state.shape[0], -1, -1, -1)
        progress_map = progress[:, None, None, None].expand(
            -1, 1, state.shape[2], state.shape[3]
        )
        return torch.cat([perception, coordinates, progress_map], dim=1)

    def _fire_mask(self, state, fire_mask):
        if fire_mask is not None:
            return fire_mask.to(state.dtype)
        return (
            torch.rand(state.shape[0], 1, *state.shape[2:], device=state.device)
            <= FIRE_RATE
        ).to(state.dtype)

    def _pose_step(self, state, expert, progress, fire_mask):
        rule_input = self._rule_input(state, progress)
        hidden = F.relu(self._selected_affine(
            rule_input, self.pose_w1, self.pose_b1, expert
        ))
        reaction = self._selected_affine(hidden, self.pose_w2, self.pose_b2, expert)
        candidate = state + reaction * fire_mask
        life = (self.alive(state) & self.alive(candidate)).to(state.dtype)
        output = (candidate * life).clamp(-8.0, 8.0)
        flows = state.new_zeros(state.shape[0], self.slots, 2, *state.shape[2:])
        assignments = state.new_zeros(state.shape[0], self.slots, *state.shape[2:])
        return output, flows, assignments, reaction

    def _edge_step(self, state, expert, progress, fire_mask):
        rule_input = self._rule_input(state, progress)
        flow_hidden = F.relu(self._selected_affine(
            rule_input, self.edge_flow_w1, self.edge_flow_b1, expert
        ))
        raw_flow = self._selected_affine(
            flow_hidden, self.edge_flow_w2, self.edge_flow_b2, expert
        )
        flows = torch.tanh(raw_flow).view(
            state.shape[0], self.slots, 2, *state.shape[2:]
        ) * self.max_flow
        slot_hidden = F.relu(self._selected_affine(
            rule_input, self.edge_slot_w1, self.edge_slot_b1, expert
        ))
        slot_logits = self._selected_affine(
            slot_hidden, self.edge_slot_w2, self.edge_slot_b2, expert
        )
        soft_assignments = F.softmax(slot_logits / 0.5, dim=1)
        if self.hard_slots:
            selected = soft_assignments.argmax(dim=1, keepdim=True)
            hard_assignments = torch.zeros_like(soft_assignments).scatter_(
                1, selected, 1.0
            )
            assignments = (
                hard_assignments
                + (soft_assignments - soft_assignments.detach())
            )
        else:
            assignments = soft_assignments
        transported = state.new_zeros(state.shape)
        for slot in range(self.slots):
            moved = advect_state(state, flows[:, slot], self.base_grid)
            transported = transported + moved * assignments[:, slot:slot + 1]

        repair_hidden = F.relu(self._selected_affine(
            rule_input, self.edge_repair_w1, self.edge_repair_b1, expert
        ))
        reaction = self._selected_affine(
            repair_hidden, self.edge_repair_w2, self.edge_repair_b2, expert
        )
        candidate = transported + reaction * fire_mask
        life = (self.alive(transported) & self.alive(candidate)).to(state.dtype)
        output = (candidate * life).clamp(-8.0, 8.0)
        return output, flows, assignments, reaction

    def step_with_aux(self, state, expert, progress, transition, fire_mask=None):
        """Run a hard-selected pose or edge expert for each batch sample."""
        fire_mask = self._fire_mask(state, fire_mask)
        if bool(transition.all()):
            return self._edge_step(state, expert, progress, fire_mask)
        if bool((~transition).all()):
            return self._pose_step(state, expert, progress, fire_mask)
        pose = self._pose_step(state, expert, progress, fire_mask)
        edge = self._edge_step(state, expert, progress, fire_mask)
        selector = transition[:, None, None, None]
        output = tuple(
            torch.where(
                selector[:, None] if pose_value.ndim == 5 else selector,
                edge_value, pose_value,
            )
            for pose_value, edge_value in zip(pose, edge)
        )
        return output

    def forward(self, state, expert, progress, transition, fire_mask=None):
        return self.step_with_aux(
            state, expert, progress, transition, fire_mask
        )[0]

    def set_stage(self, stage):
        if stage not in {"anchor", "edge", "joint"}:
            raise ValueError("stage must be anchor, edge, or joint")
        pose_trainable = stage in {"anchor", "joint"}
        edge_trainable = stage in {"edge", "joint"}
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                pose_trainable if name.startswith("pose_") else edge_trainable
            )


def save_fused_checkpoint(model, path, iteration=0, stage="joint"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "FEX2D2",
        "iteration": int(iteration),
        "stage": str(stage),
        "grid": model.grid,
        "slots": model.slots,
        "max_flow": model.max_flow,
        "hard_slots": model.hard_slots,
        "expert_hidden": model.expert_hidden,
        "flow_hidden": model.flow_hidden,
        "position_frequencies": model.position_frequencies,
        "state_dict": model.state_dict(),
    }, path)


def load_fused_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") not in {"FEX2D1", "FEX2D2"}:
        raise ValueError("not a fused expert checkpoint")
    state_dict = payload["state_dict"]
    expert_hidden = int(payload.get(
        "expert_hidden", state_dict["pose_w1"].shape[1]
    ))
    flow_hidden = int(payload.get(
        "flow_hidden", state_dict["edge_flow_w1"].shape[1]
    ))
    rule_inputs = state_dict["pose_w1"].shape[2]
    inferred_frequencies = max(0, (rule_inputs - CH * 3 - 3) // 4)
    model = FusedExpertNCA2D(
        grid=int(payload["grid"]), slots=int(payload["slots"]),
        max_flow=float(payload["max_flow"]),
        # Checkpoints written before this field were the F1a soft-slot pilot.
        hard_slots=bool(payload.get("hard_slots", False)),
        expert_hidden=expert_hidden,
        flow_hidden=flow_hidden,
        position_frequencies=int(payload.get(
            "position_frequencies", inferred_frequencies
        )),
    ).to(device)
    model.load_state_dict(state_dict)
    return model, int(payload.get("iteration", 0)), str(payload.get("stage", "joint"))
