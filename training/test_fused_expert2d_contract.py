"""CPU-fast contracts for fused pose and edge expert routing."""

import tempfile
import struct
from pathlib import Path

import numpy as np
import torch

from export_fused_expert2d import HEADER, MAGIC, WEIGHT_ORDER, export_fused
from fused_expert_nca2d import (
    FusedExpertNCA2D, load_fused_checkpoint, save_fused_checkpoint,
)
from fused_state2d import canonical_key_state
from train_cyclic import CH


def _frames(grid=17):
    frames = torch.zeros(4, 4, grid, grid)
    for pose in range(4):
        left = 3 + pose
        frames[pose, :3, 5:12, left:left + 6] = (pose + 1) / 4
        frames[pose, 3, 5:12, left:left + 6] = 1
    return frames


def test_canonical_state_preserves_visible_and_pose_code():
    frames = _frames()
    indices = torch.tensor([0, 3])
    state = canonical_key_state(frames, indices)
    assert state.shape == (2, CH, 17, 17)
    torch.testing.assert_close(state[:, :4], frames[indices])
    # Hidden offsets 6:10 are alpha-masked one-hot pose channels.
    pose_code = state[:, 10:14]
    assert pose_code[0, 0].sum() > 0 and pose_code[0, 1:].sum() == 0
    assert pose_code[1, 3].sum() > 0 and pose_code[1, :3].sum() == 0


def test_zero_initialized_pose_experts_preserve_canonical_state():
    frames = _frames()
    indices = torch.tensor([1, 2])
    state = canonical_key_state(frames, indices)
    model = FusedExpertNCA2D(grid=17)
    output = model(
        state, indices, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
        torch.ones(2, 1, 17, 17),
    )
    torch.testing.assert_close(output, state, atol=0, rtol=0)


def test_pose_parameter_banks_are_hard_selected():
    frames = _frames()
    state = canonical_key_state(frames, torch.tensor([0, 1]))
    model = FusedExpertNCA2D(grid=17)
    with torch.no_grad():
        model.pose_b2[0, 0] = 0.25
    output = model(
        state, torch.tensor([0, 1]), torch.zeros(2),
        torch.zeros(2, dtype=torch.bool), torch.ones(2, 1, 17, 17),
    )
    assert (output[0, 0] - state[0, 0]).abs().sum() > 0
    torch.testing.assert_close(output[1], state[1], atol=0, rtol=0)


def test_edge_parameter_banks_are_hard_selected():
    frames = _frames()
    indices = torch.tensor([0, 1])
    state = canonical_key_state(frames, indices)
    model = FusedExpertNCA2D(grid=17)
    with torch.no_grad():
        model.edge_flow_w2.zero_()
        model.edge_flow_b2.zero_()
        model.edge_flow_b2[0, 0::2] = 0.5
    _, flows, assignments, _ = model.step_with_aux(
        state, indices, torch.full((2,), 0.5),
        torch.ones(2, dtype=torch.bool), torch.ones(2, 1, 17, 17),
    )
    assert flows[0, :, 0].abs().mean() > 0.1
    torch.testing.assert_close(flows[1], torch.zeros_like(flows[1]), atol=0, rtol=0)
    torch.testing.assert_close(
        assignments.sum(dim=1), torch.ones_like(assignments[:, 0]),
        atol=2e-6, rtol=2e-6,
    )
    assert torch.equal(assignments, assignments.round())


def test_pose_and_edge_banks_have_finite_gradients():
    torch.manual_seed(5)
    frames = _frames()
    indices = torch.tensor([2])
    state = canonical_key_state(frames, indices)
    model = FusedExpertNCA2D(grid=17)
    pose = model(
        state + torch.randn_like(state) * 0.01, indices,
        torch.zeros(1), torch.zeros(1, dtype=torch.bool),
        torch.ones(1, 1, 17, 17),
    )
    pose.square().mean().backward()
    assert torch.isfinite(model.pose_w2.grad).all()
    assert model.pose_w2.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    edge, flows, _, _ = model.step_with_aux(
        state, indices, torch.full((1,), 0.5),
        torch.ones(1, dtype=torch.bool), torch.ones(1, 1, 17, 17),
    )
    (edge.square().mean() + flows.square().mean()).backward()
    assert torch.isfinite(model.edge_flow_w2.grad).all()
    assert model.edge_flow_w2.grad.abs().sum() > 0


def test_fused_checkpoint_roundtrip():
    model = FusedExpertNCA2D(
        grid=13, slots=3, max_flow=0.8, hard_slots=True,
        expert_hidden=96, flow_hidden=48, position_frequencies=2,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fused.pt"
        save_fused_checkpoint(model, path, iteration=81, stage="edge")
        loaded, iteration, stage = load_fused_checkpoint(path)
    assert iteration == 81 and stage == "edge"
    assert loaded.grid == 13 and loaded.slots == 3 and loaded.max_flow == 0.8
    assert loaded.hard_slots
    assert loaded.expert_hidden == 96 and loaded.flow_hidden == 48
    assert loaded.position_frequencies == 2
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(actual, expected)


def test_fx2d_export_header_weights_and_canonical_state():
    frames = _frames(grid=17)
    model = FusedExpertNCA2D(
        grid=17, slots=3, max_flow=0.8, hard_slots=True,
        expert_hidden=24, flow_hidden=12, position_frequencies=2,
    )
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkpoint = directory / "fused.pt"
        target = directory / "targets.npz"
        output = directory / "fused.fx2d"
        state_path = directory / "fused.ncs"
        save_fused_checkpoint(model, checkpoint, iteration=81, stage="edge")
        np.savez_compressed(
            target, kind=np.array("2d_cycle"),
            frames=frames.permute(0, 2, 3, 1).numpy()[None],
        )
        metadata = export_fused(
            checkpoint, target, output, state_path,
            transition_steps=19, handoff_steps=7,
        )
        binary = output.read_bytes()
        header = HEADER.unpack_from(binary)
        state_binary = state_path.read_bytes()

    assert header[:8] == (MAGIC, 17, CH, 4, 3, 24, 12, 2)
    assert abs(header[8] - 0.8) < 1e-6 and header[9:] == (0.5, 19, 7)
    expected_weights = torch.cat([
        model.state_dict()[name].detach().flatten() for name in WEIGHT_ORDER
    ]).numpy()
    actual_weights = np.frombuffer(binary, dtype="<f4", offset=HEADER.size)
    np.testing.assert_array_equal(actual_weights, expected_weights)
    assert struct.unpack_from("<4s3i", state_binary) == (b"NCS1", 17, 17, CH)
    actual_state = torch.from_numpy(
        np.frombuffer(state_binary, dtype="<f4", offset=16).copy()
    ).view(17, 17, CH).permute(2, 0, 1)
    expected_state = canonical_key_state(frames, torch.tensor([0]))[0]
    torch.testing.assert_close(actual_state, expected_state)
    assert metadata["iteration"] == 81 and metadata["transition_steps"] == 19
