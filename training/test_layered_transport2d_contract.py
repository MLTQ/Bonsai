"""CPU-fast contracts for hard-edge multi-flow 2D transport."""

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from layered_transport_metrics2d import (
    alpha_dice, boundary_f1, nonadjacent_leakage, rgb_gradient_error,
    rgb_sharpness_ratio, sharpness_ratio,
)
from layered_transport_nca2d import (
    LayeredTransportNCA2D, advect_state, controller_features,
    load_layered_checkpoint, save_layered_checkpoint,
)
from train_cyclic import CH


def test_controller_has_exactly_one_active_edge():
    edge = torch.tensor([0, 3])
    progress = torch.tensor([0.2, 0.8])
    transition = torch.tensor([True, False])
    features = controller_features(edge, progress, transition, torch.float32)
    torch.testing.assert_close(features[:, :4].sum(dim=1), torch.ones(2))
    assert torch.equal(features[:, :4].argmax(dim=1), edge)
    torch.testing.assert_close(features[:, 4], progress)
    assert torch.equal(features[:, 5].bool(), transition)


def test_zero_flow_advection_is_identity():
    model = LayeredTransportNCA2D(grid=17)
    state = torch.randn(2, CH, 17, 17)
    flow = torch.zeros(2, 2, 17, 17)
    actual = advect_state(state, flow, model.base_grid)
    torch.testing.assert_close(actual, state, atol=2e-6, rtol=2e-6)


def test_positive_x_advection_moves_impulse_right():
    model = LayeredTransportNCA2D(grid=17)
    state = torch.zeros(1, CH, 17, 17)
    state[0, 0, 8, 8] = 1
    flow = torch.zeros(1, 2, 17, 17)
    flow[:, 0] = 1
    moved = advect_state(state, flow, model.base_grid)
    maximum = torch.nonzero(moved[0, 0] == moved[0, 0].max())[0]
    assert tuple(maximum.tolist()) == (8, 9)


def test_dwell_is_identity_and_assignments_partition():
    torch.manual_seed(4)
    model = LayeredTransportNCA2D(grid=15)
    state = torch.zeros(2, CH, 15, 15)
    state[:, 3:, 3:12, 3:12] = 1
    edge = torch.tensor([0, 2])
    progress = torch.zeros(2)
    transition = torch.zeros(2, dtype=torch.bool)
    fire = torch.ones(2, 1, 15, 15)
    output, flows, assignments, _ = model.step_with_aux(
        state, edge, progress, transition, fire
    )
    torch.testing.assert_close(flows, torch.zeros_like(flows), atol=0, rtol=0)
    torch.testing.assert_close(
        assignments.sum(dim=1), torch.ones_like(assignments[:, 0]),
        atol=2e-6, rtol=2e-6,
    )
    torch.testing.assert_close(output, state, atol=2e-6, rtol=2e-6)


def test_edge_heads_are_hard_selected():
    model = LayeredTransportNCA2D(grid=13)
    with torch.no_grad():
        model.flow_w.zero_()
        model.flow_b.zero_()
        model.flow_b[0, 0::2] = 0.5
    state = torch.zeros(2, CH, 13, 13)
    state[:, 3:, 3:10, 3:10] = 1
    _, flows, _, _ = model.step_with_aux(
        state, torch.tensor([0, 1]), torch.ones(2),
        torch.ones(2, dtype=torch.bool), torch.ones(2, 1, 13, 13),
    )
    assert flows[0, :, 0].abs().mean() > 0.1
    torch.testing.assert_close(flows[1], torch.zeros_like(flows[1]), atol=0, rtol=0)


def test_transition_transport_has_finite_nonzero_gradient():
    torch.manual_seed(9)
    model = LayeredTransportNCA2D(grid=17)
    model.set_stage("flow")
    state = torch.zeros(1, CH, 17, 17)
    state[:, :4, 5:12, 4:10] = 1
    state[:, 4:, 5:12, 4:10] = torch.randn(1, CH - 4, 7, 6)
    output, flows, assignments, _ = model.step_with_aux(
        state, torch.tensor([2]), torch.tensor([0.5]),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, 1, 17, 17),
    )
    loss = output.square().mean() + flows.square().mean()
    loss.backward()
    assert model.flow_w.grad is not None
    assert torch.isfinite(model.flow_w.grad).all()
    assert model.flow_w.grad.abs().sum() > 0
    assert torch.isfinite(assignments).all()


def test_layered_checkpoint_roundtrip():
    model = LayeredTransportNCA2D(grid=11, slots=3, max_flow=0.7)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "layered.pt"
        save_layered_checkpoint(model, path, iteration=77, stage="flow")
        loaded, iteration, stage = load_layered_checkpoint(path)
    assert iteration == 77 and stage == "flow"
    assert loaded.grid == 11 and loaded.slots == 3 and loaded.max_flow == 0.7
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(actual, expected)


def test_metrics_reject_blurred_silhouette():
    target = torch.zeros(1, 4, 32, 32)
    target[:, :3, 8:24, 8:24] = 1
    target[:, 3:, 8:24, 8:24] = 1
    blurred = target.clone()
    blurred[:, 3:4] = F.avg_pool2d(target[:, 3:4], 7, stride=1, padding=3)
    blurred[:, :3] = blurred[:, 3:4]
    assert alpha_dice(blurred, target) < 0.9
    assert boundary_f1(blurred, target) < 1.0
    assert sharpness_ratio(blurred, target) < 0.8
    assert rgb_sharpness_ratio(blurred, target) < 0.8
    assert rgb_gradient_error(blurred, target) > 0


def test_leakage_separates_adjacent_and_distant_anchors():
    frames = torch.zeros(4, 4, 16, 16)
    for index in range(4):
        frames[index, :, 2 + 3 * index:4 + 3 * index, 3:7] = 1
    edge = torch.tensor([0])
    adjacent = nonadjacent_leakage(frames[1:2], frames, edge)
    distant = nonadjacent_leakage(frames[3:4], frames, edge)
    assert adjacent < 1e-4
    assert distant > 0.99
