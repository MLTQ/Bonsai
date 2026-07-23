"""Contract tests for the 2D global-controller transport experiment."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from prepare_transport2d_corpus import prepare_frames
from train_cyclic import CH, CyclicNCA
from transport_nca2d import (TransportNCA2D, advance_global,
                             load_transport_checkpoint, make_global,
                             save_transport_checkpoint, transplant_nca2,
                             warp_state)


def test_global_ring_closes_after_240_steps():
    phase = torch.tensor([0.17, 2.31])
    initial = make_global(phase)
    final = advance_global(initial, 240)
    torch.testing.assert_close(final, initial, atol=2e-6, rtol=2e-6)


def test_positive_x_flow_moves_impulse_right():
    model = TransportNCA2D(grid=9)
    state = torch.zeros(1, CH, 9, 9)
    state[0, 0, 4, 4] = 1.0
    flow = torch.zeros(1, 2, 9, 9)
    flow[:, 0] = 1.0
    moved = warp_state(state, flow, model.base_grid)
    maximum = torch.nonzero(moved[0, 0] == moved[0, 0].max())[0]
    assert tuple(maximum.tolist()) == (4, 5)


def _donor_step(donor, state, phase, fire_mask):
    pre_life = donor.alive(state)
    perception = F.conv2d(state, donor.percept_w, padding=1, groups=CH)
    condition = torch.stack([phase.sin(), phase.cos(), torch.zeros_like(phase)], dim=1)
    condition_map = condition[:, :, None, None].expand(-1, -1, *state.shape[2:])
    delta = donor.w2(F.relu(donor.w1(torch.cat([perception, condition_map], dim=1))))
    candidate = state + delta * fire_mask
    return candidate * (pre_life & donor.alive(candidate)).to(state.dtype)


def test_zero_flow_transplant_matches_behavior_zero_donor():
    torch.manual_seed(7)
    donor = CyclicNCA()
    torch.nn.init.normal_(donor.w2.weight, std=0.02)
    torch.nn.init.normal_(donor.w2.bias, std=0.01)
    treatment = TransportNCA2D(grid=12)
    transplant_nca2(donor, treatment)
    state = torch.randn(2, CH, 12, 12) * 0.03
    state[:, 3] = 1.0
    phase = torch.tensor([0.31, 3.77])
    fire = (torch.rand(2, 1, 12, 12) > 0.4).float()
    expected = _donor_step(donor, state, phase, fire)
    actual, _, flow, _ = treatment.step_with_aux(state, make_global(phase), fire)
    torch.testing.assert_close(flow, torch.zeros_like(flow), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_tn2d1_checkpoint_roundtrip():
    torch.manual_seed(3)
    model = TransportNCA2D(grid=11, max_flow=0.42)
    with torch.no_grad():
        model.flow_b[2, 0] = 0.7
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "treatment.pt"
        save_transport_checkpoint(model, path, iteration=123)
        loaded, iteration = load_transport_checkpoint(path)
    assert iteration == 123
    assert loaded.grid == 11
    assert loaded.max_flow == 0.42
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(actual, expected)


def test_corpus_preparation_preserves_relative_motion():
    sheet = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(sheet)
    positions = ((30, 30), (150, 30), (35, 160), (155, 160))
    for index, (x, y) in enumerate(positions):
        draw.rectangle((x, y, x + 30, y + 45), fill=(80 + index * 20, 120, 40))
    frames, _ = prepare_frames(sheet, grid=64)
    centroids = []
    for frame in frames:
        ys, xs = np.nonzero(frame[..., 3] > 0.5)
        centroids.append((xs.mean(), ys.mean()))
        assert np.all(frame[..., :3] <= frame[..., 3:4] + 1e-6)
    assert centroids[1][0] < centroids[0][0]
    assert centroids[3][0] < centroids[2][0]
