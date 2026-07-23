"""CPU-fast contracts for global-ring state, transport axes, and donor parity."""

import math
import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from train_cyclic3d import CH, COND, FIRE_RATE, HIDDEN, CyclicNCA3D
from transport_nca3d import (EDGE_COUNT, TransportNCA3D, advance_global,
                             load_transport_checkpoint, make_global,
                             phase_features, save_transport_checkpoint,
                             transplant_nc3c, warp_state)


class TransportNCA3DContracts(unittest.TestCase):
    def test_global_ring_and_edge_order(self):
        phases = torch.arange(EDGE_COUNT, dtype=torch.float32) * (2 * math.pi / EDGE_COUNT)
        global_state = make_global(phases)
        _, edge = phase_features(global_state)
        self.assertTrue(torch.equal(edge, torch.arange(EDGE_COUNT)))

        start = make_global(torch.tensor([0.37]))
        state = start
        for _ in range(240):
            state = advance_global(state)
        self.assertTrue(torch.allclose(start, state, atol=2e-5, rtol=0))

    def test_positive_x_flow_moves_impulse_positive_x(self):
        model = TransportNCA3D(grid=5)
        state = torch.zeros(1, 1, 5, 5, 5)
        state[0, 0, 2, 2, 2] = 1.0
        flow = torch.zeros(1, 3, 5, 5, 5)
        flow[:, 0] = 1.0
        warped = warp_state(state, flow, model.base_grid)
        peak = torch.nonzero(warped[0, 0] == warped.max(), as_tuple=False)[0]
        self.assertEqual(tuple(peak.tolist()), (2, 2, 3))

    def test_zero_flow_transplant_matches_walking_donor_step(self):
        torch.manual_seed(5)
        donor = CyclicNCA3D()
        with torch.no_grad():
            donor.w2.weight.normal_(std=0.03)
            donor.w2.bias.normal_(std=0.01)
        treatment = TransportNCA3D(grid=5)
        transplant_nc3c(donor, treatment)

        state = torch.randn(2, CH, 5, 5, 5) * 0.08
        state[:, 3] = 1.0
        phases = torch.tensor([0.31, 2.2])
        global_state = make_global(phases)
        fire = torch.ones(2, 1, 5, 5, 5)

        pre_life = donor.alive(state)
        perception = F.conv3d(state, donor.percept_w, padding=1, groups=CH)
        cond = torch.stack([phases.sin(), phases.cos(), torch.ones_like(phases)], dim=1)
        cond_map = cond[:, :, None, None, None].expand(-1, -1, 5, 5, 5)
        delta = donor.w2(F.relu(donor.w1(torch.cat([perception, cond_map], dim=1))))
        expected = state + delta * fire
        expected = (expected * (pre_life & donor.alive(expected))).clamp(-8.0, 8.0)

        actual, _ = treatment(state, global_state, fire_mask=fire)
        self.assertTrue(torch.allclose(expected, actual, atol=2e-5, rtol=1e-5))

    def test_checkpoint_round_trip(self):
        model = TransportNCA3D(grid=5)
        torch.manual_seed(9)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "transport.pt")
            save_transport_checkpoint(model, path, iteration=37)
            restored, iteration = load_transport_checkpoint(path)
        self.assertEqual(iteration, 37)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
