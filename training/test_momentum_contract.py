"""Focused contract tests for phase-space update math and NCA4 layout."""

import os
import struct
import tempfile
import unittest

import torch

from momentum_nca import POSITION_CH, STATE_CH, MomentumNCA, export_nca4, lift_state


class MomentumContractTests(unittest.TestCase):
    def test_zero_force_decays_velocity_then_advances_position(self):
        model = MomentumNCA(cond=1, hidden=8, momentum_decay=0.75)
        position = torch.zeros(1, POSITION_CH, 3, 3)
        position[:, 3] = 1.0  # keep every cell alive
        velocity = torch.full_like(position, 0.2)
        velocity[:, 3] = 0.0  # keep alpha stable for the life gate
        state = lift_state(position, velocity)

        out = model(state, torch.zeros(1, 1), fire_mask=torch.zeros(1, 1, 3, 3))

        expected_velocity = velocity * 0.75
        self.assertTrue(torch.allclose(out[:, POSITION_CH:], expected_velocity))
        self.assertTrue(torch.allclose(out[:, :POSITION_CH], position + expected_velocity))

    def test_nca4_header_and_payload_size(self):
        model = MomentumNCA(cond=1, hidden=8, momentum_decay=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.nca")
            export_nca4(model, path)
            with open(path, "rb") as f:
                magic, state, hidden, cond, position, fire, decay = struct.unpack(
                    "<4s4iff", f.read(28)
                )
            self.assertEqual((magic, state, hidden, cond, position),
                             (b"NCA4", STATE_CH, 8, 1, POSITION_CH))
            self.assertAlmostEqual(fire, 0.5)
            self.assertAlmostEqual(decay, 0.9)
            float_count = hidden * (state * 3 + cond) + hidden + position * hidden + position
            self.assertEqual(os.path.getsize(path), 28 + float_count * 4)


if __name__ == "__main__":
    unittest.main()
