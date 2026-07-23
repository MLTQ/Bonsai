"""Contract tests for hidden-only momentum update math and NCA5 layout."""

import os
import struct
import tempfile
import unittest

import torch

from hidden_momentum_nca import (MOMENTUM_CH, POSITION_CH, STATE_CH, VISIBLE_CH,
                                 HiddenMomentumNCA, export_nca5, lift_state,
                                 load_nca5)


class HiddenMomentumContractTests(unittest.TestCase):
    def test_zero_force_decays_only_hidden_velocity(self):
        model = HiddenMomentumNCA(cond=1, hidden=8, momentum_decay=0.75)
        position = torch.zeros(1, POSITION_CH, 3, 3)
        position[:, 3] = 1.0
        velocity = torch.full((1, MOMENTUM_CH, 3, 3), 0.2)
        state = lift_state(position, velocity)

        out = model(state, torch.zeros(1, 1), fire_mask=torch.zeros(1, 1, 3, 3))

        expected_velocity = velocity * 0.75
        self.assertTrue(torch.allclose(out[:, :VISIBLE_CH], position[:, :VISIBLE_CH]))
        self.assertTrue(torch.allclose(out[:, VISIBLE_CH:POSITION_CH],
                                       position[:, VISIBLE_CH:] + expected_velocity))
        self.assertTrue(torch.allclose(out[:, POSITION_CH:], expected_velocity))

    def test_nca5_header_and_payload_size(self):
        model = HiddenMomentumNCA(cond=1, hidden=8, momentum_decay=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.nca")
            export_nca5(model, path)
            with open(path, "rb") as f:
                values = struct.unpack("<4s6iff", f.read(36))
            self.assertEqual(values[:7],
                             (b"NCA5", STATE_CH, 8, 1, POSITION_CH,
                              VISIBLE_CH, MOMENTUM_CH))
            self.assertAlmostEqual(values[7], 0.5)
            self.assertAlmostEqual(values[8], 0.9)
            float_count = 8 * (STATE_CH * 3 + 1) + 8 + POSITION_CH * 8 + POSITION_CH
            self.assertEqual(os.path.getsize(path), 36 + float_count * 4)

    def test_nca5_round_trip_load(self):
        source = HiddenMomentumNCA(cond=1, hidden=8, momentum_decay=0.9)
        torch.manual_seed(17)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.copy_(torch.randn_like(parameter))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "roundtrip.nca")
            export_nca5(source, path)
            restored = HiddenMomentumNCA(cond=1, hidden=8, momentum_decay=0.1)
            load_nca5(restored, path)
        for expected, actual in zip(source.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        self.assertAlmostEqual(restored.fire_rate, source.fire_rate)
        self.assertAlmostEqual(restored.momentum_decay, source.momentum_decay)


if __name__ == "__main__":
    unittest.main()
