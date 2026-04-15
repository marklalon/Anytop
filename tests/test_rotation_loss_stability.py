from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.losses import geodesic_distance  # noqa: E402
from utils.rotation_conversions import rotation_6d_to_matrix_safe  # noqa: E402


class RotationLossStabilityTests(unittest.TestCase):
    def test_geodesic_distance_matches_known_angle(self):
        theta = math.pi / 2.0
        rot_z = torch.tensor(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 3, 3)
        identity = torch.eye(3, dtype=torch.float32).view(1, 3, 3)

        distance = geodesic_distance(rot_z, identity)

        self.assertAlmostEqual(float(distance.item()), theta, places=5)

    def test_geodesic_distance_backward_is_finite_near_identity(self):
        pred_6d = torch.tensor(
            [[1.0, 1e-5, 0.0, 0.0, 1.0, 1e-5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target_6d = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )

        pred_rot = rotation_6d_to_matrix_safe(pred_6d)
        target_rot = rotation_6d_to_matrix_safe(target_6d)
        loss = geodesic_distance(pred_rot, target_rot).sum()
        loss.backward()

        self.assertTrue(torch.isfinite(pred_6d.grad).all().item())
        self.assertLess(float(pred_6d.grad.abs().max().item()), 1e6)

    def test_rotation_6d_safe_backward_is_finite_for_near_collinear_axes(self):
        cont6d = torch.tensor(
            [[1.0, 0.0, 0.0, 1.0, 1e-8, 0.0]],
            dtype=torch.float32,
            requires_grad=True,
        )

        rotation = rotation_6d_to_matrix_safe(cont6d)
        loss = rotation.square().sum()
        loss.backward()

        self.assertTrue(torch.isfinite(cont6d.grad).all().item())
        self.assertLess(float(cont6d.grad.abs().max().item()), 1e6)


if __name__ == "__main__":
    unittest.main()