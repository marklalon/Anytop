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

    def test_zero_and_tiny_axes_are_proper_rotations_with_true_gradients(self):
        for scale in (0., 1e-8, 1e-5):
            raw = torch.full((2, 6), scale, dtype=torch.float64, requires_grad=True)
            rotation = rotation_6d_to_matrix_safe(raw)
            torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                                       torch.eye(3, dtype=raw.dtype).expand(2, -1, -1))
            torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2, dtype=raw.dtype))
            # Finite differences verify the real derivative, including at zero.
            self.assertTrue(torch.autograd.gradcheck(rotation_6d_to_matrix_safe, (raw,)))
        raw = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        rotation = rotation_6d_to_matrix_safe(raw)
        torch.testing.assert_close(rotation, torch.eye(3, dtype=raw.dtype)[None])
        # Each independent tangent direction must have a finite nonzero gradient.
        for row, col in ((0, 1), (0, 2), (1, 2)):
            grad = torch.autograd.grad(rotation[0, row, col], raw, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().max().item(), 0.)

    def test_regular_axes_keep_gram_schmidt_result_and_gradient(self):
        raw = torch.tensor([[1., .2, -.3, -.1, 1., .4]], dtype=torch.float64, requires_grad=True)
        x = torch.nn.functional.normalize(raw[:, :3], dim=-1)
        y = raw[:, 3:] - (x * raw[:, 3:]).sum(-1, keepdim=True) * x
        y = torch.nn.functional.normalize(y, dim=-1)
        z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1), dim=-1)
        expected = torch.stack((x, y, z), dim=-1)
        actual = rotation_6d_to_matrix_safe(raw)
        torch.testing.assert_close(actual, expected)
        weights = torch.arange(9, dtype=raw.dtype).reshape(1, 3, 3)
        expected_grad = torch.autograd.grad((expected * weights).sum(), raw, retain_graph=True)[0]
        actual_grad = torch.autograd.grad((actual * weights).sum(), raw)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

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