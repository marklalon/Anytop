from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import _compute_masked_teacher_losses  # noqa: E402


class _FakePhysicsTeacher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compute_losses(self, pred_motion, target_motion, *, n_joints, lengths, object_types):
        self.calls.append(
            {
                "pred_shape": tuple(pred_motion.shape),
                "target_shape": tuple(target_motion.shape),
                "n_joints": n_joints.detach().cpu().tolist(),
                "lengths": lengths.detach().cpu().tolist(),
                "object_types": list(object_types),
            }
        )
        batch_size = int(pred_motion.shape[0])
        base = torch.arange(1, batch_size + 1, device=pred_motion.device, dtype=pred_motion.dtype)
        return {
            "physics_teacher_feature_loss": base,
            "physics_teacher_margin_loss": base + 10.0,
            "physics_teacher_distance": base + 20.0,
            "physics_teacher_target_distance": base + 30.0,
            "physics_teacher_score": base + 40.0,
            "physics_teacher_target_score": base + 50.0,
        }


class GaussianDiffusionPhysicsTeacherTests(unittest.TestCase):
    def test_compute_masked_teacher_losses_only_uses_active_subset(self):
        teacher = _FakePhysicsTeacher()
        pred_motion = torch.randn(4, 3, 13, 8)
        target_motion = torch.randn(4, 3, 13, 8)
        n_joints = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        lengths = torch.tensor([8, 7, 6, 5], dtype=torch.long)
        object_types = ["A", "B", "C", "D"]
        active_mask = torch.tensor([True, False, True, False])

        losses = _compute_masked_teacher_losses(
            teacher,
            pred_motion,
            target_motion,
            n_joints=n_joints,
            lengths=lengths,
            object_types=object_types,
            active_mask=active_mask,
        )

        self.assertIsNotNone(losses)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(teacher.calls[0]["pred_shape"], (2, 3, 13, 8))
        self.assertEqual(teacher.calls[0]["target_shape"], (2, 3, 13, 8))
        self.assertEqual(teacher.calls[0]["n_joints"], [3, 5])
        self.assertEqual(teacher.calls[0]["lengths"], [8, 6])
        self.assertEqual(teacher.calls[0]["object_types"], ["A", "C"])
        self.assertTrue(torch.equal(losses["physics_teacher_feature_loss"], torch.tensor([1.0, 0.0, 2.0, 0.0])))
        self.assertTrue(torch.equal(losses["physics_teacher_margin_loss"], torch.tensor([11.0, 0.0, 12.0, 0.0])))

    def test_compute_masked_teacher_losses_skips_empty_mask(self):
        teacher = _FakePhysicsTeacher()
        pred_motion = torch.randn(4, 3, 13, 8)
        target_motion = torch.randn(4, 3, 13, 8)
        n_joints = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        lengths = torch.tensor([8, 7, 6, 5], dtype=torch.long)
        object_types = ["A", "B", "C", "D"]
        active_mask = torch.zeros(4, dtype=torch.bool)

        losses = _compute_masked_teacher_losses(
            teacher,
            pred_motion,
            target_motion,
            n_joints=n_joints,
            lengths=lengths,
            object_types=object_types,
            active_mask=active_mask,
        )

        self.assertIsNone(losses)
        self.assertEqual(len(teacher.calls), 0)


if __name__ == "__main__":
    unittest.main()