from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import (  # noqa: E402
    _compute_masked_semantic_teacher_losses,
    _normalize_teacher_batch_term,
)


class _FakeSemanticTeacher:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def compute_losses(self, pred_motion, target_motion, *, n_joints, lengths, species_labels, action_labels, temperature):
        self.calls.append(
            {
                "pred_shape": tuple(pred_motion.shape),
                "target_shape": tuple(target_motion.shape),
                "n_joints": n_joints.detach().cpu().tolist(),
                "lengths": lengths.detach().cpu().tolist(),
                "species_labels": list(species_labels),
                "action_labels": list(action_labels),
                "temperature": float(temperature),
            }
        )
        batch_size = int(pred_motion.shape[0])
        base = torch.arange(1, batch_size + 1, device=pred_motion.device, dtype=pred_motion.dtype)
        return {
            "semantic_teacher_species_ce": base,
            "semantic_teacher_action_ce": base + 10.0,
            "semantic_teacher_species_kl": base + 20.0,
            "semantic_teacher_action_kl": base + 30.0,
            "semantic_teacher_recognizability": base + 40.0,
            "semantic_teacher_target_recognizability": base + 50.0,
        }


class GaussianDiffusionSemanticTeacherTests(unittest.TestCase):
    def test_compute_masked_semantic_teacher_losses_only_uses_active_subset(self):
        teacher = _FakeSemanticTeacher()
        pred_motion = torch.randn(4, 3, 13, 8)
        target_motion = torch.randn(4, 3, 13, 8)
        n_joints = torch.tensor([3, 4, 5, 6], dtype=torch.long)
        lengths = torch.tensor([8, 7, 6, 5], dtype=torch.long)
        species_labels = ["wolf", "cat", "dog", "horse"]
        action_labels = ["run", "idle", "jump", "walk"]
        active_indices = torch.tensor([0, 2], dtype=torch.long)

        losses = _compute_masked_semantic_teacher_losses(
            teacher,
            pred_motion,
            target_motion,
            n_joints=n_joints,
            lengths=lengths,
            species_labels=species_labels,
            action_labels=action_labels,
            temperature=1.5,
            active_indices=active_indices,
        )

        self.assertIsNotNone(losses)
        self.assertEqual(len(teacher.calls), 1)
        self.assertEqual(teacher.calls[0]["pred_shape"], (2, 3, 13, 8))
        self.assertEqual(teacher.calls[0]["target_shape"], (2, 3, 13, 8))
        self.assertEqual(teacher.calls[0]["n_joints"], [3, 5])
        self.assertEqual(teacher.calls[0]["lengths"], [8, 6])
        self.assertEqual(teacher.calls[0]["species_labels"], ["wolf", "dog"])
        self.assertEqual(teacher.calls[0]["action_labels"], ["run", "jump"])
        self.assertAlmostEqual(teacher.calls[0]["temperature"], 1.5)
        self.assertTrue(torch.equal(losses["semantic_teacher_species_ce"], torch.tensor([1.0, 0.0, 2.0, 0.0])))
        self.assertTrue(torch.equal(losses["semantic_teacher_target_recognizability"], torch.tensor([51.0, 0.0, 52.0, 0.0])))

    def test_normalize_teacher_batch_term_uses_only_active_mean(self):
        values = torch.tensor([3.0, 0.0, 7.0, 0.0])
        active_mask = torch.tensor([True, False, True, False])

        normalized = _normalize_teacher_batch_term(values, active_mask)

        self.assertTrue(torch.equal(normalized, torch.tensor([5.0, 5.0, 5.0, 5.0])))
        self.assertAlmostEqual(float(normalized.mean().item()), 5.0)


if __name__ == "__main__":
    unittest.main()