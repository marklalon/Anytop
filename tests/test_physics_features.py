from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.skeleton_metadata import SkeletonMetadata  # noqa: E402
from eval.physics_features import PHYSICS_FEATURE_DIM, _INDEX_TENSOR_CACHE, extract_physics_features  # noqa: E402


def _build_metadata() -> dict[str, SkeletonMetadata]:
    metadata = SkeletonMetadata(
        object_type="TestRig",
        parents=(-1, 0, 0),
        end_effector_joints=(1, 2),
        contact_joints=(1, 2),
        symmetry_partner_indices=(-1, 2, 1),
        symmetric_joint_pairs=((1, 2),),
        is_symmetric=True,
        n_joints=3,
        joint_depths=(0, 1, 1),
        edge_child_indices=(1, 2),
        edge_parent_indices=(0, 0),
        symmetry_left_indices=(1,),
        symmetry_right_indices=(2,),
        subtree_indices=((0, 1, 2), (1,), (2,)),
        max_joint_depth=1,
    )
    return {metadata.object_type: metadata}


def _build_motion() -> torch.Tensor:
    motion = torch.zeros((1, 3, 13, 4), dtype=torch.float32)
    root_positions = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    left_positions = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    right_positions = torch.tensor(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    motion[0, 0, :3, :] = root_positions
    motion[0, 1, :3, :] = left_positions
    motion[0, 2, :3, :] = right_positions
    motion[0, 0, 3:9, :] = torch.tensor([[1.0], [0.0], [0.0], [0.0], [1.0], [0.0]], dtype=torch.float32)
    motion[0, :, 9:12, :] = 0.05
    motion[0, 1:, 12, :] = 1.0
    return motion


class PhysicsFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        _INDEX_TENSOR_CACHE.clear()

    def test_differentiable_path_accepts_inference_tensors(self):
        metadata_lookup = _build_metadata()
        n_joints = torch.tensor([3], dtype=torch.long)
        lengths = torch.tensor([4], dtype=torch.long)
        with torch.inference_mode():
            motion = _build_motion()

        features = extract_physics_features(
            motion,
            n_joints,
            lengths,
            ["TestRig"],
            metadata_lookup,
            differentiable=True,
        )

        self.assertEqual(tuple(features.shape), (1, PHYSICS_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all().item())
        self.assertFalse(features.is_inference())

    def test_differentiable_path_accepts_outer_inference_context(self):
        metadata_lookup = _build_metadata()
        n_joints = torch.tensor([3], dtype=torch.long)
        lengths = torch.tensor([4], dtype=torch.long)
        motion = _build_motion()

        with torch.inference_mode():
            features = extract_physics_features(
                motion,
                n_joints,
                lengths,
                ["TestRig"],
                metadata_lookup,
                differentiable=True,
            )

        self.assertEqual(tuple(features.shape), (1, PHYSICS_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all().item())
        self.assertFalse(features.is_inference())

    def test_differentiable_path_works_after_inference_cache_population(self):
        metadata_lookup = _build_metadata()
        n_joints = torch.tensor([3], dtype=torch.long)
        lengths = torch.tensor([4], dtype=torch.long)
        motion = _build_motion()

        with torch.inference_mode():
            cached_features = extract_physics_features(
                motion,
                n_joints,
                lengths,
                ["TestRig"],
                metadata_lookup,
                differentiable=False,
            )

        features = extract_physics_features(
            motion,
            n_joints,
            lengths,
            ["TestRig"],
            metadata_lookup,
            differentiable=True,
        )

        self.assertEqual(tuple(cached_features.shape), (1, PHYSICS_FEATURE_DIM))
        self.assertEqual(tuple(features.shape), (1, PHYSICS_FEATURE_DIM))
        self.assertTrue(torch.isfinite(features).all().item())

    def test_differentiable_path_still_backprops_for_normal_tensors(self):
        metadata_lookup = _build_metadata()
        motion = _build_motion().requires_grad_(True)
        n_joints = torch.tensor([3], dtype=torch.long)
        lengths = torch.tensor([4], dtype=torch.long)

        features = extract_physics_features(
            motion,
            n_joints,
            lengths,
            ["TestRig"],
            metadata_lookup,
            differentiable=True,
        )
        loss = features.sum()
        loss.backward()

        self.assertIsNotNone(motion.grad)
        self.assertEqual(tuple(motion.grad.shape), tuple(motion.shape))


if __name__ == "__main__":
    unittest.main()