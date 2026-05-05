import os
import sys
import unittest

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from tools.compare_motions import MotionData, _detect_and_align


def _build_motion(world_positions: np.ndarray) -> MotionData:
    num_frames, num_joints = world_positions.shape[:2]
    identity_rotations = np.zeros((num_frames, num_joints, 4), dtype=np.float64)
    identity_rotations[..., 0] = 1.0
    return MotionData(
        file_path="synthetic",
        file_format="npy",
        bone_names=["root", "child"],
        parents=np.array([-1, 0], dtype=np.int32),
        offsets=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        world_positions=world_positions,
        world_rotations=identity_rotations,
        sample_frames=[0.0, 1.0],
        sample_times=[0.0, 1.0 / 30.0],
        fps=30.0,
        num_frames=num_frames,
        num_joints=num_joints,
    )


class CompareMotionsReflectionAlignmentTest(unittest.TestCase):
    def test_detect_and_align_handles_reflection_without_quaternion_conversion(self):
        motion_a = _build_motion(
            np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
                    [[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]],
                ],
                dtype=np.float64,
            )
        )
        motion_b = _build_motion(
            np.array(
                [
                    [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
                    [[0.0, 0.0, 0.0], [1.0, -2.0, 4.0]],
                ],
                dtype=np.float64,
            )
        )

        motion_b_aligned, alignment = _detect_and_align(motion_a, motion_b)

        self.assertEqual(alignment.rotation_label, "flip_Y")
        np.testing.assert_allclose(alignment.rotation_matrix, np.diag([1.0, -1.0, 1.0]))
        np.testing.assert_allclose(motion_b_aligned.world_positions, motion_a.world_positions)
        np.testing.assert_allclose(motion_b_aligned.world_rotations, motion_b.world_rotations)


if __name__ == "__main__":
    unittest.main()
