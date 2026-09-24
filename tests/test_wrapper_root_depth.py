"""Rest-pose-only wrapper folding: the loader rule re-applied after the drops."""
import numpy as np

from motion_lib.root_collapse import wrapper_root_depth
from utils.npy_restore import _remap_skeleton_metadata


def test_zero_offset_single_child_wrapper_is_counted():
    names = ["Dummy_Root", "Bip001", "Bip001-Pelvis"]
    parents = np.array([-1, 0, 1])
    offsets = np.array([[0, 0, 0], [0, 0, 0.2], [0, 0, 0]], dtype=float)
    assert wrapper_root_depth(names, parents, offsets) == 1


def test_wrapper_with_other_children_is_kept():
    names = ["Dummy_Root", "Bip001", "Point_stick"]
    parents = np.array([-1, 0, 0])
    offsets = np.zeros((3, 3))
    assert wrapper_root_depth(names, parents, offsets) == 0


def test_semantic_or_offset_root_is_kept():
    parents = np.array([-1, 0, 1])
    offsets = np.array([[0, 0, 0], [0, 0, 0.2], [0, 0.1, 0]], dtype=float)
    assert wrapper_root_depth(["Hips", "Spine", "Neck"], parents, offsets) == 0
    moved = offsets.copy()
    moved[0] = [0, 1, 0]
    assert wrapper_root_depth(["Dummy_Root", "Spine", "Neck"], parents, moved) == 0


def test_remap_folds_dropped_wrapper_into_new_root():
    # Wrapper rotated 90 deg about X; the child's world rest transform must survive.
    half = np.sqrt(0.5)
    source_names = ["Dummy_Root", "Bip001", "Pelvis"]
    source_parents = np.array([-1, 0, 1], dtype=np.int32)
    source_offsets = np.array([[0, 0, 0], [0, 0, 0.2], [0.1, 0, 0]], dtype=np.float32)
    source_rot = np.array([[half, half, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    parents, offsets, rots = _remap_skeleton_metadata(
        source_names, source_parents, source_offsets, source_rot, ["Bip001", "Pelvis"],
    )
    assert parents.tolist() == [-1, 0]
    np.testing.assert_allclose(offsets[0], [0, -0.2, 0], atol=1e-6)
    np.testing.assert_allclose(rots[0], [half, half, 0, 0], atol=1e-6)
    np.testing.assert_allclose(offsets[1], source_offsets[2])
