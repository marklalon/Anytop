import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.animation_utils import (
    crop_animation_to_max_joints,
    select_cropped_joint_indices,
)


def _make_anim(parents, frames=3):
    parents = np.asarray(parents, dtype=np.int32)
    n = len(parents)
    rots = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, n, 1)))
    orients = Quaternions(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)))
    positions = np.arange(frames * n * 3, dtype=np.float64).reshape(frames, n, 3)
    offsets = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    return Animation(rots, positions, orients, offsets, parents)


def test_select_removes_deepest_leaf_first():
    # chain 0-1-2-3-4-5 with a shallow branch 1->6
    parents = [-1, 0, 1, 2, 3, 4, 1]
    keep, removed = select_cropped_joint_indices(parents, max_joints=5)
    assert removed == [5, 4]  # deepest leaves first
    assert keep == [0, 1, 2, 3, 6]


def test_select_tiebreak_prefers_larger_index():
    # two equal-depth leaves under joint 1
    parents = [-1, 0, 1, 1]
    keep, removed = select_cropped_joint_indices(parents, max_joints=3)
    assert removed == [3]
    assert keep == [0, 1, 2]


def test_select_returns_none_when_within_cap():
    assert select_cropped_joint_indices([-1, 0, 1], max_joints=100) is None


def test_select_keeps_set_ancestor_closed():
    # whole deep branch should be peeled before the shallow one
    parents = [-1, 0, 1, 2, 0]  # chain 0-1-2-3 plus 0-4
    keep, _ = select_cropped_joint_indices(parents, max_joints=2)
    assert keep == [0, 1]


def test_crop_animation_subsets_all_arrays_and_remaps_parents():
    parents = [-1, 0, 1, 2, 3, 4, 1]
    anim = _make_anim(parents, frames=3)
    names = [f"j{i}" for i in range(7)]

    cropped, new_names, keep = crop_animation_to_max_joints(anim, names, max_joints=5)

    assert keep == [0, 1, 2, 3, 6]
    assert new_names == ["j0", "j1", "j2", "j3", "j6"]
    assert cropped.parents.tolist() == [-1, 0, 1, 2, 1]
    assert cropped.positions.shape == (3, 5, 3)
    assert cropped.offsets.shape == (5, 3)
    assert len(cropped.orients) == 5
    # cropped parents must be a valid ancestor-closed tree (every parent < child here)
    assert all(p < i for i, p in enumerate(cropped.parents.tolist()) if p >= 0)


def test_crop_animation_is_noop_within_cap():
    parents = [-1, 0, 1]
    anim = _make_anim(parents)
    names = ["a", "b", "c"]
    out_anim, out_names, keep = crop_animation_to_max_joints(anim, names, max_joints=100)
    assert keep is None
    assert out_anim is anim
    assert out_names == names


def test_crop_rejects_name_count_mismatch():
    anim = _make_anim([-1, 0, 1])
    with pytest.raises(ValueError):
        crop_animation_to_max_joints(anim, ["only", "two"], max_joints=2)


def test_rest_pose_and_motion_crop_to_identical_set():
    # Same topology loaded independently for rest pose vs. a motion clip must crop
    # to the same joint set so the offset-count guard in get_hml_aligned_anim holds.
    parents = [-1, 0, 1, 2, 3, 4, 1, 2]
    rest = _make_anim(parents, frames=1)
    motion = _make_anim(parents, frames=10)
    names = [f"j{i}" for i in range(len(parents))]

    _, _, keep_rest = crop_animation_to_max_joints(rest, names, max_joints=6)
    _, _, keep_motion = crop_animation_to_max_joints(motion, names, max_joints=6)
    assert keep_rest == keep_motion
