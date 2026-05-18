from __future__ import annotations

import os
import sys
import tempfile

import numpy as np


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from motion_lib.Animation import Animation
from motion_lib import BVH
from motion_lib.Quaternions import Quaternions
from motion_lib.Animation import positions_global
from data_loaders.truebones.truebones_utils.motion_process import (
    append_leaf_rotation_helpers_to_animation,
    build_leaf_rotation_helper_metadata,
    get_bvh_cont6d_params,
    get_motion_features,
    get_rifke,
    needs_bvh_position_channels,
    resolve_mirrored_export_skeleton_metadata,
    reorder_animation_to_dfs,
)
from data_loaders.truebones.truebones_utils.features import to_parent_local_pos_residual
from tools.restore_glb_from_npy import (
    _bare_feature_rotation_channel_mask,
    _strip_appended_helper_joints,
)
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np


def test_leaf_rotation_helper_budget_uses_dfs_leaf_order() -> None:
    joint_names = ["Root", "LeafA", "LeafB", "LeafC", "LeafD"]
    parents = np.array([-1, 0, 0, 0, 0], dtype=np.int32)

    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=len(joint_names) + 2,
    )

    assert helper_metadata["original_leaf_joint_indices"] == [1, 2, 3, 4]
    assert helper_metadata["helper_source_leaf_indices"] == [1, 2]
    assert helper_metadata["unaugmented_leaf_indices"] == [3, 4]
    assert helper_metadata["helper_joint_indices"] == [5, 6]
    assert helper_metadata["helper_joint_count"] == 2


def test_leaf_rotation_helper_budget_prefers_complete_bilateral_pairs() -> None:
    joint_names = ["Root", "CenterLeaf", "LeftFrontFoot", "RightFrontFoot"]
    parents = np.array([-1, 0, 0, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=len(joint_names) + 2,
        offsets=offsets,
    )

    assert helper_metadata["original_leaf_joint_indices"] == [1, 2, 3]
    assert helper_metadata["helper_source_leaf_indices"] == [2, 3]
    assert helper_metadata["unaugmented_leaf_indices"] == [1]
    assert helper_metadata["helper_joint_indices"] == [4, 5]
    assert helper_metadata["helper_joint_count"] == 2



def test_append_leaf_rotation_helpers_appends_helpers_at_end() -> None:
    joint_names = ["Root", "Joint", "Leaf"]
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], 2, axis=0)
    positions[:, 0, :] = 0.0
    original_anim = Animation(
        Quaternions.id((2, 3)),
        positions,
        Quaternions.id(3),
        offsets,
        parents,
    )
    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=4,
    )

    augmented_anim, augmented_names = append_leaf_rotation_helpers_to_animation(
        original_anim,
        joint_names,
        helper_metadata,
    )

    assert augmented_anim.shape == (2, 4)
    assert augmented_names[:3] == joint_names
    assert augmented_names[3].startswith("Leaf__rot_helper_")
    assert np.array_equal(augmented_anim.parents, np.array([-1, 0, 1, 2], dtype=np.int32))
    assert np.allclose(augmented_anim.offsets[3], np.zeros(3, dtype=np.float64))
    assert np.allclose(augmented_anim.positions[:, 3, :], 0.0)


def test_mirrored_export_metadata_reparents_unpaired_helper_to_mirrored_leaf() -> None:
    object_cond = {
        "parents": np.array([-1, 0, 0, 1], dtype=np.int32),
        "offsets": np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "symmetry_partner_indices": [-1, 2, 1, -1],
        "helper_joint_indices": [3],
        "helper_source_leaf_indices": [1],
    }

    parents, offsets, joint_names = resolve_mirrored_export_skeleton_metadata(
        object_cond,
        object_cond["parents"],
        object_cond["offsets"],
        ["Root", "LeftToe", "RightToe", "LeftToeHelper"],
    )

    assert np.array_equal(parents, np.array([-1, 0, 0, 2], dtype=np.int32))
    assert np.allclose(offsets[3], np.zeros(3, dtype=np.float64))
    assert joint_names == ["Root", "LeftToe", "RightToe", "RightToeHelper"]


def test_mirrored_export_metadata_keeps_mirror_disabled_helper_on_original_side() -> None:
    object_cond = {
        "parents": np.array([-1, 0, 0, 2], dtype=np.int32),
        "offsets": np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "symmetry_partner_indices": [-1, 2, 1, -1],
        "helper_joint_indices": [3],
        "helper_source_leaf_indices": [2],
        "mirror_disabled_joint_indices": [3],
    }

    parents, offsets, joint_names = resolve_mirrored_export_skeleton_metadata(
        object_cond,
        object_cond["parents"],
        object_cond["offsets"],
        ["Root", "LeftToe", "RightToe", "RightToeHelper"],
    )

    assert np.array_equal(parents, np.array([-1, 0, 0, 2], dtype=np.int32))
    assert np.allclose(offsets, object_cond["offsets"])
    assert joint_names == ["Root", "LeftToe", "RightToe", "RightToeHelper"]


def test_reorder_animation_to_dfs_preserves_helper_augmented_bvh_roundtrip() -> None:
    joint_names = ["Root", "Branch", "Leaf", "Sibling"]
    parents = np.array([-1, 0, 1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], 2, axis=0)
    positions[:, 0, :] = 0.0
    rotations = Quaternions.id((2, 4))
    rotations[:, 1] = Quaternions.from_angle_axis(
        np.array([np.pi / 6.0, np.pi / 5.0], dtype=np.float64),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )
    rotations[:, 2] = Quaternions.from_angle_axis(
        np.array([np.pi / 7.0, np.pi / 8.0], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
    )
    original_anim = Animation(
        rotations,
        positions,
        Quaternions.id(4),
        offsets,
        parents,
    )
    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=5,
    )
    augmented_anim, augmented_names = append_leaf_rotation_helpers_to_animation(
        original_anim,
        joint_names,
        helper_metadata,
    )

    reordered_anim, reordered_names = reorder_animation_to_dfs(augmented_anim, augmented_names)

    helper_name = helper_metadata["helper_joint_names"][0]
    assert reordered_names == ["Root", "Branch", "Leaf", helper_name, "Sibling"]

    with tempfile.TemporaryDirectory() as temp_dir:
        bvh_path = os.path.join(temp_dir, "helper_preview.bvh")
        BVH.save(
            bvh_path,
            reordered_anim,
            reordered_names,
            positions=needs_bvh_position_channels(reordered_anim),
        )
        loaded_anim, loaded_names, _frame_time = BVH.load(bvh_path)

    base_positions = positions_global(reordered_anim)
    loaded_positions = positions_global(loaded_anim)
    base_index = {name: index for index, name in enumerate(reordered_names)}
    loaded_index = {name: index for index, name in enumerate(loaded_names)}
    common_names = [name for name in reordered_names if name in loaded_index]

    assert common_names == reordered_names

    max_position_error = 0.0
    for name in common_names:
        base_joint_index = base_index[name]
        loaded_joint_index = loaded_index[name]
        joint_error = float(
            np.max(
                np.linalg.norm(
                    base_positions[:, base_joint_index] - loaded_positions[:, loaded_joint_index],
                    axis=-1,
                )
            )
        )
        max_position_error = max(max_position_error, joint_error)

    assert max_position_error < 1e-5


def test_reorder_animation_to_dfs_is_idempotent() -> None:
    joint_names = ["Root", "Branch", "Leaf", "Sibling"]
    parents = np.array([-1, 0, 1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], 2, axis=0)
    positions[:, 0, :] = 0.0
    original_anim = Animation(
        Quaternions.id((2, 4)),
        positions,
        Quaternions.id(4),
        offsets,
        parents,
    )
    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=5,
    )
    augmented_anim, augmented_names = append_leaf_rotation_helpers_to_animation(
        original_anim,
        joint_names,
        helper_metadata,
    )

    reordered_once, names_once = reorder_animation_to_dfs(augmented_anim, augmented_names)
    reordered_twice, names_twice = reorder_animation_to_dfs(reordered_once, names_once)

    assert names_once == names_twice
    assert np.array_equal(reordered_once.parents, reordered_twice.parents)
    assert np.allclose(reordered_once.offsets, reordered_twice.offsets)
    assert np.allclose(reordered_once.positions, reordered_twice.positions)
    assert np.allclose(reordered_once.rotations.qs, reordered_twice.rotations.qs)



def test_helper_covered_leaf_rotation_roundtrips_through_bare_features() -> None:
    joint_names = ["Root", "Joint", "Leaf"]
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], 2, axis=0)
    positions[:, 0, :] = 0.0
    rotations = Quaternions.id((2, 3))
    leaf_rotation = Quaternions.from_angle_axis(
        np.array([np.pi / 4.0, np.pi / 4.0], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
    )
    rotations[:, 2] = leaf_rotation
    original_anim = Animation(
        rotations,
        positions,
        Quaternions.id(3),
        offsets,
        parents,
    )
    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=4,
    )
    augmented_anim, _augmented_names = append_leaf_rotation_helpers_to_animation(
        original_anim,
        joint_names,
        helper_metadata,
    )

    cont_6d_params, _r_velocity, _velocity, r_rot, global_positions = get_bvh_cont6d_params(
        augmented_anim,
        "TestCreature",
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        translation_root_index=0,
    )
    ric_positions = get_rifke(global_positions, r_rot, translation_root_index=0)
    residual_positions = to_parent_local_pos_residual(ric_positions, augmented_anim, translation_root_index=0)
    local_vel = np.repeat(r_rot[1:, None], global_positions.shape[1], axis=1) * (
        global_positions[1:] - global_positions[:-1]
    )
    features, _max_joints = get_motion_features(
        residual_positions,
        cont_6d_params,
        np.zeros((1, augmented_anim.shape[1]), dtype=np.float64),
        local_vel,
        np.zeros((augmented_anim.shape[1], 3), dtype=np.float64),
        np.zeros((augmented_anim.shape[1],), dtype=np.float64),
        max_joints=augmented_anim.shape[1],
    )

    recovered_anim, _has_animated_pos = recover_animation_from_motion_np(
        features,
        augmented_anim.parents,
        augmented_anim.offsets,
        translation_root_index=0,
    )

    assert np.allclose(
        recovered_anim.rotations[:, 2].rotation_matrix(),
        augmented_anim.rotations[:, 2].rotation_matrix(),
        atol=1e-6,
    )



def test_strip_appended_helper_joints_restores_original_joint_count() -> None:
    joint_names = ["Root", "Joint", "Leaf"]
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions = np.repeat(offsets[None, :, :], 2, axis=0)
    positions[:, 0, :] = 0.0
    original_anim = Animation(
        Quaternions.id((2, 3)),
        positions,
        Quaternions.id(3),
        offsets,
        parents,
    )
    helper_metadata = build_leaf_rotation_helper_metadata(
        joint_names,
        parents,
        max_joints=4,
    )
    augmented_anim, _augmented_names = append_leaf_rotation_helpers_to_animation(
        original_anim,
        joint_names,
        helper_metadata,
    )

    stripped_anim = _strip_appended_helper_joints(
        augmented_anim,
        original_joint_count=3,
    )

    assert stripped_anim.shape == original_anim.shape
    assert np.array_equal(stripped_anim.parents, parents)
    assert np.allclose(stripped_anim.offsets, offsets)



