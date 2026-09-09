from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions
from utils.fullbody_ik import (
    rebuild_fullbody_animation_with_ik,
    rebuild_retarget_pose_channels_with_ik,
)
from utils.exporter import animation_to_exporter_inputs
from utils.roundtrip_common import build_skeleton


def test_fullbody_ik_rebuild_restores_rigid_local_positions() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    source_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rigid_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = np.array(
        [
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    target_anim = Animation(
        Quaternions.id((1, 3)),
        local_positions,
        Quaternions.id(0),
        source_offsets,
        parents,
    )

    rebuilt_anim, mean_error, max_error = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=parents,
        iterations=10,
    )

    assert np.allclose(rebuilt_anim.positions[:, 1:, :], rigid_offsets[1:][None, :, :])
    assert np.allclose(rebuilt_anim.positions[:, 0, :], target_anim.positions[:, 0, :])
    assert mean_error == pytest.approx(0.0, abs=1e-5)
    assert max_error == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(
        positions_global(rebuilt_anim),
        positions_global(target_anim),
        atol=1e-5,
    )


def test_fullbody_ik_rebuild_requires_single_root() -> None:
    target_anim = Animation(
        Quaternions.id((1, 2)),
        np.zeros((1, 2, 3), dtype=np.float64),
        Quaternions.id(0),
        np.zeros((2, 3), dtype=np.float64),
        np.array([-1, -1], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Expected exactly one root joint"):
        rebuild_fullbody_animation_with_ik(
            target_anim,
            rigid_offsets=np.zeros((2, 3), dtype=np.float64),
            rigid_parents=np.array([-1, -1], dtype=np.int32),
        )


def test_fullbody_ik_rebuild_zeroes_exporter_nonroot_translations() -> None:
    joint_names = ["Root", "JointA", "JointB"]
    parents = np.array([-1, 0, 1], dtype=np.int32)
    source_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rigid_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = np.array(
        [
            [
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    target_anim = Animation(
        Quaternions.id((1, 3)),
        local_positions,
        Quaternions.id(0),
        source_offsets,
        parents,
    )

    rebuilt_anim, _mean_error, _max_error = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=parents,
        iterations=10,
    )

    identity_rest_rotations = np.zeros((3, 4), dtype=np.float32)
    identity_rest_rotations[:, 0] = 1.0
    skeleton = build_skeleton(joint_names, rigid_offsets, parents, identity_rest_rotations)
    _joint_rotations, _root_translation, _root_rotation, bone_translations = (
        animation_to_exporter_inputs(rebuilt_anim, skeleton)
    )

    assert bone_translations is None


def test_fullbody_ik_rebuild_can_preserve_selected_local_positions() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    source_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rigid_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = np.array(
        [
            [
                [2.0, 0.0, 0.0],
                [1.25, 0.5, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    target_anim = Animation(
        Quaternions.id((1, 3)),
        local_positions,
        Quaternions.id(0),
        source_offsets,
        parents,
    )

    rebuilt_anim, mean_error, max_error = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=parents,
        preserved_position_indices=[1],
        iterations=10,
    )

    assert np.allclose(rebuilt_anim.positions[:, 1, :], target_anim.positions[:, 1, :])
    assert np.allclose(rebuilt_anim.positions[:, 2, :], rigid_offsets[2][None, :])
    assert mean_error == pytest.approx(0.0, abs=1e-5)
    assert max_error == pytest.approx(0.0, abs=1e-5)


def test_fullbody_ik_rebuild_can_preserve_selected_local_rotations() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    source_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rigid_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    rotations = Quaternions.id((1, 3))
    rotations[:, 1] = Quaternions.from_angle_axis(
        np.array([np.pi / 4.0], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
    )
    target_anim = Animation(
        rotations,
        local_positions,
        Quaternions.id(0),
        source_offsets,
        parents,
    )

    rebuilt_anim, mean_error, max_error = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=parents,
        preserved_rotation_indices=[1],
        iterations=10,
    )

    assert np.allclose(
        rebuilt_anim.rotations.qs[:, 1, :],
        target_anim.rotations.qs[:, 1, :],
        atol=1e-6,
    )
    assert mean_error == pytest.approx(0.0, abs=1e-5)


def test_fullbody_ik_rebuild_can_constrain_inconsistent_branch_targets() -> None:
    parents = np.array([-1, 0, 1, 1], dtype=np.int32)
    rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = rest_offsets[None].copy()
    local_positions[:, 2, :] = np.array([[0.0, 0.45, 0.0]], dtype=np.float64)
    local_positions[:, 3, :] = np.array([[-0.5, 0.5, 0.0]], dtype=np.float64)
    target_anim = Animation(
        Quaternions.id((1, 4)),
        local_positions,
        Quaternions.id(0),
        rest_offsets,
        parents,
    )

    rebuilt_anim, mean_error, max_error = rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rest_offsets,
        rigid_parents=parents,
        stretch_factor=0.1,
        iterations=10,
    )

    assert not np.allclose(rebuilt_anim.positions[:, 2:, :], local_positions[:, 2:, :])
    np.testing.assert_allclose(
        np.linalg.norm(rebuilt_anim.positions[:, 2:, :], axis=-1),
        0.9,
        atol=1e-6,
    )
    assert mean_error < 0.3
    assert max_error < 0.5

    # The branch outlier is clamped instead of being converted into a fold.
    max_joint_angle = np.degrees(
        2.0 * np.arccos(np.abs(rebuilt_anim.rotations.qs[:, 1:, 0])).max()
    )
    assert max_joint_angle <= 45.0 + 1e-4


def _pose_channels_to_world_positions(
    joint_rotations: np.ndarray,
    bone_translations: np.ndarray,
    parents: np.ndarray,
    rest_offsets: np.ndarray,
    rest_rotations: np.ndarray,
) -> np.ndarray:
    """Read the exporter's pose channels back the way ``export_glb`` poses a rig."""
    from utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np

    local_rotations = quat_multiply_wxyz_np(rest_rotations[None], joint_rotations)
    local_positions = rest_offsets[None] + quat_rotate_wxyz_np(
        np.broadcast_to(rest_rotations[None], joint_rotations.shape), bone_translations,
    )
    return positions_global(
        Animation(
            Quaternions(local_rotations),
            local_positions,
            Quaternions(rest_rotations.copy()),
            rest_offsets.copy(),
            parents.copy(),
        )
    )


def test_retarget_pose_channel_ik_makes_a_stretched_chain_rigid() -> None:
    # A static wrapper root over a locomotion joint (the Bip01 shape), then a
    # two-bone chain whose retarget reached its target by doubling both bones:
    # the pose-translation channel carries the whole stretch, which is what a
    # cross-species world-space transfer leaves behind.
    frame_count = 6
    parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    rest_offsets = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    rest_rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1))
    joint_rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frame_count, 4, 1))
    bone_translations = np.zeros((frame_count, 4, 3), dtype=np.float64)
    bone_translations[:, 1, 0] = np.arange(frame_count, dtype=np.float64)   # travel
    bone_translations[:, 1, 1] = 3.0
    bone_translations[:, 2:, 0] = 1.0                                       # stretch

    rebuilt_rotations, rebuilt_translations, mean_error, max_error = (
        rebuild_retarget_pose_channels_with_ik(
            joint_rotations,
            bone_translations,
            parents=parents,
            rest_offsets=rest_offsets,
            rest_rotations=rest_rotations,
            iterations=10,
            stretch_factor=0.0,
        )
    )

    # Rigid: the stretched chain sits back on its own rest offsets ...
    assert np.allclose(rebuilt_translations[:, 2:, :], 0.0, atol=1e-6)
    # ... while the joint carrying the transport keeps its translation verbatim,
    # because that translation IS the locomotion.
    assert np.allclose(rebuilt_translations[:, 1, :], bone_translations[:, 1, :])
    assert np.isfinite(mean_error) and np.isfinite(max_error)

    rebuilt_world = _pose_channels_to_world_positions(
        rebuilt_rotations, rebuilt_translations, parents, rest_offsets, rest_rotations,
    )
    bone_lengths = np.linalg.norm(
        rebuilt_world[:, 2:, :] - rebuilt_world[:, parents[2:], :], axis=-1
    )
    assert np.allclose(bone_lengths, 1.0, atol=1e-6)


def test_retarget_pose_channel_ik_is_a_no_op_on_an_already_rigid_pose() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    rest_offsets = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64
    )
    rest_rotations = Quaternions.from_euler(
        np.array([[0.0, 0.3, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, -0.4]])
    ).qs.astype(np.float64)
    joint_rotations = Quaternions.from_euler(
        np.array([[[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]]]).reshape(3, 3)
    ).qs.astype(np.float64)[None]
    bone_translations = np.zeros((1, 3, 3), dtype=np.float64)

    before = _pose_channels_to_world_positions(
        joint_rotations, bone_translations, parents, rest_offsets, rest_rotations,
    )
    rebuilt_rotations, rebuilt_translations, _mean_error, max_error = (
        rebuild_retarget_pose_channels_with_ik(
            joint_rotations,
            bone_translations,
            parents=parents,
            rest_offsets=rest_offsets,
            rest_rotations=rest_rotations,
            iterations=4,
            stretch_factor=0.0,
        )
    )
    after = _pose_channels_to_world_positions(
        rebuilt_rotations, rebuilt_translations, parents, rest_offsets, rest_rotations,
    )

    assert max_error == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(after, before, atol=1e-5)


def test_retarget_pose_channel_ik_rejects_a_multi_root_skeleton() -> None:
    parents = np.array([-1, -1], dtype=np.int32)
    rest_offsets = np.zeros((2, 3), dtype=np.float64)
    rest_rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))

    with pytest.raises(ValueError, match="root"):
        rebuild_retarget_pose_channels_with_ik(
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (1, 2, 1)),
            np.zeros((1, 2, 3), dtype=np.float64),
            parents=parents,
            rest_offsets=rest_offsets,
            rest_rotations=rest_rotations,
        )
