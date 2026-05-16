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
from tools.restore_glb_from_npy import (
    _clamp_unobservable_joint_positions_to_rest,
    _rebuild_fullbody_animation_with_ik,
)
from Anytop.utils.exporter import animation_to_exporter_inputs
from Anytop.utils.roundtrip_common import _build_skeleton


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

    rebuilt_anim, mean_error, max_error = _rebuild_fullbody_animation_with_ik(
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
        _rebuild_fullbody_animation_with_ik(
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

    rebuilt_anim, _mean_error, _max_error = _rebuild_fullbody_animation_with_ik(
        target_anim,
        rigid_offsets=rigid_offsets,
        rigid_parents=parents,
        iterations=10,
    )

    identity_rest_rotations = np.zeros((3, 4), dtype=np.float32)
    identity_rest_rotations[:, 0] = 1.0
    skeleton = _build_skeleton(joint_names, rigid_offsets, parents, identity_rest_rotations)
    _joint_rotations, _root_translation, _root_rotation, bone_translations = (
        animation_to_exporter_inputs(rebuilt_anim, skeleton)
    )

    assert bone_translations is None


def test_clamp_unobservable_joint_positions_to_rest_zeroes_leaf_exporter_translations() -> None:
    joint_names = ["Root", "JointA", "Leaf"]
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    local_positions = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.3, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.7, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    target_anim = Animation(
        Quaternions.id((2, 3)),
        local_positions,
        Quaternions.id(0),
        offsets,
        parents,
    )
    rotation_channel_mask = np.array([True, True, False], dtype=bool)

    clamped_anim = _clamp_unobservable_joint_positions_to_rest(
        target_anim,
        rest_offsets=offsets,
        rotation_channel_mask=rotation_channel_mask,
    )

    assert np.allclose(clamped_anim.positions[:, 2, :], offsets[2][None, :])
    identity_rest_rotations = np.zeros((3, 4), dtype=np.float32)
    identity_rest_rotations[:, 0] = 1.0
    skeleton = _build_skeleton(joint_names, offsets, parents, identity_rest_rotations)
    _joint_rotations, _root_translation, _root_rotation, bone_translations = animation_to_exporter_inputs(
        clamped_anim,
        skeleton,
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

    rebuilt_anim, mean_error, max_error = _rebuild_fullbody_animation_with_ik(
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

    rebuilt_anim, mean_error, max_error = _rebuild_fullbody_animation_with_ik(
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
