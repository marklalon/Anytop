import os
import sys
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


def _load_utils_module(module_name: str) -> None:
    module_path = os.path.join(
        _ANYTOP_ROOT,
        'utils',
        f"{module_name.rsplit('.', 1)[-1]}.py",
    )
    if os.path.isfile(module_path) and module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


_load_utils_module('utils.rotation_conversions')


from motion_lib.Animation import positions_global, rotations_global
from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions
from data_loaders.truebones.truebones_utils.features import (
    get_motion,
    recover_animation_from_motion_np,
)
from data_loaders.truebones.truebones_utils.animation_utils import find_translation_root
import Anytop.utils.retarget as retarget_mod
from Anytop.utils.auto_retarget import _build_tpose_aligned_target_animation
from Anytop.utils.auto_retarget import retarget_features_npy_to_target
from Anytop.utils.rotation_numpy import quat_multiply_wxyz_np, quat_rotate_wxyz_np


def _quat_x(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg) * 0.5
    return np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0, 0.0], dtype=np.float64)


def _quat_y(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg) * 0.5
    return np.array([np.cos(angle_rad), 0.0, np.sin(angle_rad), 0.0], dtype=np.float64)


def _quat_z(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg) * 0.5
    return np.array([np.cos(angle_rad), 0.0, 0.0, np.sin(angle_rad)], dtype=np.float64)


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    dot = float(np.clip(abs(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _identity_quat(count: int) -> np.ndarray:
    qs = np.zeros((count, 4), dtype=np.float64)
    qs[:, 0] = 1.0
    return qs


class _TensorLike:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._array


def test_find_translation_root_ignores_sparse_wrapper_motion() -> None:
    positions = np.zeros((10, 3, 3), dtype=np.float64)
    positions[[6, 7], 0, 0] = np.array([0.0015, 0.0045], dtype=np.float64)
    positions[1:, 1, 1] = np.linspace(0.0, 0.18, 9, dtype=np.float64)

    anim = SimpleNamespace(positions=positions, parents=np.array([-1, 0, 1], dtype=np.int64))

    assert int(find_translation_root(anim)) == 1


def test_retarget_distributes_short_source_bone_across_longer_target_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Root': 'Root',
            'Neck': 'Neck',
            'Neck 1': 'Neck 1',
            'Head': 'Head',
        },
    )

    src_parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    src_rest_rotations = _identity_quat(4)
    # Twist around the source bone axis keeps the donor head direction on the
    # Y axis, but the old transport-frame walk rotated the target's Z-bearing
    # gap offsets out into X.
    src_rest_rotations[2] = _quat_y(55.0)

    tgt_parents = np.array([-1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.7, 0.3],
            [0.0, 0.6, 0.4],
            [0.0, 0.5, 0.1],
        ],
        dtype=np.float64,
    )

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=src_rest_rotations,
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=_identity_quat(7),
        src_joint_rotations=_identity_quat(4)[None, :, :],
        src_root_translation=np.zeros((1, 3), dtype=np.float64),
        src_root_rotation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        src_match_names=['Root', 'Neck', 'Neck 1', 'Head'],
        tgt_match_names=['Root', 'Neck', 'Neck 1', 'Neck 2', 'Neck 3', 'Neck 4', 'Head'],
        coordinate_search=False,
        verbose=False,
    )

    target_world_positions = np.asarray(result['target_world_positions'], dtype=np.float64)[0]
    expected_positions = np.zeros((7, 3), dtype=np.float64)
    expected_positions[1] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    expected_positions[2] = np.array([0.0, 2.0, 0.0], dtype=np.float64)

    cursor = expected_positions[2].copy()
    for joint_idx in (3, 4, 5, 6):
        cursor = cursor + np.array([0.0, np.linalg.norm(tgt_rest_offsets[joint_idx]), 0.0], dtype=np.float64)
        expected_positions[joint_idx] = cursor

    np.testing.assert_allclose(target_world_positions, expected_positions, atol=1e-6)


def test_retarget_preserves_zero_pose_locations_for_rigid_longer_target_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Root': 'Root',
            'Neck': 'Neck',
            'Neck 1': 'Neck 1',
            'Head': 'Head',
        },
    )

    src_parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    tgt_parents = np.array([-1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.7, 0.3],
            [0.0, 0.6, 0.4],
            [0.0, 0.5, 0.1],
        ],
        dtype=np.float64,
    )

    joint_rotations = np.tile(_identity_quat(4)[None, :, :], (2, 1, 1))
    joint_rotations[1, 1] = _quat_z(35.0)
    joint_rotations[1, 2] = _quat_x(-25.0)
    joint_rotations[1, 3] = _quat_y(15.0)

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=_identity_quat(4),
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=_identity_quat(7),
        src_joint_rotations=joint_rotations,
        src_root_translation=np.zeros((2, 3), dtype=np.float64),
        src_root_rotation=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (2, 1)),
        src_bone_translations=np.zeros((2, 4, 3), dtype=np.float64),
        src_match_names=['Root', 'Neck', 'Neck 1', 'Head'],
        tgt_match_names=['Root', 'Neck', 'Neck 1', 'Neck 2', 'Neck 3', 'Neck 4', 'Head'],
        coordinate_search=False,
        verbose=False,
    )

    assert result['bone_translations'] is not None
    np.testing.assert_allclose(
        np.asarray(result['bone_translations'], dtype=np.float64)[:, 1:, :],
        0.0,
        atol=1e-6,
    )


def test_retarget_skips_llm_and_preserves_root_motion_under_target_root_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _should_not_call_llm(*_args, **_kwargs):
        raise AssertionError("LLM mapping should not run for exact same-name self-retarget with root wrappers")

    monkeypatch.setattr(retarget_mod, '_llm_joint_mapping', _should_not_call_llm)

    src_parents = np.array([-1, 0], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    tgt_parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    tgt_rest_rotations = _identity_quat(4)
    tgt_rest_rotations[0] = _quat_x(-90.0)

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=_identity_quat(2),
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=tgt_rest_rotations,
        src_joint_rotations=np.tile(_identity_quat(2)[None, :, :], (2, 1, 1)),
        src_root_translation=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        src_root_rotation=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (2, 1)),
        src_bone_translations=None,
        src_match_names=['locator2', 'koshi'],
        tgt_match_names=['EAL1_2', 'N_ALL', 'locator2', 'koshi'],
        coordinate_search=False,
        verbose=False,
    )

    np.testing.assert_array_equal(
        np.asarray(result['src_to_tgt'], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
    )
    np.testing.assert_allclose(
        np.asarray(result['target_world_positions'], dtype=np.float64)[:, 2, :],
        np.array(
            [
                [0.0, 6.0, 0.0],
                [0.0, 7.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result['target_world_positions'], dtype=np.float64)[:, 3, :],
        np.array(
            [
                [0.0, 7.0, 0.0],
                [0.0, 8.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )

    pose_rotations = np.asarray(result['joint_rotations'], dtype=np.float64).copy()
    pose_locations = np.zeros((pose_rotations.shape[0], pose_rotations.shape[1], 3), dtype=np.float64)
    if result['bone_translations'] is not None:
        pose_locations[:] = np.asarray(result['bone_translations'], dtype=np.float64)
    pose_rotations[:, 0, :] = np.asarray(result['root_rotation'], dtype=np.float64)
    pose_locations[:, 0, :] = np.asarray(result['root_translation'], dtype=np.float64)

    reconstructed_world_positions, _ = retarget_mod._batch_pose_fk_np(
        pose_rotations,
        pose_locations,
        tgt_parents,
        tgt_rest_offsets,
        tgt_rest_rotations,
    )
    np.testing.assert_allclose(
        reconstructed_world_positions,
        np.asarray(result['target_world_positions'], dtype=np.float64),
        atol=1e-6,
    )


def test_build_target_animation_preserves_world_rotations_across_gap() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    gap_world = _quat_z(90.0)
    head_local = _quat_x(30.0)
    head_world = quat_multiply_wxyz_np(gap_world[None, :], head_local[None, :])[0]

    target_world_rotations = np.array(
        [[
            [1.0, 0.0, 0.0, 0.0],
            gap_world,
            head_world,
        ]],
        dtype=np.float64,
    )
    target_world_positions = np.array(
        [[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ]],
        dtype=np.float64,
    )
    retarget_result = {
        'joint_rotations': np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (1, 3, 1)),
        'target_world_positions': target_world_positions,
        'target_world_rotations': target_world_rotations,
        'src_to_tgt': np.array([0, 2], dtype=np.int32),
    }
    target_tp = SimpleNamespace(
        offsets=offsets,
        tpos_anim=SimpleNamespace(parents=parents),
        tpos_rots=_identity_quat(3)[None, :, :],
    )

    anim = _build_tpose_aligned_target_animation(retarget_result, target_tp)

    np.testing.assert_allclose(
        rotations_global(anim).qs,
        target_world_rotations,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        positions_global(anim),
        target_world_positions,
        atol=1e-6,
    )
    np.testing.assert_allclose(anim.rotations.qs[0, 1], gap_world, atol=1e-6)


def test_build_target_animation_keeps_pure_unmapped_branch_at_identity() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    branch_world = _quat_z(177.0)
    leaf_world = quat_multiply_wxyz_np(branch_world[None, :], _quat_x(25.0)[None, :])[0]
    retarget_result = {
        'joint_rotations': np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (1, 3, 1)),
        'target_world_positions': np.array(
            [[
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-0.9, 1.1, 0.0],
            ]],
            dtype=np.float64,
        ),
        'target_world_rotations': np.array(
            [[
                [1.0, 0.0, 0.0, 0.0],
                branch_world,
                leaf_world,
            ]],
            dtype=np.float64,
        ),
        'src_to_tgt': np.array([0], dtype=np.int32),
    }
    target_tp = SimpleNamespace(
        offsets=offsets,
        tpos_anim=SimpleNamespace(parents=parents),
        tpos_rots=_identity_quat(3)[None, :, :],
    )

    anim = _build_tpose_aligned_target_animation(retarget_result, target_tp)

    np.testing.assert_allclose(anim.rotations.qs[0, 1], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(anim.rotations.qs[0, 2], np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)


def test_tpose_aligned_roundtrip_preserves_gap_chain_and_rest_side_branch() -> None:
    parents = np.array([-1, 0, 1, 2, 0, 4], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    gap_world = np.stack([_quat_z(90.0), _quat_z(105.0)], axis=0)
    head_local = np.stack([_quat_x(30.0), _quat_x(35.0)], axis=0)
    head_world = quat_multiply_wxyz_np(gap_world, head_local)
    branch_world = np.stack([_quat_z(177.0), _quat_z(160.0)], axis=0)
    branch_leaf_local = np.stack([_quat_x(25.0), _quat_x(10.0)], axis=0)
    branch_leaf_world = quat_multiply_wxyz_np(branch_world, branch_leaf_local)

    target_world_rotations = np.stack(
        [
            _identity_quat(6),
            _identity_quat(6),
        ],
        axis=0,
    )
    target_world_rotations[:, 1] = gap_world
    target_world_rotations[:, 2] = head_world
    target_world_rotations[:, 3] = head_world
    target_world_rotations[:, 4] = branch_world
    target_world_rotations[:, 5] = branch_leaf_world

    target_world_positions = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.433013, 1.25, 0.0],
                [1.0, 0.0, 0.0],
                [0.000609, 0.052336, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-0.965926, 0.741181, 0.0],
                [-1.440618, 0.614228, 0.0],
                [1.0, 0.0, 0.0],
                [0.060307, 0.34202, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    retarget_result = {
        'joint_rotations': np.tile(_identity_quat(6)[None, :, :], (2, 1, 1)),
        'target_world_positions': target_world_positions,
        'target_world_rotations': target_world_rotations,
        'src_to_tgt': np.array([0, 2], dtype=np.int32),
    }
    target_tp = SimpleNamespace(
        offsets=offsets,
        tpos_anim=SimpleNamespace(parents=parents),
        tpos_rots=_identity_quat(6)[None, :, :],
        foot_indices=[],
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
    )

    baseline_anim = _build_tpose_aligned_target_animation(retarget_result, target_tp)
    baseline_world_rot = rotations_global(baseline_anim).qs
    baseline_world_pos = positions_global(baseline_anim)

    features, *_ = get_motion(
        baseline_anim,
        foot_contact_vel_thresh=0.002,
        object_type='Synthetic',
        max_joints=6,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=target_tp.tpos_rots,
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=target_tp.orientation_quat,
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
    )
    assert features is not None

    recovered_anim, _has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        allow_infer=True,
    )
    recovered_world_rot = rotations_global(recovered_anim).qs
    recovered_world_pos = positions_global(recovered_anim)

    for joint_idx in (1, 2):
        for frame_idx in range(2):
            assert _quat_angle_deg(recovered_world_rot[frame_idx, joint_idx], baseline_world_rot[frame_idx, joint_idx]) < 1e-5
        np.testing.assert_allclose(recovered_world_pos[:, joint_idx], baseline_world_pos[:, joint_idx], atol=1e-5)

    for joint_idx in (4, 5):
        np.testing.assert_allclose(
            recovered_anim.rotations.qs[:, joint_idx],
            np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (2, 1)),
            atol=1e-6,
        )


def _fk_world_positions(
    parents: np.ndarray,
    offsets: np.ndarray,
    world_rotations: np.ndarray,
) -> np.ndarray:
    """FK world positions from per-joint world rotations and rest offsets."""
    frame_count, joint_count = world_rotations.shape[:2]
    positions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    for j in range(joint_count):
        p = int(parents[j])
        if p < 0:
            continue
        offset_repeated = np.repeat(offsets[j:j+1][None, :, :], frame_count, axis=0)[:, 0, :]
        positions[:, j] = positions[:, p] + quat_rotate_wxyz_np(
            world_rotations[:, p], offset_repeated,
        )
    return positions


def test_tpose_aligned_roundtrip_with_nontrivial_rest_rotations() -> None:
    """Mapped joint world rotations must survive the full roundtrip even when
    unmapped gap joints have non-identity FBX rest rotations.

    Skeleton:
        root(0) -> gap1(1) -> gap2(2) -> head(3) -> head_helper(4)   # mapped chain
        root(0) -> branch(5) -> branch_leaf(6)                       # pure unmapped side branch

    src_to_tgt maps source 0 -> target 0 (root) and source 1 -> target 3 (head),
    so gap1/gap2 are gap joints (unmapped with mapped descendant), head_helper
    plays the role of the leaf-rotation helper that the real pipeline appends
    via ``augment_leaf_rotation_helpers`` so the mapped head's rotation gets
    a child slot in the HumanML3D feature encoding.

    Target world rotations follow the convention emitted by
    ``retarget_world_space_np``'s section G: unmapped joints sit at
    ``parent_world * tgt_rest_rotations[j]``, mapped joints take the source's
    animated world rotation.
    """
    parents = np.array([-1, 0, 1, 2, 3, 0, 5], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    # Non-trivial FBX rest rotations for unmapped joints.
    rest_gap1 = _quat_z(20.0)
    rest_gap2 = _quat_z(15.0)
    rest_branch = _quat_z(-45.0)
    rest_branch_leaf = _quat_z(-10.0)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # Frame 0: source at rest. Frame 1: mapped head animates by X 30°.
    head_world_anim = np.stack([identity, _quat_x(30.0)], axis=0)
    root_world = np.stack([identity, identity], axis=0)

    # Propagate gap/branch world rotations through rest deltas.
    gap1_world = quat_multiply_wxyz_np(root_world, np.stack([rest_gap1, rest_gap1], axis=0))
    gap2_world = quat_multiply_wxyz_np(gap1_world, np.stack([rest_gap2, rest_gap2], axis=0))
    # head_helper is an unmapped leaf with identity rest → world = parent (head).
    head_helper_world = head_world_anim
    branch_world = quat_multiply_wxyz_np(root_world, np.stack([rest_branch, rest_branch], axis=0))
    branch_leaf_world = quat_multiply_wxyz_np(branch_world, np.stack([rest_branch_leaf, rest_branch_leaf], axis=0))

    target_world_rotations = np.zeros((2, 7, 4), dtype=np.float64)
    target_world_rotations[:, 0] = root_world
    target_world_rotations[:, 1] = gap1_world
    target_world_rotations[:, 2] = gap2_world
    target_world_rotations[:, 3] = head_world_anim
    target_world_rotations[:, 4] = head_helper_world
    target_world_rotations[:, 5] = branch_world
    target_world_rotations[:, 6] = branch_leaf_world

    target_world_positions = _fk_world_positions(parents, offsets, target_world_rotations)

    retarget_result = {
        'joint_rotations': np.tile(_identity_quat(7)[None, :, :], (2, 1, 1)),
        'target_world_positions': target_world_positions,
        'target_world_rotations': target_world_rotations,
        'src_to_tgt': np.array([0, 3], dtype=np.int32),
    }

    # tpos_rots encodes the target skeleton's FBX rest local rotations.
    rest_local_rotations = np.stack(
        [identity, rest_gap1, rest_gap2, identity, identity, rest_branch, rest_branch_leaf],
        axis=0,
    )
    tpos_rots = rest_local_rotations[None, :, :]

    target_tp = SimpleNamespace(
        offsets=offsets,
        tpos_anim=SimpleNamespace(parents=parents),
        tpos_rots=tpos_rots,
        foot_indices=[],
        scale_factor=1.0,
        orientation_quat=identity.copy(),
        helper_metadata=None,
    )

    baseline_anim = _build_tpose_aligned_target_animation(retarget_result, target_tp)
    baseline_world_rot = rotations_global(baseline_anim).qs

    # Sanity: the built Animation's FK must already reproduce mapped + gap
    # world rotations (gap rotations follow by chain composition through
    # tgt_rest_rotations baked into local channels).
    for joint_idx in (0, 1, 2, 3):
        for frame_idx in range(2):
            assert _quat_angle_deg(
                baseline_world_rot[frame_idx, joint_idx],
                target_world_rotations[frame_idx, joint_idx],
            ) < 1e-5, f"baseline anim world_rot[{frame_idx},{joint_idx}] doesn't match input"

    features, *_ = get_motion(
        baseline_anim,
        foot_contact_vel_thresh=0.002,
        object_type='Synthetic',
        max_joints=7,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=tpos_rots,
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=target_tp.orientation_quat,
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
    )
    assert features is not None

    recovered_anim, _has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        allow_infer=True,
    )
    recovered_world_rot = rotations_global(recovered_anim).qs

    # Mapped joints (root, head) must round-trip exactly even though the gap
    # chain between root and head carries non-identity rest rotations.
    for joint_idx in (0, 3):
        for frame_idx in range(2):
            assert _quat_angle_deg(
                recovered_world_rot[frame_idx, joint_idx],
                target_world_rotations[frame_idx, joint_idx],
            ) < 1e-4, (
                f"recovered world rotation at mapped joint {joint_idx} drifted at frame {frame_idx}; "
                f"this indicates the gap-joint rest rotations are not being composed correctly"
            )

    # Gap joints should follow the rest-propagated chain (root_world * rest_chain).
    for joint_idx in (1, 2):
        for frame_idx in range(2):
            assert _quat_angle_deg(
                recovered_world_rot[frame_idx, joint_idx],
                target_world_rotations[frame_idx, joint_idx],
            ) < 1e-4, (
                f"recovered world rotation at gap joint {joint_idx} drifted at frame {frame_idx}"
            )


def test_retarget_features_npy_to_target_uses_tpose_aligned_motion_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import Anytop.utils.auto_retarget as auto_retarget_mod
    import Anytop.utils.exporter as exporter_mod
    import Anytop.utils.retarget as retarget_mod
    import Anytop.utils.roundtrip_common as roundtrip_common_mod
    import data_loaders.truebones.truebones_utils.features as features_mod

    sys.modules['utils.exporter'] = exporter_mod
    sys.modules['utils.retarget'] = retarget_mod
    sys.modules['utils.roundtrip_common'] = roundtrip_common_mod
    importlib.invalidate_caches()

    sentinel_anim = Animation(
        Quaternions(np.tile(_identity_quat(2)[None, :, :], (1, 1, 1))),
        np.zeros((1, 2, 3), dtype=np.float64),
        Quaternions(_identity_quat(2)),
        np.zeros((2, 3), dtype=np.float64),
        np.array([-1, 0], dtype=np.int32),
    )
    source_tp = SimpleNamespace(
        names=['Root', 'Head'],
        offsets=np.zeros((2, 3), dtype=np.float32),
        tpos_anim=SimpleNamespace(parents=np.array([-1, 0], dtype=np.int32)),
        tpos_rots=_identity_quat(2)[None, :, :].astype(np.float32),
    )
    target_tp = SimpleNamespace(
        names=['Root', 'Head'],
        offsets=np.zeros((2, 3), dtype=np.float64),
        tpos_anim=SimpleNamespace(parents=np.array([-1, 0], dtype=np.int32)),
        tpos_rots=_identity_quat(2)[None, :, :].astype(np.float64),
        foot_indices=[],
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
    )
    source_cond = {
        'object_type': 'Parrot',
        'original_joint_count': 2,
        'canonical_joint_names': ['Root', 'Head'],
        'orientation_reference_fbx_path': 'unused',
    }
    target_cond = {
        'object_type': 'Dragon',
        'canonical_joint_names': ['Root', 'Head'],
    }

    monkeypatch.setattr(features_mod, 'recover_animation_from_motion_np', lambda *args, **kwargs: (object(), False))
    monkeypatch.setattr(roundtrip_common_mod, '_build_skeleton', lambda *args, **kwargs: object())
    monkeypatch.setattr(
        exporter_mod,
        'animation_to_exporter_inputs',
        lambda *args, **kwargs: (
            _TensorLike(np.zeros((1, 2, 4), dtype=np.float32)),
            _TensorLike(np.zeros((1, 3), dtype=np.float32)),
            _TensorLike(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)),
            None,
        ),
    )
    monkeypatch.setattr(auto_retarget_mod, 'find_translation_root', lambda anim: 0)
    monkeypatch.setattr(retarget_mod, 'retarget_world_space_np', lambda **kwargs: {'src_to_tgt': np.array([0, 1], dtype=np.int32)})
    monkeypatch.setattr(auto_retarget_mod, '_build_tpose_aligned_target_animation', lambda *args, **kwargs: sentinel_anim)

    captured: dict[str, object] = {}

    def _fake_get_motion(anim, *args, **kwargs):
        captured['anim'] = anim
        captured['kwargs'] = kwargs
        return np.zeros((1, 2, 13), dtype=np.float32), None, None, None, None, None, None, None

    monkeypatch.setattr(features_mod, 'get_motion', _fake_get_motion)

    result = retarget_features_npy_to_target(
        np.zeros((1, 2, 13), dtype=np.float32),
        source_cond,
        'Parrot',
        target_tp,
        'Dragon',
        max_joints=2,
        source_tp=source_tp,
        target_cond=target_cond,
    )

    assert result is not None
    assert captured['anim'] is sentinel_anim
    assert captured['kwargs']['animation_input_is_tpose_aligned'] is True


def test_retarget_promotes_unmapped_effective_root_to_target_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Hips': 'Cg',
            'Ctrl': None,
            'Bip01': None,
            'Pelvis': 'Pelvis',
        },
    )

    src_parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    tgt_parents = np.array([-1, 0], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    joint_rotations = np.tile(_identity_quat(4)[None, :, :], (2, 1, 1))
    root_translation = np.zeros((2, 3), dtype=np.float64)
    root_rotation = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (2, 1))
    bone_translations = np.zeros((2, 4, 3), dtype=np.float64)
    bone_translations[1, 2, 0] = 1.0

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=_identity_quat(4),
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=_identity_quat(2),
        src_joint_rotations=joint_rotations,
        src_root_translation=root_translation,
        src_root_rotation=root_rotation,
        src_effective_root_index=2,
        src_bone_translations=bone_translations,
        src_match_names=['Hips', 'Ctrl', 'Bip01', 'Pelvis'],
        tgt_match_names=['Cg', 'Pelvis'],
        coordinate_search=False,
        verbose=False,
    )

    np.testing.assert_allclose(
        np.asarray(result['target_world_positions'], dtype=np.float64)[:, 0],
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(result['target_world_positions'], dtype=np.float64)[:, 1],
        np.array(
            [
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-6,
    )
    assert int(result['src_to_tgt'][2]) == 0
    assert int(result['src_to_tgt'][0]) == -1


def test_retarget_promotes_matched_effective_root_over_wrapper_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Hips': 'Cg',
            'Pelvis': 'Pelvis',
            'Spine': 'Spine',
        },
    )

    src_parents = np.array([-1, 0, 1, 2], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    tgt_parents = np.array([-1, 0, 1], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    joint_rotations = np.tile(_identity_quat(4)[None, :, :], (2, 1, 1))
    root_translation = np.zeros((2, 3), dtype=np.float64)
    root_rotation = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (2, 1))
    bone_translations = np.zeros((2, 4, 3), dtype=np.float64)
    bone_translations[:, 1, 1] = np.array([1.0, 1.2], dtype=np.float64)

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=_identity_quat(4),
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=_identity_quat(3),
        src_joint_rotations=joint_rotations,
        src_root_translation=root_translation,
        src_root_rotation=root_rotation,
        src_effective_root_index=1,
        src_bone_translations=bone_translations,
        src_match_names=['Hips', 'Pelvis', 'Spine', 'Head'],
        tgt_match_names=['Cg', 'Pelvis', 'Spine'],
        coordinate_search=False,
        verbose=False,
    )

    target_world_positions = np.asarray(result['target_world_positions'], dtype=np.float64)
    assert target_world_positions[1, 0, 1] > target_world_positions[0, 0, 1]
    assert int(result['src_to_tgt'][1]) == 0
    assert int(result['src_to_tgt'][0]) == -1


def test_bridge_gap_joint_uses_source_anchor_rotation_to_avoid_spine_translation_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Hips': 'Cg',
            'Pelvis': 'Pelvis',
            'Spine': 'Spine',
        },
    )

    src_parents = np.array([-1, 0, 1], dtype=np.int32)
    src_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    src_rest_rotations = _identity_quat(3)
    src_rest_rotations[1] = _quat_z(35.0)

    tgt_parents = np.array([-1, 0, 1], dtype=np.int32)
    tgt_rest_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.7, 0.0],
            [0.0, 0.8, 0.0],
        ],
        dtype=np.float64,
    )

    result = retarget_mod.retarget_world_space_np(
        src_parents=src_parents,
        src_rest_offsets=src_rest_offsets,
        src_rest_rotations=src_rest_rotations,
        tgt_parents=tgt_parents,
        tgt_rest_offsets=tgt_rest_offsets,
        tgt_rest_rotations=_identity_quat(3),
        src_joint_rotations=_identity_quat(3)[None, :, :],
        src_root_translation=np.zeros((1, 3), dtype=np.float64),
        src_root_rotation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        src_effective_root_index=1,
        src_bone_translations=None,
        src_match_names=['Hips', 'Pelvis', 'Spine'],
        tgt_match_names=['Cg', 'Pelvis', 'Spine'],
        coordinate_search=False,
        verbose=False,
    )

    target_tp = SimpleNamespace(
        offsets=tgt_rest_offsets,
        tpos_anim=SimpleNamespace(parents=tgt_parents),
        tpos_rots=_identity_quat(3)[None, :, :],
    )
    anim = _build_tpose_aligned_target_animation(result, target_tp)

    np.testing.assert_allclose(
        anim.positions[0, 2],
        tgt_rest_offsets[2],
        atol=1e-6,
    )


def test_root_promotion_shifts_descendant_chain_up_one_target_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Hips': 'Cg',
            'Pelvis': 'Pelvis',
            'Spine 1': 'Spine',
            'Spine 2': 'Spine 1',
        },
    )

    result = retarget_mod.retarget_world_space_np(
        src_parents=np.array([-1, 0, 1, 2], dtype=np.int32),
        src_rest_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        src_rest_rotations=_identity_quat(4),
        tgt_parents=np.array([-1, 0, 1, 2], dtype=np.int32),
        tgt_rest_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        tgt_rest_rotations=_identity_quat(4),
        src_joint_rotations=_identity_quat(4)[None, :, :],
        src_root_translation=np.zeros((1, 3), dtype=np.float64),
        src_root_rotation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        src_effective_root_index=1,
        src_bone_translations=None,
        src_match_names=['Hips', 'Pelvis', 'Spine 1', 'Spine 2'],
        tgt_match_names=['Cg', 'Pelvis', 'Spine', 'Spine 1'],
        coordinate_search=False,
        verbose=False,
    )

    assert int(result['src_to_tgt'][0]) == -1
    assert int(result['src_to_tgt'][1]) == 0
    assert int(result['src_to_tgt'][2]) == 1
    assert int(result['src_to_tgt'][3]) == 2


def test_root_promotion_redistributes_short_neck_chain_across_longer_target_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        retarget_mod,
        '_llm_joint_mapping',
        lambda *_args, **_kwargs: {
            'Hips': 'Cg',
            'Pelvis': 'Pelvis',
            'Spine 1': 'Spine',
            'Spine 2': 'Spine 1',
            'Spine 3': 'Spine 2',
            'Neck 1': 'Neck',
            'Neck 2': 'Neck 1',
            'Head': 'Head',
        },
    )

    result = retarget_mod.retarget_world_space_np(
        src_parents=np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        src_rest_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float64,
        ),
        src_rest_rotations=_identity_quat(9),
        tgt_parents=np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32),
        tgt_rest_offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.7, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.15, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        ),
        tgt_rest_rotations=_identity_quat(11),
        src_joint_rotations=_identity_quat(9)[None, :, :],
        src_root_translation=np.zeros((1, 3), dtype=np.float64),
        src_root_rotation=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        src_effective_root_index=1,
        src_bone_translations=None,
        src_match_names=['Hips', 'Pelvis', 'Spine 1', 'Spine 2', 'Spine 3', 'Ribcage', 'Neck 1', 'Neck 2', 'Head'],
        tgt_match_names=['Cg', 'Pelvis', 'Spine', 'Spine 1', 'Spine 2', 'Neck', 'Neck 1', 'Neck 2', 'Neck 3', 'Neck 4', 'Head'],
        coordinate_search=False,
        verbose=False,
    )

    assert int(result['src_to_tgt'][0]) == -1
    assert int(result['src_to_tgt'][1]) == 0
    assert int(result['src_to_tgt'][2]) == 1
    assert int(result['src_to_tgt'][3]) == 2
    assert int(result['src_to_tgt'][4]) == 3
    assert int(result['src_to_tgt'][5]) == 4
    assert int(result['src_to_tgt'][6]) == 6
    assert int(result['src_to_tgt'][7]) == 8
    assert int(result['src_to_tgt'][8]) == 10


def test_build_target_animation_prefers_retarget_bone_translations() -> None:
    target_tp = SimpleNamespace(
        offsets=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        tpos_anim=SimpleNamespace(parents=np.array([-1, 0, 1], dtype=np.int32)),
        tpos_rots=np.array([_identity_quat(3)], dtype=np.float64),
    )

    retarget_result = {
        'target_world_positions': np.array(
            [[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]],
            dtype=np.float64,
        ),
        'target_world_rotations': np.array([_identity_quat(3)], dtype=np.float64),
        'src_to_tgt': np.array([0, 1, 2], dtype=np.int32),
        'bone_translations': np.zeros((1, 3, 3), dtype=np.float64),
    }

    anim = _build_tpose_aligned_target_animation(retarget_result, target_tp)

    np.testing.assert_allclose(anim.positions[0, 0], np.array([0.0, 0.0, 0.0], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(anim.positions[0, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(anim.positions[0, 2], np.array([0.0, 0.5, 0.0], dtype=np.float64), atol=1e-6)

