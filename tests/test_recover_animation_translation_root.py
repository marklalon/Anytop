import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.features import build_canonical_joint_rotations, decanonicalize_delta_quaternions
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    find_translation_root,
    xz_locomotion_extent,
    get_6d_rep,
    get_common_features_from_T_pose,
    get_hml_aligned_anim,
    get_motion,
    infer_translation_root_index_from_features,
    mirror_features_with_safeguards,
    move_xz_to_origin,
    positions_global,
    recover_animation_from_motion_np,
    recover_bvh_export_animation_from_motion_np,
    recover_from_bvh_ric_np,
    recover_root_quat_and_pos_np,
)
from utils.rotation_conversions import rotation_6d_to_matrix_np


def _identity_cont6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _identity_quats(count: int) -> np.ndarray:
    quat = np.zeros((count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat


def _canon_from_offsets(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    rest_positions = np.zeros_like(offsets, dtype=np.float64)
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
    return build_canonical_joint_rotations(rest_positions, parents)


def _recover_kwargs_from_cond(cond, *, offsets=None) -> dict[str, object]:
    recover_offsets = np.asarray(offsets, dtype=np.float64) if offsets is not None else None
    canon_joint_rot = (
        _canon_from_offsets(recover_offsets, cond['parents'])
        if recover_offsets is not None
        else np.asarray(cond['canon_joint_rot'], dtype=np.float32)
    )
    return {
        'tpose_rest_rotations': np.asarray(cond['rest_rotations'], dtype=np.float32),
        'canon_joint_rot': np.asarray(canon_joint_rot, dtype=np.float32),
        'norm_schema_version': int(cond.get('norm_schema_version', 4) or 4),
    }


def _load_motion_metadata_entry(opt, motion_name: str) -> dict[str, object]:
    data_root = opt.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_root)
    motion_metadata = load_motion_metadata(data_root).get(motion_name)
    if not isinstance(motion_metadata, dict):
        raise AssertionError(f"missing motion metadata for {motion_name}")
    return motion_metadata


def _with_translation_root_index(motion_metadata, motion: np.ndarray, cond) -> dict[str, object]:
    if 'translation_root_index' in motion_metadata:
        return motion_metadata
    updated_motion_metadata = dict(motion_metadata)
    updated_motion_metadata['translation_root_index'] = infer_translation_root_index_from_features(
        motion,
        cond['parents'],
        cond['offsets'],
    )
    return updated_motion_metadata


def test_recover_animation_uses_effective_translation_root_feature_row():
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    frames = 4
    rest_quat = _identity_quats(3).astype(np.float64)
    trajectory_x = np.arange(frames, dtype=np.float64)
    rotations = Quaternions(np.tile(rest_quat[None, :, :], (frames, 1, 1)))
    positions = np.zeros((frames, 3, 3), dtype=np.float64)
    positions[:, 0] = np.array([[0.0, 0.0, 1.0]] * frames, dtype=np.float64)
    positions[:, 1] = np.array([[0.0, 1.0, 0.0]] * frames, dtype=np.float64)
    positions[:, 2] = np.stack(
        [trajectory_x, np.zeros_like(trajectory_x), -np.ones_like(trajectory_x)],
        axis=-1,
    )
    source_anim = Animation(rotations, positions, Quaternions.id(0), offsets.astype(np.float64), parents)

    features, _feature_parents, _max_joints, _feature_anim, _export_anim, _is_loop, translation_root_index, _root_translation_xz = get_motion(
        source_anim,
        FOOT_CONTACT_VEL_THRESH,
        'Synthetic',
        max_joints=3,
        offsets=offsets.astype(np.float64),
        foot_indices=[],
        tpos_rots=Quaternions(rest_quat[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
    )

    assert features is not None
    assert int(translation_root_index) == 2

    anim, has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        translation_root_index=2,
        tpose_rest_rotations=rest_quat.astype(np.float32),
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
        allow_infer=True,
    )
    global_pos = positions_global(anim)

    np.testing.assert_allclose(
        global_pos[:, 0],
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        global_pos[:, 1],
        np.array([[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        global_pos[:, 2],
        np.array([[0.0, 2.0, -1.0], [1.0, 2.0, -1.0], [2.0, 2.0, -1.0], [3.0, 2.0, -1.0]], dtype=np.float32),
        atol=1e-5,
    )
    assert has_animated_pos is False


def test_v4_get_motion_folds_effective_root_trajectory_onto_joint_zero():
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rest_quat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotations = Quaternions(np.tile(rest_quat[None, :, :], (3, 1, 1)))
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 1] = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    anim = Animation(rotations, positions, Quaternions.id(0), offsets, parents)

    features, _feature_parents, _max_joints, _feature_anim, export_anim, _is_loop, _translation_root_index, _root_translation_xz = get_motion(
        anim,
        FOOT_CONTACT_VEL_THRESH,
        'Synthetic',
        max_joints=2,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=Quaternions(rest_quat[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
    )

    assert features is not None
    np.testing.assert_allclose(features[:-1, 0, 9], np.array([1.0, 1.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(features[:, 0, :3], np.array([[0.0, 1.0, 0.0]] * 3, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(positions_global(export_anim), positions_global(anim), atol=1e-5)

    recovered_anim, _has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        translation_root_index=1,
        tpose_rest_rotations=rest_quat.astype(np.float32),
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
        allow_infer=True,
    )
    recovered_global = positions_global(recovered_anim)

    np.testing.assert_allclose(
        recovered_global[:, 0],
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-5,
    )


def test_v4_get_motion_preserves_small_effective_root_trajectory_without_threshold():
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rest_quat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotations = Quaternions(np.tile(rest_quat[None, :, :], (3, 1, 1)))
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 1] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.4, 1.0, 0.0],
            [0.8, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    anim = Animation(rotations, positions, Quaternions.id(0), offsets, parents)

    features, _feature_parents, _max_joints, _feature_anim, export_anim, _is_loop, translation_root_index, _root_translation_xz = get_motion(
        anim,
        FOOT_CONTACT_VEL_THRESH,
        'Synthetic',
        max_joints=2,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=Quaternions(rest_quat[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
    )

    assert int(translation_root_index) == 1
    np.testing.assert_allclose(features[:-1, 0, 9], np.array([0.4, 0.4], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(features[:, 1, 9], np.ones((3,), dtype=np.float32), atol=1e-5)

    recovered_anim, _has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        translation_root_index=1,
        tpose_rest_rotations=rest_quat.astype(np.float32),
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
        allow_infer=True,
    )
    assert _has_animated_pos is False
    np.testing.assert_allclose(
        positions_global(recovered_anim)[:, 0],
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.4, 1.0, 0.0],
                [0.8, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        positions_global(recovered_anim)[:, 1],
        np.array(
            [
                [0.0, 2.0, 0.0],
                [0.4, 2.0, 0.0],
                [0.8, 2.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atol=1e-5,
    )


def test_recover_bvh_export_animation_matches_export_anim_for_root_locomotion():
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rest_quat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotations = Quaternions(np.tile(rest_quat[None, :, :], (3, 1, 1)))
    positions = np.zeros((3, 2, 3), dtype=np.float64)
    positions[:, 0] = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    positions[:, 1] = np.array([[0.0, 1.0, 0.0]] * 3, dtype=np.float64)
    anim = Animation(rotations, positions, Quaternions.id(0), offsets, parents)

    features, _feature_parents, _max_joints, _feature_anim, export_anim, _is_loop, translation_root_index, _root_translation_xz = get_motion(
        anim,
        FOOT_CONTACT_VEL_THRESH,
        'Synthetic',
        max_joints=2,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=Quaternions(rest_quat[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
    )

    assert int(translation_root_index) == 0

    recovered_export_anim, _joint_names, _has_export_pos = recover_bvh_export_animation_from_motion_np(
        features,
        parents,
        offsets,
        joint_names=['Root', 'Joint'],
        translation_root_index=0,
        tpose_rest_rotations=rest_quat.astype(np.float32),
        canon_joint_rot=rest_quat.astype(np.float32),
        norm_schema_version=4,
        allow_infer=True,
    )
    np.testing.assert_allclose(
        positions_global(recovered_export_anim),
        positions_global(export_anim),
        atol=1e-5,
    )

    import torch
    from Anytop.kinematics.forward_kinematics import batched_fk_from_features

    fk_positions = batched_fk_from_features(
        torch.from_numpy(features[None, ...]).to(torch.float32),
        torch.from_numpy(offsets[None, ...]).to(torch.float32),
        torch.from_numpy(rest_quat[None, ...]).to(torch.float32),
        torch.from_numpy(rest_quat[None, ...]).to(torch.float32),
        parents,
    ).cpu().numpy()[0]

    np.testing.assert_allclose(fk_positions, positions_global(export_anim), atol=1e-5)


def test_find_translation_root_detects_single_chain_descendant():
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = np.zeros((3, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (6, len(parents), 1),
        )
    )
    positions = np.zeros((6, 3, 3), dtype=np.float64)
    positions[:, 2, 0] = np.linspace(0.0, 5.0, num=6, dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)

    assert find_translation_root(anim) == 2


def test_find_translation_root_ignores_descendants_after_branch():
    parents = np.array([-1, 0, 1, 1, 3], dtype=np.int64)
    offsets = np.zeros((5, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (6, len(parents), 1),
        )
    )
    positions = np.zeros((6, 5, 3), dtype=np.float64)
    positions[:, 4, 0] = np.linspace(0.0, 5.0, num=6, dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)

    assert find_translation_root(anim) == 0


def test_find_translation_root_limits_search_depth_to_five_descendants():
    parents = np.array([-1, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.zeros((7, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (6, len(parents), 1),
        )
    )
    positions = np.zeros((6, 7, 3), dtype=np.float64)
    positions[:, 6, 0] = np.linspace(0.0, 5.0, num=6, dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)

    assert find_translation_root(anim) == 0


def test_xz_locomotion_extent_ignores_static_origin_offset_after_initial_root_centering():
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.zeros((2, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (30, len(parents), 1),
        )
    )
    positions = np.zeros((30, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = np.linspace(10.0, 11.0, num=30, dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)
    centered_anim, root_translation_xz = move_xz_to_origin(anim)

    np.testing.assert_allclose(root_translation_xz, np.array([10.0, 0.0, 0.0], dtype=np.float64), atol=1e-8)
    assert xz_locomotion_extent(centered_anim, 1) == pytest.approx(1.0)


def test_xz_locomotion_extent_still_detects_true_locomotion_after_initial_root_centering():
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.zeros((2, 3), dtype=np.float64)
    rotations = Quaternions(
        np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            (30, len(parents), 1),
        )
    )
    positions = np.zeros((30, 2, 3), dtype=np.float64)
    positions[:, 1, 0] = np.linspace(0.0, 4.0, num=30, dtype=np.float64)

    anim = Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)
    centered_anim, root_translation_xz = move_xz_to_origin(anim)

    np.testing.assert_allclose(root_translation_xz, np.array([0.0, 0.0, 0.0], dtype=np.float64), atol=1e-8)
    assert xz_locomotion_extent(centered_anim, 1) == pytest.approx(4.0)


def test_raw_tpose_animation_input_reapplies_tpose_normalization():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpose_fbx = os.path.join(
        repo_root,
        'dataset',
        'truebones',
        'zoo',
        'Truebone_Z-OO',
        'Buffalo',
        'Buffalo-TPOSE.fbx',
    )
    if not os.path.isfile(tpose_fbx):
        pytest.skip(f'missing T-pose FBX: {tpose_fbx}')

    tp = get_common_features_from_T_pose(
        tpose_fbx,
        'Buffalo',
        augment_leaf_rotation_helpers=True,
        max_joints=53,
    )

    squared_positions_error: dict[str, float] = {}
    aligned_anim, _export_anim, _names, _root_translation_xz = get_hml_aligned_anim(
        tp.tpos_anim,
        'Buffalo',
        tp.tpos_rots,
        tp.offsets,
        squared_positions_error,
        scale_factor=float(tp.scale_factor),
        foot_indices=tp.foot_indices,
        orientation_quat=np.asarray(tp.orientation_quat, dtype=np.float64),
        helper_metadata=tp.helper_metadata,
        animation_input_is_tpose_aligned=False,
    )

    expected_identity = np.zeros((len(tp.names), 4), dtype=np.float64)
    expected_identity[:, 0] = 1.0
    np.testing.assert_allclose(aligned_anim.rotations.qs[0], expected_identity, atol=1e-5)


def test_recover_animation_matches_safeguarded_horse_target_globals():
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Horse']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    motion_path, motion_name = _find_motion_file(motion_dir, 'Horse_RunLoop_*.npy')
    raw = np.load(motion_path).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, motion_name),
        raw,
        cond,
    )
    mirrored, mirrored_offsets = mirror_features_with_safeguards(raw, cond, motion_metadata=motion_metadata)
    mirror_recover_kwargs = _recover_kwargs_from_cond(cond, offsets=mirrored_offsets)
    target_global = recover_from_bvh_ric_np(
        mirrored,
        parents=cond['parents'],
        offsets=mirrored_offsets,
        motion_metadata=motion_metadata,
        **mirror_recover_kwargs,
    )

    anim, has_animated_pos = recover_animation_from_motion_np(
        mirrored,
        cond['parents'],
        mirrored_offsets,
        motion_metadata=motion_metadata,
        **mirror_recover_kwargs,
    )
    recovered_global = positions_global(anim)

    np.testing.assert_allclose(recovered_global, target_global, atol=1e-4)
    assert has_animated_pos is True


def _find_motion_file(motion_dir: str, pattern: str) -> tuple[str, str]:
    """Find a motion file by glob pattern, returning (full_path, basename)."""
    files = sorted(glob.glob(os.path.join(motion_dir, pattern)))
    assert files, f"No files matching '{pattern}' in {motion_dir}"
    return files[0], os.path.basename(files[0])


def _recover_pre_normalized_bvh_rotations(raw: np.ndarray, cond, motion_metadata) -> Quaternions:
    parents = np.asarray(cond['parents'], dtype=np.int64)
    offsets = np.asarray(cond['offsets'], dtype=np.float64)
    r_rot_quat, _r_pos = recover_root_quat_and_pos_np(
        raw,
        parents=parents,
        offsets=offsets,
        motion_metadata=motion_metadata,
    )

    all_qs = np.zeros((raw.shape[0], raw.shape[1], 4), dtype=np.float64)
    all_qs[..., 0] = 1.0
    all_qs[:, 0] = np.asarray(r_rot_quat.qs, dtype=np.float64)
    if raw.shape[1] > 1:
        nonroot_delta = Quaternions.from_transforms(
            rotation_6d_to_matrix_np(np.asarray(raw[:, 1:, 3:9], dtype=np.float64))
        ).qs
        canon_joint_rot = np.asarray(cond.get('canon_joint_rot'), dtype=np.float64)
        if canon_joint_rot.shape[0] != raw.shape[1]:
            canon_joint_rot = _canon_from_offsets(offsets, parents)
        all_qs[:, 1:] = decanonicalize_delta_quaternions(
            nonroot_delta,
            canon_joint_rot[None, 1:, :],
        )

    return Quaternions(all_qs)


@pytest.mark.parametrize(
    ("object_type", "motion_pattern"),
    [
        ("Buffalo", "Buffalo_AlertIdle_*.npy"),
        ("Horse", "Horse_GetUp_*.npy"),
    ],
)
def test_feature_roundtrip_preserves_dataset_motion_features(object_type: str, motion_pattern: str):
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()[object_type]
    if int(cond.get('norm_schema_version', 0) or 0) < 4:
        pytest.skip('dataset roundtrip regression requires regenerated schema v4 motion features')

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    motion_path, motion_name = _find_motion_file(motion_dir, motion_pattern)

    raw = np.load(motion_path).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, motion_name),
        raw,
        cond,
    )
    anim, _has_animated_pos = recover_animation_from_motion_np(
        raw,
        cond['parents'],
        cond['offsets'],
        motion_metadata=motion_metadata,
        **_recover_kwargs_from_cond(cond),
    )

    tp = get_common_features_from_T_pose(
        cond['orientation_reference_fbx_path'],
        object_type,
        augment_leaf_rotation_helpers=True,
        max_joints=len(cond['parents']),
    )
    squared_positions_error: dict[str, float] = {}
    rebuilt, _parents, _max_joints, _feature_anim, _export_anim, _is_loop, _translation_root_index, _root_translation_xz = get_motion(
        anim,
        FOOT_CONTACT_VEL_THRESH,
        object_type,
        len(cond['parents']),
        tp.offsets,
        tp.foot_indices,
        tp.tpos_rots,
        squared_positions_error,
        scale_factor=float(cond['scale_factor']),
        orientation_quat=np.asarray(cond['orientation_quat'], dtype=np.float64),
        helper_metadata=tp.helper_metadata,
        canon_joint_rot=np.asarray(cond['canon_joint_rot'], dtype=np.float32),
        norm_schema_version=int(cond.get('norm_schema_version', 4) or 4),
    )

    assert rebuilt is not None
    rebuilt = np.asarray(rebuilt, dtype=np.float32)
    np.testing.assert_allclose(rebuilt[:, 1:, 3:9], raw[:, 1:, 3:9], atol=1e-5)
    np.testing.assert_allclose(rebuilt[:, 0, :], raw[:, 0, :], atol=2e-1)
    np.testing.assert_allclose(rebuilt[:, 1:, [0, 1, 2, 10, 11]], raw[:, 1:, [0, 1, 2, 10, 11]], atol=1e-5)


def test_from_transforms_preserves_positive_trace_ninety_degree_yaw():
    matrix = np.array(
        [[[-5.2504788e-08, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -5.2504788e-08]]],
        dtype=np.float32,
    )

    quat = Quaternions.from_transforms(matrix)

    np.testing.assert_allclose(quat.transforms(), matrix.astype(np.float64), atol=1e-6)


def test_recover_bvh_rot_np_root_rotation_consistency():
    """Verify recover_from_bvh_rot_np produces smooth, sign-consistent root rotations.

    This tests the replacement of ``rotations[:, 0] = -r_rot_quat * rotations[:, 0]``
    with ``_normalize_quaternion_signs`` to ensure the root rotation semantics
    remain correct for BVH export.
    """
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_rot_np

    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Horse']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    motion_path, motion_name = _find_motion_file(motion_dir, 'Horse_RunLoop_*.npy')
    raw = np.load(motion_path).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, motion_name),
        raw,
        cond,
    )

    global_pos, anim = recover_from_bvh_rot_np(
        raw,
        cond['parents'],
        cond['offsets'],
        motion_metadata=motion_metadata,
        **_recover_kwargs_from_cond(cond),
    )

    root_quats = anim.rotations.qs[:, 0]  # (F, 4)

    # 1. First-frame root quaternion w >= 0 (normalization invariant)
    assert root_quats[0, 0] >= 0, "First-frame root quaternion w should be >= 0"

    # 2. Temporal sign consistency: dot product with previous frame should be >= 0
    dots = np.sum(root_quats[1:] * root_quats[:-1], axis=1)
    assert np.all(dots >= -1e-6), (
        f"Root quaternion sign flips detected: min dot={np.min(dots):.6f}"
    )

    # 3. Root global positions are smooth (no large jumps)
    root_pos = global_pos[:, 0]  # (F, 3)
    root_deltas = np.linalg.norm(root_pos[1:] - root_pos[:-1], axis=1)
    assert np.all(root_deltas < 1.0), (
        f"Root position has large jumps: max delta={np.max(root_deltas):.4f}"
    )


def test_recover_bvh_rot_np_normalizes_non_root_quaternion_sign_flips():
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_rot_np

    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Alligator']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    motion_path, motion_name = _find_motion_file(motion_dir, 'Alligator_Bite1_*.npy')
    raw = np.load(motion_path).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, motion_name),
        raw,
        cond,
    )

    raw_rotations = _recover_pre_normalized_bvh_rotations(raw, cond, motion_metadata)
    raw_dots = np.sum(raw_rotations.qs[1:] * raw_rotations.qs[:-1], axis=-1)
    assert np.any(raw_dots[:, 1:] < 0.0), 'fixture should exercise non-root quaternion sign flips'

    _global_pos, anim = recover_from_bvh_rot_np(
        raw,
        cond['parents'],
        cond['offsets'],
        motion_metadata=motion_metadata,
        **_recover_kwargs_from_cond(cond),
    )

    fixed_dots = np.sum(anim.rotations.qs[1:] * anim.rotations.qs[:-1], axis=-1)
    assert np.all(fixed_dots[:, 1:] >= -1e-6), (
        f"Non-root quaternion sign flips remain after normalization: min dot={np.min(fixed_dots[:, 1:]):.6f}"
    )

    np.testing.assert_allclose(
        raw_rotations.transforms(),
        anim.rotations.transforms(),
        atol=1e-6,
    )


def test_recover_animation_hound_mirror_matches_world_x_reflection():
    opt = get_opt(None)
    cond = np.load(opt.cond_file, allow_pickle=True).item()['Hound']

    motion_dir = opt.motion_dir
    if not os.path.isabs(motion_dir):
        motion_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), motion_dir)

    motion_path, motion_name = _find_motion_file(motion_dir, 'Hound_Attack_*.npy')
    raw = np.load(motion_path).astype(np.float32, copy=False)
    motion_metadata = _with_translation_root_index(
        _load_motion_metadata_entry(opt, motion_name),
        raw,
        cond,
    )
    mirrored, mirrored_offsets = mirror_features_with_safeguards(raw, cond, motion_metadata=motion_metadata)

    clean_anim, _ = recover_animation_from_motion_np(
        raw,
        cond['parents'],
        cond['offsets'],
        motion_metadata=motion_metadata,
        **_recover_kwargs_from_cond(cond),
    )
    mirror_anim, _ = recover_animation_from_motion_np(
        mirrored,
        cond['parents'],
        mirrored_offsets,
        motion_metadata=motion_metadata,
        **_recover_kwargs_from_cond(cond, offsets=mirrored_offsets),
    )
    clean_global = positions_global(clean_anim)
    mirror_global = positions_global(mirror_anim)

    spi = np.asarray(cond['symmetry_partner_indices'], dtype=np.int64)
    perm = np.arange(len(spi), dtype=np.int64)
    perm[spi >= 0] = spi[spi >= 0]

    expected_x = clean_global[:, perm].copy()
    expected_x[..., 0] *= -1.0
    reflected_z = clean_global[:, perm].copy()
    reflected_z[..., 2] *= -1.0

    x_error = float(np.abs(expected_x - mirror_global).mean())
    z_error = float(np.abs(reflected_z - mirror_global).mean())

    assert x_error < 0.25
    assert x_error * 2.0 < z_error