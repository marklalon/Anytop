import glob
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    ROOT_XZ_STRIP_THRESHOLD,
    find_translation_root,
    xz_locomotion_extent,
    get_common_features_from_T_pose,
    get_hml_aligned_anim,
    get_motion,
    infer_translation_root_index_from_features,
    move_xz_to_origin,
    positions_global,
    recover_animation_from_motion_np,
)
from data_loaders.truebones.truebones_utils import features as features_module


def _identity_cont6d() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


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
    features = np.zeros((frames, 3, 13), dtype=np.float32)
    features[:, :, 3:9] = _identity_cont6d()

    trajectory_x = np.arange(frames, dtype=np.float32)

    # Joint 2 is the effective translation root: its RIFKE XZ stays at zero, and
    # its X velocity carries the trajectory. Joint 0 remains globally fixed.
    features[:, 0, 0] = -trajectory_x
    features[:, 0, 2] = 1.0
    features[:, 1, 0] = -trajectory_x
    features[:, 1, 1] = 1.0
    features[:, 1, 2] = 1.0
    features[:, 2, 1] = 1.0
    features[:-1, 2, 9] = 1.0

    anim, has_animated_pos = recover_animation_from_motion_np(
        features,
        parents,
        offsets,
        translation_root_index=2,
    )
    global_pos = positions_global(anim)

    np.testing.assert_allclose(global_pos[:, 0], np.array([[0.0, 0.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(global_pos[:, 1], np.array([[0.0, 1.0, 1.0]] * frames, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(
        global_pos[:, 2],
        np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )
    assert has_animated_pos is True


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
    assert xz_locomotion_extent(centered_anim, 1) > ROOT_XZ_STRIP_THRESHOLD


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
        animation_input_is_tpose_aligned=False,
    )

    expected_identity = np.zeros((len(tp.names), 4), dtype=np.float64)
    expected_identity[:, 0] = 1.0
    np.testing.assert_allclose(aligned_anim.rotations.qs[0], expected_identity, atol=1e-5)


def test_common_features_use_bind_pose_not_sampled_frame_zero(monkeypatch):
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rest_rotations = Quaternions.id(3)
    sampled_rotations = Quaternions.id((2, 3))
    sampled_positions = np.repeat(offsets[None], 2, axis=0)
    sampled_positions[0, 1] = [3.0, 0.0, 0.0]
    sampled_positions[0, 2] = [0.0, 4.0, 0.0]

    loaded_anim = Animation(sampled_rotations, sampled_positions, rest_rotations, offsets, parents)

    def fake_load(_path):
        return loaded_anim, ["Root", "Spine", "Head"], 1.0 / 30.0

    monkeypatch.setattr(features_module.FBX, "load", fake_load)

    tp = features_module.get_common_features_from_rest_pose("fake.fbx", "TestCreature")

    rest_global = positions_global(tp.tpos_anim)[0]
    sampled_global = positions_global(loaded_anim[:1])[0]

    assert not np.allclose(rest_global, sampled_global)
    np.testing.assert_allclose(tp.offsets[1:], offsets[1:] * float(tp.scale_factor), atol=1e-8)
    np.testing.assert_allclose(tp.tpos_anim.positions[0, 1:], tp.offsets[1:], atol=1e-8)


def test_common_features_apply_vertical_clamp(monkeypatch):
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    loaded_anim = Animation(
        Quaternions.id((1, 3)),
        offsets[None].copy(),
        Quaternions.id(3),
        offsets,
        parents,
    )

    monkeypatch.setattr(
        features_module.FBX,
        "load",
        lambda _path: (loaded_anim, ["Root", "Spine", "Head"], 1.0 / 30.0),
    )

    clamp_calls = []

    def fake_clamp(anim, object_type):
        clamp_calls.append((object_type, anim.positions.copy()))
        positions = anim.positions.copy()
        positions[:, 0, 1] += 7.0
        return Animation(
            anim.rotations.copy(),
            positions,
            anim.orients.copy(),
            anim.offsets.copy(),
            anim.parents.copy(),
        )

    monkeypatch.setattr(features_module, "clamp_vertical_trajectory", fake_clamp)

    tp = features_module.get_common_features_from_rest_pose("fake.fbx", "Bird")

    assert len(clamp_calls) == 1
    assert clamp_calls[0][0] == "Bird"
    assert tp.tpos_anim.positions[0, 0, 1] == pytest.approx(
        clamp_calls[0][1][0, 0, 1] + 7.0
    )


def _find_motion_file(motion_dir: str, pattern: str) -> tuple[str, str]:
    """Find a motion file by glob pattern, returning (full_path, basename)."""
    files = sorted(glob.glob(os.path.join(motion_dir, pattern)))
    assert files, f"No files matching '{pattern}' in {motion_dir}"
    return files[0], os.path.basename(files[0])


def _recover_pre_normalized_bvh_rotations(raw: np.ndarray, cond, motion_metadata) -> Quaternions:
    """Recover BVH rotations directly from own-rotation feature encoding.

    Each slot stores the joint's own local rotation — no parent-child scatter needed.
    """
    from utils.rotation_conversions import rotation_6d_to_matrix_np as _r6d_to_mat

    cont6d_params_hml_order = _r6d_to_mat(np.asarray(raw[..., :, 3:9], dtype=np.float64))
    return Quaternions.from_transforms(cont6d_params_hml_order)


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
    )

    tp = get_common_features_from_T_pose(
        cond['orientation_reference_fbx_path'],
        object_type,
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
    )

    assert rebuilt is not None
    np.testing.assert_allclose(np.asarray(rebuilt, dtype=np.float32), raw, atol=1e-5)


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

