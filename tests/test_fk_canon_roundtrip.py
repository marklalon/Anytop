from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.motion_process import get_motion  # noqa: E402
from kinematics.forward_kinematics import batched_fk_from_features  # noqa: E402
from motion_lib.Animation import Animation, positions_global  # noqa: E402
from motion_lib.Quaternions import Quaternions  # noqa: E402
from utils.quaternion import quat_conjugate, quat_multiply, quat_normalize, quat_to_matrix  # noqa: E402


def _identity_quats_torch(joint_count: int, dtype: torch.dtype) -> torch.Tensor:
    quats = torch.zeros((joint_count, 4), dtype=dtype)
    quats[:, 0] = 1.0
    return quats


def _identity_quats_np(joint_count: int) -> np.ndarray:
    quats = np.zeros((joint_count, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    return quats


def _quat_axis_deg(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half_angle = np.deg2rad(degrees) * 0.5
    return np.concatenate(([np.cos(half_angle)], axis * np.sin(half_angle)))


def _canon_variant(joint_count: int, variant: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if variant == 0:
        return _identity_quats_torch(joint_count, dtype)

    canon = quat_normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.91, 0.22, 0.19, 0.29],
                [0.88, 0.26, 0.31, 0.19],
            ],
            dtype=dtype,
        )
    )
    if variant == 2:
        canon[1:] = quat_normalize(
            torch.tensor(
                [
                    [0.84, 0.19, 0.34, 0.37],
                    [0.79, 0.28, 0.41, 0.35],
                ],
                dtype=dtype,
            )
        )
    canon[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return canon[:joint_count]


def _encode_features(
    local_quats: torch.Tensor,
    root_translation: torch.Tensor,
    stretch: torch.Tensor,
    canon_joint_rot: torch.Tensor,
) -> torch.Tensor:
    encoded_quats = local_quats.clone()
    if local_quats.shape[1] > 1:
        canon = canon_joint_rot[None, 1:, :].expand(local_quats.shape[0], -1, -1)
        encoded_quats[:, 1:] = quat_multiply(
            quat_multiply(canon, local_quats[:, 1:]),
            quat_conjugate(canon),
        )

    features = torch.zeros((local_quats.shape[0], local_quats.shape[1], 13), dtype=local_quats.dtype)
    features[:, :, 3:9] = torch.cat([quat_to_matrix(encoded_quats)[..., 0], quat_to_matrix(encoded_quats)[..., 1]], dim=-1)
    features[:, 0, :3] = root_translation
    if local_quats.shape[1] > 1:
        features[:, 1:, 9] = stretch[:, 1:]
    return features


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(quat_to_matrix(q), v.unsqueeze(-1)).squeeze(-1)


def _manual_fk(
    local_quats: torch.Tensor,
    root_translation: torch.Tensor,
    offsets: torch.Tensor,
    stretch: torch.Tensor,
    parents: np.ndarray,
) -> torch.Tensor:
    joint_count = local_quats.shape[1]
    world_q = [local_quats[:, 0]]
    world_pos = [root_translation]
    for joint_index in range(1, joint_count):
        parent_index = int(parents[joint_index])
        parent_q = world_q[parent_index]
        world_q.append(quat_multiply(parent_q, local_quats[:, joint_index]))
        local_pos = offsets[joint_index].unsqueeze(0) * stretch[:, joint_index : joint_index + 1]
        world_pos.append(world_pos[parent_index] + _quat_apply(parent_q, local_pos))
    return torch.stack(world_pos, dim=1)


def _decode_positions(features: torch.Tensor, offsets: torch.Tensor, canon_joint_rot: torch.Tensor, parents: np.ndarray) -> torch.Tensor:
    joint_count = offsets.shape[0]
    return batched_fk_from_features(
        features.unsqueeze(0),
        offsets.unsqueeze(0),
        _identity_quats_torch(joint_count, features.dtype).unsqueeze(0),
        canon_joint_rot.unsqueeze(0),
        parents,
    )[0]


@pytest.mark.parametrize("variant", [0, 1, 2], ids=["canon-id", "canon-tilted", "canon-mixed"])
def test_batched_fk_roundtrip_decodes_canonicalized_local_rotations(variant: int) -> None:
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )
    local_quats = quat_normalize(
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.83, 0.13, 0.27, 0.47],
                    [0.76, 0.31, 0.24, 0.52],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.78, 0.26, 0.39, 0.41],
                    [0.73, 0.29, 0.42, 0.45],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.74, 0.22, 0.44, 0.46],
                    [0.70, 0.36, 0.33, 0.52],
                ],
            ],
            dtype=torch.float32,
        )
    )
    root_translation = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.1, 1.1, -0.1],
            [0.2, 1.2, -0.2],
        ],
        dtype=torch.float32,
    )
    stretch = torch.ones((local_quats.shape[0], local_quats.shape[1]), dtype=torch.float32)
    canon_joint_rot = _canon_variant(local_quats.shape[1], variant)

    features = _encode_features(local_quats, root_translation, stretch, canon_joint_rot)
    decoded_positions = _decode_positions(features, offsets, canon_joint_rot, parents)
    expected_positions = _manual_fk(local_quats, root_translation, offsets, stretch, parents)

    torch.testing.assert_close(decoded_positions, expected_positions, atol=1e-5, rtol=1e-5)


def test_batched_fk_roundtrip_handles_nonroot_translation_root() -> None:
    parents = np.array([-1, 0], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    canon_joint_rot = _canon_variant(2, 1).cpu().numpy().astype(np.float32)
    rotations = Quaternions(np.tile(_identity_quats_np(2)[None, :, :], (3, 1, 1)))
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

    features, feature_parents, _max_joints, _motion_anim, _export_anim, _is_loop, translation_root_index, _root_translation_xz = get_motion(
        anim,
        foot_contact_vel_thresh=0.002,
        object_type="Synthetic",
        max_joints=2,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=Quaternions(_identity_quats_np(2)[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=True,
        canon_joint_rot=canon_joint_rot,
        norm_schema_version=4,
    )
    assert features is not None

    fk_positions = batched_fk_from_features(
        torch.from_numpy(features[None, ...]).to(torch.float32),
        torch.from_numpy(offsets[None, ...]).to(torch.float32),
        torch.from_numpy(_identity_quats_np(2)[None, ...]).to(torch.float32),
        torch.from_numpy(canon_joint_rot[None, ...]).to(torch.float32),
        feature_parents,
    ).cpu().numpy()[0]

    assert int(translation_root_index) == 1
    np.testing.assert_allclose(
        fk_positions[:, 0],
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        fk_positions[:, 1],
        np.array(
            [
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )


def test_batched_fk_roundtrip_preserves_stretch_along_rest_offsets() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    local_quats = _identity_quats_torch(3, torch.float32).unsqueeze(0).expand(3, -1, -1).clone()
    root_translation = torch.tensor([[0.0, 1.0, 0.0]] * 3, dtype=torch.float32)
    stretch = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.25, 0.8],
            [1.0, 0.85, 1.15],
        ],
        dtype=torch.float32,
    )
    canon_joint_rot = _canon_variant(3, 2)

    features = _encode_features(local_quats, root_translation, stretch, canon_joint_rot)
    decoded_positions = _decode_positions(features, offsets, canon_joint_rot, parents)
    expected_positions = _manual_fk(local_quats, root_translation, offsets, stretch, parents)

    torch.testing.assert_close(decoded_positions, expected_positions, atol=1e-5, rtol=1e-5)


def test_batched_fk_roundtrip_matches_nonidentity_rest_glb_style_geometry() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.2, 0.6, 0.3],
        ],
        dtype=np.float64,
    )
    rest_rotations = np.stack(
        [
            _quat_axis_deg(np.array([0.0, 1.0, 0.0], dtype=np.float64), 25.0),
            _quat_axis_deg(np.array([1.0, 0.0, 0.0], dtype=np.float64), 40.0),
            _quat_axis_deg(np.array([0.0, 0.0, 1.0], dtype=np.float64), -35.0),
        ],
        axis=0,
    )
    canon_joint_rot = _canon_variant(3, 2).cpu().numpy().astype(np.float32)
    local_rotations = np.stack(
        [
            [
                _quat_axis_deg(np.array([0.0, 1.0, 0.0], dtype=np.float64), 30.0),
                _quat_axis_deg(np.array([1.0, 0.0, 0.0], dtype=np.float64), 20.0),
                _quat_axis_deg(np.array([0.0, 0.0, 1.0], dtype=np.float64), -15.0),
            ],
            [
                _quat_axis_deg(np.array([0.0, 1.0, 0.0], dtype=np.float64), 18.0),
                _quat_axis_deg(np.array([1.0, 0.0, 1.0], dtype=np.float64), 28.0),
                _quat_axis_deg(np.array([0.0, 1.0, 1.0], dtype=np.float64), -22.0),
            ],
            [
                _quat_axis_deg(np.array([0.0, 1.0, 0.0], dtype=np.float64), 8.0),
                _quat_axis_deg(np.array([1.0, 1.0, 0.0], dtype=np.float64), 14.0),
                _quat_axis_deg(np.array([0.0, 1.0, 1.0], dtype=np.float64), -9.0),
            ],
        ],
        axis=0,
    )
    local_positions = np.repeat(offsets[None, :, :], local_rotations.shape[0], axis=0)
    local_positions[:, 0] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 1.2, 0.0],
        ],
        dtype=np.float64,
    )
    processed_anim = Animation(
        Quaternions(local_rotations),
        local_positions,
        Quaternions.id(0),
        offsets,
        parents,
    )

    features, feature_parents, _max_joints, _motion_anim, _export_anim, _is_loop, translation_root_index, _root_translation_xz = get_motion(
        processed_anim,
        foot_contact_vel_thresh=0.002,
        object_type="Synthetic",
        max_joints=3,
        offsets=offsets,
        foot_indices=[],
        tpos_rots=Quaternions(rest_rotations[None, :, :]),
        squared_positions_error={},
        scale_factor=1.0,
        orientation_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        helper_metadata=None,
        animation_input_is_tpose_aligned=False,
        canon_joint_rot=canon_joint_rot,
        norm_schema_version=4,
    )
    assert features is not None
    assert int(translation_root_index) == 0

    fk_positions = batched_fk_from_features(
        torch.from_numpy(features[None, ...]).to(torch.float32),
        torch.from_numpy(offsets[None, ...]).to(torch.float32),
        torch.from_numpy(rest_rotations[None, ...]).to(torch.float32),
        torch.from_numpy(canon_joint_rot[None, ...]).to(torch.float32),
        feature_parents,
    ).cpu().numpy()[0]

    np.testing.assert_allclose(fk_positions, positions_global(processed_anim), atol=1e-5)