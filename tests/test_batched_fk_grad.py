from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from kinematics.forward_kinematics import batched_fk_from_features  # noqa: E402
from utils.quaternion import quat_conjugate, quat_multiply, quat_normalize, quat_to_matrix  # noqa: E402


_BASE_LOCAL_QUATS = quat_normalize(
    torch.tensor(
        [
            [
                [0.83, 0.13, 0.27, 0.47],
                [0.76, 0.31, 0.24, 0.52],
                [0.71, 0.22, 0.56, 0.35],
                [0.69, 0.44, 0.32, 0.47],
            ],
            [
                [0.78, 0.26, 0.39, 0.41],
                [0.73, 0.29, 0.42, 0.45],
                [0.67, 0.38, 0.51, 0.37],
                [0.63, 0.35, 0.48, 0.50],
            ],
            [
                [0.74, 0.22, 0.44, 0.46],
                [0.70, 0.36, 0.33, 0.52],
                [0.65, 0.41, 0.49, 0.40],
                [0.61, 0.38, 0.45, 0.53],
            ],
        ],
        dtype=torch.float64,
    )
)
_BASE_ROOT_POSITIONS = torch.tensor(
    [
        [0.2, 1.0, -0.1],
        [0.2, 1.1, -0.1],
        [0.2, 1.2, -0.1],
    ],
    dtype=torch.float64,
)
_BASE_STRETCH = torch.tensor(
    [
        [1.08, 0.94, 1.02],
        [1.05, 0.97, 1.01],
        [1.02, 1.01, 0.99],
    ],
    dtype=torch.float64,
)


def _identity_quats(joint_count: int, dtype: torch.dtype) -> torch.Tensor:
    quats = torch.zeros((joint_count, 4), dtype=dtype)
    quats[:, 0] = 1.0
    return quats


def _canon_joint_rot(joint_count: int, dtype: torch.dtype) -> torch.Tensor:
    canon = quat_normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.91, 0.22, 0.19, 0.29],
                [0.88, 0.26, 0.31, 0.19],
                [0.84, 0.19, 0.34, 0.37],
            ],
            dtype=dtype,
        )
    )
    canon[0] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype)
    return canon[:joint_count]


def _canonicalize_rot_quats(local_quats: torch.Tensor, canon_joint_rot: torch.Tensor) -> torch.Tensor:
    encoded = local_quats.clone()
    if local_quats.shape[1] > 1:
        canon = canon_joint_rot[None, 1:, :].expand(local_quats.shape[0], -1, -1)
        encoded[:, 1:] = quat_multiply(
            quat_multiply(canon, local_quats[:, 1:]),
            quat_conjugate(canon),
        )
    return encoded


def _matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def _feature_sample(joint_count: int, canon_joint_rot: torch.Tensor, *, root_y_shift: float = 0.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    frame_count = _BASE_LOCAL_QUATS.shape[0]
    local_quats = _BASE_LOCAL_QUATS[:frame_count, :joint_count].to(dtype=dtype)
    encoded_quats = _canonicalize_rot_quats(local_quats, canon_joint_rot.to(dtype=dtype))
    features = torch.zeros((frame_count, joint_count, 13), dtype=dtype)
    features[:, :, 3:9] = _matrix_to_cont6d(quat_to_matrix(encoded_quats))
    features[:, 0, :3] = _BASE_ROOT_POSITIONS[:frame_count].to(dtype=dtype)
    features[:, 0, 1] = features[:, 0, 1] + root_y_shift
    if joint_count > 1:
        features[:, 1:, 9] = _BASE_STRETCH[:frame_count, : joint_count - 1].to(dtype=dtype)
    return features


def _make_diffusion() -> GaussianDiffusion:
    return GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        lambda_fk=1.0,
    )


def _build_fk_loss_inputs(dtype: torch.dtype = torch.float32):
    specs = [
        {
            "object_type": "buffalo",
            "parents": np.array([-1, 0, 1], dtype=np.int64),
            "offsets": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.2, 0.1],
                    [0.1, 0.7, 0.2],
                ],
                dtype=dtype,
            ),
            "root_y_shift": 0.0,
        },
        {
            "object_type": "deer",
            "parents": np.array([-1, 0, 0, 2], dtype=np.int64),
            "offsets": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.4, 0.3, 0.0],
                    [0.0, 0.5, 0.2],
                    [0.2, 0.4, 0.3],
                ],
                dtype=dtype,
            ),
            "root_y_shift": 0.3,
        },
    ]
    frame_count = _BASE_LOCAL_QUATS.shape[0]
    batch_size = len(specs)
    max_joints = max(len(spec["parents"]) for spec in specs)

    features = torch.zeros((batch_size, frame_count, max_joints, 13), dtype=dtype)
    offsets = torch.zeros((batch_size, max_joints, 3), dtype=dtype)
    rest_rotations = torch.zeros((batch_size, max_joints, 4), dtype=dtype)
    rest_rotations[:, :, 0] = 1.0
    canon_joint_rot = torch.zeros((batch_size, max_joints, 4), dtype=dtype)
    canon_joint_rot[:, :, 0] = 1.0
    actual_joints = []
    object_types = []
    parents_list = []
    joint_mask = torch.zeros((batch_size, 1, 1, max_joints), dtype=dtype)

    for batch_index, spec in enumerate(specs):
        joint_count = len(spec["parents"])
        actual_joints.append(joint_count)
        object_types.append(spec["object_type"])
        parents_list.append(spec["parents"])
        offsets[batch_index, :joint_count] = spec["offsets"]
        canon_joint_rot[batch_index, :joint_count] = _canon_joint_rot(joint_count, dtype)
        features[batch_index, :, :joint_count] = _feature_sample(
            joint_count,
            canon_joint_rot[batch_index, :joint_count],
            root_y_shift=spec["root_y_shift"],
            dtype=dtype,
        )
        joint_mask[batch_index, 0, 0, :joint_count] = 1.0

    target_denorm = features.permute(0, 2, 3, 1).contiguous()
    temp_mask = torch.ones((batch_size, 1, 1, frame_count), dtype=dtype)
    model_kwargs = {
        "y": {
            "offsets": offsets,
            "rest_rotations": rest_rotations,
            "canon_joint_rot": canon_joint_rot,
            "parents": parents_list,
            "object_type": object_types,
            "norm_schema_version": torch.full((batch_size,), 4, dtype=torch.long),
        }
    }
    return target_denorm, model_kwargs, temp_mask, joint_mask, torch.tensor(actual_joints, dtype=torch.long)


_NONIDENTITY_REST = quat_normalize(
    torch.tensor(
        [
            [0.97, 0.10, 0.17, 0.13],
            [0.88, 0.31, 0.22, 0.27],
            [0.82, 0.24, 0.39, 0.34],
        ],
        dtype=torch.float64,
    )
)


@pytest.mark.parametrize(
    ("joint_index", "channel", "rest_kind"),
    [
        (0, 3, "identity"),  # root rotation channel: quat_multiply(rot_q[:, :, :1], root_rest) path
        (1, 3, "identity"),  # non-root: canon conjugation + parent-cumulative rest composition
        (2, 6, "identity"),  # deeper non-root joint, different 6D channel
        (1, 4, "nonidentity"),  # exercise the non-identity-rest geometric path that was fixed
        (2, 8, "nonidentity"),
    ],
    ids=[
        "root-ch3-id",
        "nonroot-j1-ch3-id",
        "nonroot-j2-ch6-id",
        "nonroot-j1-ch4-nonid",
        "nonroot-j2-ch8-nonid",
    ],
)
def test_batched_fk_backward_matches_finite_difference_for_rotation_channel(
    joint_index: int, channel: int, rest_kind: str
) -> None:
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1],
            [0.1, 0.7, 0.2],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    if rest_kind == "identity":
        rest_rotations = _identity_quats(3, torch.float64).unsqueeze(0)
    else:
        rest_rotations = _NONIDENTITY_REST.clone().unsqueeze(0)
    canon_joint_rot = _canon_joint_rot(3, torch.float64).unsqueeze(0)
    features = _feature_sample(3, canon_joint_rot[0], dtype=torch.float64).unsqueeze(0)
    features.requires_grad_()

    loss = batched_fk_from_features(
        features,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    loss.backward()

    autograd_grad = float(features.grad[0, 1, joint_index, channel].item())
    eps = 1e-4

    with torch.no_grad():
        positive = features.detach().clone()
        negative = features.detach().clone()
        positive[0, 1, joint_index, channel] += eps
        negative[0, 1, joint_index, channel] -= eps

    positive_loss = batched_fk_from_features(
        positive,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    negative_loss = batched_fk_from_features(
        negative,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    finite_difference = float(((positive_loss - negative_loss) / (2.0 * eps)).item())

    assert np.isfinite(autograd_grad)
    assert autograd_grad == pytest.approx(finite_difference, rel=5e-3, abs=5e-4)


def test_batched_fk_backward_matches_finite_difference_for_stretch_channel() -> None:
    parents = np.array([-1, 0, 1], dtype=np.int64)
    offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.2, 0.1],
            [0.1, 0.7, 0.2],
        ],
        dtype=torch.float64,
    ).unsqueeze(0)
    rest_rotations = _identity_quats(3, torch.float64).unsqueeze(0)
    canon_joint_rot = _canon_joint_rot(3, torch.float64).unsqueeze(0)
    features = _feature_sample(3, canon_joint_rot[0], dtype=torch.float64).unsqueeze(0)
    features.requires_grad_()

    loss = batched_fk_from_features(
        features,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    loss.backward()

    autograd_grad = float(features.grad[0, 1, 1, 9].item())
    eps = 1e-4

    with torch.no_grad():
        positive = features.detach().clone()
        negative = features.detach().clone()
        positive[0, 1, 1, 9] += eps
        negative[0, 1, 1, 9] -= eps

    positive_loss = batched_fk_from_features(
        positive,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    negative_loss = batched_fk_from_features(
        negative,
        offsets,
        rest_rotations,
        canon_joint_rot,
        parents,
    ).square().mean()
    finite_difference = float(((positive_loss - negative_loss) / (2.0 * eps)).item())

    assert np.isfinite(autograd_grad)
    assert autograd_grad == pytest.approx(finite_difference, rel=5e-3, abs=5e-4)


def test_fk_feature_loss_is_zero_for_perfect_prediction() -> None:
    diffusion = _make_diffusion()
    target_denorm, model_kwargs, temp_mask, joint_mask, actual_joints = _build_fk_loss_inputs()

    pos_loss, vel_loss = diffusion.fk_feature_loss(
        target_denorm,
        target_denorm.clone(),
        temp_mask,
        joint_mask,
        actual_joints,
        model_kwargs,
    )

    assert float(pos_loss.item()) == pytest.approx(0.0, abs=1e-8)
    assert float(vel_loss.item()) == pytest.approx(0.0, abs=1e-8)


def test_fk_feature_loss_keeps_padded_joint_gradients_zero() -> None:
    diffusion = _make_diffusion()
    joint_count = 2
    max_joints = 4
    frame_count = _BASE_LOCAL_QUATS.shape[0]
    dtype = torch.float32
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = torch.zeros((1, max_joints, 3), dtype=dtype)
    offsets[0, :joint_count] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.3, 0.2],
        ],
        dtype=dtype,
    )
    rest_rotations = torch.zeros((1, max_joints, 4), dtype=dtype)
    rest_rotations[:, :, 0] = 1.0
    canon_joint_rot = torch.zeros((1, max_joints, 4), dtype=dtype)
    canon_joint_rot[:, :, 0] = 1.0
    canon_joint_rot[0, :joint_count] = _canon_joint_rot(joint_count, dtype)

    features = torch.zeros((1, frame_count, max_joints, 13), dtype=dtype)
    features[0, :, :joint_count] = _feature_sample(joint_count, canon_joint_rot[0, :joint_count], dtype=dtype)
    target_denorm = features.permute(0, 2, 3, 1).contiguous()
    pred_denorm = target_denorm.clone().detach()
    pred_denorm[0, 1, 9, 1] += 0.05
    pred_denorm[0, 2, 3, 0] = 1.5
    pred_denorm[0, 3, 9, 0] = 0.8
    pred_denorm.requires_grad_()

    temp_mask = torch.ones((1, 1, 1, frame_count), dtype=dtype)
    joint_mask = torch.zeros((1, 1, 1, max_joints), dtype=dtype)
    joint_mask[:, :, :, :joint_count] = 1.0
    actual_joints = torch.tensor([joint_count], dtype=torch.long)
    model_kwargs = {
        "y": {
            "offsets": offsets,
            "rest_rotations": rest_rotations,
            "canon_joint_rot": canon_joint_rot,
            "parents": [parents],
            "object_type": ["buffalo"],
            "norm_schema_version": torch.tensor([4], dtype=torch.long),
        }
    }

    pos_loss, vel_loss = diffusion.fk_feature_loss(
        target_denorm,
        pred_denorm,
        temp_mask,
        joint_mask,
        actual_joints,
        model_kwargs,
    )
    (pos_loss + vel_loss).backward()

    assert float(pred_denorm.grad[0, joint_count:].abs().max().item()) == pytest.approx(0.0, abs=1e-9)
    assert float(pred_denorm.grad[0, :joint_count].abs().sum().item()) > 0.0


def test_fk_feature_loss_backward_handles_mixed_topologies() -> None:
    diffusion = _make_diffusion()
    target_denorm, model_kwargs, temp_mask, joint_mask, actual_joints = _build_fk_loss_inputs()
    pred_denorm = target_denorm.clone().detach()
    pred_denorm[0, 1, 9, 1] += 0.05
    pred_denorm[1, 2, 9, 0] -= 0.04
    pred_denorm.requires_grad_()

    pos_loss, vel_loss = diffusion.fk_feature_loss(
        target_denorm,
        pred_denorm,
        temp_mask,
        joint_mask,
        actual_joints,
        model_kwargs,
    )
    total_loss = pos_loss + vel_loss
    total_loss.backward()

    assert bool(torch.isfinite(total_loss).item())
    assert bool(torch.isfinite(pred_denorm.grad).all().item())
    assert float(pred_denorm.grad[0, : int(actual_joints[0].item())].abs().sum().item()) > 0.0
    assert float(pred_denorm.grad[1, : int(actual_joints[1].item())].abs().sum().item()) > 0.0