"""
Production-ready NPY roundtrip utilities.

Functions for encoding, recovering, and loading AnyTop's 13-channel NPY motion features.

"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_rest_positions(offsets: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Forward-kinematics on the rest pose -> (J, 3) global positions."""
    joint_count = len(parents)
    positions = np.zeros((joint_count, 3), dtype=np.float64)
    for joint_idx in range(joint_count):
        parent_idx = parents[joint_idx]
        if parent_idx >= 0:
            positions[joint_idx] = positions[parent_idx] + offsets[joint_idx]
        else:
            positions[joint_idx] = offsets[joint_idx].copy()
    return positions


def get_cont6d_params_own(anim: Any, r_rot: Any) -> np.ndarray:
    """Compute 6D rotation features — each bone stores its OWN rotation."""
    quat_params = anim.rotations
    return quat_params.rotation_matrix(cont6d=True)


def detect_motion_loop(positions: np.ndarray) -> bool:
    """Return True if the last frame's root-relative pose ≈ first frame's."""
    if positions.shape[0] < 2:
        return False
    per_joint_dist = np.linalg.norm(positions[-1] - positions[0], axis=-1)
    return bool(np.mean(per_joint_dist) < 0.05)


def compute_terminal_local_velocity(global_positions, r_rot, is_loop, prev_velocity=None):
    """Terminal-frame local velocity for feature export."""
    joints_num = global_positions.shape[1]
    terminal = np.zeros((joints_num, 3), dtype=np.float32)
    if prev_velocity is not None:
        delta = global_positions[-1] - global_positions[-2]
        terminal = (r_rot[-1] * delta).astype(np.float32)
    if is_loop and global_positions.shape[0] >= 2:
        wrap_delta = global_positions[0] - global_positions[-1]
        wrap_vel = (r_rot[0] * wrap_delta).astype(np.float32)
        if np.linalg.norm(wrap_vel) < np.linalg.norm(terminal):
            terminal = wrap_vel
    return terminal.astype(np.float32)


# ── feature encoding ──────────────────────────────────────────────────────────

def coerce_feature_payload(features_or_payload: Any) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
    """Unpack a roundtrip payload back into (features_tensor, payload_dict).

    Accepts either a dict payload or a plain (F, J, 13) ndarray.
    """
    if isinstance(features_or_payload, dict):
        payload = features_or_payload
        features = np.asarray(payload["features"])
        return features, payload
    return np.asarray(features_or_payload), None


def _require_translation_root_index(
    translation_root_index: Optional[int],
    joint_count: int,
    context: str,
) -> int:
    if translation_root_index is None:
        raise ValueError(f"{context} requires translation_root_index to be provided explicitly")
    try:
        index = int(translation_root_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid translation_root_index: {translation_root_index}") from exc
    if index < 0 or index >= int(joint_count):
        raise ValueError(
            f"{context} translation_root_index out of range: {index} for {joint_count} joints"
        )
    return index


def _recover_from_production_motion_features(
    features_arr: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    translation_root_index: int,
    anim_pos_threshold: float,
):
    """Recover production bare `(F, J, 13)` features from get_motion_features().

    The training/export pipeline stores the root-facing quaternion on joint row 0
    and stores each non-root row's parent local rotation in its 6D slot. That is
    different from the self-contained debug payload path, where every row stores
    its own local rotation directly.

    Bare dataset tensors therefore need a different inverse:
      - recover global joint positions from the RIFKE channels
      - reconstruct each non-leaf joint's local rotation from any child row that
        encodes it
      - solve local positions back from the recovered target global positions

    Leaf/helper joint local rotations are not explicitly encoded in the bare
    tensor, so they stay at identity here. This still preserves world-space bone
    heads and the encoded non-leaf rotations exactly.
    """
    from motion_lib.Animation import Animation, positions_global, rotations_global
    from motion_lib.Quaternions import Quaternions as Qcls

    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from utils.rotation_conversions import rotation_6d_to_matrix_np as _r6d_to_mat

    frame_count, joint_count, channel_count = features_arr.shape
    if channel_count != 13:
        raise ValueError(f"Expected 13 channels, got {channel_count}")

    # Production bare features store each parent's local rotation on its child row.
    rot_mats = np.repeat(
        np.eye(3, dtype=np.float64)[None, None, :, :],
        frame_count,
        axis=0,
    )
    rot_mats = np.repeat(rot_mats, joint_count, axis=1)
    parent_rot_mats = _r6d_to_mat(np.asarray(features_arr[:, 1:, 3:9], dtype=np.float64))
    for child_idx, parent_idx in enumerate(parents[1:], start=1):
        rot_mats[:, parent_idx] = parent_rot_mats[:, child_idx - 1]

    all_rots = Qcls.from_transforms(rot_mats)
    target_global = recover_from_bvh_ric_np(features_arr, translation_root_index=translation_root_index)

    positions = offsets[None].repeat(frame_count, axis=0).copy()
    recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
    recovered_global = positions_global(recovered_anim)
    per_joint_err = np.abs(target_global - recovered_global).max(axis=(0, 2))
    animated_joints = sorted(
        joint_idx for joint_idx in range(joint_count) if per_joint_err[joint_idx] > anim_pos_threshold
    )

    if animated_joints:
        for joint_idx in animated_joints:
            if joint_idx == 0 or parents[joint_idx] < 0:
                positions[:, joint_idx] = target_global[:, joint_idx]
                continue
            temp_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
            temp_global_rots = rotations_global(temp_anim)
            temp_global_pos = positions_global(temp_anim)
            parent_idx = parents[joint_idx]
            positions[:, joint_idx] = (
                -temp_global_rots[:, parent_idx]
            ) * (target_global[:, joint_idx] - temp_global_pos[:, parent_idx])
        recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)

    has_animated_pos = bool(
        animated_joints and any(joint_idx > 0 and parents[joint_idx] >= 0 for joint_idx in animated_joints)
    ) or any(
        np.any(np.ptp(recovered_anim.positions[:, joint_idx], axis=0) > 1e-4)
        for joint_idx in range(1, joint_count)
    )

    return recovered_anim, has_animated_pos


# ── recovery ──────────────────────────────────────────────────────────────────

def recover_from_features(
    features: Any,
    parents: np.ndarray,
    offsets: np.ndarray,
    translation_root_index: Optional[int] = None,
    anim_pos_threshold: float = 0.01,
    motion_metadata: Optional[dict[str, object]] = None,
):
    """Recover an Animation from a 13-channel NPY feature tensor.

    Accepts either a dict payload or a plain (F, J, 13) ndarray.

    Dict payloads use the self-contained own-rotation roundtrip layout written
    by build_npy_metadata_payload(...). Plain arrays are treated as production
    dataset features written by get_motion_features(...).

    Bare dataset tensors can read ``translation_root_index`` from
    ``motion_metadata`` when it is available, avoiding expensive inference.
    When neither ``translation_root_index`` nor ``motion_metadata`` is provided,
    falls back to ``infer_translation_root_index_from_features``.

    Returns:
        (anim, has_animated_pos) — the reconstructed Animation and a bool
        indicating whether non-root position channels are animated.
    """
    from motion_lib.Animation import Animation, positions_global, rotations_global
    from motion_lib.Quaternions import Quaternions
    from utils.rotation_conversions import rotation_6d_to_matrix_np as _r6d_to_mat

    features_arr, payload = coerce_feature_payload(features)

    frame_count, joint_count, channel_count = features_arr.shape
    assert channel_count == 13, f"Expected 13 channels, got {channel_count}"

    if payload is None:
        if translation_root_index is None:
            # Prefer metadata cache to avoid expensive inference.
            if motion_metadata is not None and "translation_root_index" in motion_metadata:
                trans_root = int(motion_metadata["translation_root_index"])
            else:
                from data_loaders.truebones.truebones_utils.motion_process import infer_translation_root_index_from_features

                trans_root = infer_translation_root_index_from_features(
                    features_arr,
                    parents,
                    offsets,
                    anim_pos_threshold=anim_pos_threshold,
                )
        else:
            trans_root = _require_translation_root_index(
                translation_root_index,
                joint_count,
                context="production motion features",
            )
        return _recover_from_production_motion_features(
            features_arr,
            parents,
            offsets,
            trans_root,
            anim_pos_threshold,
        )

    # ── 1. Translation root ─────────────────────────────────────────────
    if translation_root_index is None and payload is not None:
        translation_root_index = payload.get("translation_root_index")
    trans_root = _require_translation_root_index(
        translation_root_index,
        joint_count,
        context="roundtrip payload",
    )

    # ── 2. Restore initial XZ offset ─────────────────────────────────---
    initial_translation_root_xz = np.zeros(2, dtype=np.float64)
    if payload is not None and "initial_translation_root_xz" in payload:
        initial_translation_root_xz = np.asarray(
            payload["initial_translation_root_xz"], dtype=np.float64,
        )

    # ── 3. Root quaternion ──────────────────────────────────────────────
    r_rot_6d = features_arr[:, 0, 3:9]
    r_rot_mat = _r6d_to_mat(r_rot_6d)
    from motion_lib.Quaternions import Quaternions as Qcls
    r_rot = Qcls.from_transforms(r_rot_mat)

    # ── 4. Root position from velocity integration + Y ──────────────────
    #   get_motion_features stores velocity[t] = displacement from t to t+1
    r_pos = np.zeros((frame_count, 3), dtype=np.float64)
    r_pos[1:, [0, 2]] = features_arr[:-1, trans_root, [9, 11]]
    r_pos = (-r_rot) * r_pos
    r_pos = np.cumsum(r_pos, axis=0)
    r_pos[:, 0] += initial_translation_root_xz[0]
    r_pos[:, 2] += initial_translation_root_xz[1]
    r_pos[:, 1] = features_arr[:, 0, 1]

    # ── 5. Joint rotations from 6D ──────────────────────────────────────
    own_rot_6d = features_arr[..., 3:9]
    rot_mats = _r6d_to_mat(own_rot_6d)
    all_rots = Qcls.from_transforms(rot_mats)

    # ── 6. Positions — non-root = rest offsets; root from RIFKE ─────────
    positions = offsets[None].repeat(frame_count, axis=0).copy()
    root_ric = np.asarray(features_arr[:, 0, :3], dtype=np.float64)
    root_global = (-r_rot) * root_ric
    root_global[:, 0] += r_pos[:, 0]
    root_global[:, 2] += r_pos[:, 2]
    positions[:, 0] = root_global

    # ── 7. Handle non-root translation root ─────────────────────────────
    if trans_root != 0 and parents[trans_root] >= 0:
        anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
        global_rots = rotations_global(anim)
        global_pos = positions_global(anim)
        parent_idx = parents[trans_root]
        positions[:, trans_root] = (-global_rots[:, parent_idx]) * (r_pos - global_pos[:, parent_idx])

    # ── 8. Reconcile with RIFKE truth ───────────────────────────────────
    target_global = (-r_rot[:, None]) * np.asarray(features_arr[..., :3], dtype=np.float64)
    target_global[..., 0] += r_pos[:, 0:1]
    target_global[..., 2] += r_pos[:, 2:3]

    recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
    recovered_global = positions_global(recovered_anim)
    per_joint_err = np.abs(target_global - recovered_global).max(axis=(0, 2))
    animated_joints = sorted(
        joint_idx for joint_idx in range(joint_count) if per_joint_err[joint_idx] > anim_pos_threshold
    )

    if animated_joints:
        for joint_idx in animated_joints:
            if joint_idx == 0 or parents[joint_idx] < 0:
                positions[:, joint_idx] = target_global[:, joint_idx]
                continue
            temp_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)
            temp_global_rots = rotations_global(temp_anim)
            temp_global_pos = positions_global(temp_anim)
            parent_idx = parents[joint_idx]
            positions[:, joint_idx] = (
                -temp_global_rots[:, parent_idx]
            ) * (target_global[:, joint_idx] - temp_global_pos[:, parent_idx])
        recovered_anim = Animation(all_rots, positions, Qcls.id(0), offsets, parents)

    has_animated_pos = bool(
        animated_joints and any(joint_idx > 0 and parents[joint_idx] >= 0 for joint_idx in animated_joints)
    ) or any(
        np.any(np.ptp(recovered_anim.positions[:, joint_idx], axis=0) > 1e-4)
        for joint_idx in range(1, joint_count)
    )

    return recovered_anim, has_animated_pos


# ── (end) ──────────────────────────────────────────────────────────────────────
