"""
Production-ready NPY roundtrip utilities.

Functions for encoding and recovering AnyTop's 13-channel NPY motion features
with the translation-root XZ offset preserved for exact GLB roundtrip.

Usage::

    from utils.npy_roundtrip_utils import build_roundtrip_feature_payload, recover_from_features

    # Encode
    payload = build_roundtrip_feature_payload(
        anim, object_type="Horse", offsets=..., parents=..., bone_names=...,
    )
    np.save("roundtrip_features.npy", payload, allow_pickle=True)

    # Decode the saved payload
    loaded = np.load("roundtrip_features.npy", allow_pickle=True).item()
    recovered_anim, has_animated_pos = recover_from_features(
        loaded, parents, offsets,
    )
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


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

def extract_raw_features(
    anim: Any,
    object_type: str,
    offsets: np.ndarray,
    parents: np.ndarray,
    bone_names: list[str],
    max_joints: int = 85,
) -> np.ndarray:
    """Extract the 13-channel NPY motion features (no HML transforms).

    Uses ``anim.rotations[:, 0]`` as the root-facing quaternion *r_rot*.
    No scaling, centering, or face-orientation transforms are applied.
    """
    from motion_lib.Animation import positions_global

    from data_loaders.truebones.truebones_utils.motion_process import (
        _find_translation_root,
        get_contact_state,
        get_motion_features,
        get_rifke,
        get_terminal_contact_state,
    )
    from data_loaders.truebones.truebones_utils.param_utils import FOOT_CONTACT_VEL_THRESH
    from data_loaders.truebones.truebones_utils.physics_joint_annotation import _infer_contact_joints

    global_pos = positions_global(anim)
    r_rot = anim.rotations[:, 0].copy()
    cont_6d_params = get_cont6d_params_own(anim, r_rot)

    translation_root_index = _find_translation_root(anim)
    positions = get_rifke(global_pos, r_rot, translation_root_index=translation_root_index)

    rest_pos = compute_rest_positions(offsets, parents)
    foot_indices, _contact_source = _infer_contact_joints(
        object_type, bone_names, parents.tolist(), rest_pos,
    )
    foot_contact = get_contact_state(global_pos, foot_indices, FOOT_CONTACT_VEL_THRESH)

    local_vel = r_rot[1:, None] * (global_pos[1:] - global_pos[:-1])
    prev_velocity = local_vel[-1] if local_vel.shape[0] > 0 else None
    is_loop = detect_motion_loop(positions)
    terminal_local_vel = compute_terminal_local_velocity(
        global_pos, r_rot, is_loop, prev_velocity=prev_velocity,
    )
    terminal_contact = get_terminal_contact_state(
        global_pos, foot_indices, FOOT_CONTACT_VEL_THRESH, is_loop,
    )

    features, _max_joints = get_motion_features(
        positions,
        cont_6d_params,
        foot_contact,
        local_vel,
        terminal_local_vel,
        terminal_contact,
        max_joints,
    )
    return features


# ── payload ───────────────────────────────────────────────────────────────────

def build_roundtrip_feature_payload(
    anim: Any,
    object_type: str,
    offsets: np.ndarray,
    parents: np.ndarray,
    bone_names: list[str],
) -> dict[str, Any]:
    """Build a serializable NPY payload for exact GLB roundtrip recovery.

    The 13-channel motion tensor does not preserve the translation root's
    initial world-space XZ offset, so store it alongside the tensor in the
    same .npy payload.

    Returns a dict that can be saved via ``np.save(path, payload, allow_pickle=True)``
    and loaded via ``np.load(path, allow_pickle=True).item()``.
    """
    from data_loaders.truebones.truebones_utils.motion_process import _find_translation_root
    from motion_lib.Animation import positions_global

    features = extract_raw_features(anim, object_type, offsets, parents, bone_names)
    global_pos = positions_global(anim)
    translation_root_index = int(_find_translation_root(anim))
    initial_translation_root_xz = np.asarray(
        global_pos[0, translation_root_index, [0, 2]], dtype=np.float64,
    )
    return {
        "features": features,
        "translation_root_index": translation_root_index,
        "initial_translation_root_xz": initial_translation_root_xz,
    }


def coerce_feature_payload(features_or_payload: Any) -> tuple[np.ndarray, Optional[dict[str, Any]]]:
    """Unpack a roundtrip payload back into (features_tensor, payload_dict).

    Accepts either the full dict (from ``build_roundtrip_feature_payload``) or
    a plain (F, J, 13) ndarray for backward compatibility.
    """
    if isinstance(features_or_payload, dict):
        payload = features_or_payload
        features = np.asarray(payload["features"])
        return features, payload
    return np.asarray(features_or_payload), None


# ── recovery ──────────────────────────────────────────────────────────────────

def recover_from_features(
    features: Any,
    parents: np.ndarray,
    offsets: np.ndarray,
    pos_err_threshold: float = 0.01,
):
    """Recover an Animation from a 13-channel NPY feature tensor.

    Accepts either:
    - A dict payload produced by :func:`build_roundtrip_feature_payload`
    - A plain (F, J, 13) ndarray

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

    # ── 1. Translation root ─────────────────────────────────────────────
    if payload is not None and "translation_root_index" in payload:
        trans_root = int(payload["translation_root_index"])
    else:
        xz_abs_max = np.max(np.abs(features_arr[:, :, [0, 2]]), axis=(0, 2))
        zero_xz = np.flatnonzero(xz_abs_max <= 1e-5)
        trans_root = int(zero_xz[0]) if zero_xz.size > 0 else int(np.argmin(xz_abs_max))

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
        joint_idx for joint_idx in range(joint_count) if per_joint_err[joint_idx] > pos_err_threshold
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
