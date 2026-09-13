"""
Production-ready NPY roundtrip utilities.

Functions for encoding, recovering, and loading AnyTop's 12-channel NPY motion features.

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

    Accepts either a dict payload or a plain (F, J, FEATS_LEN) ndarray.
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
    translation_root_index: Optional[int] = None,
    anim_pos_threshold: float = 0.01,
    motion_metadata: Optional[dict[str, object]] = None,
):
    """Recover an Animation from a 12-channel NPY feature tensor.

    Thin front door over ``recover_animation_from_motion_np`` -- the one decode
    used by the BVH preview, the GLB restore and the retarget tools -- so every
    reader gets the same quaternion sign normalization and zero-offset-leaf
    pinning. Accepts either a plain ``(F, J, FEATS_LEN)`` ndarray or a legacy
    dict payload (``{"features": ..., "translation_root_index": ...}``).

    The translation root comes from, in order, the explicit argument, the dict
    payload, ``motion_metadata``; otherwise it is the hierarchy root (0), which
    is what preprocessing's root collapse guarantees for every cond entry.

    Returns:
        (anim, has_animated_pos) — the reconstructed Animation and a bool
        indicating whether non-root position channels are animated.
    """
    from data_loaders.truebones.truebones_utils.features import recover_animation_from_motion_np
    from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN

    features_arr, payload = coerce_feature_payload(features)
    if features_arr.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (F, J, C), got {features_arr.shape}")
    frame_count, joint_count, channel_count = features_arr.shape
    if channel_count != FEATS_LEN:
        raise ValueError(f"Expected {FEATS_LEN} channels, got {channel_count}")

    if translation_root_index is None and payload is not None:
        translation_root_index = payload.get("translation_root_index")
    if translation_root_index is None and motion_metadata is not None:
        translation_root_index = motion_metadata.get("translation_root_index")
    if translation_root_index is None:
        translation_root_index = 0
    translation_root_index = int(translation_root_index)
    if translation_root_index < 0 or translation_root_index >= joint_count:
        raise ValueError(
            f"translation_root_index out of range: {translation_root_index} for {joint_count} joints"
        )

    return recover_animation_from_motion_np(
        features_arr,
        np.asarray(parents, dtype=np.int32),
        np.asarray(offsets, dtype=np.float32),
        translation_root_index=translation_root_index,
        anim_pos_threshold=anim_pos_threshold,
    )
