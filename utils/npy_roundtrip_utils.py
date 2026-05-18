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
