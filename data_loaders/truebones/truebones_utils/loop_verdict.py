"""What a clip's loop verdict writes into its stored feature tensor.

The verdict (``is_loop`` in action_labels.jsonl) is not only a training label:
preprocessing writes the LAST velocity row of the tensor from it. A loop's row
is the wrap delta ``pos[0] - pos[-1]`` -- the step playback takes from the last
frame back to the first -- so the circular roll and the tile seam in
``dataset._prepare_sample`` read a physically consistent velocity; a one-shot
clip's row repeats the previous frame's velocity (see
``features._compute_terminal_local_velocity``).

So a verdict flipped by hand in the review UI has to reach the tensor too, or
the flag and the seam row disagree until the species is preprocessed again.
This module recomputes that row from the stored tensor alone -- numpy only, so
the review server can call it without importing the torch-backed pipeline.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Channel layout of a stored (T, J, 12) clip, as written by get_motion_features:
# RIC position 0:3, own 6D rotation 3:9, local velocity 9:12.
_POS = slice(0, 3)
_VEL = slice(9, 12)


def terminal_velocity_for_verdict(features, is_loop, translation_root_index):
    """The (J, 3) terminal velocity row a stored clip carries under ``is_loop``.

    Mirrors ``features._compute_terminal_local_velocity`` on the tensor instead
    of the animation. The root-facing frame is identity in this feature space
    (``get_bvh_cont6d_params``), so the world wrap delta ``gp[0] - gp[-1]`` is
    the RIC delta plus the root's XZ travel over the clip -- every joint's RIC
    position subtracts the same root XZ, and that travel is the sum of the root
    velocity rows short of the terminal one. The one-shot row is the previous
    frame's velocity, copied exactly.

    The loop row reproduces preprocessing's to float64 rounding (~1e-16 on
    every stored clip), not bit for bit: the tensor's RIC and velocity
    channels each already carry one rounding of the global positions the
    pipeline subtracted directly, and those bits are gone. The training loader
    casts to float32, where the two are identical.
    """
    features = np.asarray(features)
    frame_count, joint_count = features.shape[0], features.shape[1]
    terminal = np.zeros((joint_count, 3), dtype=np.float64)
    if frame_count < 2:
        return terminal.astype(features.dtype)
    root = int(translation_root_index)
    if not 0 <= root < joint_count:
        raise ValueError(
            f"translation_root_index {root} is out of bounds for {joint_count} joints"
        )
    velocity = np.asarray(features[..., _VEL], dtype=np.float64)
    if is_loop:
        positions = np.asarray(features[..., _POS], dtype=np.float64)
        terminal = positions[0] - positions[-1]
        root_travel = velocity[:-1, root, :].sum(axis=0)
        terminal[:, 0] -= root_travel[0]
        terminal[:, 2] -= root_travel[2]
    else:
        terminal = velocity[-2]
    return terminal.astype(features.dtype)


def apply_loop_verdict(features, is_loop, translation_root_index):
    """A copy of ``features`` whose terminal velocity row matches ``is_loop``."""
    features = np.asarray(features)
    updated = features.copy()
    if features.shape[0] >= 1:
        updated[-1, :, _VEL] = terminal_velocity_for_verdict(
            features, is_loop, translation_root_index
        )
    return updated


def rewrite_terminal_row(motion_path, is_loop, translation_root_index) -> bool:
    """Rewrite one stored clip's terminal row in place for ``is_loop``.

    Returns False without touching the file when the row already matches, so a
    toggle to the verdict the tensor already carries is a no-op on disk -- a
    pipeline-written loop row matches to rounding, not bitwise, hence the
    tolerance (four orders above float64 noise, nine below any real velocity).
    The write goes through a temp file and ``os.replace`` so a crash cannot
    leave a truncated tensor behind.
    """
    motion_path = Path(motion_path)
    features = np.load(motion_path)
    updated = apply_loop_verdict(features, is_loop, translation_root_index)
    if np.allclose(updated, features, rtol=0.0, atol=1e-12):
        return False
    tmp_path = motion_path.with_name(motion_path.name + ".tmp.npy")
    np.save(tmp_path, updated)
    os.replace(tmp_path, motion_path)
    return True
