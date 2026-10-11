"""Writing a generated sample: the last per-sample fixes and the .npy + BVH.

The BVH preview goes through the same decode as ``tools/restore_glb_from_npy``
(``utils.npy_restore``), so what the preview shows is what the GLB restore
will produce.
"""
from os.path import join as pjoin

import numpy as np
import torch

from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR
from utils.npy_restore import write_feature_bvh


def _zero_root_ric_xz(motion_np, translation_root_index):
    """Clear the translation root's RIC X/Z channels before export.

    ch0/ch2 of the root are structurally zero (get_rifke subtracts its own XZ
    from every joint); the world XZ path lives in ch9/ch11, which the exporter
    integrates into r_pos. Model noise in ch0/ch2 would offset the whole
    skeleton away from that path, so it is cleared. Unconditional: the old
    --loop-only drift-cancelling step is gone, and root XZ is now integrated
    the same way for looping and non-looping samples alike.
    """
    if motion_np.ndim != 3:
        return
    _frame_count, joint_count, feature_count = motion_np.shape
    root_index = int(translation_root_index)
    if feature_count < 12 or root_index < 0 or root_index >= joint_count:
        return

    motion_np[:, root_index, 0] = 0.0
    motion_np[:, root_index, 2] = 0.0


def _get_batch_translation_root_index(model_kwargs, sample_idx, fallback=0):
    y = model_kwargs.get('y', {}) if isinstance(model_kwargs, dict) else {}
    value = y.get('translation_root_index', fallback)
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)
        if value.numel() == 0:
            return int(fallback)
        return int(value[min(sample_idx, value.numel() - 1)].item())
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)
        if value.size == 0:
            return int(fallback)
        return int(value[min(sample_idx, value.size - 1)])
    if isinstance(value, (list, tuple)):
        if not value:
            return int(fallback)
        return int(value[min(sample_idx, len(value) - 1)])
    return int(value)


def _bvh_preview_options(args):
    """The BVH preview's decode options, named like restore_glb_from_npy's."""
    return {
        'fullbody_ik': bool(getattr(args, 'fullbody_ik', False)),
        'stretch_factor': float(getattr(args, 'stretch_factor', DEFAULT_IK_STRETCH_FACTOR)),
    }


def _export_motion(task):
    """Write one generated sample as .npy plus its BVH preview.

    The preview is the skeleton-only GLB's animation written as BVH: the same
    decode ``tools/restore_glb_from_npy.py`` runs (``utils.npy_restore``), on the
    cond skeleton in HML space, with the same optional full-body IK.
    """
    motion_np, cond_entry, npy_name, joint_names, out_path, fps, preview_options = task
    np.save(pjoin(out_path, npy_name), motion_np)
    write_feature_bvh(
        motion_np,
        cond_entry,
        pjoin(out_path, npy_name.replace('.npy', '.bvh')),
        fps=fps,
        joint_names=joint_names,
        **preview_options,
    )
    return npy_name
