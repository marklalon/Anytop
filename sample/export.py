"""Writing a generated sample: the last per-sample fixes and the .npy + BVH.

The BVH preview goes through the same decode as ``tools/restore_glb_from_npy``
(``utils.npy_restore``), so what the preview shows is what the GLB restore
will produce.
"""
from os.path import join as pjoin

import numpy as np

from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR
from utils.npy_restore import write_feature_bvh


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
