import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(
            type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(
            type(ndarray)))
    return ndarray


def cleanexit():
    import sys
    import os
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])

def freeze_joints(x, joints_to_freeze):
    # Freezes selected joint *rotations* as they appear in the first frame
    # x [bs, [root+n_joints], joint_dim(6), seqlen]
    frozen = x.detach().clone()
    frozen[:, joints_to_freeze, :, :] = frozen[:, joints_to_freeze, :, :1]
    return frozen


# ── Object type inference from filenames ───────────────────────────────────

import glob as _glob
import os as _os
from collections.abc import Container as _Container


def infer_object_type_from_filename(
    filename: str,
    valid_types: _Container[str] | None = None,
) -> str | None:
    """Infer an object type key from a motion/fbx filename.

    Handles these patterns (in priority order):

        ``{Type}___{Action}_{ID}.ext``     — triple underscore (e.g. ``Horse___RunToStop_29.npy``)
        ``{Type}_{Action}_{ID}.ext``        — single underscore (e.g. ``Sea_Lion_Swim_42.npy``)
        ``{Type}-{Action}.ext``             — hyphen (e.g. ``Wyvern-Tpose.fbx``)

    When *valid_types* is provided the extracted candidate(s) are validated
    against that container.  Multi-word types (e.g. ``Sea_Lion``) are handled
    via progressive prefix matching.

    Args:
        filename:   A file path or plain filename.
        valid_types: Optional set/container of known object types for validation.

    Returns:
        The inferred object type, or ``None`` if inference fails.
    """
    stem = _os.path.splitext(_os.path.basename(filename))[0]
    if not stem:
        return None

    # 1. Triple-underscore separator  (highest priority)
    sep_triple = "___"
    if sep_triple in stem:
        candidate = stem.split(sep_triple, 1)[0]
        if valid_types is None or candidate in valid_types:
            return candidate

    # 2. Progressive single-underscore prefix matching
    #    (handles multi-word types like "Sea_Lion")
    if valid_types is not None and "_" in stem:
        parts = stem.split("_")
        best: str | None = None
        for i in range(1, len(parts)):
            candidate = "_".join(parts[:i])
            if candidate in valid_types:
                best = candidate  # keep going for a longer match
        if best is not None:
            return best

    # 3. Single underscore — first token (blind, when no valid_types)
    if "_" in stem:
        first_token = stem.split("_", 1)[0]
        if first_token:
            if valid_types is None or first_token in valid_types:
                return first_token

    # 4. Hyphen separator (for FBX stems like "Wyvern-Tpose")
    if "-" in stem:
        first_token = stem.split("-", 1)[0]
        if first_token:
            if valid_types is None or first_token in valid_types:
                return first_token

    return None


def infer_object_type_from_fbx_dir(
    directory: str,
    valid_types: _Container[str] | None = None,
) -> str | None:
    """Infer object type from the first FBX file found in *directory*.

    Args:
        directory:   Path to a directory containing ``.fbx`` / ``.FBX`` files.
        valid_types: Optional set/container of known object types for validation.

    Returns:
        The inferred object type, or ``None`` if the directory is empty or
        inference fails.
    """
    fbxs = sorted(
        _glob.glob(_os.path.join(directory, "*.[Ff][Bb][Xx]"))
    )
    if not fbxs:
        return None
    return infer_object_type_from_filename(fbxs[0], valid_types=valid_types)
