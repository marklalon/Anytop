"""Same-rig motion sources for img2img editing."""
import os

import numpy as np
import torch

from data_loaders.truebones.data.dataset import resample_motion_features
from data_loaders.truebones.truebones_utils.canonical_features import (
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
    stored_to_working,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT


_REFERENCE_MOTION_PREPROCESS_SUFFIXES = {'.fbx', '.glb', '.gltf'}


def _validate_edit_reference_shape(features, target_type, target_cond):
    """A bare feature array has no species metadata; validate its target layout."""
    if features.ndim != 3:
        raise ValueError(
            f"Reference motion must have shape (T, J, F), got {features.shape}"
        )
    target_joints = len(target_cond['parents'])
    if features.shape[1] != target_joints:
        raise ValueError(
            f"Reference motion has {features.shape[1]} joints, but target "
            f"'{target_type}' requires {target_joints}. Editing requires features "
            "on the target skeleton in the same joint order. Use the motion "
            "autoencoder for cross-skeleton transfer."
        )


def _validate_reference_motion_path(reference_motion_path):
    suffix = os.path.splitext(reference_motion_path)[1].lower()
    if suffix == '.npy':
        return suffix

    if suffix not in _REFERENCE_MOTION_PREPROCESS_SUFFIXES:
        raise ValueError(
            f"Unsupported reference motion format: {suffix or '<no extension>'}. "
            "Supported formats: .npy, .fbx, .glb, .gltf"
        )
    return suffix


def _prepare_img2img_reference_bundle(
    reference_motion_path,
    target_type,
    target_cond,
    *,
    max_joints,
    target_feature_len,
    batch_size,
    requested_output_frame_count,
    internal_num_frames=None,
    requested_visible_frame_count=None,
    min_length=20,
    preloaded_features=None,
):
    # ``preloaded_features`` is already in the working layout; a stored clip
    # read from disk is brought into it.
    if preloaded_features is not None:
        ref_raw = np.asarray(preloaded_features, dtype=np.float32)
    else:
        ref_raw = stored_to_working(np.load(reference_motion_path))
    _validate_edit_reference_shape(ref_raw, target_type, target_cond)

    loaded_reference_frame_count, loaded_reference_joint_count, ref_feats = ref_raw.shape
    if internal_num_frames is not None:
        max_reference_frames = int(internal_num_frames) * MAX_SOURCE_FRAMES_MULT
        if loaded_reference_frame_count > max_reference_frames:
            raise ValueError(
                f"Reference motion has {loaded_reference_frame_count} frames; maximum is "
                f"2N={max_reference_frames} for model window N={int(internal_num_frames)}."
            )
    # Fixed-window model: always run at native window length (num_frames);
    # resample reference up to it like pure generation, then resample output
    # to target_output_frames afterwards.
    output_frame_count = int(requested_output_frame_count)
    max_source_frames = max(int(min_length), output_frame_count * MAX_SOURCE_FRAMES_MULT)
    if loaded_reference_frame_count > max_source_frames:
        visible_frames = output_frame_count if requested_visible_frame_count is None else int(requested_visible_frame_count)
        source_frames = min(max_source_frames, max(int(min_length), visible_frames))
        ref_raw = ref_raw[:source_frames]
    # The reference is a one-shot clip, --loop or not: every frame it ships is
    # motion, and it fills the window end to end (endpoint resampling), so
    # reference frame 0 is window frame 0 and reference frame L-1 is window
    # frame T-1. --loop says the WINDOW is a cycle -- it is the model's is_loop
    # condition and it picks the export mapping -- but it is the model, not this
    # fill, that has to close the cycle over the frames it is given.
    reference_source_frame_count = int(ref_raw.shape[0])
    if ref_raw.shape[0] != output_frame_count:
        ref_raw = resample_motion_features(ref_raw, output_frame_count)

    mark_canonical_cond_entry(target_cond)
    ref_canonical = np.nan_to_num(
        physical_hml_to_canonical(ref_raw, target_cond),
        copy=True,
    ).astype(np.float32)

    if loaded_reference_joint_count < max_joints:
        pad = np.zeros(
            (output_frame_count, max_joints - loaded_reference_joint_count, ref_canonical.shape[2]),
            dtype=np.float32,
        )
        ref_canonical = np.concatenate([ref_canonical, pad], axis=1)

    ref_tensor = torch.from_numpy(ref_canonical).permute(1, 2, 0)
    ref_feat = ref_tensor.shape[1]
    if ref_feat < target_feature_len:
        pad = torch.zeros(
            (max_joints, target_feature_len - ref_feat, output_frame_count),
            dtype=torch.float32,
        )
        ref_tensor = torch.cat([ref_tensor, pad], dim=1)
    elif ref_feat > target_feature_len:
        ref_tensor = ref_tensor[:, :target_feature_len, :]
    ref_motion = ref_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
    return {
        'reference_motion': ref_motion,
        'reference_physical': ref_raw,
        'output_frame_count': output_frame_count,
        'loaded_reference_frame_count': loaded_reference_frame_count,
        'reference_source_frame_count': reference_source_frame_count,
        'loaded_reference_joint_count': loaded_reference_joint_count,
    }


def _load_raw_edit_source(path, reference_cond_path, cache_cond):
    """Run source skeleton preprocessing and feature extraction, with no retarget.

    The preprocessing pipeline infers the locomotion root policy from the file name.
    """
    from data_loaders.truebones.truebones_utils.dataset_pipeline import (
        encode_motion_files_on_rest_pose, _inherit_canonical_stats_from_dataset,
    )
    from data_loaders.truebones.truebones_utils.joint_embedding_text import (
        _build_t5_text_cache, build_joint_embedding_texts,
    )
    # Use the same wrapper-folding fixed point as dataset preprocessing, so the
    # transport root is joint zero, as required by InputProcess.
    name = os.path.splitext(os.path.basename(path))[0]
    cond, _, _, results = encode_motion_files_on_rest_pose(name, path, [path])
    motion = results[0]['motion']
    _inherit_canonical_stats_from_dataset(name, cond, reference_cond_path)
    first = next(iter(cache_cond.values()))
    cache = _build_t5_text_cache(cache_cond, first['joints_names_embs_meta']['t5_name'])
    width = np.asarray(first['joints_names_embs']).shape[-1]
    cond['joints_names_embs'] = np.stack([
        cache.get(text, np.zeros(width, np.float32)) for text in build_joint_embedding_texts(cond)
    ])
    return motion, cond


def _extract_edit_reference(reference_motion_path, target_cond, target_type,
                            reference_cond_path, cond_dict, output_dir):
    """Raw animation -> feature ``.npy`` on the target skeleton, for an edit.

    The file's own bind pose is its rest pose, so the clip is self-contained: it
    uses the dataset preprocessing pipeline, and no retarget happens.
    An edit lives on the target skeleton, so the rig the file carries must be
    the target's own.
    """
    features, source_cond = _load_raw_edit_source(
        reference_motion_path, reference_cond_path, cond_dict)
    source_parents = np.asarray(source_cond['parents'])
    target_parents = np.asarray(target_cond['parents'])
    if source_parents.shape != target_parents.shape or not np.array_equal(source_parents, target_parents):
        raise ValueError(
            f"{reference_motion_path} carries a {len(source_parents)}-joint skeleton that is not "
            f"'{target_type}' ({len(target_parents)} joints, different hierarchy). Editing runs on the "
            "target skeleton, so the file must be the same rig. Use the motion autoencoder "
            "for cross-skeleton transfer."
        )
    base = os.path.splitext(os.path.basename(reference_motion_path))[0]
    out_npy = os.path.join(output_dir, f"_reference_{base}.npy")
    np.save(out_npy, features)
    print(f"  Reference features {features.shape} (from the file's own bind pose) -> {out_npy}")
    return out_npy
