"""``--reference_motion`` handling: edit and source-encoder paths.

A reference is either a preprocessed ``.npy`` in feature space (its source
skeleton must be a cond species) or a raw ``.fbx/.glb/.gltf`` (cond-free
source, encode only). Edit (skip_timesteps / inpaint / outpaint) takes a
same-species ``.npy``, which ``_prepare_img2img_reference_bundle`` turns into
the ``[B, J, F, T]`` tensor the sampler clamps to / starts from. Encode
extracts features on the source rig and routes them only to the motion
encoder.
"""
import os

import numpy as np
import torch

from data_loaders.truebones.data.dataset import resample_motion_features
from data_loaders.truebones.truebones_utils.canonical_features import (
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from sample.generation_runtime import (
    _load_default_cond_cache,
    _lookup_object_type_case_insensitive,
)
from data_loaders.truebones.truebones_utils.dataset_sources import species_lookup_map
from utils.misc import infer_object_type_from_filename


_REFERENCE_MOTION_PREPROCESS_SUFFIXES = {'.fbx', '.glb', '.gltf'}


def _resolve_reference_source_type(
    reference_motion_path,
    cond_dict,
    *,
    target_type=None,
    default_cond_file=None,
    actual_cond_file=None,
):
    # Match the filename against the cond's own species first: only that knows how
    # many leading tokens are the species name, so a multi-token species
    # ("FEP_MagmaDemon_Attack01_1.npy") resolves whole instead of to its pack
    # prefix. The blind parse stays as the fallback for a source that lives only
    # in the checkpoint's default cond, which is searched below.
    source_type = infer_object_type_from_filename(
        reference_motion_path,
        valid_types=species_lookup_map(cond_dict),
    )
    blind_type = source_type or infer_object_type_from_filename(
        reference_motion_path,
        valid_types=None,
    )
    default_cond_cache = None

    if source_type is None and blind_type and default_cond_file and actual_cond_file:
        default_cond_cache = _load_default_cond_cache(default_cond_file, actual_cond_file)
        if default_cond_cache:
            source_type = _lookup_object_type_case_insensitive(default_cond_cache.keys(), blind_type)

    used_target_fallback = False
    if source_type is None and target_type is not None:
        source_type = target_type
        used_target_fallback = True

    return source_type, default_cond_cache, blind_type, used_target_fallback


def _build_source_cond_dict(cond_dict, source_type, default_cond_cache=None):
    source_cond_dict = dict(cond_dict)
    if source_type in source_cond_dict:
        return source_cond_dict
    if not default_cond_cache:
        raise ValueError(
            f"source type '{source_type}' not found in cond file and no default cond available."
        )
    for key, value in default_cond_cache.items():
        if key not in source_cond_dict:
            source_cond_dict[key] = value
    return source_cond_dict


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
    requested_visible_frame_count=None,
    min_length=20,
    preloaded_features=None,
):
    if preloaded_features is not None:
        ref_raw = np.asarray(preloaded_features, dtype=np.float32)
    else:
        ref_raw = np.load(reference_motion_path).astype(np.float32)
    if ref_raw.ndim != 3:
        raise ValueError(
            f"Reference motion must have shape (T, J, F), got {ref_raw.shape}"
        )

    loaded_reference_frame_count, loaded_reference_joint_count, ref_feats = ref_raw.shape
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


def _load_raw_encoder_reference(path, reference_cond_path, cache_cond, action_group=None):
    """Run source skeleton preprocessing and feature extraction, with no retarget.

    ``action_group`` overrides the file-name decision of whether the clip takes
    the locomotion root policy.
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
    cond, _, _, results = encode_motion_files_on_rest_pose(name, path, [path], action_group=action_group)
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
                            reference_cond_path, cond_dict, output_dir, action_group=None):
    """Raw animation -> feature ``.npy`` on the target skeleton, for an edit.

    The file's own bind pose is its rest pose, so the clip is self-contained: it
    is preprocessed exactly like an encoded reference, and no retarget happens.
    An edit lives on the target skeleton, so the rig the file carries must be
    the target's own.
    """
    features, source_cond = _load_raw_encoder_reference(
        reference_motion_path, reference_cond_path, cond_dict, action_group=action_group)
    source_parents = np.asarray(source_cond['parents'])
    target_parents = np.asarray(target_cond['parents'])
    if source_parents.shape != target_parents.shape or not np.array_equal(source_parents, target_parents):
        raise ValueError(
            f"{reference_motion_path} carries a {len(source_parents)}-joint skeleton that is not "
            f"'{target_type}' ({len(target_parents)} joints, different hierarchy). Editing runs on the "
            "target skeleton, so the file must be the same rig; drop --skip_timesteps / --inpaint_* / "
            "--outpaint to drive the target with it as an encoded reference instead."
        )
    base = os.path.splitext(os.path.basename(reference_motion_path))[0]
    out_npy = os.path.join(output_dir, f"_reference_{base}.npy")
    np.save(out_npy, features)
    print(f"  Reference features {features.shape} (from the file's own bind pose) -> {out_npy}")
    return out_npy


def _prepare_encoder_reference_bundle(features, source_cond, *, batch_size, n_frames,
                                      visible_frames, min_length=20):
    """Source features use the existing window resampling rule on their own rig."""
    from data_loaders.reference import collate_reference, reference_sample
    from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
    if features.ndim != 3 or features.shape[1] != len(source_cond['parents']):
        raise ValueError('Reference features do not match the source skeleton')
    if int(source_cond.get('translation_root_index', 0)) != 0:
        raise ValueError('Reference source must use the canonical joint-zero translation root; reprocess the source')
    if features.shape[2] != 12 or not np.isfinite(features).all() or len(features) < 1:
        raise ValueError('Reference requires finite [T, J_source, 12] features')
    # This packs canonical features using SOURCE statistics / rest, never target
    # indices. The existing crop budget and endpoint interpolation are shared.
    bundle = _prepare_img2img_reference_bundle(
        None, None, source_cond, max_joints=features.shape[1], target_feature_len=12,
        batch_size=1, requested_output_frame_count=n_frames,
        requested_visible_frame_count=visible_frames, min_length=min_length,
        preloaded_features=features,
    )
    sample = reference_sample(bundle['reference_physical'], source_cond,
                              source_cond['joints_names_embs'])
    canvas = bundle['reference_motion'].expand(batch_size, -1, -1, -1)
    return collate_reference(
        [sample] * batch_size, canvas,
        canvas.new_zeros(batch_size, canvas.shape[1], source_cond['joints_names_embs'].shape[-1]),
        canvas.new_zeros(batch_size, canvas.shape[1], JOINT_STRUCT_DIM),
    )
