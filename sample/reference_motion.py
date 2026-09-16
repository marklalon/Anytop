"""``--reference_motion`` handling: source-species resolution, cross-species
retarget, and packing the reference into the sampler's window.

A reference is either a preprocessed ``.npy`` in feature space (its source
skeleton must be a cond species, retargeted when it differs from the target)
or a raw ``.fbx/.glb/.gltf`` (cond-free source, always retargeted). Either way
the result is a feature-space clip on the TARGET skeleton, which
``_prepare_img2img_reference_bundle`` turns into the ``[B, J, F, T]`` tensor
the sampler clamps to / starts from.
"""
import os

import numpy as np
import torch

from data_loaders.truebones.data.dataset import (
    _drop_loop_closing_frame,
    resample_motion_features,
)
from data_loaders.truebones.truebones_utils.canonical_features import (
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.dataset_sources import (
    build_species_file_tokens,
    species_file_token,
    species_lookup_map,
)
from data_loaders.truebones.truebones_utils.motion_process import (
    tpose_features_from_cond,
)
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from sample.generation_runtime import (
    _load_default_cond_cache,
    _lookup_object_type_case_insensitive,
)
from utils.misc import infer_object_type_from_filename
from utils.npy_restore import write_feature_bvh


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


def _reference_crosses_skeletons(source_type, target_type):
    return bool(
        source_type
        and target_type
        and source_type.upper() != target_type.upper()
    )


def _should_retarget_reference(source_type, target_type):
    return _reference_crosses_skeletons(source_type, target_type)


def _build_retarget_cond_dict(cond_dict, source_type, default_cond_cache=None):
    retarget_cond_dict = dict(cond_dict)
    if source_type in retarget_cond_dict:
        return retarget_cond_dict
    if not default_cond_cache:
        raise ValueError(
            f"source type '{source_type}' not found in cond file and no default cond available for retarget."
        )
    for key, value in default_cond_cache.items():
        if key not in retarget_cond_dict:
            retarget_cond_dict[key] = value
    return retarget_cond_dict


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


def _require_cond_translation_root_index(cond_entry, *, object_type, context):
    try:
        root = int(cond_entry['translation_root_index'])
    except KeyError:
        raise KeyError(
            f"{context}: cond_dict['{object_type}'] is missing 'translation_root_index'. "
            "Regenerate dataset artifacts to populate it."
        )
    n_joints = len(cond_entry['parents'])
    if not 0 <= root < n_joints:
        raise ValueError(
            f"{context}: translation_root_index={root} out of range [0, {n_joints}) "
            f"for '{object_type}'"
        )
    return root


def _retarget_reference_motion(
    ref_motion_path,
    source_type,
    target_type,
    cond_dict,
    opt,
    output_dir,
    fps,
):
    """Retarget a reference motion .npy from ``source_type`` to ``target_type``.

    Thin wrapper around ``utils.retarget_pipeline.retarget_features_npy_to_target``.
    Loads source features, builds target TPoseFeatures, delegates the math, then
    writes the retargeted .npy and an inspection .bvh under ``output_dir``.
    """
    from utils.retarget_pipeline import (
        retarget_features_npy_to_target,
    )

    src_cond = cond_dict[source_type]
    tgt_cond = dict(cond_dict[target_type])
    # Intermediate artefacts are files, so they are named by the file token
    # rather than the '/'-bearing canonical key.
    file_tokens = build_species_file_tokens(cond_dict)
    source_token = file_tokens[source_type]
    target_token = file_tokens[target_type]

    print(f"\n### Cross-species retarget: {source_type} → {target_type}")

    ref_raw = np.load(ref_motion_path).astype(np.float32)
    print(f"  Source motion shape: {ref_raw.shape}")

    tgt_cond['translation_root_index'] = _require_cond_translation_root_index(
        tgt_cond,
        object_type=target_type,
        context='Cross-species reference retarget',
    )

    # Both skeletons reconstructed from cond (no mesh read).
    src_tp = tpose_features_from_cond(src_cond, source_type)
    tgt_tp = tpose_features_from_cond(tgt_cond, target_type)

    target_features = retarget_features_npy_to_target(
        ref_raw,
        src_cond,
        source_type,
        tgt_tp,
        target_type,
        opt.max_joints,
        source_tp=src_tp,
        target_cond=tgt_cond,
    )

    if target_features is None:
        raise RuntimeError(
            f"retarget_features_npy_to_target returned None "
            f"({source_type} → {target_type}). Check source/target cond entries and joint overlap."
        )

    # Save retargeted .npy.
    base = os.path.splitext(os.path.basename(ref_motion_path))[0]
    out_npy = os.path.join(output_dir, f"_retargeted_{source_token}_to_{target_token}__{base}.npy")
    np.save(out_npy, target_features)
    print(f"  Retargeted features {target_features.shape} → {out_npy}")

    _write_inspection_bvh(target_features, tgt_cond, out_npy, fps, object_type=target_type)

    return out_npy


def _write_inspection_bvh(features, cond_entry, out_npy, fps, *, object_type=None):
    """BVH next to a retargeted .npy, through the same decode as the GLB restore."""
    out_bvh = out_npy.replace('.npy', '.bvh')
    try:
        write_feature_bvh(features, cond_entry, out_bvh, fps=fps, object_type=object_type)
        print(f"  Retargeted BVH (for inspection) → {out_bvh}")
    except Exception as e:
        print(f"  [WARN] Failed to write inspection BVH: {e}")


def _retarget_reference_motion_from_file(
    reference_motion_path,
    target_type,
    cond_dict,
    opt,
    output_dir,
    fps,
):
    """Retarget raw .fbx/.glb/.gltf onto target_type (cond-free source).
    Only the target's cond/T-pose is required."""
    from utils.retarget_pipeline import (
        retarget_animation_file_to_target,
    )

    tgt_cond = dict(cond_dict[target_type])
    target_token = build_species_file_tokens(cond_dict)[target_type]

    # Resolve a source species hint when the raw file belongs to a registered
    # skeleton. The raw path remains cond-free; this hint is used only to apply
    # the same species-prefix joint-name canonicalization as dataset cond.
    base = os.path.splitext(os.path.basename(reference_motion_path))[0]
    source_object_type = infer_object_type_from_filename(
        reference_motion_path,
        valid_types=species_lookup_map(cond_dict),
    )
    # Intermediate artefacts are files, so a registered source is named by its
    # file token: the canonical key carries a '/'-bearing namespace
    # (``truebones/zoo/Buffalo``) that would turn the filename into a path
    # through directories that do not exist.
    source_label = species_file_token(cond_dict, source_object_type) if source_object_type else (
        infer_object_type_from_filename(
            reference_motion_path,
            valid_types=None,
        ) or base
    )

    print(
        f"\n### Reference retarget (cond-free source): {reference_motion_path} → {target_type}"
    )

    tgt_cond['translation_root_index'] = _require_cond_translation_root_index(
        tgt_cond,
        object_type=target_type,
        context='Cond-free reference retarget',
    )

    # Target rest-pose from cond; source skeleton/motion from the animation file.
    tgt_tp = tpose_features_from_cond(tgt_cond, target_type)

    target_features = retarget_animation_file_to_target(
        reference_motion_path,
        tgt_tp,
        target_type,
        opt.max_joints,
        tgt_cond,
        source_object_type=source_object_type,
    )

    if target_features is None:
        raise RuntimeError(
            f"retarget_animation_file_to_target returned None "
            f"({reference_motion_path} → {target_type}). Check the target cond entry "
            f"and joint-name overlap with the source file."
        )

    out_npy = os.path.join(output_dir, f"_retargeted_{source_label}_to_{target_token}__{base}.npy")
    np.save(out_npy, target_features)
    print(f"  Retargeted features {target_features.shape} → {out_npy}")

    _write_inspection_bvh(target_features, tgt_cond, out_npy, fps, object_type=target_type)

    return out_npy


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
    loop=False,
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
    if loop:
        # --loop asks for a closed window, so the reference fills it periodically,
        # exactly as the loader prepares a loop clip (dataset._prepare_sample):
        # drop a closing key if the clip ships one, then resample periodically
        # at step L/T.
        #
        # The generated window is exported with the SAME mapping (periodic when
        # --loop), so window frame t is reference source time t*L/T in both
        # directions: the round trip is the identity, a --inpaint_frames range
        # lands on the reference poses it names, and the step the model is told
        # about (resample_speed_cond = L/T) is the step the reference actually
        # moves at. Endpoint (open) resampling would instead pin the
        # reference's ends and leave the window's wrap step at 1 source frame
        # against (L-1)/(T-1) inside -- the uneven seam this convention exists
        # to remove, and under a clamp it lands in the output.
        ref_raw = _drop_loop_closing_frame(ref_raw)
    reference_source_frame_count = int(ref_raw.shape[0])
    if ref_raw.shape[0] != output_frame_count:
        ref_raw = resample_motion_features(
            ref_raw, output_frame_count, periodic=bool(loop),
        )

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
        'output_frame_count': output_frame_count,
        'loaded_reference_frame_count': loaded_reference_frame_count,
        'reference_source_frame_count': reference_source_frame_count,
        'loaded_reference_joint_count': loaded_reference_joint_count,
    }
