# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import concurrent.futures
import os
import sys

# Ensure both the Anytop dir (for bare ``utils.*`` / ``data_loaders.*`` imports)
# and its parent (for ``Anytop.utils.*`` imports made by submodules like
# ``utils/retarget.py``) are on sys.path when running as a script. Insert
# repo-root first then Anytop second so Anytop's ``utils/`` wins over the
# unrelated ``<repo_root>/utils/`` directory for bare imports.
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ANYTOP_ROOT))
sys.path.insert(0, _ANYTOP_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    ensure_joint_name_embeddings,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    get_common_features_from_T_pose,
    get_motion,
    recover_bvh_export_animation_from_motion_np,
)
from data_loaders.truebones.truebones_utils.features import resolve_feature_translation_root_index
from motion_lib import BVH
from os.path import join as pjoin
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    ClassifierFreeReferenceModel,
    create_model_and_diffusion_general_skeleton,
    load_model,
    model_supports_reference_conditioning,
    resolve_t5_out_dim,
)
from utils.parser_util import generate_args
from utils.misc import infer_object_type_from_filename


_REFERENCE_MOTION_PREPROCESS_SUFFIXES = {'.fbx', '.glb', '.gltf'}


def validate_reference_mode_configuration(reference_mode, reference_motion_path=None, skip_timesteps=0, model=None):
    mode = str(reference_mode or 'img2img').strip().lower()
    if mode not in {'img2img', 'controlnet'}:
        raise ValueError(f"Unsupported reference_mode '{reference_mode}'.")
    if mode == 'controlnet':
        if reference_motion_path is None:
            raise ValueError("--reference_mode controlnet requires --reference_motion.")
        if skip_timesteps is not None and int(skip_timesteps) != 0:
            print(
                f"[generate] WARNING: --reference_mode controlnet requires --skip_timesteps 0; "
                f"forcing skip_timesteps from {skip_timesteps} to 0."
            )
        skip_timesteps = 0
        if model is not None and not model_supports_reference_conditioning(model):
            raise ValueError(
                "Loaded checkpoint does not support --reference_mode controlnet. Load or retrain a checkpoint with --reference_cond enabled."
            )
    # Non-None values pass through (caller handles defaults per-mode);
    # controlnet always returns 0.
    return mode, int(skip_timesteps) if skip_timesteps is not None else 0


def resolve_reference_scale(reference_scale):
    if reference_scale is None:
        return 1.0
    return float(reference_scale)


def _lookup_object_type_case_insensitive(object_types, requested_type):
    if requested_type is None:
        return None
    return next(
        (object_type for object_type in object_types if object_type.upper() == requested_type.upper()),
        None,
    )


def _load_default_cond_cache(default_cond_file, actual_cond_file):
    if not default_cond_file:
        return None

    default_real = os.path.realpath(default_cond_file)
    actual_real = os.path.realpath(actual_cond_file)
    try:
        if os.path.samefile(default_real, actual_real):
            return None
    except FileNotFoundError:
        if default_real == actual_real:
            return None

    return np.load(default_cond_file, allow_pickle=True).item()


def _resolve_reference_source_type(
    reference_motion_path,
    cond_dict,
    *,
    target_type=None,
    default_cond_file=None,
    actual_cond_file=None,
):
    blind_type = infer_object_type_from_filename(
        reference_motion_path,
        valid_types=None,
    )
    source_type = _lookup_object_type_case_insensitive(cond_dict.keys(), blind_type)
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


def _should_retarget_reference(source_type, target_type, reference_mode):
    return _reference_crosses_skeletons(source_type, target_type)


def _resolve_source_cond_entry(source_type, cond_dict, default_cond_cache=None):
    if not source_type:
        return None
    source_cond_entry = cond_dict.get(source_type)
    if source_cond_entry is None and default_cond_cache:
        source_cond_entry = default_cond_cache.get(source_type)
    return source_cond_entry


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


def _validate_reference_sampling_request(
    *,
    inpaint_enabled,
    reference_mode,
    cross_species_reference,
):
    return None


def _prepare_reference_motion_path(
    reference_motion_path,
    source_type,
    source_cond,
    opt,
    output_dir,
):
    suffix = os.path.splitext(reference_motion_path)[1].lower()
    if suffix == '.npy':
        return reference_motion_path

    if suffix not in _REFERENCE_MOTION_PREPROCESS_SUFFIXES:
        raise ValueError(
            f"Unsupported reference motion format: {suffix or '<no extension>'}. "
            "Supported formats: .npy, .fbx, .glb, .gltf"
        )

    if source_cond is None:
        raise KeyError(
            f"Missing cond entry for reference motion object_type '{source_type}'. "
            "Cannot preprocess non-NPY reference motion."
        )

    tpose_path = source_cond.get('orientation_reference_fbx_path')
    if not tpose_path or not os.path.isfile(tpose_path):
        raise FileNotFoundError(
            f"Reference motion preprocessing requires a valid orientation_reference_fbx_path "
            f"for '{source_type}', not found: {tpose_path!r}"
        )

    cond_parents = source_cond.get('parents')
    preprocess_max_joints = len(cond_parents) if cond_parents is not None else int(opt.max_joints)

    print(f"  Preprocessing reference motion {suffix} -> .npy using object_type={source_type}")
    source_tp = get_common_features_from_T_pose(
        tpose_path,
        source_type,
        augment_leaf_rotation_helpers=True,
        max_joints=preprocess_max_joints,
    )
    scale_factor = float(source_cond.get('scale_factor', source_tp.scale_factor))
    squared_positions_error = {}
    source_features, *_ = get_motion(
        reference_motion_path,
        FOOT_CONTACT_VEL_THRESH,
        source_type,
        preprocess_max_joints,
        source_tp.offsets,
        source_tp.foot_indices,
        source_tp.tpos_rots,
        squared_positions_error,
        scale_factor=scale_factor,
        orientation_quat=source_tp.orientation_quat,
        helper_metadata=source_tp.helper_metadata,
    )
    if source_features is None:
        raise RuntimeError(
            f"Failed to preprocess reference motion '{reference_motion_path}' into feature-space NPY"
        )

    source_features = np.asarray(source_features, dtype=np.float32)
    if source_features.shape[1] != len(source_tp.names):
        raise RuntimeError(
            "Reference motion preprocessing produced a joint count that does not match "
            "the helper-aware T-pose feature skeleton"
        )

    base = os.path.splitext(os.path.basename(reference_motion_path))[0]
    out_npy = os.path.join(output_dir, f"_reference_features_{source_type}__{base}.npy")
    np.save(out_npy, source_features, allow_pickle=False)
    return out_npy


def _get_reference_normalization_stats(
    cond_entry,
    *,
    object_type,
    joint_count,
    feature_count,
    context,
):
    if cond_entry is None:
        raise KeyError(
            f"Missing cond entry for {context} object_type '{object_type}'."
        )

    mean = np.asarray(cond_entry['mean'], dtype=np.float32)
    std = np.asarray(cond_entry['std'], dtype=np.float32)
    if mean.ndim != 2 or std.ndim != 2:
        raise ValueError(
            f"{context} normalization stats for '{object_type}' must have shape (J, F), "
            f"got mean={mean.shape}, std={std.shape}"
        )
    if mean.shape[0] != joint_count or std.shape[0] != joint_count:
        raise ValueError(
            f"{context} normalization stats for '{object_type}' expect {joint_count} joints, "
            f"got mean={mean.shape[0]}, std={std.shape[0]}"
        )
    if mean.shape[1] < feature_count or std.shape[1] < feature_count:
        raise ValueError(
            f"{context} normalization stats for '{object_type}' need at least {feature_count} feature channels, "
            f"got mean={mean.shape[1]}, std={std.shape[1]}"
        )
    return mean[:, :feature_count], std[:, :feature_count] + 1e-6


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

    Thin wrapper around ``utils.auto_retarget.retarget_features_npy_to_target``.
    Loads source features, builds target TPoseFeatures, delegates the math, then
    writes the retargeted .npy and an inspection .bvh under ``output_dir``.
    """
    from Anytop.utils.auto_retarget import retarget_features_npy_to_target
    from data_loaders.truebones.truebones_utils.features import get_common_features_from_T_pose

    src_cond = cond_dict[source_type]
    tgt_cond = cond_dict[target_type]

    src_tpose_path = src_cond.get('orientation_reference_fbx_path')
    tgt_tpose_path = tgt_cond.get('orientation_reference_fbx_path')
    for label, path in (('source', src_tpose_path), ('target', tgt_tpose_path)):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cross-species retarget requires {label} T-pose file "
                f"(cond_dict['{source_type if label == 'source' else target_type}']"
                f"['orientation_reference_fbx_path']), not found: {path!r}"
            )

    print(f"\n### Cross-species retarget: {source_type} → {target_type}")

    ref_raw = np.load(ref_motion_path).astype(np.float32)
    print(f"  Source motion shape: {ref_raw.shape}")

    tgt_tp = get_common_features_from_T_pose(
        tgt_tpose_path, target_type,
        augment_leaf_rotation_helpers=True,
        max_joints=opt.max_joints,
    )

    target_features = retarget_features_npy_to_target(
        ref_raw,
        src_cond,
        source_type,
        tgt_tp,
        target_type,
        opt.max_joints,
        source_tp=None,  # loaded lazily from src_cond['orientation_reference_fbx_path']
        target_cond=tgt_cond,
    )

    if target_features is None:
        raise RuntimeError(
            f"retarget_features_npy_to_target returned None "
            f"({source_type} → {target_type}). Check source T-pose FBX and joint overlap."
        )

    # Save retargeted .npy
    base = os.path.splitext(os.path.basename(ref_motion_path))[0]
    out_npy = os.path.join(output_dir, f"_retargeted_{source_type}_to_{target_type}__{base}.npy")
    np.save(out_npy, target_features)
    print(f"  Retargeted features {target_features.shape} → {out_npy}")

    # Inspection-friendly BVH sibling
    try:
        out_bvh = out_npy.replace('.npy', '.bvh')
        out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
            target_features,
            np.asarray(tgt_cond['parents'], dtype=np.int32),
            np.asarray(tgt_cond['offsets'], dtype=np.float32),
            list(tgt_cond.get('canonical_bvh_joint_names', tgt_cond['joints_names'])),
            allow_infer=True,
            tpose_rest_rotations=tgt_tp.tpos_rots[0],
        )
        if out_anim is not None:
            BVH.save(
                out_bvh, out_anim, joint_names,
                frametime=1.0 / fps, positions=has_animated_pos,
            )
            print(f"  Retargeted BVH (for inspection) → {out_bvh}")
    except Exception as e:
        print(f"  [WARN] Failed to write inspection BVH: {e}")

    return out_npy


def _prepare_reference_prior_bundle(
    reference_motion_path,
    source_type,
    source_cond,
    *,
    max_joints,
    target_feature_len,
    batch_size,
    requested_output_frame_count=None,
):
    if source_cond is None:
        raise KeyError(
            f"Missing cond entry for reference prior object_type '{source_type}'."
        )

    ref_raw = np.load(reference_motion_path).astype(np.float32)
    if ref_raw.ndim != 3:
        raise ValueError(
            f"Reference prior motion must have shape (T, J, F), got {ref_raw.shape}"
        )

    loaded_reference_frame_count, ref_joints, ref_feats = ref_raw.shape
    if requested_output_frame_count is None:
        output_frame_count = loaded_reference_frame_count
    else:
        output_frame_count = min(loaded_reference_frame_count, requested_output_frame_count)
    if loaded_reference_frame_count > output_frame_count:
        ref_raw = ref_raw[:output_frame_count]
    source_parents = np.asarray(source_cond['parents'], dtype=np.int64)
    source_offsets = np.asarray(source_cond['offsets'], dtype=np.float32)
    source_name_embs = np.asarray(source_cond['joints_names_embs'], dtype=np.float32)

    if ref_joints != source_parents.shape[0]:
        raise RuntimeError(
            f"Reference prior joint count mismatch for '{source_type}': motion has {ref_joints} joints, "
            f"cond expects {source_parents.shape[0]}"
        )
    if ref_joints > max_joints:
        raise RuntimeError(
            f"Reference prior joint count {ref_joints} exceeds model max_joints {max_joints}"
        )

    # When the caller supplies a requested output length, match img2img's
    # effective frame-count rule so both reference modes drive the same
    # output-length semantics. Normalize real features first, then zero-pad
    # feature channels in normalized space so padded channels stay exactly
    # zero.
    if ref_feats > target_feature_len:
        ref_raw = ref_raw[:, :, :target_feature_len]
    normalized_feature_dim = ref_raw.shape[2]
    source_mean, source_std = _get_reference_normalization_stats(
        source_cond,
        object_type=source_type,
        joint_count=ref_joints,
        feature_count=normalized_feature_dim,
        context='reference prior',
    )
    ref_norm = np.nan_to_num(
        (ref_raw - source_mean[None, :, :normalized_feature_dim])
        / source_std[None, :, :normalized_feature_dim],
        copy=True,
    ).astype(np.float32)
    if normalized_feature_dim < target_feature_len:
        feat_pad = np.zeros(
            (ref_norm.shape[0], ref_norm.shape[1], target_feature_len - normalized_feature_dim),
            dtype=np.float32,
        )
        ref_norm = np.concatenate([ref_norm, feat_pad], axis=2)
    if ref_joints < max_joints:
        joint_pad = np.zeros((ref_norm.shape[0], max_joints - ref_joints, ref_norm.shape[2]), dtype=np.float32)
        ref_norm = np.concatenate([ref_norm, joint_pad], axis=1)

    ref_tensor = torch.from_numpy(ref_norm).permute(1, 2, 0).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    translation_root_index = int(resolve_feature_translation_root_index(
        ref_raw,
        parents=source_parents,
        offsets=source_offsets,
        allow_infer=True,
        context=f"reference prior motion '{reference_motion_path}'",
    ))

    reference_name_embs = np.zeros((max_joints, source_name_embs.shape[1]), dtype=np.float32)
    reference_name_embs[:ref_joints] = source_name_embs
    reference_name_embs = torch.from_numpy(reference_name_embs).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    reference_conditioning_kwargs = {
        'reference_n_joints': torch.full((batch_size,), ref_joints, dtype=torch.long),
        'reference_lengths': torch.full((batch_size,), output_frame_count, dtype=torch.long),
        'reference_translation_root_index': torch.full((batch_size,), translation_root_index, dtype=torch.long),
        'reference_parents': [source_parents.copy() for _ in range(batch_size)],
        'reference_joints_names_embs': reference_name_embs,
    }
    return ref_tensor, reference_conditioning_kwargs, loaded_reference_frame_count, output_frame_count


def _prepare_img2img_reference_bundle(
    reference_motion_path,
    target_type,
    target_cond,
    *,
    max_joints,
    target_feature_len,
    batch_size,
    requested_output_frame_count,
):
    ref_raw = np.load(reference_motion_path).astype(np.float32)
    if ref_raw.ndim != 3:
        raise ValueError(
            f"Reference motion must have shape (T, J, F), got {ref_raw.shape}"
        )

    loaded_reference_frame_count, loaded_reference_joint_count, ref_feats = ref_raw.shape
    output_frame_count = min(loaded_reference_frame_count, requested_output_frame_count)
    if loaded_reference_frame_count > output_frame_count:
        ref_raw = ref_raw[:output_frame_count]

    obj_mean, obj_std = _get_reference_normalization_stats(
        target_cond,
        object_type=target_type,
        joint_count=loaded_reference_joint_count,
        feature_count=ref_feats,
        context='reference motion',
    )
    ref_norm = np.nan_to_num(
        (ref_raw - obj_mean[None, :, :ref_feats]) / obj_std[None, :, :ref_feats],
        copy=True,
    ).astype(np.float32)

    if loaded_reference_joint_count < max_joints:
        pad = np.zeros(
            (output_frame_count, max_joints - loaded_reference_joint_count, ref_norm.shape[2]),
            dtype=np.float32,
        )
        ref_norm = np.concatenate([ref_norm, pad], axis=1)

    ref_tensor = torch.from_numpy(ref_norm).permute(1, 2, 0)
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
        'reference_conditioning_kwargs': None,
        'output_frame_count': output_frame_count,
        'loaded_reference_frame_count': loaded_reference_frame_count,
        'loaded_reference_joint_count': loaded_reference_joint_count,
    }


def _prepare_reference_for_mode(
    reference_motion_path,
    *,
    reference_mode,
    source_type,
    source_cond,
    target_type,
    target_cond,
    max_joints,
    target_feature_len,
    batch_size,
    requested_output_frame_count,
):
    mode = str(reference_mode or 'img2img').strip().lower()
    if mode == 'controlnet':
        ref_motion, reference_conditioning_kwargs, loaded_reference_frame_count, output_frame_count = _prepare_reference_prior_bundle(
            reference_motion_path,
            target_type,
            target_cond,
            max_joints=max_joints,
            target_feature_len=target_feature_len,
            batch_size=batch_size,
            requested_output_frame_count=requested_output_frame_count,
        )
        loaded_reference_joint_count = int(reference_conditioning_kwargs['reference_n_joints'][0].item())
        return {
            'reference_motion': ref_motion,
            'reference_conditioning_kwargs': reference_conditioning_kwargs,
            'output_frame_count': output_frame_count,
            'loaded_reference_frame_count': loaded_reference_frame_count,
            'loaded_reference_joint_count': loaded_reference_joint_count,
        }

    return _prepare_img2img_reference_bundle(
        reference_motion_path,
        target_type,
        target_cond,
        max_joints=max_joints,
        target_feature_len=target_feature_len,
        batch_size=batch_size,
        requested_output_frame_count=requested_output_frame_count,
    )


def _export_motion(task):
    motion_np, parents_np, offsets, npy_name, joint_names, out_path, fps, tpose_rest_rotations = task
    out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_np,
        parents_np,
        offsets,
        joint_names,
        allow_infer=True,
        tpose_rest_rotations=tpose_rest_rotations,
    )
    np.save(pjoin(out_path, npy_name), motion_np)
    if out_anim is not None:
        BVH.save(
            pjoin(out_path, npy_name.replace('.npy', '.bvh')),
            out_anim,
            joint_names,
            frametime=1.0 / fps,
            positions=has_animated_pos,
        )
    return npy_name


def _sample_batch(
    diffusion,
    model,
    model_kwargs,
    sampling_method,
    sample_shape,
    ddim_eta,
    seed,
    device,
    reference_motion=None,
    reference_conditioning_kwargs=None,
    reference_mode='img2img',
    reference_scale=1.0,
    skip_timesteps=0,
    inpaint_mask=None,
    repaint_jump_length=0,
    repaint_jump_n_sample=1,
):
    if repaint_jump_length < 0:
        raise ValueError("repaint_jump_length must be >= 0")
    if repaint_jump_n_sample < 1:
        raise ValueError("repaint_jump_n_sample must be >= 1")

    reference_mode, skip_timesteps = validate_reference_mode_configuration(
        reference_mode,
        reference_motion_path=reference_motion,
        skip_timesteps=skip_timesteps,
    )

    inpainting = inpaint_mask is not None
    if inpainting:
        if reference_motion is None:
            raise ValueError("inpaint_mask given without a reference_motion")
        if int(reference_motion.shape[-1]) != int(sample_shape[-1]):
            raise ValueError(
                "Motion inpainting requires reference_motion frame count to match target sample length; "
                f"got reference {reference_motion.shape[-1]} and target {sample_shape[-1]}"
            )
        if sampling_method == 'plms':
            # PLMS carries an Adams-Bashforth eps history (old_eps); an
            # external per-step clamp would desync that history from the
            # trajectory. Not supported for inpainting.
            raise ValueError(
                "PLMS does not support motion inpainting; use "
                "--sampling_method ddpm (recommended) or ddim."
            )

    def _prepared_cross_limb_unreliable_mask_from_inpaint_mask(inpaint_mask_):
        if inpaint_mask_ is None:
            return None
        if inpaint_mask_.dim() != 4 or inpaint_mask_.shape[2] != 1:
            raise ValueError(
                f"inpaint_mask must have shape [B, J, 1, T], got {tuple(inpaint_mask_.shape)}"
            )
        raw_cross_limb_unreliable_mask = inpaint_mask_.squeeze(2).permute(0, 2, 1).contiguous()
        reliable_tpose = torch.zeros(
            (
                raw_cross_limb_unreliable_mask.shape[0],
                1,
                raw_cross_limb_unreliable_mask.shape[2],
            ),
            device=raw_cross_limb_unreliable_mask.device,
            dtype=raw_cross_limb_unreliable_mask.dtype,
        )
        return torch.cat([reliable_tpose, raw_cross_limb_unreliable_mask], dim=1).transpose(0, 1).contiguous()

    def _copy_model_kwargs_for_loop(cross_limb_unreliable_mask_, reference_motion_, reference_conditioning_kwargs_, use_reference_conditioning_):
        if model_kwargs is None:
            loop_model_kwargs = {}
            loop_y = {}
        else:
            loop_model_kwargs = dict(model_kwargs)
            loop_y = dict(model_kwargs.get('y', {}))
        if cross_limb_unreliable_mask_ is None:
            loop_y.pop('cross_limb_unreliable_mask', None)
        else:
            loop_y['cross_limb_unreliable_mask'] = cross_limb_unreliable_mask_
        if use_reference_conditioning_:
            loop_y['reference_motion'] = reference_motion_
            loop_y['reference_scale'] = float(reference_scale)
            if reference_conditioning_kwargs_:
                loop_y.update(reference_conditioning_kwargs_)
        else:
            for key in list(loop_y.keys()):
                if key.startswith('reference_'):
                    loop_y.pop(key, None)
            loop_y.pop('reference_cond_mask', None)
        loop_model_kwargs['y'] = loop_y
        return loop_model_kwargs

    def _run_loop(noise, init_image, skip_ts, inpaint_mask_, inpaint_reference_, cross_limb_unreliable_mask_, use_reference_conditioning_):
        common_kwargs = dict(
            model=reference_cfg_model if use_reference_conditioning_ else model,
            shape=sample_shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=_copy_model_kwargs_for_loop(
                cross_limb_unreliable_mask_,
                reference_conditioning_motion,
                reference_conditioning_kwargs,
                use_reference_conditioning_,
            ),
            device=device,
            init_image=init_image,
            skip_timesteps=skip_ts,
        )
        # Only p_* / ddim_* loops accept the inpaint kwargs; plms is rejected
        # above whenever an inpaint mask is present.
        inpaint_kwargs = dict(
            inpaint_mask=inpaint_mask_, inpaint_reference=inpaint_reference_
        )
        repaint_kwargs = dict(
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
        )
        if sampling_method == 'ddim':
            return diffusion.ddim_sample_loop(
                progress=True,
                eta=ddim_eta,
                **inpaint_kwargs,
                **repaint_kwargs,
                **common_kwargs,
            )
        if sampling_method == 'plms':
            return diffusion.plms_sample_loop(
                progress=True,
                **common_kwargs,
            )
        if sampling_method in ('p', 'ddpm'):
            return diffusion.p_sample_loop(
                progress=True,
                dump_steps=None,
                const_noise=False,
                **inpaint_kwargs,
                **repaint_kwargs,
                **common_kwargs,
            )
        raise ValueError(f'Unknown sampling_method: {sampling_method}')

    reference_conditioning_motion = None
    if reference_conditioning_kwargs is None:
        reference_conditioning_kwargs = {}
    reference_cfg_model = model
    if reference_mode == 'controlnet' and reference_motion is not None:
        reference_conditioning_motion = reference_motion.to(device, non_blocking=True)
        reference_cfg_model = ClassifierFreeReferenceModel(model)

    if inpainting and reference_mode == 'img2img' and skip_timesteps > 0:
        # Localized img2img inpainting: start the reverse process from the
        # reference noised to the requested skip timestep, but clamp the known
        # (unmasked) region back to the ORIGINAL reference at every step. This
        # keeps skip_timesteps local to the inpaint mask instead of varying the
        # preserved context outside it.
        ref = reference_motion.to(device, non_blocking=True)
        mask = inpaint_mask.to(device, non_blocking=True)
        prepared_cross_limb_unreliable_mask = _prepared_cross_limb_unreliable_mask_from_inpaint_mask(mask)

        fixseed(seed)
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=ref,
            skip_ts=skip_timesteps,
            inpaint_mask_=mask,
            inpaint_reference_=ref,
            cross_limb_unreliable_mask_=prepared_cross_limb_unreliable_mask,
            use_reference_conditioning_=False,
        )

    fixseed(seed)
    if inpainting:
        # Motion inpainting (RePaint-style imputation), skip_timesteps == 0: the
        # latent starts from PURE NOISE everywhere (so masked joints/frames are
        # truly generated, not a noised copy of the reference), and the
        # reference is used only as the per-step clamp source for the known
        # (unmasked) region. We deliberately do NOT route the reference
        # through init_image, and denoise the full schedule.
        mask = inpaint_mask.to(device, non_blocking=True)
        prepared_cross_limb_unreliable_mask = _prepared_cross_limb_unreliable_mask_from_inpaint_mask(mask)
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=None,
            skip_ts=0,
            inpaint_mask_=mask,
            inpaint_reference_=reference_motion.to(device, non_blocking=True),
            cross_limb_unreliable_mask_=prepared_cross_limb_unreliable_mask,
            use_reference_conditioning_=reference_mode == 'controlnet',
        )
    if reference_mode == 'controlnet' and reference_motion is not None:
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=None,
            skip_ts=0,
            inpaint_mask_=None,
            inpaint_reference_=None,
            cross_limb_unreliable_mask_=None,
            use_reference_conditioning_=True,
        )
    if reference_motion is not None and skip_timesteps > 0:
        # img2img-style: noise the whole reference to an intermediate step.
        # skip_timesteps: how many of the noisiest timesteps to skip.
        # Higher = start from less noisy state = more faithful to reference.
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=reference_motion.to(device, non_blocking=True),
            skip_ts=skip_timesteps,
            inpaint_mask_=None,
            inpaint_reference_=None,
            cross_limb_unreliable_mask_=None,
            use_reference_conditioning_=False,
        )
    # Plain generation: no reference => full denoising from pure noise.
    return _run_loop(
        noise=torch.randn(sample_shape, device=device),
        init_image=None,
        skip_ts=0,
        inpaint_mask_=None,
        inpaint_reference_=None,
        cross_limb_unreliable_mask_=None,
        use_reference_conditioning_=False,
    )


def main(args=None, cond_dict=None):
    if args is None:
        args = generate_args()

    fixseed(args.seed)

    skip_timesteps_raw = getattr(args, 'skip_timesteps', None)

    # Early check for inpaint before ~30s model load (reused below for the
    # skip_timesteps fast-fail).
    _inpaint_early = bool(
        str(getattr(args, 'inpaint_joints', '') or '').strip()
        or str(getattr(args, 'inpaint_frames', '') or '').strip()
    )

    # img2img + reference_motion + no inpaint + no explicit skip_timesteps =>
    # fast-fail because the user must decide how faithful to the reference
    # (skip_timesteps=0 means maximum variation from pure noise).
    if (str(getattr(args, 'reference_mode', 'img2img')).strip().lower() == 'img2img'
            and getattr(args, 'reference_motion', None)
            and not _inpaint_early
            and skip_timesteps_raw is None):
        sys.exit(
            "ERROR: --skip_timesteps is required when using --reference_motion "
            "in img2img mode without --inpaint_joints/--inpaint_frames.\n"
            "  Higher values (e.g. 80-100) produce motion more faithful to the reference;\n"
            "  lower values (e.g. 20-40) allow more model-driven variation.\n"
            "  When combined with --inpaint_joints, the default is 0 (skip disabled)."
        )

    # Fail fast (before the ~30s model load) if inpaint flags are set without a
    # reference motion: the masked region needs a known region to clamp to,
    # otherwise it would silently degrade to plain generation.
    if _inpaint_early and not getattr(args, 'reference_motion', None):
        sys.exit(
            "ERROR: --inpaint_joints / --inpaint_frames require --reference_motion "
            "(the reference is the known region held fixed while the masked region "
            "is regenerated). Pass --reference_motion <path>, or drop the inpaint "
            "flags for plain generation."
        )

    try:
        reference_mode, skip_timesteps = validate_reference_mode_configuration(
            getattr(args, 'reference_mode', 'img2img'),
            reference_motion_path=getattr(args, 'reference_motion', None),
            skip_timesteps=skip_timesteps_raw,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    # --inpaint_joints with --skip_timesteps omitted: default to 0 (skip disabled).
    if _inpaint_early and skip_timesteps_raw is None:
        skip_timesteps = 0

    # Fail fast: --reference_scale is only effective in controlnet mode.
    if getattr(args, 'reference_scale', None) is not None \
            and str(getattr(args, 'reference_mode', 'img2img')).strip().lower() != 'controlnet':
        sys.exit(
            "ERROR: --reference_scale is only effective with --reference_mode controlnet. "
            "In img2img mode, use --skip_timesteps to control faithfulness to the reference."
        )

    # Fail fast: --skip_timesteps is only effective in img2img mode.
    if getattr(args, 'skip_timesteps', None) is not None \
            and str(getattr(args, 'reference_mode', 'img2img')).strip().lower() == 'controlnet':
        sys.exit(
            "ERROR: --skip_timesteps is not supported with --reference_mode controlnet. "
            "In controlnet mode, use --reference_scale to control faithfulness to the reference."
        )

    opt = get_opt(args.device)
    if cond_dict is None:
        if args.cond_path:
            cond_dict = np.load(args.cond_path, allow_pickle=True).item()
            actual_cond_file = args.cond_path
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
            actual_cond_file = opt.cond_file
    else:
        actual_cond_file = opt.cond_file

    n_joints_in_cond = max(
        len(np.asarray(cond_dict[object_key]['parents']))
        for object_key in cond_dict
    )
    if n_joints_in_cond > opt.max_joints:
        print(
            f'[generate] detected cond max joints {n_joints_in_cond} > '
            f'opt.max_joints={opt.max_joints}; raising to {n_joints_in_cond}'
        )
        opt.max_joints = n_joints_in_cond

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    n_frames = int(args.motion_length * fps)
    max_joints = opt.max_joints
    dist_util.setup_dist(args.device)
    object_type = args.object_type
    if out_path == '':
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            'samples_{}_{}_seed{}'.format(name, niter, args.seed),
        )
    os.makedirs(out_path, exist_ok=True)

    print('Creating model and diffusion...')
    resolve_t5_out_dim(args, cond_source=actual_cond_file)
    sampling_steps = int(getattr(args, 'sampling_steps', 100))
    sampling_method = str(getattr(args, 'sampling_method', 'ddim')).lower()
    if sampling_steps > 0:
        if sampling_method == 'ddim':
            args.timestep_respacing = f'ddim{sampling_steps}'
        elif sampling_method == 'plms':
            args.timestep_respacing = f'ddim{sampling_steps}'
        else:
            args.timestep_respacing = str(sampling_steps)
    else:
        args.timestep_respacing = ''
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f'Loading checkpoints from [{args.model_path}]...')
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print('EMA checkpoint detected, loading model_avg weights.')
        state_dict = state_dict['model_avg']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    load_model(model, state_dict)
    try:
        reference_mode, skip_timesteps = validate_reference_mode_configuration(
            reference_mode,
            reference_motion_path=getattr(args, 'reference_motion', None),
            skip_timesteps=skip_timesteps,
            model=model,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    print('Validating precomputed joint-name embeddings from cond.npy...')
    ensure_joint_name_embeddings(
        cond_dict,
        expected_embedding_dim=args.t5_out_dim,
        cond_source=actual_cond_file,
    )
    model.to(dist_util.dev())
    model.eval()

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    reference_motion_path = getattr(args, 'reference_motion', None)
    reference_scale = resolve_reference_scale(getattr(args, 'reference_scale', None))

    inpaint_joints_arg = str(getattr(args, 'inpaint_joints', '') or '').strip()
    inpaint_frames_arg = str(getattr(args, 'inpaint_frames', '') or '').strip()
    inpaint_include_subtree = bool(getattr(args, 'inpaint_include_subtree', True))
    inpaint_enabled = bool(inpaint_joints_arg or inpaint_frames_arg)
    repaint_jump_length = int(getattr(args, 'repaint_jump_length', 0))
    repaint_jump_n_sample = int(getattr(args, 'repaint_jump_n_sample', 1))
    repaint_enabled = (
        inpaint_enabled
        and repaint_jump_length > 0
        and repaint_jump_n_sample > 1
    )

    # ── Resolve --object_type ───────────────────────────────────────────────
    # --object_type: look up directly in cond (user-provided first, then default).
    # --reference_motion: infer source type from filename, look up in cond the same way.
    # If source != target → retarget.
    explicit_object_type = args.object_type

    if not reference_motion_path and not explicit_object_type:
        sys.exit(
            "ERROR: must supply at least one of --reference_motion or --object_type. "
            "Pass --object_type for pure-random generation, --reference_motion for "
            "reference-guided generation (object_type auto-inferred from filename), "
            "or both to retarget the reference into a different target skeleton."
        )

    # (inpaint-requires-reference is enforced early, before the model load)

    # 1) Resolve target object_type
    if explicit_object_type:
        # Case A: explicit --object_type provided.
        # Look up case-insensitively in cond_dict.
        target_type = _lookup_object_type_case_insensitive(cond_dict.keys(), explicit_object_type)
        if target_type is None:
            available = ', '.join(sorted(cond_dict.keys()))
            sys.exit(
                f"ERROR: object_type '{explicit_object_type}' not found in cond file. "
                f"Available: {available}"
            )
    elif reference_motion_path:
        # Case B: no --object_type, infer from reference motion filename.
        target_type = infer_object_type_from_filename(
            reference_motion_path, valid_types=cond_dict.keys()
        )
        if target_type is None:
            available = ', '.join(sorted(cond_dict.keys()))
            sys.exit(
                f"ERROR: Cannot infer object_type from reference motion filename: "
                f"{reference_motion_path}\nAvailable object types: {available}\n"
                "Rename the file to follow the naming convention "
                "(e.g., 'ObjectType___action_id.npy') or pass --object_type explicitly."
            )
    else:
        target_type = None  # unreachable

    # 2) Resolve reference source type (for retarget decision)
    source_type = None
    _default_cond_cache = None
    source_type_used_target_fallback = False
    blind_type = None

    if reference_motion_path:
        source_type, _default_cond_cache, blind_type, source_type_used_target_fallback = _resolve_reference_source_type(
            reference_motion_path,
            cond_dict,
            target_type=target_type,
            default_cond_file=getattr(opt, 'cond_file', None),
            actual_cond_file=actual_cond_file,
        )
        if source_type is None and blind_type:
            available = ', '.join(sorted(cond_dict.keys()))
            if _default_cond_cache:
                default_available = ', '.join(sorted(_default_cond_cache.keys()))
                sys.exit(
                    f"ERROR: source type '{blind_type}' (inferred from reference motion "
                    f"{reference_motion_path}) not found in any cond file. "
                    f"Available in user cond: {available}\n"
                    f"Available in default cond: {default_available}"
                )
            sys.exit(
                f"ERROR: source type '{blind_type}' (inferred from reference motion "
                f"{reference_motion_path}) not found in cond file. "
                f"Available: {available}"
            )
    cross_species_reference = _reference_crosses_skeletons(source_type, target_type)
    should_retarget_reference = _should_retarget_reference(
        source_type,
        target_type,
        reference_mode,
    )

    object_type = target_type  # downstream code keeps reading `object_type`
    if reference_motion_path:
        if source_type_used_target_fallback:
            print(
                f"Reference motion object_type inference was invalid"
                f" ({blind_type or 'no match'}); falling back to target object_type: {target_type}"
            )
        if should_retarget_reference:
            print(f"Reference motion object_type: {source_type} (will retarget to {target_type})")
        elif cross_species_reference:
            print(f"Reference motion object_type: {source_type} (will retarget to {target_type})")
        else:
            inferred_display = source_type if source_type else target_type
            print(f"Reference motion object_type: {inferred_display}")

    # Create thread pool for export.
    # Threads still benefit from GIL release inside np.save / np.savetxt (C code),
    # and recover_animation_from_motion_np uses numpy ops that often release the GIL.
    num_workers = 8
    export_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
    try:
        print(f'\n### Sampling object_type: {object_type}')
        print(f'  method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

        # Prepare reference motion (normalize + reshape)
        ref_motion = None
        reference_conditioning_kwargs = None
        output_frame_count = n_frames

        source_cond_entry = _resolve_source_cond_entry(
            source_type,
            cond_dict,
            _default_cond_cache,
        )

        prepared_reference_path = reference_motion_path
        if reference_motion_path:
            prepared_reference_path = _prepare_reference_motion_path(
                reference_motion_path,
                source_type,
                source_cond_entry,
                opt,
                out_path,
            )

        effective_reference_path = prepared_reference_path
        if should_retarget_reference:
            retarget_cond_dict = _build_retarget_cond_dict(
                cond_dict,
                source_type,
                _default_cond_cache,
            )

            effective_reference_path = _retarget_reference_motion(
                prepared_reference_path,
                source_type=source_type,
                target_type=target_type,
                cond_dict=retarget_cond_dict,
                opt=opt,
                output_dir=out_path,
                fps=fps,
            )

        _validate_reference_sampling_request(
            inpaint_enabled=inpaint_enabled,
            reference_mode=reference_mode,
            cross_species_reference=cross_species_reference,
        )

        if effective_reference_path:
            reference_bundle = _prepare_reference_for_mode(
                effective_reference_path,
                reference_mode=reference_mode,
                source_type=source_type,
                source_cond=source_cond_entry,
                target_type=object_type,
                target_cond=cond_dict[object_type],
                max_joints=max_joints,
                target_feature_len=model.feature_len,
                batch_size=args.batch_size,
                requested_output_frame_count=n_frames,
            )
            ref_motion = reference_bundle['reference_motion']
            reference_conditioning_kwargs = reference_bundle['reference_conditioning_kwargs']
            output_frame_count = reference_bundle['output_frame_count']
            loaded_reference_frame_count = reference_bundle['loaded_reference_frame_count']
            loaded_reference_joint_count = reference_bundle['loaded_reference_joint_count']

            if output_frame_count != n_frames:
                print(f'  Reference motion overrides frame count: {n_frames} -> {output_frame_count}')
            print(f'  Reference motion loaded: {effective_reference_path}')
            if prepared_reference_path != reference_motion_path:
                print(f'    Preprocessed from original: {reference_motion_path}')
            if effective_reference_path != prepared_reference_path:
                print(f'    Retargeted from preprocessed: {prepared_reference_path}')
            elif cross_species_reference and reference_mode == 'controlnet':
                print(f'    Controlnet prior path retargeted the reference into target-space before encoding')
            if reference_mode == 'controlnet':
                print(
                    f'    Reference prior input: [{loaded_reference_frame_count} frames, {loaded_reference_joint_count} joints] '
                    f'-> Output target: [{output_frame_count} frames, {max_joints} joints]'
                )
            else:
                print(
                    f'    Original: [{loaded_reference_frame_count} frames, {loaded_reference_joint_count} joints] '
                    f'-> Target: [{output_frame_count} frames, {max_joints} joints]'
                )
            if inpaint_enabled and skip_timesteps > 0:
                print(f'    Mode: inpaint + skip_timesteps={skip_timesteps} '
                      '(masked region starts from an img2img-noised reference; '
                      'unmasked region stays clamped to the original reference)')
            elif inpaint_enabled and reference_mode == 'controlnet':
                print(
                    f'    Mode: inpainting + controlnet '
                    f'(full schedule from pure noise, reference_scale={reference_scale})'
                )
            elif reference_mode == 'controlnet':
                print(
                    f'    Mode: controlnet prior conditioning '
                    f'(full schedule from pure noise, reference_scale={reference_scale})'
                )
            elif inpaint_enabled:
                print('    Mode: inpainting (reference is the clamped known region; '
                      'skip_timesteps=0, denoising full schedule from pure noise)')
            else:
                print(f'    skip_timesteps: {skip_timesteps} (higher = more faithful to reference)')
            if inpaint_enabled:
                if repaint_enabled:
                    print(
                        f'    RePaint resampling: on '
                        f'(jump_length={repaint_jump_length}, '
                        f'jump_n_sample={repaint_jump_n_sample})'
                    )
                else:
                    print(
                        '    RePaint resampling: off '
                        '(single reverse pass; known region is still clamped every step)'
                    )

        # Build inpaint mask (masked region = regenerated, rest clamped to ref).
        inpaint_mask = None
        if inpaint_enabled:
            if ref_motion is None:
                sys.exit(
                    "ERROR: --inpaint_* is set but the reference motion could "
                    "not be loaded; cannot inpaint without a known region."
                )
            inpaint_mask = build_inpaint_mask(
                cond_dict[object_type],
                inpaint_joints_arg,
                inpaint_include_subtree,
                inpaint_frames_arg,
                args.batch_size,
                opt.max_joints,
                output_frame_count,
            )

        # Create condition with effective frame count
        obj_batch = [object_type] * args.batch_size
        _, model_kwargs = create_condition(
            obj_batch,
            cond_dict,
            output_frame_count,
            args.temporal_window,
            max_joints=opt.max_joints,
            feature_len=opt.feature_len
        )
        sample = _sample_batch(
            diffusion=diffusion,
            model=model,
            model_kwargs=model_kwargs,
            sampling_method=sampling_method,
            sample_shape=(args.batch_size, max_joints, model.feature_len, output_frame_count),
            ddim_eta=ddim_eta,
            seed=args.seed,
            device=dist_util.dev(),
            reference_motion=ref_motion,
            reference_conditioning_kwargs=reference_conditioning_kwargs,
            reference_mode=reference_mode,
            reference_scale=reference_scale,
            skip_timesteps=skip_timesteps,
            inpaint_mask=inpaint_mask,
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
        )

        # Pre-compute filenames with a single directory scan
        existing_npy_files = [
            f for f in os.listdir(out_path)
            if f.startswith(object_type) and f.endswith('.npy')
        ]
        base_index = len(existing_npy_files)

        # Extract T-pose rest rotations (6D → quaternion)
        _tff = cond_dict[object_type].get('tpos_first_frame')
        _tpose_rest_rotations = None
        if _tff is not None:
            from utils.rotation_conversions import rotation_6d_to_matrix_np
            from motion_lib.Quaternions import Quaternions
            _rot6d = np.asarray(_tff[:, 3:9], dtype=np.float64)
            _tpose_rest_rotations = Quaternions.from_transforms(
                rotation_6d_to_matrix_np(_rot6d)
            ).qs

        # Collect export tasks (in-process, no pickling needed)
        joint_names = cond_dict[object_type].get(
            'canonical_bvh_joint_names',
            cond_dict[object_type]['joints_names'],
        )
        export_tasks = []
        for sample_idx, motion in enumerate(sample):
            n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
            motion = motion[:n_joints]
            parents = model_kwargs['y']['parents'][sample_idx]
            mean = cond_dict[object_type]['mean'][None, :]
            std = cond_dict[object_type]['std'][None, :]
            motion_np = motion.cpu().permute(2, 0, 1).numpy() * std + mean
            offsets = cond_dict[object_type]['offsets']

            npy_name = f'{object_type}_#{base_index + sample_idx}.npy'
            export_tasks.append((
                motion_np,
                parents,  # already np.ndarray, shared in-process
                offsets,
                npy_name,
                joint_names,
                out_path,
                fps,
                _tpose_rest_rotations,
            ))

        # Parallel export using ThreadPoolExecutor.
        # np.save / np.savetxt release the GIL (C-level I/O), so threads
        # can overlap I/O with each other and with the host sampler loop.
        results = list(
            tqdm(export_pool.map(_export_motion, export_tasks),
                  total=len(export_tasks),
                  desc=f'{object_type} export')
        )
        for npy_name in results:
            print(f'    Created motion: {npy_name}')
    finally:
        export_pool.shutdown(wait=True)

    return out_path


def _parse_frame_ranges(spec, n_frames):
    """Parse '40-90' / '0-20,150-180' / '30' into a set of frame indices,
    inclusive and clipped to [0, n_frames - 1]. Empty spec => all frames.
    """
    if not spec:
        return set(range(n_frames))
    frames = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo_str, hi_str = chunk.split('-', 1)
            lo, hi = int(lo_str), int(hi_str)
        else:
            lo = hi = int(chunk)
        if lo > hi:
            lo, hi = hi, lo
        lo = max(0, lo)
        hi = min(n_frames - 1, hi)
        frames.update(range(lo, hi + 1))
    if not frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {n_frames} frames, indices 0..{n_frames - 1})"
        )
    return frames


def _resolve_inpaint_joint_indices(cond_entry, names_arg, include_subtree):
    """Resolve comma-separated joint names to a set of joint indices.

    Names are matched against the union of the raw / canonical / canonical_bvh
    alias lists (all same length and index order). When include_subtree is set,
    every descendant of a selected joint is added too. Empty names_arg => all
    real joints.
    """
    raw_names = list(cond_entry['joints_names'])
    n_joints = len(raw_names)
    canon = list(cond_entry.get('canonical_joint_names', raw_names))
    canon_bvh = list(cond_entry.get('canonical_bvh_joint_names', raw_names))

    if not names_arg:
        base = set(range(n_joints))
    else:
        alias_to_index = {}
        for idx in range(n_joints):
            for alias in (raw_names[idx], canon[idx], canon_bvh[idx]):
                if alias is not None:
                    alias_to_index.setdefault(str(alias), idx)
        base = set()
        invalid = []
        for token in names_arg.split(','):
            token = token.strip()
            if not token:
                continue
            if token in alias_to_index:
                base.add(alias_to_index[token])
            else:
                invalid.append(token)
        if invalid:
            table = ['  idx | raw | canonical | canonical_bvh']
            for idx in range(n_joints):
                table.append(f'  {idx:>3} | {raw_names[idx]} | {canon[idx]} | {canon_bvh[idx]}')
            raise ValueError(
                f"--inpaint_joints: unknown joint name(s) {invalid}.\n"
                "Accepted names (any of the three aliases):\n" + '\n'.join(table)
            )

    if not include_subtree or not base:
        return base, n_joints

    parents = np.asarray(cond_entry['parents'], dtype=np.int64)
    children = [[] for _ in range(n_joints)]
    for j in range(n_joints):
        p = int(parents[j])
        if 0 <= p < n_joints:
            children[p].append(j)
    selected = set(base)
    stack = list(base)
    while stack:
        cur = stack.pop()
        for child in children[cur]:
            if child not in selected:
                selected.add(child)
                stack.append(child)
    return selected, n_joints


def build_inpaint_mask(
    cond_entry,
    inpaint_joints_arg,
    inpaint_include_subtree,
    inpaint_frames_arg,
    batch_size,
    max_joints,
    n_frames,
):
    """Build the inpainting mask tensor [B, max_joints, 1, n_frames].

    Convention: 1.0 = regenerate (free), 0.0 = keep reference (clamped).
    Padding joints (index >= n_joints) stay 0.0. The regenerated region is
    selected-joints x selected-frames; everything else is held to the
    reference during sampling.
    """
    joint_indices, n_joints = _resolve_inpaint_joint_indices(
        cond_entry, inpaint_joints_arg, inpaint_include_subtree
    )
    frame_indices = _parse_frame_ranges(inpaint_frames_arg, n_frames)
    if not joint_indices:
        raise ValueError("--inpaint_joints resolved to an empty joint set")

    mask = np.zeros((max_joints, 1, n_frames), dtype=np.float32)
    j_idx = np.fromiter(
        (j for j in joint_indices if 0 <= j < n_joints), dtype=np.int64
    )
    f_idx = np.fromiter(
        (f for f in frame_indices if 0 <= f < n_frames), dtype=np.int64
    )
    if j_idx.size and f_idx.size:
        mask[np.ix_(j_idx, [0], f_idx)] = 1.0

    n_regen_joints = int(j_idx.size)
    n_regen_frames = int(f_idx.size)
    print(
        f'  Inpaint mask: regenerating {n_regen_joints}/{n_joints} joints x '
        f'{n_regen_frames}/{n_frames} frames '
        f'(joints={sorted(int(j) for j in j_idx)[:20]}'
        f'{"..." if n_regen_joints > 20 else ""}, '
        f'subtree={"on" if inpaint_include_subtree else "off"})'
    )

    mask_t = torch.from_numpy(mask).unsqueeze(0).expand(
        batch_size, -1, -1, -1
    ).contiguous()
    return mask_t


def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len):
    """Build model_kwargs for a batch of object_types.
    """
    batches = list()
    for object_type in object_types:
        if object_type not in cond_dict:
            available = ', '.join(sorted(cond_dict.keys()))
            raise KeyError(
                f"Unknown object_type '{object_type}'. Available object types in cond file: {available}"
            )
        batch = list()
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        tpos_first_frame = (tpos_first_frame - mean) / (std + 1e-6)
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = cond_dict[object_type]['joints_names_embs']
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(tpos_first_frame)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batch.append(max_joints)
        batch.append(object_type)
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
