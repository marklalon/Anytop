# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys
from dataclasses import dataclass

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
    resample_motion_features,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    FOOT_CONTACT_VEL_THRESH,
    get_common_features_from_T_pose,
    get_motion,
    recover_bvh_export_animation_from_motion_np,
)
from motion_lib import BVH, FBX
from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions
from os.path import join as pjoin
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion_general_skeleton,
    load_model,
    model_supports_global_energy_conditioning,
    resolve_t5_out_dim,
    unwrap_anytop_model,
)
from utils.parser_util import generate_args
from utils.misc import infer_object_type_from_filename


_REFERENCE_MOTION_PREPROCESS_SUFFIXES = {'.fbx', '.glb', '.gltf'}


@dataclass
class GenerationRuntime:
    opt: object
    cond_dict: dict
    actual_cond_file: str
    model: torch.nn.Module
    diffusion: object
    model_path: str
    device: int
    sampling_method: str
    sampling_steps: int
    amp_dtype: str
    cond_path: str

    def validate_args(self, args):
        expected_model = os.path.realpath(self.model_path)
        actual_model = os.path.realpath(args.model_path)
        if actual_model != expected_model:
            raise ValueError(
                f"GenerationRuntime was prepared for model_path={self.model_path!r}, "
                f"but task requested {args.model_path!r}"
            )
        if int(getattr(args, 'device', 0)) != int(self.device):
            raise ValueError("GenerationRuntime cannot be reused across different --device values")
        sampling_steps = int(getattr(args, 'sampling_steps', 100))
        sampling_method = str(getattr(args, 'sampling_method', 'ddim')).lower()
        if sampling_method != self.sampling_method or sampling_steps != self.sampling_steps:
            raise ValueError(
                "GenerationRuntime cannot be reused when --sampling_method or "
                "--sampling_steps changes"
            )
        amp_dtype = str(getattr(args, 'amp_dtype', 'fp32')).lower()
        if amp_dtype != self.amp_dtype:
            raise ValueError("GenerationRuntime cannot be reused across different --amp_dtype values")


def _load_generation_cond(args, opt, cond_dict=None):
    if cond_dict is None:
        if args.cond_path:
            return np.load(args.cond_path, allow_pickle=True).item(), args.cond_path
        return np.load(opt.cond_file, allow_pickle=True).item(), opt.cond_file
    return cond_dict, opt.cond_file


def _normalize_optional_path(path):
    return os.path.realpath(path) if path else ''


def _raise_opt_max_joints_for_cond(opt, cond_dict):
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


def _configure_sampling_args(args):
    sampling_steps = int(getattr(args, 'sampling_steps', 100))
    sampling_method = str(getattr(args, 'sampling_method', 'ddim')).lower()
    if sampling_steps > 0:
        if sampling_method == 'ddim':
            args.timestep_respacing = f'ddim{sampling_steps}'
        else:
            args.timestep_respacing = str(sampling_steps)
    else:
        args.timestep_respacing = ''
    return sampling_method, sampling_steps


def _resolve_inference_amp_dtype(args):
    """Resolve the effective AMP dtype for inference.

    Sampling runs under a single top-level ``torch.autocast`` context (applied in
    ``_sample_batch``); this only validates bf16 availability and returns the
    effective dtype string ('bf16' or 'fp32'). It does not mutate the model.
    """
    amp_dtype_arg = str(getattr(args, 'amp_dtype', 'fp32')).lower()
    if amp_dtype_arg != 'bf16':
        return amp_dtype_arg
    _amp_device = dist_util.dev()
    if _amp_device.type == 'cuda' and torch.cuda.is_bf16_supported():
        print('bf16 autocast enabled for sampling via torch.autocast; softmax/layernorm stay fp32.')
        return 'bf16'
    print(
        '[generate] WARNING: --amp_dtype bf16 requested but the active device is CPU or lacks '
        'bf16 support; falling back to fp32.'
    )
    return 'fp32'


def prepare_generation_runtime(args=None, cond_dict=None):
    if args is None:
        args = generate_args()

    dist_util.setup_dist(args.device)
    opt = get_opt(args.device)
    cond_dict, actual_cond_file = _load_generation_cond(args, opt, cond_dict)
    _raise_opt_max_joints_for_cond(opt, cond_dict)

    print('Creating model and diffusion...')
    resolve_t5_out_dim(args, cond_source=actual_cond_file)
    sampling_method, sampling_steps = _configure_sampling_args(args)
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f'Loading checkpoints from [{args.model_path}]...')
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'model_avg' in state_dict:
        print('EMA checkpoint detected, loading model_avg weights.')
        state_dict = state_dict['model_avg']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    load_model(model, state_dict)

    print('Validating precomputed joint-name embeddings from cond.npy...')
    ensure_joint_name_embeddings(
        cond_dict,
        expected_embedding_dim=args.t5_out_dim,
        cond_source=actual_cond_file,
    )
    model.to(dist_util.dev())
    model.eval()
    amp_dtype = _resolve_inference_amp_dtype(args)

    return GenerationRuntime(
        opt=opt,
        cond_dict=cond_dict,
        actual_cond_file=actual_cond_file,
        model=model,
        diffusion=diffusion,
        model_path=args.model_path,
        device=int(getattr(args, 'device', 0)),
        sampling_method=sampling_method,
        sampling_steps=sampling_steps,
        amp_dtype=amp_dtype,
        cond_path=_normalize_optional_path(getattr(args, 'cond_path', '') or ''),
    )


def validate_reference_configuration(
    reference_motion_path=None,
    skip_timesteps=0,
    global_energy=None,
):
    # --global_energy is optional with --reference_motion. When omitted,
    # the model uses the global-energy CFG drop path (unconditional energy
    # token, FiLM sublayer bypassed). When provided, it overrides with the
    # explicit z-score value — useful for controlled energy sweeps on
    # reference-guided generation.
    return int(skip_timesteps) if skip_timesteps is not None else 0


def _finalize_output_lengths(requested_frames, min_length, internal_num_frames):
    """Validate the requested output frame count M and derive the playspeed
    conditioning value. Returns ``(requested_output_frames, target_output_frames,
    playspeed_cond_value)``.
    """
    if requested_frames < min_length or requested_frames > 2 * internal_num_frames:
        sys.exit(
            f"ERROR: num_frames M={requested_frames} outside "
            f"[min_length={min_length}, 2*num_frames={2 * internal_num_frames}]"
        )
    playspeed = float(requested_frames) / float(internal_num_frames)
    return requested_frames, requested_frames, playspeed


def resolve_global_energy_condition(model, global_energy, batch_size):
    if global_energy is None:
        return None
    if not model_supports_global_energy_conditioning(model):
        raise ValueError(
            "Loaded checkpoint does not support global energy conditioning. Load or retrain a checkpoint with --global_energy_cond enabled."
        )

    unwrapped_model = unwrap_anytop_model(model)
    running_mean = unwrapped_model.global_energy_running_mean.detach().to(device='cpu', dtype=torch.float32).clone()
    running_var = unwrapped_model.global_energy_running_var.detach().to(device='cpu', dtype=torch.float32).clone()
    running_std = torch.sqrt(running_var.clamp_min(1e-6))
    # CLI value is in normalized space (Z-score against training distribution).
    # De-normalize to raw space so that downstream _build_global_energy_token
    # re-normalizes it correctly: raw = norm * running_std + running_mean.
    raw = running_mean.clone()
    raw[0] = float(global_energy) * running_std[0] + running_mean[0]
    if not torch.isfinite(raw).all():
        raise ValueError("--global_energy must be finite")
    return raw.unsqueeze(0).expand(batch_size, -1).clone()


def _compute_global_energy_from_reference(ref_motion, n_joints, playspeed_cond=None):
    """Extract raw global energy [mean, std] from a reference motion tensor.

    ``ref_motion`` must be a (B, J, F, T) tensor in the model feature space.
    Returns a (B, 1) float32 tensor on the same device with column
    ``[global_energy]`` ready for ``_build_global_energy_token``.
    """
    from Anytop.model.anytop import GlobalEnergyExtractor

    return GlobalEnergyExtractor.compute_global_energy_condition(
        ref_motion,
        n_joints,
        playspeed_cond=playspeed_cond,
    )



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


def _should_retarget_reference(source_type, target_type):
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


def _reference_skeleton_from_tpose(tp):
    """Return (names, parents) of the reference skeleton.

    Under own-rotation encoding no leaf rotation helpers are appended,
    so the full tp.names list is the reference skeleton.
    """
    return list(tp.names), np.asarray(tp.tpos_anim.parents, dtype=np.int32)


def _reindex_animation_subset(raw_anim, names, keep_indices):
    """Drop all joints except ``keep_indices`` and reindex parents/arrays.

    ``keep_indices`` must already be in ascending (DFS pre-order) order and must
    be closed under the parent relation (every kept joint's parent is kept), which
    the caller guarantees before calling.
    """
    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    new_names = [names[i] for i in keep_indices]
    new_parents = np.array(
        [old_to_new[int(raw_anim.parents[i])] if int(raw_anim.parents[i]) >= 0 else -1
         for i in keep_indices],
        dtype=np.int32,
    )
    new_anim = Animation(
        Quaternions(raw_anim.rotations.qs[:, keep_indices].copy()),
        raw_anim.positions[:, keep_indices].copy(),
        Quaternions(raw_anim.orients.qs[keep_indices].copy()),
        raw_anim.offsets[keep_indices].copy(),
        new_parents,
    )
    return new_anim, new_names


def _align_reference_skeleton(raw_anim, names, expected_names, expected_parents, fname, source_type=None):
    """Validate/repair the raw reference skeleton against the dataset reference skeleton.

    ``get_motion`` assumes the raw animation's joints match the dataset's canonical
    skeleton index-for-index, then appends leaf-rotation helpers to reach the feature
    joint count. That assumption is silent: a reference carrying extra terminal bones
    (e.g. Blender ``*_end`` tip bones materialised on FBX/GLB export) can coincidentally
    match the *augmented* joint count and get mismatched joint-by-joint, scrambling the
    whole skeleton.

    Defense:
      * If the skeleton already matches the reference, return it unchanged.
      * Otherwise strip terminal (leaf) bones whose names are not in the reference
        — this removes ``*_end``-style tip bones while preserving DFS order — and
        re-validate.
      * If it still does not match (missing joints, reordered joints, or an
        *internal* extra bone that cannot be safely dropped), raise with a clear
        diff instead of silently producing corrupt motion.
    """
    expected_name_set = set(expected_names)

    if list(names) == list(expected_names):
        return raw_anim, list(names)

    # Identify leaf joints (no children) whose names are not in the reference.
    parents = np.asarray(raw_anim.parents, dtype=np.int32)
    has_children = np.zeros(len(names), dtype=bool)
    has_children[parents[parents >= 0]] = True
    unexpected_leaves = [
        i for i, name in enumerate(names)
        if name not in expected_name_set and not has_children[i]
    ]

    if unexpected_leaves:
        keep_indices = [i for i in range(len(names)) if i not in set(unexpected_leaves)]
        # Reject if dropping orphans an expected joint (its parent was removed) —
        # that means the extra bone is internal and cannot be safely stripped.
        keep_set = set(keep_indices)
        for i in keep_indices:
            p = int(parents[i])
            if p >= 0 and p not in keep_set:
                break
        else:
            stripped_names = [names[i] for i in unexpected_leaves]
            raw_anim, names = _reindex_animation_subset(raw_anim, names, keep_indices)
            print(
                f"[generate] WARNING: stripped {len(stripped_names)} terminal bone(s) "
                f"from reference {fname} not present in the '{','.join(expected_names[:1])}...' "
                f"reference skeleton: {stripped_names[:10]}"
                f"{'...' if len(stripped_names) > 10 else ''}. These are typically Blender "
                f"'*_end' tip bones; the clip is processed on the canonical "
                f"{len(expected_names)}-joint skeleton."
            )

    # Final structural validation (names AND parents must match exactly).
    if list(names) == list(expected_names) and np.array_equal(
        np.asarray(raw_anim.parents, dtype=np.int32), expected_parents
    ):
        return raw_anim, list(names)

    extra = [n for n in names if n not in expected_name_set]
    missing = [n for n in expected_names if n not in set(names)]
    type_label = f" (object_type='{source_type}')" if source_type else ""
    raise ValueError(
        f"Reference motion skeleton of '{fname}'{type_label} does not match the dataset reference "
        f"skeleton ({len(expected_names)} joints, root "
        f"'{expected_names[0] if expected_names else '?'}') and cannot be auto-aligned.\n"
        f"  input joints  : {len(names)}\n"
        f"  extra joints  ({len(extra)}): {extra[:10]}{'...' if len(extra) > 10 else ''}\n"
        f"  missing joints({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
        f"  Re-export the reference on the dataset's canonical skeleton. (A same-count but "
        f"differently-ordered skeleton would otherwise be silently scrambled.)"
    )


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
    # Pass the cond entry's recorded face joints so the reproduced orientation_quat
    # matches the feature space the dataset (and the model's mean/std) were built in;
    # omitting it would silently fall back to default face joints for any object_type
    # that was authored with custom ones, misaligning the reference.
    source_tp = get_common_features_from_T_pose(
        tpose_path,
        source_type,
        face_joints=source_cond.get('face_joint_names') or None,
        max_joints=preprocess_max_joints,
    )
    scale_factor = float(source_cond.get('scale_factor', source_tp.scale_factor))

    # Defense: align the raw reference skeleton to the dataset's canonical skeleton
    # before get_motion blindly maps it index-by-index against the T-pose. This strips
    # stray terminal '*_end' tip bones (common on Blender FBX/GLB export) and fast-fails
    # on a real structural mismatch instead of silently scrambling the joints.
    fname = os.path.basename(reference_motion_path)
    raw_anim, names, _frame_time = FBX.load(reference_motion_path)
    anim_len = len(raw_anim)
    expected_names, expected_parents = _reference_skeleton_from_tpose(source_tp)
    raw_anim, names = _align_reference_skeleton(
        raw_anim, names, expected_names, expected_parents, fname, source_type
    )

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
        slice_inds=[0, anim_len],
        preloaded=(raw_anim, names),
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

    Thin wrapper around ``utils.auto_retarget.retarget_features_npy_to_target``.
    Loads source features, builds target TPoseFeatures, delegates the math, then
    writes the retargeted .npy and an inspection .bvh under ``output_dir``.
    """
    from Anytop.utils.auto_retarget import (
        retarget_features_npy_to_target,
    )
    from data_loaders.truebones.truebones_utils.features import get_common_features_from_T_pose

    src_cond = cond_dict[source_type]
    tgt_cond = dict(cond_dict[target_type])

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

    tgt_cond['translation_root_index'] = _require_cond_translation_root_index(
        tgt_cond,
        object_type=target_type,
        context='Cross-species reference retarget',
    )

    tgt_tp = get_common_features_from_T_pose(
        tgt_tpose_path, target_type,
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
            translation_root_index=tgt_cond.get('translation_root_index'),
            allow_infer=tgt_cond.get('translation_root_index') is None,
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


def _retarget_reference_motion_from_file(
    reference_motion_path,
    target_type,
    cond_dict,
    opt,
    output_dir,
    fps,
):
    """Retarget a raw .fbx/.glb/.gltf reference onto ``target_type``, cond-free on
    the source side.

    Unlike :func:`_retarget_reference_motion` (which consumes a feature .npy and
    needs the source object's cond entry), this reads the source skeleton and
    motion straight from the animation file via ``FBX.load`` and delegates to
    ``utils.auto_retarget.retarget_animation_file_to_target``. It works even when
    the source object_type is not present in ``cond_dict`` — only the target's
    cond/T-pose is required. Output artifacts (retargeted .npy + inspection .bvh)
    match the feature-npy retarget path.
    """
    from Anytop.utils.auto_retarget import (
        retarget_animation_file_to_target,
    )
    from data_loaders.truebones.truebones_utils.features import get_common_features_from_T_pose

    tgt_cond = dict(cond_dict[target_type])
    tgt_tpose_path = tgt_cond.get('orientation_reference_fbx_path')
    if not tgt_tpose_path or not os.path.isfile(tgt_tpose_path):
        raise FileNotFoundError(
            f"Reference retarget requires the target T-pose file "
            f"(cond_dict['{target_type}']['orientation_reference_fbx_path']), "
            f"not found: {tgt_tpose_path!r}"
        )

    # Source label is for output naming only — never used to look up cond or to
    # drive any object_type-dependent processing on the source.
    base = os.path.splitext(os.path.basename(reference_motion_path))[0]
    source_label = infer_object_type_from_filename(reference_motion_path, valid_types=None) or base

    print(
        f"\n### Reference retarget (cond-free source): {reference_motion_path} → {target_type}"
    )

    tgt_cond['translation_root_index'] = _require_cond_translation_root_index(
        tgt_cond,
        object_type=target_type,
        context='Cond-free reference retarget',
    )

    tgt_tp = get_common_features_from_T_pose(
        tgt_tpose_path, target_type,
        max_joints=opt.max_joints,
    )

    target_features = retarget_animation_file_to_target(
        reference_motion_path,
        tgt_tp,
        target_type,
        opt.max_joints,
        tgt_cond,
    )

    if target_features is None:
        raise RuntimeError(
            f"retarget_animation_file_to_target returned None "
            f"({reference_motion_path} → {target_type}). Check target T-pose FBX "
            f"and joint-name overlap with the source file."
        )

    out_npy = os.path.join(output_dir, f"_retargeted_{source_label}_to_{target_type}__{base}.npy")
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
            translation_root_index=tgt_cond.get('translation_root_index'),
            allow_infer=tgt_cond.get('translation_root_index') is None,
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
    # The model is a fixed-window model: every training clip is resampled to
    # num_frames (see dataset._resample_motion_features), so the temporal
    # transformer and the loop-phase positional embedding are only ever valid at
    # the native window length. Always run the model at that native window
    # (requested_output_frame_count == num_frames) and resample the reference up
    # to it, exactly like the pure-generation path. The requested output length
    # is honored afterwards by resampling the sampled motion to
    # target_output_frames. Previously this used min(loaded, requested), which
    # ran the model at a shorter, never-trained window whenever the (cropped)
    # reference was shorter than num_frames -- breaking loop closure and quality
    # for num_frames < internal num_frames.
    output_frame_count = int(requested_output_frame_count)
    max_source_frames = max(int(min_length), output_frame_count * 2)
    if loaded_reference_frame_count > max_source_frames:
        visible_frames = output_frame_count if requested_visible_frame_count is None else int(requested_visible_frame_count)
        source_frames = min(max_source_frames, max(int(min_length), visible_frames))
        ref_raw = ref_raw[:source_frames]
    reference_source_frame_count = int(ref_raw.shape[0])
    if ref_raw.shape[0] != output_frame_count:
        ref_raw = resample_motion_features(ref_raw, output_frame_count)

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
        'output_frame_count': output_frame_count,
        'loaded_reference_frame_count': loaded_reference_frame_count,
        'reference_source_frame_count': reference_source_frame_count,
        'loaded_reference_joint_count': loaded_reference_joint_count,
    }


def _export_motion(task):
    motion_np, parents_np, offsets, npy_name, joint_names, out_path, fps, tpose_rest_rotations, translation_root_index = task
    out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_np,
        parents_np,
        offsets,
        joint_names,
        translation_root_index=translation_root_index,
        allow_infer=translation_root_index is None,
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
    skip_timesteps=0,
    inpaint_mask=None,
    autocast_dtype=None,
):
    skip_timesteps = int(skip_timesteps) if skip_timesteps is not None else 0

    inpainting = inpaint_mask is not None
    if inpainting:
        if reference_motion is None:
            raise ValueError("inpaint_mask given without a reference_motion")
        if int(reference_motion.shape[-1]) != int(sample_shape[-1]):
            raise ValueError(
                "Motion inpainting requires reference_motion frame count to match target sample length; "
                f"got reference {reference_motion.shape[-1]} and target {sample_shape[-1]}"
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

    def _copy_model_kwargs_for_loop(cross_limb_unreliable_mask_):
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
        loop_model_kwargs['y'] = loop_y
        return loop_model_kwargs

    def _autocast_context():
        # Single top-level autocast context around the reverse-diffusion loop;
        # the model is invoked many times deep inside the diffusion sampler, so
        # this is the natural call site to apply standard bf16 autocast.
        if autocast_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=autocast_dtype)

    def _run_loop(noise, init_image, skip_ts, inpaint_mask_, inpaint_reference_, cross_limb_unreliable_mask_):
        common_kwargs = dict(
            model=model,
            shape=sample_shape,
            noise=noise,
            clip_denoised=False,
            model_kwargs=_copy_model_kwargs_for_loop(cross_limb_unreliable_mask_),
            device=device,
            init_image=init_image,
            skip_timesteps=skip_ts,
        )
        # Only p_* / ddim_* loops accept the inpaint kwargs.
        inpaint_kwargs = dict(
            inpaint_mask=inpaint_mask_, inpaint_reference=inpaint_reference_
        )
        with _autocast_context():
            if sampling_method == 'ddim':
                return diffusion.ddim_sample_loop(
                    progress=True,
                    eta=ddim_eta,
                    **inpaint_kwargs,
                    **common_kwargs,
                )
            if sampling_method in ('p', 'ddpm'):
                return diffusion.p_sample_loop(
                    progress=True,
                    dump_steps=None,
                    const_noise=False,
                    **inpaint_kwargs,
                    **common_kwargs,
                )
        raise ValueError(f'Unknown sampling_method: {sampling_method}')

    if inpainting and skip_timesteps > 0:
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
        )

    fixseed(seed)
    if inpainting:
        # Motion inpainting, skip_timesteps == 0: the
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
        )
    # Plain generation: no reference => full denoising from pure noise.
    return _run_loop(
        noise=torch.randn(sample_shape, device=device),
        init_image=None,
        skip_ts=0,
        inpaint_mask_=None,
        inpaint_reference_=None,
        cross_limb_unreliable_mask_=None,
    )


def _generate_all_species(
    cond_dict,
    cond_max_joints,
    opt,
    args,
    n_frames,
    playspeed_cond_value,
    target_output_frames,
    global_energy_condition,
    model,
    diffusion,
    sampling_method,
    inference_autocast_dtype,
    out_path,
    fps,
):
    """Generate exactly one motion per species in mixed-species batches.

    Each batch packs up to ``args.batch_size`` different species into a single
    forward pass.  The model and sampler are batch-agnostic — all conditioning
    is per-sample — so this is both correct and more efficient than looping
    over species one at a time.
    """
    all_species = sorted(cond_dict.keys())
    batch_size = int(args.batch_size)
    species_batches = [all_species[i:i + batch_size] for i in range(0, len(all_species), batch_size)]

    output_frame_count = int(n_frames)
    total_species = len(all_species)
    print(f'\n### Multi-species generation: {total_species} species, '
          f'{len(species_batches)} batch(es) of batch_size={batch_size}')
    print(f'  All species: {", ".join(all_species)}')

    for batch_idx, batch_species in enumerate(species_batches, 1):
            actual_bs = len(batch_species)
            batch_max_joints = max(
                len(np.asarray(cond_dict[sp]['parents'])) for sp in batch_species
            )
            if batch_max_joints > cond_max_joints:
                raise RuntimeError(
                    f"Batch {batch_idx} max_joints={batch_max_joints} exceeds "
                    f"cond/model max_joints={cond_max_joints}"
                )

            print(f'\n--- Batch {batch_idx}/{len(species_batches)} ({actual_bs} species, '
                  f'max_joints={batch_max_joints}): {", ".join(batch_species)} ---')

            # Build model kwargs for this heterogeneous batch.
            _, model_kwargs = create_condition(
                list(batch_species),
                cond_dict,
                output_frame_count,
                args.temporal_window,
                max_joints=batch_max_joints,
                feature_len=opt.feature_len,
                loop=getattr(args, 'loop', False),
            )
            if global_energy_condition is not None:
                model_kwargs['y']['global_energy_cond'] = (
                    global_energy_condition[:actual_bs].clone()
                )
            model_kwargs['y']['playspeed_cond'] = torch.full(
                (actual_bs,), playspeed_cond_value, dtype=torch.float32, device=dist_util.dev(),
            )

            # Sample the whole batch in one forward pass.
            print(f'  Sampling {actual_bs} species × 1 motion each ...')
            sample = _sample_batch(
                diffusion=diffusion,
                model=model,
                model_kwargs=model_kwargs,
                sampling_method=sampling_method,
                sample_shape=(actual_bs, batch_max_joints, model.feature_len, output_frame_count),
                ddim_eta=float(getattr(args, 'ddim_eta', 0.0)),
                seed=args.seed,
                device=dist_util.dev(),
                autocast_dtype=inference_autocast_dtype,
            )

            # ── Per-sample export with per-species metadata ──────────
            export_tasks = []
            for sample_idx, motion in enumerate(sample):
                sp = batch_species[sample_idx]
                sp_entry = cond_dict[sp]
                n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
                motion = motion[:n_joints]
                parents = model_kwargs['y']['parents'][sample_idx]
                motion_np = (motion.cpu().permute(2, 0, 1).numpy()
                             * sp_entry['std'][None, :] + sp_entry['mean'][None, :])

                if target_output_frames != output_frame_count:
                    motion_np = resample_motion_features(motion_np, target_output_frames)

                translation_root_index = _get_batch_translation_root_index(
                    model_kwargs, sample_idx,
                    fallback=sp_entry.get('translation_root_index', 0),
                )
                if _root_xz_locomotion_is_degenerate(
                    sp_entry['std'], translation_root_index
                ):
                    _suppress_degenerate_root_xz_velocity(motion_np, translation_root_index)
                if getattr(args, 'loop', False):
                    _close_loop_root_xz_via_velocity(motion_np, translation_root_index)

                joint_names = sp_entry.get(
                    'canonical_bvh_joint_names', sp_entry['joints_names'],
                )

                # T-pose rest rotations (per-species)
                _tpose_rr = None
                _tff = sp_entry.get('tpos_first_frame')
                if _tff is not None:
                    from utils.rotation_conversions import rotation_6d_to_matrix_np
                    from motion_lib.Quaternions import Quaternions as _QQ
                    _rot6d = np.asarray(_tff[:, 3:9], dtype=np.float64)
                    _tpose_rr = _QQ.from_transforms(rotation_6d_to_matrix_np(_rot6d)).qs

                # Count existing outputs so repeated runs don't overwrite.
                existing = [f for f in os.listdir(out_path)
                            if f.startswith(sp) and f.endswith('.npy')]
                npy_name = f'{sp}_#{(len(existing))}.npy'
                export_tasks.append((
                    motion_np, parents, sp_entry['offsets'], npy_name, joint_names,
                    out_path, fps, _tpose_rr, translation_root_index,
                ))

            for task in tqdm(export_tasks, desc=f'batch {batch_idx} export'):
                npy_name = _export_motion(task)
                print(f'    Created: {npy_name}')


def main(args=None, cond_dict=None, runtime=None):
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

    # NOTE: the "--reference_motion needs --skip_timesteps (or inpaint)" check is
    # deferred until after the reference frame count R is known, because an
    # R < M length extension auto-enables outpaint (a temporal inpaint) and so
    # does not require --skip_timesteps. See the crop/outpaint block below.

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

    # --inpaint_joints with --skip_timesteps omitted: default to 0 (skip disabled).
    if _inpaint_early and skip_timesteps_raw is None:
        skip_timesteps_raw = 0

    try:
        skip_timesteps = validate_reference_configuration(
            reference_motion_path=getattr(args, 'reference_motion', None),
            skip_timesteps=skip_timesteps_raw,
            global_energy=getattr(args, 'global_energy', None),
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    if runtime is None:
        runtime = prepare_generation_runtime(args, cond_dict=cond_dict)
    else:
        runtime.validate_args(args)
        # If the task specifies a different --cond_path, reload cond and update
        # the runtime in-place so the model/diffusion can still be shared.
        task_cond = _normalize_optional_path(getattr(args, 'cond_path', '') or '')
        if task_cond != runtime.cond_path:
            new_cond_dict, new_actual_cond_file = _load_generation_cond(args, runtime.opt)
            _raise_opt_max_joints_for_cond(runtime.opt, new_cond_dict)
            runtime.cond_dict = new_cond_dict
            runtime.actual_cond_file = new_actual_cond_file
            runtime.cond_path = task_cond

    opt = runtime.opt
    cond_dict = runtime.cond_dict
    actual_cond_file = runtime.actual_cond_file
    model = runtime.model
    diffusion = runtime.diffusion
    sampling_method = runtime.sampling_method
    sampling_steps = runtime.sampling_steps
    inference_autocast_dtype = torch.bfloat16 if runtime.amp_dtype == 'bf16' else None

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = opt.fps
    internal_num_frames = int(getattr(args, 'num_frames', 60))
    min_length = int(getattr(args, 'min_length', 20))
    n_frames = internal_num_frames
    cond_max_joints = opt.max_joints

    reference_present = bool(getattr(args, 'reference_motion', None))
    motion_frames = getattr(args, 'num_frames', None)
    if motion_frames is None and not reference_present:
        # Pure-random generation with no --num_frames defaults to 60 frames.
        motion_frames = 60

    # Output lengths are finalized here when --num_frames is known. When it
    # is omitted together with --reference_motion they are deferred until the
    # reference frame count R is known (the output length then defaults to R,
    # clamped to [min_length, 2*num_frames]).
    requested_output_frames = target_output_frames = playspeed_cond_value = None
    if motion_frames is not None:
        requested_output_frames, target_output_frames, playspeed_cond_value = (
            _finalize_output_lengths(
                motion_frames,
                min_length,
                internal_num_frames,
            )
        )
    object_type = args.object_type
    if out_path == '':
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            'samples_{}_{}_seed{}'.format(name, niter, args.seed),
        )
    os.makedirs(out_path, exist_ok=True)

    try:
        global_energy_condition = resolve_global_energy_condition(
            model,
            getattr(args, 'global_energy', None),
            args.batch_size,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    reference_motion_path = getattr(args, 'reference_motion', None)

    inpaint_joints_arg = str(getattr(args, 'inpaint_joints', '') or '').strip()
    inpaint_frames_arg = str(getattr(args, 'inpaint_frames', '') or '').strip()
    inpaint_include_subtree = bool(getattr(args, 'inpaint_include_subtree', True))

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

    # ── Multi-species "all" mode ─────────────────────────────────────────
    if str(explicit_object_type or '').lower() == 'all':
        if reference_motion_path:
            sys.exit(
                "ERROR: --object_type all is incompatible with --reference_motion. "
                "Pass --object_type <Species> for reference-guided generation."
            )
        _generate_all_species(
            cond_dict=cond_dict,
            cond_max_joints=cond_max_joints,
            opt=opt,
            args=args,
            n_frames=n_frames,
            playspeed_cond_value=playspeed_cond_value,
            target_output_frames=target_output_frames,
            global_energy_condition=global_energy_condition,
            model=model,
            diffusion=diffusion,
            sampling_method=sampling_method,
            inference_autocast_dtype=inference_autocast_dtype,
            out_path=out_path,
            fps=fps,
        )
        return out_path

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
    #
    # Raw animation references (.fbx/.glb/.gltf) are handled cond-free: the source
    # skeleton + motion are read straight from the file, so we neither infer the
    # source object_type nor look it up in cond. Only .npy references (already in
    # feature space) still need a source cond entry for their T-pose/skeleton.
    reference_is_raw_anim = bool(reference_motion_path) and (
        os.path.splitext(reference_motion_path)[1].lower()
        in _REFERENCE_MOTION_PREPROCESS_SUFFIXES
    )
    source_type = None
    _default_cond_cache = None
    source_type_used_target_fallback = False
    blind_type = None

    if reference_motion_path and not reference_is_raw_anim:
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
    # Raw-anim references always retarget onto the target skeleton (cond-free path);
    # .npy references retarget only when their source type differs from the target.
    should_retarget_reference = reference_is_raw_anim or _should_retarget_reference(
        source_type,
        target_type,
    )

    object_type = target_type  # downstream code keeps reading `object_type`
    max_joints = len(np.asarray(cond_dict[object_type]['parents']))
    if max_joints > cond_max_joints:
        raise RuntimeError(
            f"target object_type '{object_type}' has {max_joints} joints, "
            f"exceeding cond/model max_joints={cond_max_joints}"
        )
    if max_joints < cond_max_joints:
        print(
            f"[generate] using target joint count {max_joints} for sampling "
            f"instead of cond max_joints={cond_max_joints}"
        )
    if reference_motion_path:
        if reference_is_raw_anim:
            print(
                f"Reference motion: raw animation file (source skeleton extracted "
                f"from file, cond-free; will retarget to {target_type})"
            )
        else:
            if source_type_used_target_fallback:
                print(
                    f"Reference motion object_type inference was invalid"
                    f" ({blind_type or 'no match'}); falling back to target object_type: {target_type}"
                )
            if should_retarget_reference:
                print(f"Reference motion object_type: {source_type} (will retarget to {target_type})")
            else:
                inferred_display = source_type if source_type else target_type
                print(f"Reference motion object_type: {inferred_display}")

    print(f'\n### Sampling object_type: {object_type}')
    print(f'  method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

    # Prepare reference motion (normalize + reshape)
    ref_motion = None
    output_frame_count = n_frames

    # Length-mode flags, finalized inside the reference block below.
    user_inpaint_active = bool(inpaint_joints_arg or inpaint_frames_arg)
    outpaint_active = False
    two_pass_outpaint = False
    single_pass_outpaint = False
    auto_outpaint_range = None

    prepared_reference_path = reference_motion_path
    effective_reference_path = reference_motion_path
    if reference_is_raw_anim:
        # Cond-free source path: read the source skeleton + motion straight
        # from the .fbx/.glb/.gltf and retarget onto the target. No source
        # cond entry, no source object_type inference, no source-side
        # feature-space preprocessing.
        effective_reference_path = _retarget_reference_motion_from_file(
            reference_motion_path,
            target_type=target_type,
            cond_dict=cond_dict,
            opt=opt,
            output_dir=out_path,
            fps=fps,
        )
    elif reference_motion_path:
        source_cond_entry = _resolve_source_cond_entry(
            source_type,
            cond_dict,
            _default_cond_cache,
        )

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

    if effective_reference_path:
        ref_features_full = np.load(effective_reference_path).astype(np.float32)
        if ref_features_full.ndim != 3:
            raise ValueError(
                f"Reference motion must have shape (T, J, F), got {ref_features_full.shape}"
            )
        R = int(ref_features_full.shape[0])

        # Finalize output lengths now that the reference frame count R is
        # known. If --num_frames wasn't specified, use R clamped to
        # [min_length, 2*num_frames].
        if requested_output_frames is None:
            auto_frames = int(np.clip(R, min_length, 2 * internal_num_frames))
            requested_output_frames, target_output_frames, playspeed_cond_value = (
                _finalize_output_lengths(auto_frames, min_length, internal_num_frames)
            )
            if auto_frames == R:
                print(f'  Using reference native length R={R} frames')
            else:
                print(
                    f'  Reference R={R} frames clamped to {auto_frames} '
                    f'(variable-length window [{min_length}, {2 * internal_num_frames}])'
                )
        M = int(requested_output_frames)

        # Crop (R > M) / outpaint (R < M) the reference to exactly M frames so
        # the rest of the pipeline runs at the requested length. The appended
        # [R, M) frames are padded by holding the last reference frame; that
        # placeholder is only a carrier — it is always regenerated.
        if R > M:
            ref_features_full = ref_features_full[:M]
            print(f'  Reference cropped: R={R} > M={M} -> using first {M} frames')
        elif R < M:
            outpaint_active = True
            pad = np.repeat(ref_features_full[-1:], M - R, axis=0)
            ref_features_full = np.concatenate([ref_features_full, pad], axis=0)
            auto_outpaint_range = f'{R}-{M - 1}'
            print(f'  Reference outpaint: R={R} < M={M} -> appended frames [{R}, {M - 1}]')

        # The appended [R, M) tail must be generated from PURE NOISE on the
        # full reverse schedule (a noised copy of the held-last-frame
        # placeholder would bias it toward the frozen final pose). A single
        # reverse pass has one start timestep, so it cannot give the tail a
        # from-noise start while ALSO honoring skip_timesteps / an explicit
        # inpaint selection elsewhere. When the user asks for that
        # combination we split into two passes:
        #   pass 1: outpaint the tail from noise -> a complete M-frame reference
        #   pass 2: run the requested skip / inpaint on that completed reference
        # Otherwise the extension is a single pure-noise outpaint pass.
        two_pass_outpaint = outpaint_active and (
            skip_timesteps > 0 or user_inpaint_active
        )
        single_pass_outpaint = outpaint_active and not two_pass_outpaint

        # Deferred fast-fail: plain reference img2img (no inpaint, no
        # outpaint) requires an explicit --skip_timesteps so the user
        # consciously chooses how faithful to the reference to be.
        if not outpaint_active and not user_inpaint_active and skip_timesteps_raw is None:
            sys.exit(
                "ERROR: --skip_timesteps is required when using --reference_motion "
                "without --inpaint_joints/--inpaint_frames and without a length "
                "extension (R < num_frames).\n"
                "  Higher values (e.g. 80-100) produce motion more faithful to the reference;\n"
                "  lower values (e.g. 20-40) allow more model-driven variation."
            )

        reference_bundle = _prepare_img2img_reference_bundle(
            effective_reference_path,
            object_type,
            cond_dict[object_type],
            max_joints=max_joints,
            target_feature_len=model.feature_len,
            batch_size=args.batch_size,
            requested_output_frame_count=n_frames,
            requested_visible_frame_count=target_output_frames,
            preloaded_features=ref_features_full,
            min_length=min_length,
        )
        ref_motion = reference_bundle['reference_motion']
        output_frame_count = reference_bundle['output_frame_count']
        loaded_reference_frame_count = reference_bundle['loaded_reference_frame_count']
        reference_source_frame_count = reference_bundle.get('reference_source_frame_count', loaded_reference_frame_count)
        loaded_reference_joint_count = reference_bundle['loaded_reference_joint_count']

        print(f'  Reference motion loaded: {effective_reference_path}')
        if reference_is_raw_anim:
            if effective_reference_path != reference_motion_path:
                print(f'    Retargeted from raw animation file: {reference_motion_path}')
        else:
            if prepared_reference_path != reference_motion_path:
                print(f'    Preprocessed from original: {reference_motion_path}')
            if effective_reference_path != prepared_reference_path:
                print(f'    Retargeted from preprocessed: {prepared_reference_path}')
        print(
            f'    Original: [{loaded_reference_frame_count} frames, {loaded_reference_joint_count} joints] '
            f'-> Internal target: [{output_frame_count} frames, {max_joints} joints]'
        )
        if two_pass_outpaint:
            pass2_desc = (
                f'inpaint (skip_timesteps={skip_timesteps})' if user_inpaint_active
                else f'img2img (skip_timesteps={skip_timesteps})'
            )
            print(
                f'    Mode: two-pass outpaint '
                f'(pass 1: fill appended frames [{R}, {M - 1}] from pure noise; '
                f'pass 2: {pass2_desc})'
            )
        elif single_pass_outpaint:
            print('    Mode: outpaint (appended frames from pure noise, full schedule; '
                  'retained frames clamped to reference)')
        elif user_inpaint_active and skip_timesteps > 0:
            print(f'    Mode: inpaint + skip_timesteps={skip_timesteps} '
                  '(masked region starts from an img2img-noised reference; '
                  'unmasked region stays clamped to the original reference)')
        elif user_inpaint_active:
            print('    Mode: inpainting (reference is the clamped known region; '
                  'skip_timesteps=0, denoising full schedule from pure noise)')
        else:
            print(f'    skip_timesteps: {skip_timesteps} (higher = more faithful to reference)')
        # Global energy conditioning: when --global_energy is not
        # explicitly provided alongside --reference_motion, auto-extract
        # it from the reference. When explicitly provided, it overrides
        # with the user-supplied z-score — useful for controlled energy
        # sweeps on reference-guided generation.
        if ref_motion is not None and global_energy_condition is not None:
            print(
                f'    Using explicit --global_energy={getattr(args, "global_energy", None):.4f} '
                f'(z-score, reference-guided generation)'
            )
        elif ref_motion is not None:
            if model_supports_global_energy_conditioning(model):
                _ref_n_joints = torch.full(
                    (args.batch_size,),
                    loaded_reference_joint_count,
                    dtype=torch.long,
                )
                global_energy_condition = _compute_global_energy_from_reference(
                    ref_motion,
                    _ref_n_joints,
                    playspeed_cond=float(reference_source_frame_count) / float(output_frame_count),
                )
                if global_energy_condition is not None:
                    ge_raw = float(global_energy_condition[0, 0])
                    # Normalize using the model's running stats for display
                    _uw_model = unwrap_anytop_model(model)
                    _rm = _uw_model.global_energy_running_mean.to(device='cpu', dtype=torch.float32)
                    _rs = torch.sqrt(
                        _uw_model.global_energy_running_var.to(device='cpu', dtype=torch.float32).clamp_min(1e-6)
                    )
                    ge_norm = (ge_raw - float(_rm[0])) / float(_rs[0])
                    print(
                        f'    Global energy auto-extracted from reference: '
                        f'{ge_norm:.4f} (normalized z-score)'
                    )
            
    if (user_inpaint_active or outpaint_active) and ref_motion is None:
        sys.exit(
            "ERROR: --inpaint_* / length extension is set but the reference "
            "motion could not be loaded; cannot inpaint without a known region."
        )

    # Create condition with effective frame count (shared across passes).
    obj_batch = [object_type] * args.batch_size
    _action_tags_raw = str(getattr(args, 'action_tags', '') or '').strip()
    _action_tags_per_obj = None
    if _action_tags_raw:
        _tag_list = [t.strip() for t in _action_tags_raw.replace(';', ',').split(',') if t.strip()]
        if _tag_list:
            from data_loaders.truebones.truebones_utils.motion_labels import ACTION_TAGS
            _valid_tags = {t.lower() for t in ACTION_TAGS}
            _unknown = [t for t in _tag_list if t.lower() not in _valid_tags]
            if _unknown:
                sys.exit(
                    f"ERROR: unknown action tag(s): {', '.join(sorted(set(_unknown)))}. "
                    f"Valid tags: {', '.join(sorted(_valid_tags))}"
                )
            _action_tags_per_obj = [_tag_list] * args.batch_size
        # Fast-fail if the model was trained without action-tag conditioning.
        if _action_tags_per_obj is not None:
            _uw = unwrap_anytop_model(model)
            if not getattr(_uw, 'action_tag_cond', False):
                sys.exit(
                    'ERROR: --action_tags was passed but this checkpoint was trained '
                    'without --action_tag_cond. Action tags will have no effect.'
                )
    _, model_kwargs = create_condition(
        obj_batch,
        cond_dict,
        output_frame_count,
        args.temporal_window,
        max_joints=max_joints,
        feature_len=opt.feature_len,
        loop=getattr(args, 'loop', False),
        action_tags=_action_tags_per_obj,
    )
    if global_energy_condition is not None:
        model_kwargs['y']['global_energy_cond'] = global_energy_condition.clone()
    model_kwargs['y']['playspeed_cond'] = torch.full(
        (args.batch_size,),
        playspeed_cond_value,
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    def _build_inpaint_mask_for(frames_arg, joints_arg, warn_remap=False):
        internal_frames = _map_frame_ranges_to_internal(
            frames_arg,
            source_frames=target_output_frames,
            target_frames=output_frame_count,
            warn_remap=warn_remap,
        )
        return build_inpaint_mask(
            cond_dict[object_type],
            joints_arg,
            inpaint_include_subtree,
            internal_frames,
            args.batch_size,
            max_joints,
            output_frame_count,
        )

    def _run_sample(reference_motion, skip_ts, inpaint_mask):
        return _sample_batch(
            diffusion=diffusion,
            model=model,
            model_kwargs=model_kwargs,
            sampling_method=sampling_method,
            sample_shape=(args.batch_size, max_joints, model.feature_len, output_frame_count),
            ddim_eta=ddim_eta,
            seed=args.seed,
            device=dist_util.dev(),
            reference_motion=reference_motion,
            skip_timesteps=skip_ts,
            inpaint_mask=inpaint_mask,
            autocast_dtype=inference_autocast_dtype,
        )

    if two_pass_outpaint:
        # Pass 1: outpaint the appended tail (all joints) from pure noise to
        # complete the reference; [0, R) stays clamped to the real reference.
        print('  [two-pass] pass 1/2: outpaint appended frames from pure noise')
        outpaint_mask = _build_inpaint_mask_for(auto_outpaint_range, '')
        completed_reference = _run_sample(ref_motion, 0, outpaint_mask)
        # Pass 2: apply the requested skip / inpaint to the completed reference.
        print('  [two-pass] pass 2/2: applying requested skip/inpaint to the completed reference')
        pass2_mask = (
            _build_inpaint_mask_for(inpaint_frames_arg, inpaint_joints_arg, warn_remap=True)
            if user_inpaint_active else None
        )
        sample = _run_sample(completed_reference, skip_timesteps, pass2_mask)
    elif single_pass_outpaint:
        outpaint_mask = _build_inpaint_mask_for(auto_outpaint_range, '')
        sample = _run_sample(ref_motion, 0, outpaint_mask)
    elif user_inpaint_active:
        user_mask = _build_inpaint_mask_for(inpaint_frames_arg, inpaint_joints_arg, warn_remap=True)
        sample = _run_sample(ref_motion, skip_timesteps, user_mask)
    else:
        # Plain img2img (reference present) or plain generation (ref_motion None).
        sample = _run_sample(ref_motion, skip_timesteps, None)

    # Count existing .npy outputs so repeated runs don't overwrite.
    base_index = sum(
        1 for f in os.listdir(out_path)
        if f.startswith(object_type) and f.endswith('.npy')
    )

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
    # Inpaint Y-anchor: parse user --inpaint_frames into contiguous spans
    # (user-frame indexing, already aligned with the post-trim motion_np
    # frame axis). The correction is applied per joint via vel_y
    # integration with dual-end ramp anchoring.
    inpaint_y_spans = None
    if user_inpaint_active and inpaint_frames_arg:
        inpaint_y_spans = _contiguous_frame_runs(
            _parse_frame_ranges(inpaint_frames_arg, target_output_frames)
        )
    export_tasks = []
    for sample_idx, motion in enumerate(sample):
        n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
        motion = motion[:n_joints]
        parents = model_kwargs['y']['parents'][sample_idx]
        mean = cond_dict[object_type]['mean'][None, :]
        std = cond_dict[object_type]['std'][None, :]
        motion_np = motion.cpu().permute(2, 0, 1).numpy() * std + mean

        if target_output_frames != output_frame_count:
            motion_np = resample_motion_features(
                motion_np,
                target_output_frames,
            )

        # Resolve the known per-species translation root index (the joint that
        # carries the locomotion XZ velocity). This MUST be passed explicitly to
        # BVH export: inferring it from the generated features (allow_infer) is
        # unreliable for skeletons whose translation root is not joint 0 (e.g.
        # Horse Bip01 at index 2). A wrong index integrates the wrong joint's
        # velocity channels — for non-translation-root joints those channels are
        # degenerate (zero-variance, std floored to 1.0), so the model emits
        # ~N(0,1) noise there and the wrong integration produces large root drift.
        translation_root_index = _get_batch_translation_root_index(
            model_kwargs,
            sample_idx,
            fallback=cond_dict[object_type].get('translation_root_index', 0),
        )

        # In-place species (zero XZ locomotion in training) have a floored
        # root-velocity std; the model emits noise there that would integrate
        # into root drift. Suppress it so the export stays in-place.
        if _root_xz_locomotion_is_degenerate(
            cond_dict[object_type]['std'], translation_root_index
        ):
            _suppress_degenerate_root_xz_velocity(motion_np, translation_root_index)

        if inpaint_y_spans:
            _reanchor_inpaint_root_y_via_velocity(motion_np, inpaint_y_spans)
        if getattr(args, 'loop', False):
            _close_loop_root_xz_via_velocity(motion_np, translation_root_index)

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
            translation_root_index,
        ))

    for task in tqdm(export_tasks, desc=f'{object_type} export'):
        npy_name = _export_motion(task)
        print(f'    Created motion: {npy_name}')

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


def _map_frame_ranges_to_internal(spec, source_frames, target_frames, warn_remap=False):
    if not spec or int(source_frames) == int(target_frames):
        return spec
    source_frames = int(source_frames)
    target_frames = int(target_frames)
    if source_frames <= 0 or target_frames <= 0:
        raise ValueError(
            f"Cannot map frame ranges with source_frames={source_frames}, target_frames={target_frames}"
        )
    scale = float(target_frames - 1) / float(source_frames - 1) if source_frames > 1 else 0.0
    mapped_frames = set()
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
        hi = min(source_frames - 1, hi)
        if lo > hi:
            continue
        start = max(0, min(target_frames - 1, int(np.floor(float(lo) * scale))))
        end = max(0, min(target_frames - 1, int(np.ceil(float(hi) * scale))))
        mapped_frames.update(range(start, end + 1))
    if not mapped_frames:
        raise ValueError(
            f"--inpaint_frames '{spec}' selected no valid frames "
            f"(motion has {source_frames} frames, indices 0..{source_frames - 1})"
        )
    internal_runs = _contiguous_frame_runs(mapped_frames)
    internal_spec = ','.join(
        f'{start}-{end}' if start != end else str(start)
        for start, end in internal_runs
    )

    # The frame range was given in visible/output-frame space but the model
    # samples at a different internal length, so the mask had to be rescaled.
    # floor/ceil widening + the resolution drop (visible -> internal -> visible)
    # mean the regenerated region will NOT line up with the requested integer
    # frames; warn with the effective visible boundaries so the drift is not
    # silent. To inpaint exact frames, set --num_frames to the model's
    # num_frames so visible == internal and no remapping happens.
    if warn_remap:
        inv_scale = float(source_frames - 1) / float(target_frames - 1) if target_frames > 1 else 0.0
        effective_runs = [
            (
                int(np.floor(float(start) * inv_scale)),
                int(np.ceil(float(end) * inv_scale)),
            )
            for start, end in internal_runs
        ]
        effective_spec = ','.join(
            f'{a}-{b}' if a != b else str(a) for a, b in effective_runs
        )
        print(
            f'\033[33m  [WARN] --inpaint_frames remapped: requested visible frames '
            f"'{spec}' (output length {source_frames}) -> internal frames "
            f"'{internal_spec}' (sampler length {target_frames}). "
            f'Effective regenerated visible region is ~{effective_spec}, not the '
            f'exact frames requested (boundaries drift ~1-2 frames from floor/ceil '
            f'widening and the {source_frames}->{target_frames} resolution change).\033[0m'
        )
    return internal_spec


def _contiguous_frame_runs(frame_set):
    """Convert a set of frame indices into a list of [start, end] inclusive
    runs of consecutive frames, sorted ascending.
    """
    if not frame_set:
        return []
    sorted_frames = sorted(frame_set)
    runs = []
    start = prev = sorted_frames[0]
    for f in sorted_frames[1:]:
        if f == prev + 1:
            prev = f
            continue
        runs.append((start, prev))
        start = prev = f
    runs.append((start, prev))
    return runs


def _reanchor_inpaint_root_y_via_velocity(motion_np, spans):
    """Inpaint-side fix for Y misalignment: per contiguous inpaint span
    [a, b] (frame indices into ``motion_np``) and per joint, replace the
    absolute Y channel (pos[..., 1]) with a vel-Y integral anchored at the
    last clamped frame ``a-1`` and linearly ramped to match the clamped Y
    at ``b+1``.

    Background: features are ``[pos(3) || rot(6) || vel(3) || foot(1)]`` (13ch).
    For EVERY joint, pos[1] is the absolute world Y — ``get_rifke`` only
    subtracts root XZ before encoding, leaving Y untouched, and the recover
    path likewise reads pos[1] verbatim without adding root Y back. So in an
    inpaint span the model regresses each joint's pos[1] toward its dataset
    mean (visible as the whole skeleton sinking), while vel[1] (ch 10) — the
    per-joint world Y velocity — stays near zero. Integrating vel_y from the
    left-clamped Y plus a linear ramp to the right-clamped Y closes both
    seams in the integral sense and preserves per-frame Y micro-structure.

    Applied independently to every joint via a vectorized batched cumsum.
    For joints whose columns were clamped to the reference (e.g. when
    ``--inpaint_joints`` selects a subset), the inputs are already
    consistent so the correction is mathematically a no-op.

    No-op when an inpaint span has no left or no right clamped neighbour
    (touches frame 0 or frame F-1).

    Args:
        motion_np: (F, J, C) feature tensor. Modified in place via a basic
            slice on the channel axis (preserves view semantics).
        spans: list of ``(a, b)`` inclusive frame ranges marking inpaint
            regions; both ``a-1`` and ``b+1`` must lie inside [0, F-1].
    """
    if not spans:
        return
    F, J, C = motion_np.shape
    if C < 11 or J == 0:
        return
    # Basic slicing on the channel axis returns a view, so assignments
    # through ``pos_y`` write straight back into ``motion_np``.
    pos_y = motion_np[:, :, 1]   # (F, J) view
    vel_y = motion_np[:, :, 10]  # (F, J) view — vel[f] = pos[f+1] - pos[f]
    for a, b in spans:
        if a < 1 or b > F - 2 or a > b:
            # Need both a-1 and b+1 as clamped anchors; otherwise leave alone.
            continue
        L = b - a + 1
        # Forward integrate from clamped pos_y[a-1] across the span:
        #   y_int[k] = pos_y[a-1] + sum_{i=a-1..k-1} vel_y[i]  for k in [a, b]
        integrated = pos_y[a - 1:a] + np.cumsum(
            vel_y[a - 1:b], axis=0, dtype=np.float64,
        )  # (L, J)
        # Bridge to the right anchor: if we also stepped one more by vel_y[b]
        # we should land at pos_y[b+1]. Distribute the residual linearly so
        # adjusted_y[a-1] is unchanged and adjusted_y[b+1] would land on target.
        integrated_at_b_plus_1 = integrated[-1] + vel_y[b]
        adjust = pos_y[b + 1] - integrated_at_b_plus_1  # (J,)
        # Ramp factor for k in [a, b]: (k - (a-1)) / ((b+1) - (a-1)) = (k - a + 1) / (L + 1)
        ramp = (np.arange(1, L + 1, dtype=np.float64) / float(L + 1))[:, None]  # (L, 1)
        pos_y[a:b + 1] = (integrated + adjust[None, :] * ramp).astype(
            pos_y.dtype, copy=False,
        )


def _root_xz_locomotion_is_degenerate(std, translation_root_index):
    """True when the species has no XZ locomotion to generate.

    ``get_mean_std`` floors any sub-1e-5 (zero-variance) std block to 1.0. A
    species whose translation root never translates in XZ during training (e.g.
    Tukan — all clips are in-place fly/idle, raw root XZ velocity is exactly 0)
    therefore ends up with the root XZ velocity std floored to 1.0 instead of a
    genuine ~0.01 scale value. At generation the model has no signal to learn on
    those channels and emits ~N(0,1) noise; denormalizing with std=1.0 turns that
    noise into a full unit-per-frame velocity, which integrates into large root
    drift. Real locomotion stds in this scaled feature space are ~0.001–0.015, so
    a value at/near the 1.0 floor is an unambiguous "no locomotion" marker.
    """
    std = np.asarray(std)
    root_index = int(translation_root_index)
    if std.ndim != 2 or std.shape[1] < 12 or root_index < 0 or root_index >= std.shape[0]:
        return False
    return bool(np.all(std[root_index, [9, 11]] >= 0.5))


def _suppress_degenerate_root_xz_velocity(motion_np, translation_root_index):
    """Zero the generated root XZ velocity for in-place species (see
    :func:`_root_xz_locomotion_is_degenerate`). Operates in-place on ``motion_np``
    so both the saved .npy and the exported BVH stay consistent (in-place, no
    drift). The locomotion-root RIC X/Z channels (0, 2) are zero by construction
    under RIFKE, so they are pinned too for cleanliness."""
    if motion_np.ndim != 3:
        return
    _, joint_count, feature_count = motion_np.shape
    root_index = int(translation_root_index)
    if feature_count < 12 or root_index < 0 or root_index >= joint_count:
        return
    motion_np[:, root_index, 0] = 0.0
    motion_np[:, root_index, 2] = 0.0
    motion_np[:, root_index, 9] = 0.0
    motion_np[:, root_index, 11] = 0.0


def _close_loop_root_xz_via_velocity(motion_np, translation_root_index):
    """Close loop root XZ drift by distributing velocity residual.

    The feature recover path reconstructs root XZ by integrating channels 9
    and 11 over frames ``0..F-2``. The translation-root RIC X/Z channels 0 and
    2 should be zero by construction because RIFKE subtracts that root's XZ
    before encoding. For generated loop clips, tiny non-zero residuals in both
    places become a visible first/last root-position seam when the BVH loops.
    Subtracting the mean velocity residual from each transition preserves the
    local root motion shape while making the integrated endpoint match the
    start; zeroing the root RIC X/Z removes representation noise only.
    """
    if motion_np.ndim != 3:
        return
    frame_count, joint_count, feature_count = motion_np.shape
    root_index = int(translation_root_index)
    if frame_count < 2 or feature_count < 12 or root_index < 0 or root_index >= joint_count:
        return

    motion_np[:, root_index, 0] = 0.0
    motion_np[:, root_index, 2] = 0.0

    transition_count = frame_count - 1
    drift_x = np.sum(motion_np[:-1, root_index, 9], dtype=np.float64)
    drift_z = np.sum(motion_np[:-1, root_index, 11], dtype=np.float64)
    if abs(drift_x) <= 1e-8 and abs(drift_z) <= 1e-8:
        motion_np[-1, root_index, 9] = 0.0
        motion_np[-1, root_index, 11] = 0.0
        return

    motion_np[:-1, root_index, 9] -= np.asarray(drift_x / transition_count, dtype=motion_np.dtype)
    motion_np[:-1, root_index, 11] -= np.asarray(drift_z / transition_count, dtype=motion_np.dtype)
    motion_np[-1, root_index, 9] = 0.0
    motion_np[-1, root_index, 11] = 0.0


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


def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len, loop=False, action_tags=None):
    """Build model_kwargs for a batch of object_types.

    Parameters
    ----------
    action_tags : list of list[str] or None
        Per-object action tag lists (e.g. ``[['locomotion', 'attack'], ...]``).
        When provided, must have the same length as *object_types*. Each element
        may be a list of tag strings, a single string, or ``None``.
    """
    batches = list()
    circular_mask = bool(loop)
    for i, object_type in enumerate(object_types):
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
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames, circular=circular_mask))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batch.append(max_joints)
        metadata = {
            'is_loop': bool(loop),
            'loop_full_cycle': bool(loop),
            'translation_root_index': cond_dict[object_type].get('translation_root_index', 0),
        }
        if 'species_emb' in cond_dict[object_type]:
            metadata['species_emb'] = cond_dict[object_type]['species_emb']
        if action_tags is not None and i < len(action_tags):
            tags = action_tags[i]
            if tags is not None:
                metadata['action_tags'] = tags
        batch.append(metadata)
        batch.append(object_type)
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
