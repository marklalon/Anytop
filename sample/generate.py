# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import json
import os
import sys
from dataclasses import dataclass

# Ensure both the Anytop dir (for bare ``utils.*`` / ``data_loaders.*`` imports)
# and its parent (for ``utils.*`` imports made by submodules like
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
from data_loaders.truebones.truebones_utils.canonical_features import (
    build_canonical_rest_feature,
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
    physical_hml_to_canonical,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    build_species_file_tokens,
    resolve_species_key,
    species_lookup_map,
)
from data_loaders.truebones.truebones_utils.get_opt import DEFAULT_COND_PATH, get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    tpose_features_from_cond,
    recover_bvh_export_animation_from_motion_np,
)
from motion_lib import BVH
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
    # Preloaded T5 conditioner reused by --species_tags to avoid a second T5 load.
    t5_conditioner: object = None

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


def _checkpoint_cond_path(model_path):
    """The cond.npy sitting next to a checkpoint, if training left one there.

    A training run copies its cond into save_dir, so a checkpoint carries its own
    complete inference contract -- species, skeletons, and baked species tags --
    and generation never has to reach for a dataset directory.
    """
    if not model_path:
        return ''
    candidate = os.path.join(os.path.dirname(os.path.abspath(model_path)), 'cond.npy')
    return candidate if os.path.isfile(candidate) else ''


def _resolve_generation_cond_path(args):
    """--cond_path, else the checkpoint's own snapshot, else the default dataset."""
    explicit = getattr(args, 'cond_path', '') or ''
    if explicit:
        return explicit
    checkpoint_cond = _checkpoint_cond_path(getattr(args, 'model_path', ''))
    if checkpoint_cond:
        return checkpoint_cond
    return DEFAULT_COND_PATH


def _load_generation_cond(args, opt, cond_dict=None):
    if cond_dict is None:
        return load_cond(opt.cond_file), opt.cond_file
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
    # cond.npy is the whole inference contract, so it is resolved before opt:
    # get_opt derives the dataset sources from it and configures dataset_tags,
    # falling back to the cond's baked species tags when no dataset dir exists.
    opt = get_opt(args.device, _resolve_generation_cond_path(args), cond_dict=cond_dict)
    cond_dict, actual_cond_file = _load_generation_cond(args, opt, cond_dict)
    _raise_opt_max_joints_for_cond(opt, cond_dict)

    print('Creating model and diffusion...')
    # Use in-memory cond_dict to avoid a second np.load().
    resolve_t5_out_dim(args, cond_source=cond_dict)
    sampling_method, sampling_steps = _configure_sampling_args(args)
    model, diffusion = create_model_and_diffusion_general_skeleton(args)

    print(f'Loading checkpoints from [{args.model_path}]...')
    # Load checkpoint to CUDA if available, else CPU.
    device = dist_util.dev()
    if device is None or device.type != 'cuda':
        device = torch.device('cpu')
    state_dict = torch.load(args.model_path, map_location=device)
    if 'model_avg' in state_dict:
        print('EMA checkpoint detected, loading model_avg weights.')
        state_dict = state_dict['model_avg']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    assert model is not None, 'BUG: create_model_and_diffusion_general_skeleton returned None for model'
    # model.to(device) may return None (CUDA 12.8 + torch 2.7.1); parameter move is in-place.
    model.to(device)
    load_model(model, state_dict)

    print('Validating precomputed joint-name embeddings from cond.npy...')
    ensure_joint_name_embeddings(
        cond_dict,
        expected_embedding_dim=args.t5_out_dim,
        cond_source=actual_cond_file,
    )
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
    # CLI value is a z-score; de-normalize to raw space for _build_global_energy_token.
    raw = running_mean.clone()
    raw[0] = float(global_energy) * running_std[0] + running_mean[0]
    if not torch.isfinite(raw).all():
        raise ValueError("--global_energy must be finite")
    return raw.unsqueeze(0).expand(batch_size, -1).clone()


def _compute_global_energy_from_reference(ref_motion, n_joints, playspeed_cond=None):
    """Extract raw global energy [mean, std] from reference motion (B, J, F, T) tensor."""
    from model.anytop import GlobalEnergyExtractor

    return GlobalEnergyExtractor.compute_global_energy_condition(
        ref_motion,
        n_joints,
        playspeed_cond=playspeed_cond,
    )



def _lookup_object_type_case_insensitive(object_types, requested_type):
    """Resolve user/filename species text to a canonical cond key.

    Kept under its old name because several call sites pass a bare ``.keys()``
    view; the resolution itself now goes through the shared rule (exact key,
    unique namespace suffix, then bare name taking the first dataset), so
    ``--object_type Horse`` and ``--object_type zoo_upgrade/Horse`` both work.
    """
    if requested_type is None:
        return None
    keys = object_types if isinstance(object_types, dict) else {key: None for key in object_types}
    try:
        return resolve_species_key(keys, requested_type)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")


def _load_default_cond_cache(default_cond_file, actual_cond_file):
    """Load the checkpoint's own cond snapshot as a secondary source-species pool.

    Used only to name the *source* skeleton of a reference clip when the user
    passed a narrow ``--cond_path`` that does not contain it. It is the
    checkpoint's cond, not a hard-coded dataset directory: the species a
    checkpoint knows are exactly the ones it was trained on.
    """
    if not default_cond_file or not os.path.isfile(default_cond_file):
        return None

    default_real = os.path.realpath(default_cond_file)
    actual_real = os.path.realpath(actual_cond_file)
    try:
        if os.path.samefile(default_real, actual_real):
            return None
    except FileNotFoundError:
        if default_real == actual_real:
            return None

    return load_cond(default_cond_file)


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

    Thin wrapper around ``utils.auto_retarget.retarget_features_npy_to_target``.
    Loads source features, builds target TPoseFeatures, delegates the math, then
    writes the retargeted .npy and an inspection .bvh under ``output_dir``.
    """
    from utils.auto_retarget import (
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

    # Inspection BVH.
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
    """Retarget raw .fbx/.glb/.gltf onto target_type (cond-free source).
    Only the target's cond/T-pose is required."""
    from utils.auto_retarget import (
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
    source_label = source_object_type or infer_object_type_from_filename(
        reference_motion_path,
        valid_types=None,
    ) or base

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

    # Inspection BVH.
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
    physical_energy_features=None,
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
    max_source_frames = max(int(min_length), output_frame_count * 2)
    if loaded_reference_frame_count > max_source_frames:
        visible_frames = output_frame_count if requested_visible_frame_count is None else int(requested_visible_frame_count)
        source_frames = min(max_source_frames, max(int(min_length), visible_frames))
        ref_raw = ref_raw[:source_frames]
    reference_source_frame_count = int(ref_raw.shape[0])
    # Snapshot the physical (pre-canonical, pre-resample) reference so
    # global-energy extraction runs in the same feature space as training
    # running stats (physical HML, not canonical-standardized).
    if physical_energy_features is None:
        raise ValueError(
            "physical_energy_features is required for global-energy extraction; "
            "the caller must provide the real (unpadded) physical frames so the "
            "energy statistic is computed in the same space as training running stats."
        )
    reference_physical_motion = np.array(
        physical_energy_features,
        dtype=np.float32,
        copy=True,
    )
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
        'output_frame_count': output_frame_count,
        'loaded_reference_frame_count': loaded_reference_frame_count,
        'reference_source_frame_count': reference_source_frame_count,
        'loaded_reference_joint_count': loaded_reference_joint_count,
        'reference_physical_motion': reference_physical_motion,
    }


def _export_motion(task):
    (motion_np, parents_np, offsets, npy_name, joint_names, out_path, fps,
     tpose_rest_rotations, translation_root_index, rigid_bone) = task
    out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_np(
        motion_np,
        parents_np,
        offsets,
        joint_names,
        translation_root_index=translation_root_index,
        allow_infer=translation_root_index is None,
        tpose_rest_rotations=tpose_rest_rotations,
        rigid_bone=rigid_bone,
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
        # Top-level autocast around the reverse-diffusion loop (model invoked
        # many times inside the sampler, so this is the natural call site).
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
        # Start from noised reference but clamp unmasked region to original reference at every step.
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
        # skip_timesteps=0: start from pure noise; reference is only the per-step clamp source for unmasked region.
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
        # img2img: noise the whole reference to an intermediate step (higher = more faithful).
        return _run_loop(
            noise=torch.randn(sample_shape, device=device),
            init_image=reference_motion.to(device, non_blocking=True),
            skip_ts=skip_timesteps,
            inpaint_mask_=None,
            inpaint_reference_=None,
            cross_limb_unreliable_mask_=None,
        )
    # Plain generation: full denoising from pure noise.
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
    # Canonical keys carry '/', so output filenames use the file token instead.
    species_file_tokens = build_species_file_tokens(cond_dict)
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
                mark_canonical_cond_entry(sp_entry)
                n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
                motion = motion[:n_joints]
                parents = model_kwargs['y']['parents'][sample_idx]
                # Decode with full cond entry (carries rest geometry + global standardization stats).
                motion_physical = canonical_to_physical_hml(motion.unsqueeze(0), sp_entry)[0]
                motion_np = motion_physical.cpu().permute(2, 0, 1).numpy()

                if target_output_frames != output_frame_count:
                    motion_np = resample_motion_features(motion_np, target_output_frames)

                translation_root_index = _get_batch_translation_root_index(
                    model_kwargs, sample_idx,
                    fallback=sp_entry.get('translation_root_index', 0),
                )
                if getattr(args, 'loop', False):
                    _close_loop_root_xz_via_velocity(motion_np, translation_root_index)

                joint_names = sp_entry.get(
                    'canonical_bvh_joint_names', sp_entry['joints_names'],
                )

                # T-pose rest rotations (per-species)
                _tpose_rr = sp_entry.get('tpose_rest_rotations')
                if _tpose_rr is not None:
                    _tpose_rr = np.asarray(_tpose_rr, dtype=np.float32)

                # Count existing outputs so repeated runs don't overwrite.
                sp_token = species_file_tokens[sp]
                existing = [f for f in os.listdir(out_path)
                            if f.startswith(sp_token) and f.endswith('.npy')]
                npy_name = f'{sp_token}_{(len(existing))}.npy'
                export_tasks.append((
                    motion_np, parents, sp_entry['offsets'], npy_name, joint_names,
                    out_path, fps, _tpose_rr, translation_root_index,
                    bool(getattr(args, 'rigidbone', False)),
                ))

            for task in tqdm(export_tasks, desc=f'batch {batch_idx} export'):
                npy_name = _export_motion(task)
                print(f'    Created: {npy_name}')


def main(args=None, cond_dict=None, runtime=None):
    if args is None:
        args = generate_args()

    fixseed(args.seed)

    skip_timesteps_raw = getattr(args, 'skip_timesteps', None)

    # Check inpaint flags early (before ~30s model load).
    _inpaint_early = bool(
        str(getattr(args, 'inpaint_joints', '') or '').strip()
        or str(getattr(args, 'inpaint_frames', '') or '').strip()
    )

    # --skip_timesteps check deferred until after reference length R is known
    # (R < M auto-enables outpaint, which does not need --skip_timesteps).

    # Fail fast if inpaint flags are set without reference motion.
    if _inpaint_early and not getattr(args, 'reference_motion', None):
        sys.exit(
            "ERROR: --inpaint_joints / --inpaint_frames require --reference_motion "
            "(the reference is the known region held fixed while the masked region "
            "is regenerated). Pass --reference_motion <path>, or drop the inpaint "
            "flags for plain generation."
        )

    if _inpaint_early and skip_timesteps_raw is None:
        skip_timesteps_raw = 0  # inpaint without skip: denoise full schedule

    skip_timesteps = int(skip_timesteps_raw) if skip_timesteps_raw is not None else 0

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

    # Model native window from checkpoint args.json (args.num_frames = user intent, may be None).
    _ckpt_args_path = os.path.join(os.path.dirname(args.model_path), 'args.json')
    _ckpt_num_frames = 60
    if os.path.isfile(_ckpt_args_path):
        with open(_ckpt_args_path, 'r') as _f:
            _ckpt_args = json.load(_f)
        _ckpt_num_frames = int(_ckpt_args.get('num_frames', 60))

    internal_num_frames = _ckpt_num_frames
    min_length = int(getattr(args, 'min_length', 20))
    n_frames = internal_num_frames
    cond_max_joints = opt.max_joints

    reference_present = bool(getattr(args, 'reference_motion', None))
    motion_frames = getattr(args, 'num_frames', None)
    if motion_frames is None and not reference_present:
        motion_frames = _ckpt_num_frames  # default to native window

    # Output lengths: known now if --num_frames given; otherwise deferred until
    # reference frame count R is known (defaults to R clamped to [min_length, 2*num_frames]).
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
    reference_motion_suffix = (
        _validate_reference_motion_path(reference_motion_path)
        if reference_motion_path else ''
    )

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
        if str(getattr(args, 'species_tags', '') or '').strip():
            sys.exit(
                "ERROR: --species_tags is incompatible with --object_type all "
                "(a single tag set cannot restyle every species). Pass "
                "--object_type <Species> to restyle one species."
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
        # Token map, not the raw keys: canonical keys contain '/', which cannot
        # appear in a filename. A unique bare name still matches its plain form.
        target_type = infer_object_type_from_filename(
            reference_motion_path, valid_types=species_lookup_map(cond_dict)
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
    # Raw animation references (.fbx/.glb/.gltf) are cond-free (source from file);
    # .npy references need a source cond entry for their T-pose/skeleton.
    reference_is_raw_anim = bool(reference_motion_path) and (
        reference_motion_suffix in _REFERENCE_MOTION_PREPROCESS_SUFFIXES
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
            # The checkpoint's own snapshot, not a hard-coded dataset directory.
            default_cond_file=_checkpoint_cond_path(getattr(args, 'model_path', '')),
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
    # Raw-anim references always retarget (cond-free); .npy only when source != target.
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

    print(f'\nSampling object_type: {object_type}  method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

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

        # Finalize output lengths from R (if --num_frames not specified).
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

        # Unpadded physical frames for global-energy extraction (before outpaint padding).
        physical_energy_features = np.array(
            ref_features_full[:M] if R >= M else ref_features_full,
            dtype=np.float32,
            copy=True,
        )

        # Crop (R > M) or outpaint-pad (R < M) reference to exactly M frames.
        if R > M:
            ref_features_full = ref_features_full[:M]
            print(f'  Reference cropped: R={R} > M={M} -> using first {M} frames')
        elif R < M:
            outpaint_active = True
            pad = np.repeat(ref_features_full[-1:], M - R, axis=0)
            ref_features_full = np.concatenate([ref_features_full, pad], axis=0)
            auto_outpaint_range = f'{R}-{M - 1}'
            print(f'  Reference outpaint: R={R} < M={M} -> appended frames [{R}, {M - 1}]')

        # Appended [R, M) frames need a pure-noise start (full schedule), which
        # conflicts with skip_timesteps/explicit inpaint. When both are present,
        # split into two passes: pass 1 outpaints the tail from noise, pass 2
        # applies the requested skip/inpaint on the completed reference.
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
            physical_energy_features=physical_energy_features,
            min_length=min_length,
        )
        ref_motion = reference_bundle['reference_motion']
        output_frame_count = reference_bundle['output_frame_count']
        loaded_reference_frame_count = reference_bundle['loaded_reference_frame_count']
        loaded_reference_joint_count = reference_bundle['loaded_reference_joint_count']
        reference_physical_motion = reference_bundle.get('reference_physical_motion')

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
        # Auto-extract global energy from reference when --global_energy not explicitly provided.
        if ref_motion is not None and global_energy_condition is not None:
            print(
                f'    Using explicit --global_energy={getattr(args, "global_energy", None):.4f} '
                f'(z-score, reference-guided generation)'
            )
        elif ref_motion is not None:
            if model_supports_global_energy_conditioning(model) and reference_physical_motion is not None:
                # Extract energy from physical reference (same space as training running stats).
                _ref_phys = torch.from_numpy(
                    np.ascontiguousarray(reference_physical_motion, dtype=np.float32)
                ).permute(1, 2, 0).unsqueeze(0).expand(args.batch_size, -1, -1, -1)
                _ref_n_joints = torch.full(
                    (args.batch_size,),
                    int(reference_physical_motion.shape[1]),
                    dtype=torch.long,
                )
                global_energy_condition = _compute_global_energy_from_reference(
                    _ref_phys,
                    _ref_n_joints,
                    playspeed_cond=None,
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

    # ── --species_tags: restyle the target species' motion descriptor ────────
    _species_emb_override = None
    _species_tags = _parse_species_tags(getattr(args, 'species_tags', ''))
    if _species_tags:
        # Fast-fail if checkpoint ignores species descriptor.
        _uw = unwrap_anytop_model(model)
        if not (getattr(_uw, 'species_cond', False) or getattr(_uw, 'species_joint_cond', False)):
            sys.exit(
                'ERROR: --species_tags was passed but this checkpoint was trained without '
                '--species_cond or --species_joint_cond; the species descriptor is unused, so '
                'the tags would have no effect.'
            )
        _target_cond_entry = cond_dict[object_type]
        if 'species_emb' not in _target_cond_entry:
            sys.exit(
                f"ERROR: cond entry for '{object_type}' has no baked 'species_emb'; regenerate "
                "cond.npy with species embeddings before using --species_tags."
            )
        _override_t5_name = _resolve_species_t5_name(_target_cond_entry)
        _override_dim = int(np.asarray(_target_cond_entry['species_emb']).shape[-1])
        # Reuse baked species_emb if tags match an existing cond entry (same T5 + dim).
        _cached = _find_cached_species_emb(
            _species_tags, cond_dict, _override_t5_name, _override_dim,
        )
        if _cached is None:
            # The active --cond_path may be a small custom cond lacking the
            # species whose baked tags match; fall back to the default cond DB.
            _default_cond_cache = _load_default_cond_cache(
                getattr(opt, 'cond_file', None), actual_cond_file,
            )
            if _default_cond_cache:
                _cached = _find_cached_species_emb(
                    _species_tags, _default_cond_cache, _override_t5_name, _override_dim,
                )
        if _cached is not None:
            _species_emb_override, _cache_src = _cached
            print(
                f"[generate] species tags {_species_tags} match baked descriptor of "
                f"'{_cache_src}'; reusing cached species_emb (skipped T5)."
            )
        else:
            _species_emb_override = _encode_species_tags_override(
                _species_tags,
                _target_cond_entry,
                _override_dim,
                t5_conditioner=getattr(runtime, 't5_conditioner', None),
            )
        _default_text = ' '.join(
            (_target_cond_entry.get('species_emb_meta') or {}).get('embedding_text', '').split()
        )
        print(
            f"[generate] species descriptor for '{object_type}' overridden: "
            f"'{_default_text}' -> '{' '.join(_species_tags)}'"
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
        species_emb_override=_species_emb_override,
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

    # Joint-inpaint vertical reseat: capture the reference actually
    # used to clamp the known joints so the regenerated subtree can be dropped
    # back onto its grounded vertical frame during export. Pure joint inpaint
    # only (no --inpaint_frames, which already reanchors Y temporally).
    reseat_reference = None
    reseat_free_joints = None
    if user_inpaint_active and inpaint_joints_arg and not inpaint_frames_arg:
        reseat_reference = completed_reference if two_pass_outpaint else ref_motion
        reseat_free_joints, _ = _resolve_inpaint_joint_indices(
            cond_dict[object_type], inpaint_joints_arg, inpaint_include_subtree
        )

    # Output filenames use the species FILE TOKEN, not the canonical cond key:
    # the key contains '/'. A species whose bare name is unique across the cond
    # keeps that plain name, so single-dataset runs produce today's filenames.
    object_file_token = build_species_file_tokens(cond_dict)[object_type]

    # Count existing .npy outputs so repeated runs don't overwrite.
    base_index = sum(
        1 for f in os.listdir(out_path)
        if f.startswith(object_file_token) and f.endswith('.npy')
    )

    _tpose_rest_rotations = cond_dict[object_type].get('tpose_rest_rotations')
    if _tpose_rest_rotations is not None:
        _tpose_rest_rotations = np.asarray(_tpose_rest_rotations, dtype=np.float32)

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
    mark_canonical_cond_entry(cond_dict[object_type])
    for sample_idx, motion in enumerate(sample):
        n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
        motion = motion[:n_joints]
        parents = model_kwargs['y']['parents'][sample_idx]
        # Decode with the full per-species cond entry (rest geometry + global
        # standardization stats), not a minimal dict that would drop the stats.
        motion_physical = canonical_to_physical_hml(motion.unsqueeze(0), cond_dict[object_type])[0]
        motion_np = motion_physical.cpu().permute(2, 0, 1).numpy()

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

        if inpaint_y_spans:
            _reanchor_inpaint_root_y_via_velocity(motion_np, inpaint_y_spans)
        elif reseat_reference is not None:
            ref_phys = canonical_to_physical_hml(
                reseat_reference[sample_idx][:n_joints].to(motion.device).unsqueeze(0),
                cond_dict[object_type],
            )[0]
            ref_motion_np = ref_phys.cpu().permute(2, 0, 1).numpy()
            if target_output_frames != output_frame_count:
                ref_motion_np = resample_motion_features(ref_motion_np, target_output_frames)
            reseat_delta = _reground_inpaint_joint_y(
                motion_np, ref_motion_np, reseat_free_joints, parents,
            )
            if reseat_delta:
                print(
                    f'    Inpaint reseat: shifted regenerated subtree world-Y by '
                    f'{reseat_delta:+.4f} to re-ground onto the reference'
                )
        if getattr(args, 'loop', False):
            _close_loop_root_xz_via_velocity(motion_np, translation_root_index)

        offsets = cond_dict[object_type]['offsets']

        npy_name = f'{object_file_token}_{base_index + sample_idx}.npy'
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
            bool(getattr(args, 'rigidbone', False)),
        ))

    for task in tqdm(export_tasks, desc=f'{object_file_token} export'):
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

    # Frame range remapped from visible to internal length; floor/ceil widening
    # causes ~1-2 frame drift. To inpaint exact frames, set --num_frames to match the model's
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
    """Fix inpaint Y misalignment per joint via vel-Y integral + linear ramp.

    For each inpaint span [a, b], replaces pos[..., 1] with cumulative vel-Y
    from a-1 anchored to pos_y[a-1], then ramps to match pos_y[b+1]. This
    closes the Y seam while preserving per-frame articulation.

    No-op when a span touches frame 0 or F-1 (no neighbour on one side).
    """
    if not spans:
        return
    F, J, C = motion_np.shape
    if C < 11 or J == 0:
        return
    # Slice views -> writes propagate to motion_np.
    pos_y = motion_np[:, :, 1]   # (F, J) world Y
    vel_y = motion_np[:, :, 10]  # (F, J) vel[f] = pos[f+1] - pos[f]
    for a, b in spans:
        if a < 1 or b > F - 2 or a > b:
            continue  # no clamped anchor on both sides
        L = b - a + 1
        integrated = pos_y[a - 1:a] + np.cumsum(
            vel_y[a - 1:b], axis=0, dtype=np.float64,
        )  # (L, J)
        integrated_at_b_plus_1 = integrated[-1] + vel_y[b]
        adjust = pos_y[b + 1] - integrated_at_b_plus_1  # (J,) residual to close
        ramp = (np.arange(1, L + 1, dtype=np.float64) / float(L + 1))[:, None]  # (L, 1)
        pos_y[a:b + 1] = (integrated + adjust[None, :] * ramp).astype(
            pos_y.dtype, copy=False,
        )


def _reground_inpaint_joint_y(motion_np, ref_motion_np, free_joint_indices, parents):
    """Re-ground regenerated subtree Y onto the reference.

    ``--inpaint_joints`` frees a subset of joints while the rest stay clamped
    to the reference. The free joints live in the model's own vertical frame
    which can float above the grounded body. This computes a constant offset
    at the subtree *boundary* (free joints whose parent is clamped) and shifts
    the whole free subtree to meet the reference's grounded height.

        boundary = { j in free : parent(j) not in free }
        delta    = mean_{j in boundary, t}( ref_y - gen_y )
        gen_y[:, free] += delta
    """
    if ref_motion_np is None or motion_np.ndim != 3:
        return 0.0
    F, J, C = motion_np.shape
    if C < 2 or F == 0 or ref_motion_np.shape[:2] != (F, J):
        return 0.0
    free = [int(j) for j in free_joint_indices if 0 <= int(j) < J]
    if not free:
        return 0.0
    free_set = set(free)
    parents = np.asarray(parents).reshape(-1)
    # Boundary joints: free joints whose parent is clamped — anchor the reseat here.
    boundary = [
        j for j in free
        if 0 <= int(parents[j]) < J and int(parents[j]) not in free_set
    ]
    if not boundary:
        boundary = free
    b_idx = np.asarray(boundary, dtype=np.int64)
    delta = float(np.mean(
        ref_motion_np[:, b_idx, 1].astype(np.float64)
        - motion_np[:, b_idx, 1].astype(np.float64)
    ))
    if delta == 0.0 or not np.isfinite(delta):
        return 0.0
    f_idx = np.asarray(free, dtype=np.int64)
    motion_np[:, f_idx, 1] += np.asarray(delta, dtype=motion_np.dtype)
    return delta


def _close_loop_root_xz_via_velocity(motion_np, translation_root_index):
    """Close loop root XZ drift by distributing velocity residual across frames.
    Zeroes the root RIC X/Z (ch 0, 2) and subtracts mean vel-XZ residual (ch 9, 11)
    so the integrated endpoint matches the start."""

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


def _parse_species_tags(raw):
    """Split a raw ``--species_tags`` string into a clean list of tags.

    Accepts comma- or semicolon-separated tags and drops empty entries.
    """
    return [t.strip() for t in str(raw or '').replace(';', ',').split(',') if t.strip()]


def _resolve_species_t5_name(cond_entry):
    """Return the T5 model name used to bake this species' descriptor (species_emb_meta >
    joints_names_embs_meta > t5-base)."""
    for meta_key in ('species_emb_meta', 'joints_names_embs_meta'):
        meta = cond_entry.get(meta_key)
        if isinstance(meta, dict) and meta.get('t5_name'):
            return str(meta['t5_name'])
    return 't5-base'


# Process-local T5 cache keyed by t5 model name for --species_tags re-encoding.
_SPECIES_T5_CACHE = {}


def _get_species_t5_conditioner(t5_name, *, preloaded=None):
    """Return a T5 conditioner, preferring preloaded (if name matches) > cached > fresh."""
    if preloaded is not None and str(getattr(preloaded, 'name', '')) == str(t5_name):
        return preloaded
    cached = _SPECIES_T5_CACHE.get(t5_name)
    if cached is not None:
        return cached
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[generate] Loading T5 '{t5_name}' on {device.upper()} for species-tag re-encoding ...")
    from model.conditioners import T5Conditioner

    conditioner = T5Conditioner(
        name=t5_name,
        finetune=False,
        word_dropout=0.0,
        normalize_text=False,
        device=device,
        autocast_dtype=None,
        local_files_only=True,
    )
    _SPECIES_T5_CACHE[t5_name] = conditioner
    return conditioner


def _encode_species_tags_override(tags, cond_entry, expected_dim, t5_conditioner=None):
    """Re-encode species tags via T5 into a [expected_dim] species_emb override."""
    species_text = ' '.join(tags)
    t5_name = _resolve_species_t5_name(cond_entry)
    print(f"[generate] Re-encoding species tags {tags} via T5 '{t5_name}' ...")
    conditioner = _get_species_t5_conditioner(t5_name, preloaded=t5_conditioner)
    with torch.no_grad():
        tokens = conditioner.tokenize_entries([species_text])
        emb = conditioner(tokens).detach().cpu().numpy().astype(np.float32, copy=False)[0]
    if emb.shape[-1] != int(expected_dim):
        raise ValueError(
            f"--species_tags re-encoding produced dim {emb.shape[-1]} but the model expects "
            f"{expected_dim} (t5_out_dim). The T5 model '{t5_name}' does not match the one used "
            "to build cond.npy."
        )
    return emb


def _find_cached_species_emb(tags, cond_dict, t5_name, expected_dim):
    """Reuse baked species_emb when requested tags match an existing cond entry (same T5 + dim).
    T5 mean-pooling is deterministic given tokenized text, so this is exact."""
    target_text = ' '.join(tags)
    for entry in cond_dict.values():
        if not isinstance(entry, dict):
            continue
        emb = entry.get('species_emb')
        meta = entry.get('species_emb_meta')
        if emb is None or not isinstance(meta, dict):
            continue
        if str(meta.get('t5_name') or '') != str(t5_name):
            continue
        # Normalize whitespace for comparison.
        if ' '.join(str(meta.get('embedding_text', '')).split()) != target_text:
            continue
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[-1] == int(expected_dim):
            return emb, str(entry.get('object_type') or '?')
    return None


def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len, loop=False, action_tags=None, species_emb_override=None):
    """Build model_kwargs for a batch of object_types.

    action_tags: per-object list of tag strings (or None).
    species_emb_override: [t5_out_dim] vector replacing baked species_emb for all objects.
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
        mark_canonical_cond_entry(cond_dict[object_type])
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        rest_pose = np.nan_to_num(build_canonical_rest_feature(cond_dict[object_type]))
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = cond_dict[object_type]['joints_names_embs']
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(rest_pose)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames, circular=circular_mask))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(max_joints)
        metadata = {
            'is_loop': bool(loop),
            'loop_full_cycle': bool(loop),
            'translation_root_index': cond_dict[object_type].get('translation_root_index', 0),
        }
        if 'species_emb' in cond_dict[object_type]:
            metadata['species_emb'] = cond_dict[object_type]['species_emb']
        if species_emb_override is not None:
            metadata['species_emb'] = species_emb_override
        if action_tags is not None and i < len(action_tags):
            tags = action_tags[i]
            if tags is not None:
                metadata['action_tags'] = tags
        batch.append(metadata)
        batch.append(object_type)
        batch.append({
            # Generation decodes per sample with the full cond entry, so y does
            # not carry the global stats here (sampling itself never decodes).
            'rest_pose_physical': cond_dict[object_type]['rest_pose'],
            'rest_pos_ric_hml': cond_dict[object_type]['rest_pos_ric_hml'],
            'feature_space': cond_dict[object_type].get('feature_space', 'canonical_motion_v3'),
        })
        batches.append(batch)

    return truebones_batch_collate(batches)


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
