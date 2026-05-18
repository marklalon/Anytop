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

from data_loaders.tensors import create_padded_relation, truebones_collate
from data_loaders.truebones.data.dataset import (
    create_temporal_mask_for_window,
    ensure_joint_name_embeddings,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.features import (
    recover_bvh_export_animation_from_motion_with_object_cond_np,
)
from motion_lib import BVH
from os.path import join as pjoin
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion_general_skeleton,
    load_model,
    resolve_t5_out_dim,
)
from utils.parser_util import generate_args
from utils.misc import infer_object_type_from_filename


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
        out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_with_object_cond_np(
            target_features,
            tgt_cond,
            list(tgt_cond.get('canonical_bvh_joint_names', tgt_cond['joints_names'])),
            allow_infer=True,
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


def _export_motion(task):
    motion_np, object_cond, npy_name, joint_names, out_path, fps = task
    out_anim, joint_names, has_animated_pos = recover_bvh_export_animation_from_motion_with_object_cond_np(
        motion_np,
        object_cond,
        joint_names,
        allow_infer=True,
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
):
    fixseed(seed)

    # If reference motion is provided, use it as init_image with skip_timesteps.
    # skip_timesteps: how many of the noisiest timesteps to skip.
    # Higher = start from less noisy state = more faithful to reference.
    # e.g., skip_timesteps=80 with 100 total steps means:
    #   - reference is noised to t=19 (light noise)
    #   - only 20 denoising steps performed
    if reference_motion is not None and skip_timesteps > 0:
        init_image = reference_motion.to(device, non_blocking=True)
        noise = torch.randn(sample_shape, device=device)
    else:
        init_image = None
        noise = torch.randn(sample_shape, device=device)
        skip_timesteps = 0  # No reference => full denoising from pure noise

    common_kwargs = dict(
        model=model,
        shape=sample_shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        init_image=init_image,
        skip_timesteps=skip_timesteps,
    )

    if sampling_method == 'ddim':
        return diffusion.ddim_sample_loop(
            progress=True,
            eta=ddim_eta,
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
            skip_timesteps=skip_timesteps,
            dump_steps=None,
            const_noise=False,
            **common_kwargs,
        )
    raise ValueError(f'Unknown sampling_method: {sampling_method}')


def main(args=None, cond_dict=None):
    if args is None:
        args = generate_args()

    fixseed(args.seed)
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
    skip_timesteps = int(getattr(args, 'skip_timesteps', 80))

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

    # 1) Resolve target object_type
    if explicit_object_type:
        # Case A: explicit --object_type provided.
        # Look up case-insensitively in cond_dict.
        target_type = next(
            (k for k in cond_dict if k.upper() == explicit_object_type.upper()),
            None,
        )
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
    needs_retarget = False
    _default_cond_cache = None

    if reference_motion_path:
        # Blind inference from filename, then look up in cond.
        blind_type = infer_object_type_from_filename(
            reference_motion_path, valid_types=None
        )
        if blind_type:
            # Look up case-insensitively in user-provided cond_dict first.
            source_type = next(
                (k for k in cond_dict if k.upper() == blind_type.upper()),
                None,
            )
            if source_type is None:
                # Not in user-provided cond — try default cond.
                default_cond_file = getattr(opt, 'cond_file', None)
                if default_cond_file and not os.path.samefile(
                    os.path.realpath(default_cond_file),
                    os.path.realpath(actual_cond_file),
                ):
                    _default_cond_cache = np.load(default_cond_file, allow_pickle=True).item()
                    source_type = next(
                        (k for k in _default_cond_cache if k.upper() == blind_type.upper()),
                        None,
                    )
            if source_type is None:
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
    if source_type and target_type and source_type.upper() != target_type.upper():
        needs_retarget = True

    object_type = target_type  # downstream code keeps reading `object_type`
    if reference_motion_path:
        if needs_retarget:
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
        print(f'    method={sampling_method} steps={sampling_steps or "full"} batch_size={args.batch_size}')

        # Prepare reference motion (normalize + reshape)
        ref_motion = None
        effective_n_frames = n_frames  # May be overridden by reference motion

        effective_reference_path = reference_motion_path
        if needs_retarget:
            # Build a cond_dict that contains both source and target.
            retarget_cond_dict = dict(cond_dict)
            if source_type not in cond_dict:
                # Source type not in user-provided cond — use cached default cond.
                if _default_cond_cache:
                    for k, v in _default_cond_cache.items():
                        if k not in retarget_cond_dict:
                            retarget_cond_dict[k] = v
                else:
                    sys.exit(
                        f"ERROR: source type '{source_type}' not found in cond file "
                        f"and no default cond available for retarget."
                    )

            effective_reference_path = _retarget_reference_motion(
                reference_motion_path,
                source_type=source_type,
                target_type=target_type,
                cond_dict=retarget_cond_dict,
                opt=opt,
                output_dir=out_path,
                fps=fps,
            )

        if effective_reference_path:
            ref_raw = np.load(effective_reference_path).astype(np.float32)
            # ref_raw shape: [frames, joints, features]
            ref_frames = ref_raw.shape[0]
            ref_joints = ref_raw.shape[1]

            # Use reference motion's frame count (truncated to model's max frames)
            effective_n_frames = min(ref_frames, n_frames)
            if effective_n_frames != n_frames:
                print(f'  Reference motion overrides frame count: {n_frames} -> {effective_n_frames}')

            # Truncate reference if longer than max output
            if ref_frames > effective_n_frames:
                ref_raw = ref_raw[:effective_n_frames]

            # Normalize using the same object_type's stats (same-skeleton only)
            obj_mean = cond_dict[object_type]['norm_mean']
            obj_std = np.asarray(cond_dict[object_type]['norm_std'], dtype=np.float32) + 1e-6

            # Normalize using same stats as training data
            ref_norm = np.nan_to_num((ref_raw - obj_mean[None, :]) / obj_std[None, :], copy=True).astype(np.float32)

            # Pad joints to max_joints if needed
            if ref_joints < max_joints:
                pad = np.zeros((effective_n_frames, max_joints - ref_joints, ref_norm.shape[2]), dtype=np.float32)
                ref_norm = np.concatenate([ref_norm, pad], axis=1)

            # Convert to model input shape: [batch, joints, features, frames]
            ref_tensor = torch.from_numpy(ref_norm).permute(1, 2, 0)  # [joints, features, frames]
            # Ensure feature dim matches model expectation
            ref_feat = ref_tensor.shape[1]
            target_feat = model.feature_len
            if ref_feat < target_feat:
                pad = torch.zeros((max_joints, target_feat - ref_feat, effective_n_frames), dtype=torch.float32)
                ref_tensor = torch.cat([ref_tensor, pad], dim=1)
            elif ref_feat > target_feat:
                ref_tensor = ref_tensor[:, :target_feat, :]
            ref_tensor = ref_tensor.unsqueeze(0).expand(args.batch_size, -1, -1, -1)
            ref_motion = ref_tensor
            print(f'  Reference motion loaded: {effective_reference_path}')
            if effective_reference_path != reference_motion_path:
                print(f'    (retargeted from original: {reference_motion_path})')
            print(f'    Original: [{ref_frames} frames, {ref_joints} joints] -> Target: [{effective_n_frames} frames, {max_joints} joints]')
            print(f'    skip_timesteps: {skip_timesteps} (higher = more faithful to reference)')

        # Create condition with effective frame count
        obj_batch = [object_type] * args.batch_size
        _, model_kwargs = create_condition(
            obj_batch,
            cond_dict,
            effective_n_frames,
            args.temporal_window,
            max_joints=opt.max_joints,
            feature_len=opt.feature_len
        )
        sample = _sample_batch(
            diffusion=diffusion,
            model=model,
            model_kwargs=model_kwargs,
            sampling_method=sampling_method,
            sample_shape=(args.batch_size, max_joints, model.feature_len, effective_n_frames),
            ddim_eta=ddim_eta,
            seed=args.seed,
            device=dist_util.dev(),
            reference_motion=ref_motion,
            skip_timesteps=skip_timesteps,
        )

        # Pre-compute filenames with a single directory scan
        existing_npy_files = [
            f for f in os.listdir(out_path)
            if f.startswith(object_type) and f.endswith('.npy')
        ]
        base_index = len(existing_npy_files)

        # Collect export tasks (in-process, no pickling needed)
        joint_names = cond_dict[object_type].get(
            'canonical_bvh_joint_names',
            cond_dict[object_type]['joints_names'],
        )
        object_cond = cond_dict[object_type]
        export_tasks = []
        for sample_idx, motion in enumerate(sample):
            n_joints = model_kwargs['y']['n_joints'][sample_idx].item()
            motion = motion[:n_joints]
            norm_mean = cond_dict[object_type]['norm_mean'][None, :]
            norm_std = cond_dict[object_type]['norm_std'][None, :]
            motion_np = motion.cpu().permute(2, 0, 1).numpy() * norm_std + norm_mean

            npy_name = f'{object_type}_#{base_index + sample_idx}.npy'
            export_tasks.append((
                motion_np,
                object_cond,
                npy_name,
                joint_names,
                out_path,
                fps,
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


def _build_condition_item(object_type, cond_dict, n_frames, temporal_window, max_joints, feature_len):
    object_cond = cond_dict[object_type]
    parents = np.asarray(object_cond['parents'], dtype=np.int64)
    n_joints = len(parents)
    return {
        'inp': torch.zeros((n_joints, feature_len, n_frames), dtype=torch.float32),
        'n_joints': n_joints,
        'lengths': int(n_frames),
        'parents': parents,
        'offsets': torch.from_numpy(np.asarray(object_cond['offsets'], dtype=np.float32)),
        'rest_rotations': torch.from_numpy(np.asarray(object_cond['rest_rotations'], dtype=np.float32)),
        'canon_joint_rot': torch.from_numpy(np.asarray(object_cond['canon_joint_rot'], dtype=np.float32)),
        'norm_schema_version': int(object_cond.get('norm_schema_version', 0) or 0),
        'temporal_mask': torch.as_tensor(create_temporal_mask_for_window(temporal_window, n_frames)),
        'graph_dist': create_padded_relation(object_cond['joints_graph_dist'], max_joints, n_joints),
        'joints_relations': create_padded_relation(object_cond['joint_relations'], max_joints, n_joints),
        'object_type': object_type,
        'joints_names_embs': torch.from_numpy(np.asarray(object_cond['joints_names_embs'], dtype=np.float32)),
        'tpos_first_frame': torch.from_numpy(np.asarray(object_cond['tpos_first_frame'], dtype=np.float32)),
        'norm_mean': torch.from_numpy(np.asarray(object_cond['norm_mean'], dtype=np.float32)),
        'norm_std': torch.from_numpy(np.asarray(object_cond['norm_std'], dtype=np.float32)),
    }


def create_condition(object_types, cond_dict, n_frames, temporal_window, max_joints, feature_len):
    """Build model_kwargs for a batch of object_types.
    """
    batch_items = list()
    for object_type in object_types:
        if object_type not in cond_dict:
            available = ', '.join(sorted(cond_dict.keys()))
            raise KeyError(
                f"Unknown object_type '{object_type}'. Available object types in cond file: {available}"
            )
        batch_items.append(_build_condition_item(object_type, cond_dict, n_frames, temporal_window, max_joints, feature_len))

    return truebones_collate(batch_items)


if __name__ == '__main__':
    main()
