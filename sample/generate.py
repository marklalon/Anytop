# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import sys

# Ensure both the Anytop dir (for bare ``utils.*`` / ``data_loaders.*`` imports)
# and its parent (for ``utils.*`` imports made by submodules like
# ``utils/retarget_core.py``) are on sys.path when running as a script. Insert
# repo-root first then Anytop second so Anytop's ``utils/`` wins over the
# unrelated ``<repo_root>/utils/`` directory for bare imports.
_ANYTOP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_ANYTOP_ROOT))
sys.path.insert(0, _ANYTOP_ROOT)

import numpy as np
import torch
from tqdm import tqdm

from data_loaders.truebones.data.dataset import ensure_joint_name_embeddings
from data_loaders.truebones.truebones_utils.canonical_features import (
    canonical_to_physical_hml,
    mark_canonical_cond_entry,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import (
    build_species_file_tokens,
    species_lookup_map,
)
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.param_utils import MAX_SOURCE_FRAMES_MULT
from sample.conditioning import (
    _resolve_action_condition,
    _resolve_species_emb_override,
    _wrap_action_label_cfg,
    create_condition,
)
from sample.export import (
    _bvh_preview_options,
    _export_motion,
    _get_batch_translation_root_index,
    _zero_root_ric_xz,
)
from sample.generation_runtime import (
    _checkpoint_cond_loader,
    _checkpoint_cond_path,
    _checkpoint_native_num_frames,
    _lookup_object_type_case_insensitive,
    _normalize_optional_path,
    _raise_opt_max_joints_for_cond,
    prepare_generation_runtime,
)
from sample.inpaint import (
    _contiguous_frame_runs,
    _map_frame_ranges_to_internal,
    _parse_frame_ranges,
    _reanchor_inpaint_root_y_via_velocity,
    _reground_inpaint_joint_y,
    _resolve_inpaint_joint_indices,
    build_inpaint_mask,
)
from sample.output_lengths import (
    _all_species_output_lengths,
    _finalize_output_lengths,
    _resample_window_to_output,
    _resolve_auto_output_lengths,
    fit_reference_to_output,
    resolve_loop_condition,
)
from sample.reference_motion import (
    _REFERENCE_MOTION_PREPROCESS_SUFFIXES,
    _build_retarget_cond_dict,
    _prepare_img2img_reference_bundle,
    _resolve_reference_source_type,
    _retarget_reference_motion,
    _retarget_reference_motion_from_file,
    _should_retarget_reference,
    _validate_reference_motion_path,
)
from sample.sampling import _sample_batch
from utils import dist_util
from utils.fixseed import fixseed
from utils.misc import infer_object_type_from_filename
from utils.parser_util import generate_args


def _generate_all_species(
    cond_dict,
    cond_max_joints,
    opt,
    args,
    n_frames,
    species_output_lengths,
    model,
    diffusion,
    sampling_method,
    inference_autocast_dtype,
    out_path,
    fps,
    action_condition=None,
):
    """Generate exactly one motion per species in mixed-species batches.

    Each batch packs up to ``args.batch_size`` different species into a single
    forward pass.  The model and sampler are batch-agnostic — all conditioning
    is per-sample — so this is both correct and more efficient than looping
    over species one at a time.

    ``action_condition`` (from ``--action_label``) is applied to every species,
    which is exactly what the flag means here: the same prompt performed by each
    skeleton.

    ``species_output_lengths`` is ``{species: (target_output_frames,
    resample_speed_cond, is_loop)}``. Length is per species for the same reason
    the conditioning is: a Pigeon's walk cycle is not a Horse's, so one shared
    number would put every species but the average one off its training
    distribution. All three values are already per-sample in the batch (the
    speed and the loop flag are conditioning channels, the frame count only
    affects the export resample), so nothing about the shared forward pass
    changes.
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
    if action_condition is not None:
        print(f'  Action label: {action_condition["action_label"]!r} '
              f'(group={action_condition["action_group"]})')
    sampling_model = _wrap_action_label_cfg(model, args, action_condition)

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

            batch_roster = ", ".join(
                f'{sp}({species_output_lengths[sp][0]}f'
                f'{", loop" if species_output_lengths[sp][2] else ""})'
                for sp in batch_species
            )
            print(f'\n--- Batch {batch_idx}/{len(species_batches)} ({actual_bs} species, '
                  f'max_joints={batch_max_joints}): {batch_roster} ---')

            _, model_kwargs = create_condition(
                list(batch_species),
                cond_dict,
                output_frame_count,
                max_joints=batch_max_joints,
                feature_len=opt.feature_len,
                # Same meaning as in main(), per sample: a closed window, so
                # every temporal resample of it is periodic.
                loop=[species_output_lengths[sp][2] for sp in batch_species],
                action_condition=action_condition,
            )
            model_kwargs['y']['resample_speed_cond'] = torch.tensor(
                [species_output_lengths[sp][1] for sp in batch_species],
                dtype=torch.float32, device=dist_util.dev(),
            )

            print(f'  Sampling {actual_bs} species × 1 motion each ...')
            sample = _sample_batch(
                diffusion=diffusion,
                model=sampling_model,
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
                # Decode with full cond entry (carries rest geometry + global standardization stats).
                motion_physical = canonical_to_physical_hml(motion.unsqueeze(0), sp_entry)[0]
                motion_np = motion_physical.cpu().permute(2, 0, 1).numpy()

                motion_np = _resample_window_to_output(
                    motion_np, species_output_lengths[sp][0], output_frame_count,
                    species_output_lengths[sp][2],
                )

                translation_root_index = _get_batch_translation_root_index(
                    model_kwargs, sample_idx,
                    fallback=sp_entry.get('translation_root_index', 0),
                )
                _zero_root_ric_xz(motion_np, translation_root_index)

                joint_names = list(sp_entry.get(
                    'canonical_bvh_joint_names', sp_entry['joints_names'],
                ))

                # Count existing outputs so repeated runs don't overwrite.
                sp_token = species_file_tokens[sp]
                existing = [f for f in os.listdir(out_path)
                            if f.startswith(sp_token) and f.endswith('.npy')]
                npy_name = f'{sp_token}_{(len(existing))}.npy'
                export_tasks.append((
                    motion_np, sp_entry, npy_name, joint_names, out_path, fps,
                    _bvh_preview_options(args),
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
        # If the task specifies a different --cond_path, reload that cond and
        # re-derive opt for it (species entries, dataset_tags/subsets, baked
        # tags) so the runtime reflects the task's cond rather than the
        # checkpoint's. The model/diffusion stay shared.
        task_cond = _normalize_optional_path(getattr(args, 'cond_path', '') or '')
        if task_cond != runtime.cond_path:
            new_cond_dict = load_cond(task_cond)
            # Same gate the first cond went through: a task that swaps in another
            # cond.npy must not slip an older joint-name embedding schema past
            # the check prepare_generation_runtime already ran.
            ensure_joint_name_embeddings(
                new_cond_dict,
                expected_embedding_dim=args.t5_out_dim,
                cond_source=task_cond,
            )
            new_opt = get_opt(
                runtime.device, task_cond, cond_dict=new_cond_dict, inference=True
            )
            _raise_opt_max_joints_for_cond(new_opt, new_cond_dict)
            runtime.opt = new_opt
            runtime.cond_dict = new_cond_dict
            runtime.actual_cond_file = task_cond
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
    _ckpt_num_frames = _checkpoint_native_num_frames(args.model_path)

    internal_num_frames = _ckpt_num_frames
    min_length = int(getattr(args, 'min_length', 20))
    n_frames = internal_num_frames
    cond_max_joints = opt.max_joints

    motion_frames = getattr(args, 'num_frames', None)

    # Output length, in priority order:
    #   1. --num_frames itself. An explicit number is the user's decision and
    #      outranks everything, reference included (the reference is outpainted
    #      when R < M and cropped when R > M, loop or not --
    #      fit_reference_to_output): it is finalized right here, and every
    #      fallback below is guarded on requested_output_frames still being None.
    #   2. a --reference_motion's own frame count R -- with no number given, the
    #      reference IS the requested length.
    #   3. the training-length prior for --action_label, resolved once
    #      --object_type is known (below, and in the all-species branch).
    #   4. the checkpoint's native window.
    requested_output_frames = target_output_frames = resample_speed_cond_value = None
    if motion_frames is not None:
        requested_output_frames, target_output_frames, resample_speed_cond_value = (
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

    ddim_eta = float(getattr(args, 'ddim_eta', 0.0))
    reference_motion_path = getattr(args, 'reference_motion', None)
    reference_motion_suffix = (
        _validate_reference_motion_path(reference_motion_path)
        if reference_motion_path else ''
    )

    inpaint_joints_arg = str(getattr(args, 'inpaint_joints', '') or '').strip()
    inpaint_frames_arg = str(getattr(args, 'inpaint_frames', '') or '').strip()
    inpaint_include_subtree = bool(getattr(args, 'inpaint_include_subtree', True))
    # --loop is the whole loop condition: the model is asked for a closed window
    # (y['is_loop'], the circular phase table, the loader's periodic window
    # resample), and the sampled window is exported with the matching periodic
    # resample so nothing downstream breaks the cycle open. A reference is not
    # part of that: it only supplies the verdict under 'auto' and is otherwise a
    # one-shot clip. 'auto' needs the action label and the target species
    # (resolve_loop_condition), so the bool is fixed further down, right before
    # the length that depends on it.
    loop_mode = getattr(args, 'loop', 'auto')

    # ── Resolve --object_type ───────────────────────────────────────────────
    # --object_type: look up directly in cond (user-provided first, then default).
    # --reference_motion: infer source type from filename, look up in cond the same way.
    # If source != target → retarget.
    explicit_object_type = args.object_type

    # A --cond_path whose file holds exactly one species makes --object_type
    # redundant: there is only one possible target, so use it (e.g.
    # `--cond_path outputs/new_skeleton_horse/cond.npy --loop`).
    if (
        not reference_motion_path
        and not explicit_object_type
        and str(getattr(args, 'cond_path', '') or '').strip()
        and len(cond_dict) == 1
    ):
        explicit_object_type = next(iter(cond_dict))
        print(
            f"[generate] --object_type omitted; the cond file {args.cond_path} "
            f"contains a single species, using it: {explicit_object_type}"
        )

    if not reference_motion_path and not explicit_object_type:
        sys.exit(
            "ERROR: must supply at least one of --reference_motion or --object_type. "
            "Pass --object_type for pure-random generation, --reference_motion for "
            "reference-guided generation (object_type auto-inferred from filename), "
            "or both to retarget the reference into a different target skeleton. "
            "(A single-species --cond_path also auto-selects its species.)"
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
        # --action_label must be honoured here too. The all-species path returns
        # before the single-species resolve below, so without this the prompt was
        # silently dropped (and a label on a checkpoint that cannot use it went
        # unreported). The condition is species-independent: it is word ids into
        # the checkpoint's own vocabulary.
        _all_action_condition = _resolve_action_condition(args, model)
        _species_output_lengths = _all_species_output_lengths(
            cond_dict,
            _all_action_condition,
            explicit=(
                None if requested_output_frames is None
                else (target_output_frames, resample_speed_cond_value)
            ),
            min_length=min_length,
            internal_num_frames=internal_num_frames,
            default_frames=_ckpt_num_frames,
            loop_mode=loop_mode,
            fallback_cond_loader=_checkpoint_cond_loader(args, actual_cond_file),
        )
        _generate_all_species(
            cond_dict=cond_dict,
            cond_max_joints=cond_max_joints,
            opt=opt,
            args=args,
            n_frames=n_frames,
            species_output_lengths=_species_output_lengths,
            model=model,
            diffusion=diffusion,
            sampling_method=sampling_method,
            inference_autocast_dtype=inference_autocast_dtype,
            out_path=out_path,
            fps=fps,
            action_condition=_all_action_condition,
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

    # Resolved here, ahead of every reference/retarget step, because
    # an unset ``--num_frames`` needs the label -- and because a bad label should
    # fail before minutes of retargeting, not after. The loop condition comes
    # first: the auto length only pools loop clips when the window is a loop.
    # With a reference both wait for the clip itself (further down): its length
    # is the window, and --loop auto reads whether it closes off the tensor.
    _action_condition = _resolve_action_condition(args, model)
    loop_condition = None
    if not reference_motion_path:
        loop_condition = resolve_loop_condition(
            loop_mode,
            cond_dict,
            object_type,
            _action_condition,
            fallback_cond_loader=_checkpoint_cond_loader(args, actual_cond_file),
        )
    if requested_output_frames is None and not reference_motion_path:
        # A reference outranks the label: with one present the length is its own
        # R, finalized from the loaded reference further down.
        requested_output_frames, target_output_frames, resample_speed_cond_value = (
            _resolve_auto_output_lengths(
                cond_dict,
                object_type,
                _action_condition,
                min_length=min_length,
                internal_num_frames=internal_num_frames,
                default_frames=_ckpt_num_frames,
                loop=loop_condition,
                fallback_cond_loader=_checkpoint_cond_loader(args, actual_cond_file),
            )
        )

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

        # The loop condition, off the reference as it will fill the window:
        # retargeted onto the target skeleton, so the target's root indexes it.
        # This verdict is the ONLY thing the reference's loopiness decides: it
        # becomes the model's is_loop condition, and with it the export mapping.
        # Nothing below reads the clip as a cycle -- it is fitted, filled and
        # clamped as a one-shot, every frame it ships kept, and closing the
        # cycle is the model's job.
        loop_condition = resolve_loop_condition(
            loop_mode,
            cond_dict,
            object_type,
            _action_condition,
            reference_features=ref_features_full,
            translation_root_index=int(cond_dict[object_type].get('translation_root_index', 0)),
        )

        # Finalize output lengths from R (if --num_frames not specified).
        if requested_output_frames is None:
            auto_frames = int(np.clip(R, min_length, MAX_SOURCE_FRAMES_MULT * internal_num_frames))
            requested_output_frames, target_output_frames, resample_speed_cond_value = (
                _finalize_output_lengths(auto_frames, min_length, internal_num_frames)
            )
            if auto_frames == R:
                print(f'  Using reference native length R={R} frames')
            else:
                print(
                    f'  Reference R={R} frames clamped to {auto_frames} '
                    f'(variable-length window [{min_length}, '
                    f"{MAX_SOURCE_FRAMES_MULT * internal_num_frames}])"
                )
        M = int(requested_output_frames)

        # Fit the reference to M: crop (R > M) or outpaint-pad (R < M). A loop
        # reference is no exception -- R < M appends frames from noise, which
        # under is_loop is exactly where the model gets to close the cycle.
        ref_features_full, auto_outpaint_range, fit_note = fit_reference_to_output(
            ref_features_full, M,
        )
        outpaint_active = auto_outpaint_range is not None
        if fit_note:
            print(fit_note)

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
            min_length=min_length,
        )
        ref_motion = reference_bundle['reference_motion']
        output_frame_count = reference_bundle['output_frame_count']
        loaded_reference_frame_count = reference_bundle['loaded_reference_frame_count']
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

    if (user_inpaint_active or outpaint_active) and ref_motion is None:
        sys.exit(
            "ERROR: --inpaint_* / length extension is set but the reference "
            "motion could not be loaded; cannot inpaint without a known region."
        )

    # Create condition with effective frame count (shared across passes).
    obj_batch = [object_type] * args.batch_size
    _sampling_model = _wrap_action_label_cfg(model, args, _action_condition)

    # ── --species_tags: restyle the target species' motion descriptor ────────
    _species_emb_override = _resolve_species_emb_override(
        args,
        model,
        cond_dict,
        object_type,
        default_cond_file=_checkpoint_cond_path(
            getattr(args, 'model_path', ''),
        ),
        actual_cond_file=actual_cond_file,
        t5_conditioner=getattr(runtime, 't5_conditioner', None),
    )

    _, model_kwargs = create_condition(
        obj_batch,
        cond_dict,
        output_frame_count,
        max_joints=max_joints,
        feature_len=opt.feature_len,
        loop=loop_condition,
        action_condition=_action_condition,
        species_emb_override=_species_emb_override,
    )
    model_kwargs['y']['resample_speed_cond'] = torch.full(
        (args.batch_size,),
        resample_speed_cond_value,
        dtype=torch.float32,
        device=dist_util.dev(),
    )

    def _build_inpaint_mask_for(frames_arg, joints_arg, warn_remap=False):
        # Output frames name REFERENCE frames (a mask only exists where a
        # reference does), so they map the way the reference fills the window:
        # end to end, one-shot, --loop or not.
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
            model=_sampling_model,
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

    # Collect export tasks (in-process, no pickling needed)
    joint_names = list(cond_dict[object_type].get(
        'canonical_bvh_joint_names',
        cond_dict[object_type]['joints_names'],
    ))
    preview_options = _bvh_preview_options(args)
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

        motion_np = _resample_window_to_output(
            motion_np, target_output_frames, output_frame_count, loop_condition,
        )

        # The per-species translation root (the joint carrying the locomotion
        # XZ velocity; the hierarchy root for every collapsed cond skeleton).
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
            # Same mapping as the exported motion: _reground_inpaint_joint_y
            # pairs these two frame by frame.
            ref_motion_np = _resample_window_to_output(
                ref_motion_np, target_output_frames, output_frame_count, loop_condition,
            )
            reseat_delta = _reground_inpaint_joint_y(
                motion_np, ref_motion_np, reseat_free_joints, parents,
            )
            if reseat_delta:
                print(
                    f'    Inpaint reseat: shifted regenerated subtree world-Y by '
                    f'{reseat_delta:+.4f} to re-ground onto the reference'
                )
        _zero_root_ric_xz(motion_np, translation_root_index)

        npy_name = f'{object_file_token}_{base_index + sample_idx}.npy'
        export_tasks.append((
            motion_np,
            cond_dict[object_type],
            npy_name,
            joint_names,
            out_path,
            fps,
            preview_options,
        ))

    for task in tqdm(export_tasks, desc=f'{object_file_token} export'):
        npy_name = _export_motion(task)
        print(f'    Created motion: {npy_name}')

    return out_path


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
