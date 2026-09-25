"""Checkpoint + cond loading for generation: the ``GenerationRuntime``.

Everything here answers "which weights, which cond.npy, which device" -- the
part of a generation that is shared across tasks and therefore prepared once
(``prepare_generation_runtime``) and reused by ``sample.generate.main``.
"""
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.truebones.data.dataset import ensure_joint_name_embeddings
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.species_descriptor_table import (
    bind_cond_species_embs,
    load_checkpoint_species_descriptor_table,
)
from utils import dist_util
from utils.model_util import (
    bind_checkpoint_action_conditioning,
    create_model_and_diffusion_general_skeleton,
    load_checkpoint_weights,
    load_model,
    resolve_t5_out_dim,
)
from utils.parser_util import generate_args


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
    # The checkpoint's species descriptor table (None when the checkpoint has no
    # species conditioning): the only source of species_emb at generation.
    species_table: object = None

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
        sampling_method = str(getattr(args, 'sampling_method', 'ddpm')).lower()
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
    """``--cond_path``, else the checkpoint's own snapshot. Never the dataset.

    The checkpoint's cond is the contract the weights were trained against --
    the species it knows, their skeletons, their baked statistics. The dataset's
    cond is a moving target: it is re-preprocessed, re-merged and re-baked
    between training runs, so reaching for it would silently sample a checkpoint
    against conditioning it never saw. A checkpoint with no cond.npy next to it
    is therefore an error, not a reason to substitute the dataset's.
    """
    explicit = getattr(args, 'cond_path', '') or ''
    if explicit:
        return explicit
    checkpoint_cond = _checkpoint_cond_path(getattr(args, 'model_path', ''))
    if checkpoint_cond:
        return checkpoint_cond
    sys.exit(
        f"ERROR: no cond.npy next to the checkpoint "
        f"{os.path.dirname(os.path.abspath(getattr(args, 'model_path', '') or '.'))}. "
        "A checkpoint carries its own cond snapshot (training copies it into "
        "save_dir); generation reads that and never the dataset's cond, which "
        "has moved on since. Copy the cond the run was trained with next to the "
        "checkpoint, or name one explicitly with --cond_path."
    )


def _load_checkpoint_species_table(args):
    """The descriptor table next to the checkpoint, or None when species cond is off."""
    if not (getattr(args, 'species_cond', False) or getattr(args, 'species_joint_cond', False)):
        return None
    table = load_checkpoint_species_descriptor_table(args.model_path, 'the checkpoint')
    if table.embedding_dim != int(args.t5_out_dim):
        raise ValueError(
            f"species descriptor table {table.source} is {table.embedding_dim}d but the "
            f"model expects t5_out_dim={args.t5_out_dim}"
        )
    return table


def bind_species_table(species_table, cond_dict, cond_source):
    """Condition every cond entry on the checkpoint's table row for its species_tags."""
    if species_table is not None:
        bind_cond_species_embs(cond_dict, species_table, f"cond {cond_source}")


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
    sampling_method = str(getattr(args, 'sampling_method', 'ddpm')).lower()
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
    ``_sample_batch``); this validates bf16 availability and returns the
    effective dtype string ('bf16' or 'fp32'). It does not mutate the model.

    On CUDA it also turns TF32 on for fp32 matmuls, the same setting --compile
    training uses: fp32+TF32 costs ~5% per forward over bf16, while bf16's output
    rounding inflates the scorer's jerk/spectral-flatness terms
    (docs/bf16_precision_issues.md). This is process-global and
    ``fixseed`` leaves it alone.
    """
    amp_dtype_arg = str(getattr(args, 'amp_dtype', 'fp32')).lower()
    _amp_device = dist_util.dev()
    if _amp_device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    if amp_dtype_arg != 'bf16':
        return amp_dtype_arg
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
    # cond.npy is the whole inference contract, so it is resolved before opt.
    # ``inference=True``: dataset_tags comes from the cond's own baked species
    # tags and no dataset directory is touched, so a checkpoint generates the
    # same motion on a machine that has never held the training data.
    opt = get_opt(
        args.device,
        _resolve_generation_cond_path(args),
        cond_dict=cond_dict,
        inference=True,
    )
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
    payload = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict, _state_dict_avg, checkpoint_metadata = load_checkpoint_weights(
        payload, args.model_path, prefer_ema=True)
    if payload.get('model_avg') is not None:
        print('EMA checkpoint detected, loading model_avg weights.')
    assert model is not None, 'BUG: create_model_and_diffusion_general_skeleton returned None for model'
    # model.to(device) may return None (CUDA 12.8 + torch 2.7.1); parameter move is in-place.
    model.to(device)
    load_model(model, state_dict)
    # The frozen word table came out of the checkpoint with the weights; this
    # certifies it and the contract it was trained under. Inference reads no
    # sidecar and no dataset directory for the action condition.
    bind_checkpoint_action_conditioning(model, checkpoint_metadata, args.model_path)

    print('Validating precomputed joint-name embeddings from cond.npy...')
    ensure_joint_name_embeddings(
        cond_dict,
        expected_embedding_dim=args.t5_out_dim,
        cond_source=actual_cond_file,
    )
    species_table = _load_checkpoint_species_table(args)
    bind_species_table(species_table, cond_dict, actual_cond_file)
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
        cond_path=_normalize_optional_path(_resolve_generation_cond_path(args)),
        species_table=species_table,
    )


def _checkpoint_native_num_frames(model_path, default=60):
    """The window the checkpoint was trained on, from the args.json next to it.

    ``args.num_frames`` at generation is the user's requested output length (or
    ``None``); the model's own window only lives in the training args.
    """
    ckpt_args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    if not os.path.isfile(ckpt_args_path):
        return int(default)
    with open(ckpt_args_path, 'r') as f:
        return int(json.load(f).get('num_frames', default))


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


def _checkpoint_cond_loader(args, actual_cond_file):
    """A no-argument loader for the checkpoint's own cond snapshot, or ``None``.

    Returns ``None`` (nothing to load) when the snapshot IS the active cond,
    which is the ordinary case -- so the caller never pays for a second np.load
    unless it is really looking at a different, narrower cond file. The load is
    memoized because the all-species path asks once per species.
    """
    default_cond_file = _checkpoint_cond_path(getattr(args, 'model_path', ''))
    if not default_cond_file or os.path.realpath(default_cond_file) == os.path.realpath(actual_cond_file):
        return None
    cache = []

    def load():
        if not cache:
            cache.append(_load_default_cond_cache(default_cond_file, actual_cond_file))
        return cache[0]

    return load
