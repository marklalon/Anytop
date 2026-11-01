"""Explicit backbone/cache/checkpoint compatibility and optional generation adapter."""
import hashlib
import json
from pathlib import Path

import torch

from model.motion_detail_refiner import MotionDetailRefiner, RefinerConfig
from utils.model_util import unwrap_anytop_model
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_FEATURE_SPACE, canonical_to_physical_hml, physical_hml_to_canonical,
)

DETAIL_SCHEMA = 1


def normalize_backbone_contract(contract):
    """Drop the known legacy ADAPTER field from otherwise strict base compatibility.

    Cached backbone motions do not depend on how a later refiner composes them.
    The adapter composition is now versioned separately in its architecture.
    """
    result = dict(contract)
    marker = result.pop('position_contract', None)
    if marker not in (None, 'base_plus_fk_difference_v1'):
        raise ValueError('Unsupported legacy position contract')
    return result


def tree_to(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def backbone_contract(args, cond_path):
    return {
        'schema': DETAIL_SCHEMA, 'feature_space': CANONICAL_FEATURE_SPACE,
        'backbone_sha256': file_hash(args.model_path),
        'args_sha256': file_hash(Path(args.model_path).with_name('args.json')),
        'cond_sha256': file_hash(cond_path),
        'weights': 'online' if getattr(args, 'no_ema', False) else 'ema_preferred',
        'sampling_method': args.sampling_method, 'sampling_steps': args.sampling_steps,
        'ddim_eta': float(args.ddim_eta), 'amp_dtype': args.amp_dtype,
        'action_cfg': float(args.action_label_cfg_scale),
        'time_contract': 'source_frames_over_window_at_reference_fps_v1',
    }


@torch.no_grad()
def attach_detail_action(model, y):
    model = unwrap_anytop_model(model)
    if not getattr(model, 'action_label_cond', False):
        raise ValueError('Detail refiner requires an action-conditioned backbone')
    device = model.action_word_embeddings.device
    batch = len(y['n_joints'])
    if 'action_word_ids' in y:
        ids = y['action_word_ids'].to(device)
        slots = y['action_slot_ids'].to(device)
        mask = y['action_word_mask'].to(device)
        action = model._assemble_action_slot_channels(ids, slots, mask, torch.float32)
    else:
        action = torch.zeros(batch, 3 * model.action_word_embeddings.shape[1], device=device)
    y = dict(y)
    y['detail_action'] = action.detach()
    return y


def decode_batch(sample, y):
    return canonical_to_physical_hml(sample, y)


def load_refiner(path, contract, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != DETAIL_SCHEMA or not isinstance(payload.get('contract'), dict)
            or normalize_backbone_contract(payload['contract']) != normalize_backbone_contract(contract)):
        raise ValueError('Refiner backbone/cond/sampler contract mismatch; rebuild and validate the cache')
    if payload.get('architecture', {}).get('calibration_version') != 3:
        raise ValueError('Unsupported refiner checkpoint; only calibration_version=3 is supported')
    config = RefinerConfig(**payload['architecture'])
    expected_composition = config.composition_contract
    if payload.get('composition_contract', config.composition_contract) != expected_composition:
        raise ValueError('Refiner composition contract mismatch')
    net = MotionDetailRefiner(config).to(device)
    net.load_state_dict(payload['refiner'])
    net.eval().requires_grad_(False)
    return net, payload


@torch.no_grad()
def refine_generated_batch(sample, y, model, args, output_dir, output_frames=None):
    """Called once after sampling, before the common physical decode/export path."""
    path = getattr(args, 'refiner_path', '')
    args._detail_export_base = None
    alpha = float(getattr(args, 'refiner_alpha', 1.))
    if not path or alpha == 0:
        return sample
    if getattr(args, 'reference_motion', '') or getattr(args, 'outpaint', False) or getattr(args, 'inpaint_joints', '') or getattr(args, 'inpaint_frames', ''):
        raise ValueError('Refinement of reference/edit/inpaint/outpaint requires separate validation')
    # Cache the loaded adapter on this generation's args; never change backbone.
    if not hasattr(args, '_detail_adapter'):
        from sample.generation_runtime import _resolve_generation_cond_path
        contract = backbone_contract(args, _resolve_generation_cond_path(args))
        args._detail_adapter = load_refiner(path, contract, sample.device)[0]
    net = args._detail_adapter
    context = attach_detail_action(model, tree_to(y, sample.device))
    if output_frames is not None:
        if isinstance(output_frames, int):
            output_frames = [output_frames] * len(sample)
        # Exclude frequencies that would alias when the common export resamples
        # the native window down. Both clocks retain the same source duration.
        limits = []
        for b, count in enumerate(output_frames):
            count = int(count)
            t = int(y['lengths'][b])
            source = float(y['resample_speed_cond'][b]) * t
            loop = bool(y['is_loop'][b])
            step = source / count if loop else (source - 1) / max(1, count - 1)
            limits.append(0.5 * net.config.reference_fps / step)
        context['detail_max_hz'] = limits
    physical = decode_batch(sample, context)
    args._detail_export_base = (physical.detach().cpu(), tree_to(context, 'cpu'))
    refined, delta = net(physical, context, seed=args.refiner_seed, alpha=alpha)
    encoded = physical_hml_to_canonical(refined, context)
    # Never introduce canonical round-trip drift into roots, padding or unused channels.
    result = sample.clone()
    for b in range(len(sample)):
        n, t = int(y['n_joints'][b]), int(y['lengths'][b])
        if bool(delta[b, :n, :, :t].count_nonzero()):
            nonroot = [j for j, p in enumerate(y['parents'][b][:n]) if int(p) >= 0]
            result[b, nonroot, :9, :t] = encoded[b, nonroot, :9, :t]
    folder = Path(output_dir) / 'refiner_comparisons'
    folder.mkdir(parents=True, exist_ok=True)
    # The comparison export writes the frozen backbone's motion beside the
    # refined one (see base_export_motion); only the reproduction info is logged.
    index = len(list(folder.glob('batch_*.json')))
    args._detail_comparison_dir = folder
    samples = []
    for b in range(len(sample)):
        n, t = int(y['n_joints'][b]), int(y['lengths'][b])
        window = delta[b, :n, :, :t]
        samples.append({
            'index': b, 'n_joints': n, 'lengths': t, 'is_loop': bool(y['is_loop'][b]),
            'output_frames': (None if output_frames is None else int(
                output_frames[b] if isinstance(output_frames, (list, tuple)) else output_frames)),
            'delta_abs_max_rad': float(window.abs().max()) if window.numel() else 0.0,
        })
    diagnostics = {
        'backbone_seed': args.seed, 'detail_seed': args.refiner_seed, 'alpha': alpha,
        'refiner_path': str(path), 'refiner_sha256': file_hash(path), 'samples': samples,
    }
    (folder / f'batch_{index:05d}.json').write_text(
        json.dumps(diagnostics, indent=2), encoding='utf-8')
    return result


def base_export_motion(args, row):
    """The frozen backbone's decoded motion for one sample, ``[T, N, 9]`` numpy.

    ``None`` when this run does not refine (no base was cached), which is also
    what tells the caller not to write a comparison export.
    """
    cached = getattr(args, '_detail_export_base', None)
    if cached is None:
        return None
    base, y = cached
    n, t = int(y['n_joints'][row]), int(y['lengths'][row])
    return base[row, :n, :, :t].permute(2, 0, 1).numpy()


@torch.no_grad()
def transport_export_positions(motion, args, row, output_frames, window_frames, loop, *, invert=True):
    """Reapply the current-bone contract on the final export sampling grid.

    Temporal interpolation/inversion does not commute with rotation transport.
    Preserve the resampled frozen base's geometry, rather than the intermediate
    window's geometry, while keeping the exported refiner rotations.
    """
    cached = getattr(args, '_detail_export_base', None)
    if cached is None:
        return motion
    from sample.output_lengths import _resample_window_to_output
    from model.motion_detail_refiner import fk, transport_positions
    from utils.rotation_conversions import rotation_6d_to_matrix_safe
    base, y = cached
    n, t = int(y['n_joints'][row]), int(y['lengths'][row])
    before = _resample_window_to_output(base[row, :n, :, :t].permute(2, 0, 1).numpy(),
        output_frames, window_frames, loop, invert=invert)
    before = torch.from_numpy(before.copy())
    after = torch.from_numpy(motion.copy())
    parents = [int(p) for p in y['parents'][row][:n]]
    rest = y['rest_pos_ric_hml'][row, :n].to(before)
    gb, _ = fk(rotation_6d_to_matrix_safe(before[..., 3:9].contiguous()), rest, parents)
    ga, _ = fk(rotation_6d_to_matrix_safe(after[..., 3:9].contiguous()), rest, parents)
    after[..., :3] = transport_positions(before[..., :3], gb, ga, parents)
    for j, p in enumerate(parents):
        if p < 0:
            after[:, j] = before[:, j]
    return after.numpy()
