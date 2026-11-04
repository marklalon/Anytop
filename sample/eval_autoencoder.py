"""Reconstruct explicit preprocessed .npy clips with a motion AE/VAE checkpoint.

The entire clip is resampled to the checkpoint's training window, then the
reconstruction is restored to the input frame count. No dataset filter,
augmentation, diffusion sampling, or raw animation preprocessing is used.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANYTOP_ROOT.parent))
sys.path.insert(0, str(ANYTOP_ROOT))

import numpy as np
import torch

from data_loaders.truebones.data.dataset import ensure_joint_name_embeddings, resample_motion_features
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_FEATURE_SPACE, canonical_to_physical_hml, mark_canonical_cond_entry,
    physical_hml_to_canonical, stored_to_working, working_to_stored,
)
from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key, species_lookup_map
from data_loaders.truebones.truebones_utils.param_utils import FPS
from model.motion_autoencoder import DECODER_CONTENT_VERSION, MotionAutoencoder, to_working_motion
from sample.conditioning import create_condition
from sample.export import _bvh_preview_options, _export_motion
from sample.output_lengths import _resample_window_to_output, reference_loop_verdict
from utils.fullbody_ik import DEFAULT_IK_STRETCH_FACTOR
from utils.misc import infer_object_type_from_filename

SKELETON_KEYS = ('n_joints', 'joints_names_embs', 'rest_geom', 'joint_struct',
                 'graph_dist', 'joints_relations')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True, help='Training autoencoder.pt checkpoint')
    p.add_argument('--npy', dest='motion_paths', nargs='+', required=True,
                   help='Preprocessed stored-layout .npy clips, each shaped (T, J, 12)')
    p.add_argument('--cond_path', help='Explicit skeleton/statistics cond; otherwise use training cond')
    p.add_argument('--object_type', help='Source skeleton key; otherwise infer separately from each filename')
    p.add_argument('--output_dir', help='Defaults to outputs/<run>/eval_autoencoder_step<step>')
    p.add_argument('--loop', choices=('auto', 'on', 'off'), default='auto')
    p.add_argument('--loop_cycles', type=int, default=1, help='Number of cycles in each loop clip')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--amp_dtype', choices=('fp32', 'fp16'), default='fp32')
    p.add_argument('--fullbody_ik', action='store_true', help='Apply full-body IK to BVH previews')
    p.add_argument('--stretch_factor', type=float, default=DEFAULT_IK_STRETCH_FACTOR)
    return p


def resolve_cond_path(checkpoint, model_path, explicit=None):
    if explicit:
        return Path(explicit)
    snapshot = model_path.parent / 'cond.npy'
    training_path = Path(checkpoint['args']['cond_path'])
    candidates = (snapshot, training_path if training_path.is_absolute() else ANYTOP_ROOT / training_path)
    for candidate in candidates:
        if candidate.is_file():
            expected = checkpoint['cond_sha256']
            if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
                raise ValueError(f'{candidate} differs from the training cond. '
                                 'Pass --cond_path explicitly to evaluate with a different cond.')
            return candidate
    raise FileNotFoundError('Training cond not found; pass --cond_path pointing to the training cond.npy')


def load_motion(path, cond):
    if path.suffix.lower() != '.npy':
        raise ValueError(f'Only preprocessed .npy motion is supported: {path}')
    motion = np.load(path, allow_pickle=False)
    expected_joints = len(cond['parents'])
    if motion.ndim != 3 or motion.shape[0] < 1 or motion.shape[1:] != (expected_joints, 12):
        raise ValueError(f'{path}: expected (T, {expected_joints}, 12) with T >= 1, got {motion.shape}')
    if not np.issubdtype(motion.dtype, np.number) or not np.isfinite(motion).all():
        raise ValueError(f'{path}: motion must contain finite numeric features')
    if int(cond.get('translation_root_index', 0)) != 0:
        raise ValueError('Motion must use the canonical joint-zero translation root; reprocess the source')
    return motion


@torch.inference_mode()
def reconstruct(model, stored, object_type, cond, *, frames, is_loop, cycles, device, amp_dtype):
    """Use the same static conditions and 9-channel canonical input as training."""
    working = stored_to_working(stored)
    window = resample_motion_features(working, frames, periodic=is_loop)
    canonical = physical_hml_to_canonical(window, cond)
    target = torch.from_numpy(np.asarray(canonical[..., :9], dtype=np.float32)).permute(1, 2, 0)[None].to(device)
    _, kwargs = create_condition([object_type], {object_type: cond}, frames,
                                 max_joints=stored.shape[1], feature_len=12,
                                 loop=is_loop, loop_cycles=cycles)
    y = {key: value.to(device) if torch.is_tensor(value) else value for key, value in kwargs['y'].items()}
    skeleton = {key: y[key] for key in SKELETON_KEYS}
    frame = torch.ones((1, frames), dtype=torch.bool, device=device)
    loop = {key: y[key] for key in ('is_loop', 'loop_tile_count')}
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_dtype == 'fp16'):
        # encode defaults to sample=False, including for a variational checkpoint.
        latent = model.encode(target, skeleton, frame, loop)
        prediction = model.decode(latent, skeleton, frame, loop)
    prediction = prediction.float()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Autoencoder produced nonfinite features')
    mse = float((prediction - target).square().mean())
    physical = canonical_to_physical_hml(to_working_motion(prediction), cond)[0].permute(2, 0, 1).cpu().numpy()
    physical = _resample_window_to_output(physical, len(stored), frames, is_loop)
    return working_to_stored(physical, is_loop=is_loop), mse


def main(argv=None):
    args = parser().parse_args(argv)
    if args.loop_cycles < 1:
        raise ValueError('--loop_cycles must be positive')
    device = torch.device('cuda:0' if args.device == '0' else args.device)
    if args.amp_dtype == 'fp16' and device.type != 'cuda':
        raise ValueError('--amp_dtype fp16 requires CUDA')
    torch.set_num_threads(4)
    model_path = Path(args.model_path).resolve()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if checkpoint['kind'] != 'motion_autoencoder_7a':
        raise ValueError('Expected a train_motion_autoencoder.py checkpoint')
    if checkpoint['canonical_version'] != CANONICAL_FEATURE_SPACE:
        raise ValueError('Checkpoint canonical feature version does not match this code')
    if checkpoint['decoder_content_version'] != DECODER_CONTENT_VERSION:
        raise ValueError('Checkpoint decoder content version does not match this code')
    model = MotionAutoencoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    frames = int(checkpoint['args']['num_frames'])
    cond_path = resolve_cond_path(checkpoint, model_path, args.cond_path)
    cond_dict = load_cond(str(cond_path))
    # Validate every input before writing any output; never overwrite a source clip.
    jobs = []
    stems = set()
    out = Path(args.output_dir) if args.output_dir else (
        ANYTOP_ROOT / 'outputs' / model_path.parent.name / f'eval_autoencoder_step{checkpoint["step"]}')
    for value in args.motion_paths:
        path = Path(value).resolve()
        object_type = (resolve_species_key(cond_dict, args.object_type) if args.object_type
                       else infer_object_type_from_filename(str(path), valid_types=species_lookup_map(cond_dict)))
        if object_type is None:
            raise ValueError(f'Cannot resolve skeleton for {path}; pass --object_type with a key from {cond_path}')
        cond = cond_dict[object_type]
        mark_canonical_cond_entry(cond)
        ensure_joint_name_embeddings({object_type: cond}, expected_embedding_dim=model.config['t5_out_dim'],
                                     cond_source=str(cond_path))
        stored = load_motion(path, cond)
        is_loop = (reference_loop_verdict(stored, 0)[0] if args.loop == 'auto' else args.loop == 'on')
        if path.stem in stems:
            raise ValueError(f'Duplicate input filename stem: {path.stem}; use separate output directories')
        stems.add(path.stem)
        for label in ('original', 'reconstruction'):
            if (out / f'{path.stem}_{label}.npy').resolve() == path:
                raise ValueError('Output would overwrite the input motion; choose another --output_dir')
        jobs.append((path, object_type, cond, stored, is_loop))
    out.mkdir(parents=True, exist_ok=True)
    print(f'[AE] checkpoint={model_path}, step={checkpoint["step"]}, window={frames}, cond={cond_path}')
    records = []
    for path, object_type, cond, stored, is_loop in jobs:
        reconstruction, mse = reconstruct(model, stored, object_type, cond, frames=frames,
                                           is_loop=is_loop, cycles=args.loop_cycles,
                                           device=device, amp_dtype=args.amp_dtype)
        names = list(cond.get('canonical_bvh_joint_names', cond['joints_names']))
        for label, motion in (('original', stored), ('reconstruction', reconstruction)):
            _export_motion((motion, cond, f'{path.stem}_{label}.npy', names,
                            str(out), FPS, _bvh_preview_options(args)))
        records.append(dict(motion_path=str(path), object_type=object_type, frames=len(stored),
                            window_frames=frames, is_loop=is_loop, loop_cycles=args.loop_cycles,
                            canonical_window_mse=mse))
        print(f'[AE] {path.name}: {len(stored)} -> {frames} -> {len(stored)} frames, '
              f'loop={is_loop}, canonical MSE={mse:.6g}')
    report = dict(model_path=str(model_path), step=checkpoint['step'], cond_path=str(cond_path),
                  latent='posterior mean' if model.variational else 'deterministic', samples=records)
    (out / 'reconstruction_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[AE] Saved original/reconstruction .npy and .bvh pairs to {out.resolve()}')


if __name__ == '__main__':
    main()
