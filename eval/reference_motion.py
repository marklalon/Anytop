"""Capacity audit and reproducible reference-swap / species-FiLM ablations.

python -m eval.reference_motion capacity --cond_path dataset/cond.npy --output outputs/reference_motion_capacity.json
python -m eval.reference_motion evaluate --model_path save/ref/model000049999.pt --tasks eval/reference_tasks.json --output outputs/reference_eval
"""
import argparse
import json
from pathlib import Path

import numpy as np


QUANTILES = (0, 10, 50, 90, 100)
CANDIDATES = ((2, 8), (4, 8), (4, 16), (8, 32))


def capacity_distribution(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if not len(counts) or np.any(counts < 1):
        raise ValueError('Capacity audit needs positive valid joint counts')
    return {
        'samples': len(counts),
        'joint_quantiles': np.percentile(counts, QUANTILES).tolist(),
        'input_dim_quantiles': np.percentile(9 * counts + 2, QUANTILES).tolist(),
        'candidates': {f'{k}x{d}': {
            'latent_dim_per_frame': k * d,
            'ratio_quantiles': np.percentile(k * d / (9 * counts + 2), QUANTILES).tolist(),
            'max_ratio': float((k * d / (9 * counts + 2)).max()),
            'uncompressed_samples': int((k * d >= 9 * counts + 2).sum()),
        } for k, d in CANDIDATES},
    }


def audit_capacity(cond_path):
    from data_loaders.truebones.data.dataset import Truebones, EmptySplitError
    report = {'cond_path': str(cond_path), 'quantiles': list(QUANTILES), 'splits': {}}
    for split in ('train', 'val'):
        try:
            dataset = Truebones(split=split, num_frames=60, objects_subset='all',
                                cond_path=cond_path).motion_dataset
        except EmptySplitError as exc:
            report['splits'][split] = {'samples': 0, 'unavailable': str(exc)}
            continue
        counts = dataset.sampler_index_joint_counts()
        report['splits'][split] = capacity_distribution(counts)
        # Also report each skeleton once, so prolific source clips cannot hide
        # a tiny rig that fails the compression assumption.
        species = {dataset.data_dict[name]['object_type'] for name in dataset.name_list}
        report['splits'][split]['skeletons'] = capacity_distribution([
            len(dataset.cond_dict[key]['parents']) for key in sorted(species)])
        report['splits'][split]['weak_compression_at_default'] = [
            {'species': key, 'valid_joints': len(dataset.cond_dict[key]['parents']),
             'ratio': 16 / (9 * len(dataset.cond_dict[key]['parents']) + 2)}
            for key in sorted(species) if 16 / (9 * len(dataset.cond_dict[key]['parents']) + 2) > .5
        ]
    return report


def motion_curves(features, cond, fps=30, torso=None, samples=60):
    """Skeleton-independent scalar curves in physical HML space.

    Torso endpoints are annotated per skeleton, never a source/target joint map.
    No heuristic angle is reported when endpoints have not been annotated.
    """
    from data_loaders.truebones.truebones_utils.canonical_features import _length_scale_from_rest
    from data_loaders.truebones.truebones_utils.features import recover_from_bvh_ric_np
    features = np.asarray(features)
    if features.ndim != 3 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError('Metrics need finite physical features with at least two frames')
    root = int(cond['translation_root_index'])
    length = float(_length_scale_from_rest(cond['rest_pos_ric_hml']))
    positions = recover_from_bvh_ric_np(features, translation_root_index=root)
    root_speed = np.linalg.norm(features[:, root, [9, 11]], axis=-1) * fps / length
    velocity = np.diff(positions, axis=0) * fps / length
    energy = np.mean(np.sum(velocity ** 2, axis=-1), axis=1)
    curves = {'root_speed': root_speed, 'motion_energy': energy}
    if torso is not None:
        base, tip = map(int, torso)
        direction = positions[:, tip] - positions[:, base]
        curves['torso_pitch'] = np.arctan2(direction[:, 1], np.linalg.norm(direction[:, [0, 2]], axis=-1))
    return {key: np.interp(np.linspace(0, 1, samples), np.linspace(0, 1, len(v)), v)
            for key, v in curves.items()}


def compare_curves(source, target):
    result = {}
    for key in source.keys() & target.keys():
        a, b = source[key], target[key]
        corr = float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-8 else None
        result[key] = {'rmse': float(np.sqrt(np.mean((a - b) ** 2))), 'correlation': corr}
    return result


def evaluate(args):
    from sample.generate import main as generate
    from sample.generation_runtime import prepare_generation_runtime
    from utils.parser_util import generate_args
    from data_loaders.truebones.truebones_utils.dataset_sources import resolve_species_key
    from eval.motion_quality.quality import MotionQualityScorer
    from data_loaders.truebones.data.dataset import resample_motion_features
    from data_loaders.truebones.truebones_utils.cond_schema import load_cond
    tasks = json.loads(Path(args.tasks).read_text(encoding='utf-8'))['tasks']
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = None
    scorer = MotionQualityScorer() if args.score else None
    source_conds = load_cond(str(Path(args.model_path).resolve().parent / 'cond.npy'))
    results = []
    for task in tasks:
        reference = np.load(task['reference_motion'])
        frames = int(task.get('num_frames', len(reference)))
        for species_active in (True, False):
            for use_ref in (True, False):
                for seed in task.get('seeds', [10]):
                    name = f"{task['id']}_species{int(species_active)}_ref{int(use_ref)}_seed{seed}"
                    folder = output / name
                    # Never overwrite a prior run or score old clips as new ones.
                    if folder.exists():
                        raise FileExistsError(f'Evaluation directory already exists: {folder}')
                    argv = ['--model_path', args.model_path, '--output_dir', str(folder),
                            '--object_type', task['target_type'], '--batch_size', '1',
                            '--seed', str(seed), '--num_frames', str(frames),
                            '--loop', task.get('loop', 'off'), '--species_active', str(species_active).lower(),
                            '--sampling_method', 'ddim', '--sampling_steps', str(args.sampling_steps)]
                    if task.get('action_label'):
                        argv += ['--action_label', task['action_label']]
                    if task.get('cond_path'):
                        argv += ['--cond_path', task['cond_path']]
                    if use_ref:
                        argv += ['--reference_mode', 'encode', '--reference_motion', task['reference_motion'],
                                 '--reference_source_type', task['source_type'],
                                 '--ref_cfg_scale', str(args.ref_cfg_scale)]
                    options = generate_args(argv)
                    if runtime is None:
                        runtime = prepare_generation_runtime(options)
                    generate(options, runtime=runtime)
                    target = resolve_species_key(runtime.cond_dict, task['target_type'])
                    metric_conds = {**source_conds, **runtime.cond_dict}
                    source = resolve_species_key(metric_conds, task['source_type'])
                    clips = sorted(p for p in folder.glob('*.npy') if not p.name.startswith('_'))
                    if len(clips) != 1:
                        raise ValueError(f'Expected exactly one generated clip: {folder}')
                    generated = np.load(clips[0])
                    metrics = compare_curves(
                        motion_curves(resample_motion_features(reference, frames),
                                      metric_conds[source], torso=task.get('source_torso')),
                        motion_curves(generated, runtime.cond_dict[target], torso=task.get('target_torso')))
                    row = dict(name=name, task=task, reference_active=use_ref, species_active=species_active,
                               metrics=metrics, output=str(clips[0]))
                    if scorer:
                        scorer.register_cond({target: runtime.cond_dict[target]})
                        row['quality'] = scorer.evaluate([generated], target, task.get('eval_label', task.get('action_label', '')),
                                                         names=[str(clips[0])]).as_dict()
                    results.append(row)
                    (output / 'report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    capacity = sub.add_parser('capacity')
    capacity.add_argument('--cond_path', required=True)
    capacity.add_argument('--output', required=True)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--model_path', required=True)
    evaluation.add_argument('--tasks', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--sampling_steps', type=int, default=50)
    evaluation.add_argument('--ref_cfg_scale', type=float, default=1)
    evaluation.add_argument('--score', action='store_true', help='Include MotionQualityScorer plausibility / prior')
    args = parser.parse_args()
    if args.command == 'capacity':
        report = audit_capacity(args.cond_path)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
