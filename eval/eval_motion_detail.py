"""Held-out seeds, all-band calibration, minimal correction and geometry checks.

Targets remain train-split empirical statistics. This is seed generalization,
not an estimate of unseen-GT-clip quality or a replacement for visual review.
"""
import argparse
import json
from pathlib import Path

import torch

from model.motion_detail_refiner import compose, sample_context, spectral_project
from sample.detail_refinement import file_hash, load_refiner
from tools.cache_motion_detail import repeat_context
from train.train_motion_detail import load_groups
from train.detail_objectives import detail_features, statistics_loss, empirical_target
from train.detail_metrics import preservation_metrics
from utils.rotation_conversions import rotation_6d_to_matrix_safe


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--refiner', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--detail-seeds', type=int, nargs='+', default=[2**40, 2**40 + 1])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    if len(set(a.detail_seeds)) < 2 or min(a.detail_seeds) < 2**31:
        p.error('Use at least two held-out detail seeds >= 2**31')
    manifest, groups = load_groups(a.cache_dir, a.device)
    net, checkpoint = load_refiner(a.refiner, manifest['contract'], a.device)
    if checkpoint['cache_manifest_sha256'] != file_hash(Path(a.cache_dir) / 'manifest.json'):
        raise ValueError('Evaluation cache differs from checkpoint training/held-out seed contract')
    thresholds_path = Path(a.refiner).with_name('acceptance.json')
    thresholds = json.loads(thresholds_path.read_text(encoding='utf-8'))
    if thresholds.get('calibration_version') != net.config.calibration_version:
        raise ValueError('Acceptance thresholds do not match the refiner calibration version')
    rows = []
    def stat(features, target, scale):
        return statistics_loss(features, target, scale)
    for i, group in enumerate(groups):
        base = group['validation_base']
        y = repeat_context(group['y'], len(base))
        target = empirical_target(detail_features(group['gt'],
            repeat_context(group['y'], len(group['gt'])), net.config),
            group.get('clips'))
        scale = checkpoint['normalization'][i].to(base)
        base_stat = float(stat(detail_features(base, y, net.config), target, scale))
        outputs, seed_rows, matched_rows = [], [], []
        matched_y = repeat_context(group['y'], len(group['gt']))
        for seed in a.detail_seeds:
            matched_out, _ = net(group['gt'], matched_y, seed=seed)
            matched_row = preservation_metrics(matched_out, group['gt'], matched_y, net.config)
            matched_row['stat'] = float(stat(
                empirical_target(detail_features(matched_out, matched_y, net.config),
                    group.get('clips')), target, scale))
            matched_rows.append(matched_row)
            out, delta = net(base, y, seed=seed)
            outputs.append(out)
            row = preservation_metrics(out, base, y, net.config)
            features = detail_features(out, y, net.config)
            row.update({'detail_seed': seed, 'stat': float(stat(features, target, scale)),
                        'max_residual_rad': float(torch.linalg.vector_norm(delta, dim=2).max()),
                        'feature_rms': float(features.square().mean().sqrt())})
            # Random all-band residual with the learned residual's RMS budget.
            random_out = base.clone()
            for b in range(len(base)):
                n, t, loop, dt, parents = sample_context(y, b, net.config)
                gen = torch.Generator(device=base.device).manual_seed(seed + b)
                noise = torch.randn((t, n, 3), device=base.device, generator=gen)
                noise = spectral_project(noise, dt, loop, high=net.config.output_max_hz)
                noise[:, [j for j, p in enumerate(parents) if p < 0]] = 0
                noise *= delta[b, :n, :, :t].square().mean().sqrt() / noise.square().mean().sqrt().clamp_min(1e-9)
                peak = torch.linalg.vector_norm(noise, dim=-1).amax(0, keepdim=True)[..., None]
                noise *= net.config.max_angle_rad / peak.clamp_min(net.config.max_angle_rad)
                x = compose(base[b, :n, :, :t].permute(2, 0, 1), noise,
                            y['rest_pos_ric_hml'][b, :n], parents, net.config.composition)
                random_out[b, :n, :, :t] = x.permute(1, 2, 0)
            row['random_control_stat'] = float(stat(detail_features(random_out, y, net.config), target, scale))
            seed_rows.append(row)
        rotations = [rotation_6d_to_matrix_safe(v[:, :, 3:9].permute(0, 3, 1, 2)) for v in outputs]
        diversity = float((rotations[0] - rotations[1]).square().mean().sqrt())
        metrics = {key: max(r[key] for r in seed_rows) for key in preservation_metrics(outputs[0], base, y, net.config)}
        metrics['max_residual_rad'] = max(r['max_residual_rad'] for r in seed_rows)
        metrics['calibration_stat_ratio'] = max(r['stat'] for r in seed_rows) / max(base_stat, 1e-8)
        checks = {k: value <= thresholds[k] for k, value in metrics.items() if k in thresholds}
        checks['matched_train_inputs_preserved'] = all(
            r['stat'] <= thresholds['matched_stat_increase_max']
            and r['position_change_rms_relative'] <= thresholds['matched_position_change_rms_relative_max']
            and r['local_rotation_change_rms'] <= thresholds['matched_local_rotation_change_rms_max']
            for r in matched_rows)
        matched = base_stat <= thresholds['matched_base_stat_max']
        if matched:
            checks['matched_statistics_preserved'] = max(r['stat'] for r in seed_rows) <= base_stat + thresholds['matched_stat_increase_max']
            checks['matched_minimal_correction'] = (
                metrics['position_change_rms_relative'] <= thresholds['matched_position_change_rms_relative_max']
                and metrics['local_rotation_change_rms'] <= thresholds['matched_local_rotation_change_rms_max'])
        else:
            checks['calibration_statistics_improved'] = metrics['calibration_stat_ratio'] <= thresholds['calibration_stat_ratio_max']
            checks['better_than_random_control'] = all(r['stat'] <= r['random_control_stat'] for r in seed_rows)
        rows.append({'condition': json.loads(group['key']), 'independent_gt_clips': len(set(group['clips'])),
                     'base_stat': base_stat, 'metrics': metrics, 'detail_seed_rotation_rms': diversity,
                     'matched_base': matched, 'matched_train_inputs': matched_rows,
                     'checks': checks, 'passed': all(checks.values()), 'seeds': seed_rows})
    report = {'scope': 'held-out seeds against train-only GT statistics; visual review required',
              'cache_sha256': file_hash(Path(a.cache_dir) / 'manifest.json'),
              'refiner_sha256': file_hash(a.refiner), 'thresholds': thresholds,
              'thresholds_sha256': file_hash(thresholds_path),
              'passed': all(r['passed'] for r in rows), 'groups': rows}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[detail eval] passed={report["passed"]}; report={output}')


if __name__ == '__main__':
    main()
