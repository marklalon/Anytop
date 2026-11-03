"""Preview the exact training selection without initializing a model or GPU."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loaders.truebones.truebones_utils.cond_schema import load_cond
from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags, assert_cond_species_tags_current
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_labels import load_motion_metadata
from data_loaders.truebones.truebones_utils.sample_filter import resolve_sample_filter


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--cond_path', required=True)
    parser.add_argument('--sample_filter', required=True)
    parser.add_argument('--split', choices=['train', 'val', 'all'], default='train')
    parser.add_argument('--min_length', type=int, default=20)
    parser.add_argument('--sample_limit', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=60)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', type=Path, help='Optional JSON report with final clip IDs and metadata')
    args = parser.parse_args()
    opt = get_opt('cpu', args.cond_path)
    cond = load_cond(opt.cond_file)
    metadata = {source.namespace: load_motion_metadata(source.root) for source in opt.sources}
    report = resolve_sample_filter(
        args.sample_filter, cond, opt.sources, metadata,
        {key: dataset_tags().tags_for(key) for key in cond}, split=args.split,
        min_length=args.min_length, sample_limit=args.sample_limit,
        num_frames=args.num_frames, seed=args.seed,
    )
    selected_objects = {r['object_type'] for r in report['clips']}
    assert_cond_species_tags_current({k: v for k, v in cond.items() if k in selected_objects})
    summary = {k: v for k, v in report.items() if k not in ('config', 'clips', 'too_short')}
    summary['trainable_count'] = len(report['clips'])
    summary['too_short_count'] = len(report['too_short'])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report['too_short']:
        print('Clips rejected by min_length:\n' + '\n'.join(report['too_short']))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Report: {args.output}')


if __name__ == '__main__':
    main()
