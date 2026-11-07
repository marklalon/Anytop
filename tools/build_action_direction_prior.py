"""Build checkpoint direction statistics without changing model weights.

Prefer the saved sample_selection.json. --rebuild-from-dataset explicitly opts
into reconstructing eligibility from saved training args and available data;
that data must still represent the corpus used to train the checkpoint.
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.action_direction_prior import (  # noqa: E402
    build_direction_prior, prior_from_dataset, save_direction_prior,
)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--rebuild-from-dataset', action='store_true')
    parser.add_argument('--cond-path', help='Training cond override when its saved path moved')
    options = parser.parse_args(argv)
    directory = Path(options.checkpoint_dir).resolve()
    cond_path = str(Path(options.cond_path).resolve()) if options.cond_path else None
    selection = directory / 'sample_selection.json'
    if selection.is_file():
        snapshot = json.loads(selection.read_text(encoding='utf-8'))
        prior = build_direction_prior(
            (row['clip_id'], row['action_label']) for row in snapshot['clips']
        )
    else:
        if not options.rebuild_from_dataset:
            parser.error('No saved sample_selection.json. To reconstruct from the original '
                         'training data, pass --rebuild-from-dataset.')
        from train.train_motion_diffusion import create_training_data_loader
        from utils.fixseed import fixseed
        args = Namespace(**json.loads((directory / 'args.json').read_text(encoding='utf-8')))
        if cond_path:
            args.cond_path = cond_path
        args.action_label_cond = False
        args.action_conditioning = None
        args.species_descriptor_table = None
        args.num_workers = 0
        args.main_process_prefetch_batches = 0
        os.chdir(REPO_ROOT)
        fixseed(args.seed)
        data = create_training_data_loader(args)
        prior = prior_from_dataset(data.dataset)
    print(save_direction_prior(prior, directory))


if __name__ == '__main__':
    main()
