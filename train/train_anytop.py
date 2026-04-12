# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on motions.
"""
import sys
import os
import json
import re
import shutil

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion_general_skeleton
from utils.ml_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform #required


def find_latest_checkpoint(save_dir, prefix='model'):
    if not save_dir or not os.path.isdir(save_dir):
        return ''
    candidates = []
    for file_name in os.listdir(save_dir):
        match = re.fullmatch(rf'{re.escape(prefix)}(\d+)\.pt', file_name)
        if match:
            candidates.append((int(match.group(1)), os.path.join(save_dir, file_name)))
    if not candidates:
        return ''
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def clear_training_artifacts(save_dir):
    if not os.path.isdir(save_dir):
        return
    for file_name in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file_name)
        if re.fullmatch(r'model\d+\.pt', file_name) or re.fullmatch(r'opt\d+\.pt', file_name):
            os.remove(file_path)
            continue
        if file_name == 'args.json':
            os.remove(file_path)
            continue
        if file_name.startswith('model') and file_name.endswith('.pt.samples') and os.path.isdir(file_path):
            shutil.rmtree(file_path)

def prepare_save_dir(args):
    save_dir = args.save_dir
    if save_dir is None:
        save_root = os.path.join(os.getcwd(), 'save')
        os.makedirs(save_root, exist_ok=True)
        prefix = "AnyTop"
        if args.model_prefix is not None:
            prefix = args.model_prefix
        model_name = f'{prefix}_dataset_truebones_bs_{args.batch_size}_latentdim_{args.latent_dim}'
        save_dir = os.path.join(save_root, model_name)
        args.save_dir = save_dir

    if save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    os.makedirs(save_dir, exist_ok=True)

    if args.auto_resume:
        if not getattr(args, 'resume_checkpoint', ''):
            latest_checkpoint = find_latest_checkpoint(save_dir, prefix='model')
            if not latest_checkpoint:
                print(f'[INFO] auto_resume was requested but no checkpoint was found in save_dir [{save_dir}]. Starting fresh training.')
                args.resume_checkpoint = ''
                clear_training_artifacts(save_dir)
            else:
                args.resume_checkpoint = latest_checkpoint
                if not getattr(args, 'load_optimizer_state', False):
                    args.load_optimizer_state = True
                print(f'[INFO] Auto-resuming AnyTop from {args.resume_checkpoint}')
        else:
            if not getattr(args, 'load_optimizer_state', False):
                args.load_optimizer_state = True
            print(f'[INFO] Auto-resuming AnyTop from {args.resume_checkpoint}')
    elif not getattr(args, 'resume_checkpoint', ''):
        args.resume_checkpoint = ''
        clear_training_artifacts(save_dir)
    return save_dir

def create_training_data_loader(args):
    return get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        split=getattr(args, 'train_split', 'train'),
        temporal_window=args.temporal_window,
        t5_name=args.t5_name,
        balanced=args.balanced,
        objects_subset=args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        sample_limit=args.sample_limit,
        use_reference_conditioning=getattr(args, 'use_reference_conditioning', True),
        motion_name_keywords=getattr(args, 'motion_name_keywords', ''),
    )

def run_training(args):
    fixseed(
        args.seed,
        cudnn_benchmark=getattr(args, 'cudnn_benchmark', True),
        allow_tf32=getattr(args, 'allow_tf32', True),
    )
    save_dir = prepare_save_dir(args)
    args.checkpoint_step_numbering = 'completed_steps'

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=args.save_dir)
    ml_platform.report_args(args, name='Args')

    args_path = os.path.join(save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    data = create_training_data_loader(args)

    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())
    ml_platform.watch_model(model)
    print("Training...")
    TrainLoop(args, ml_platform, model, diffusion, data).run_loop()
    ml_platform.close()

def main():
    args = train_args()
    run_training(args)

if __name__ == "__main__":
    main()
