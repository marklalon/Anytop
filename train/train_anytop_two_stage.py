import json
import os
import sys
from copy import deepcopy


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train_anytop import run_training
from utils.parser_util import train_two_stage_args


def _resolve_stage_save_dir(args, stage_name, default_suffix):
    explicit_dir = getattr(args, f'{stage_name}_save_dir', '')
    if explicit_dir:
        return explicit_dir
    if args.experiment_root:
        return os.path.join(args.experiment_root, default_suffix)
    save_root = os.path.join(os.getcwd(), 'save')
    prefix = args.model_prefix or 'AnyTopTwoStage'
    return os.path.join(save_root, f'{prefix}_{default_suffix}')


def _find_latest_model_checkpoint(save_dir):
    if not save_dir or not os.path.isdir(save_dir):
        return ''
    candidates = []
    for file_name in os.listdir(save_dir):
        if not (file_name.startswith('model') and file_name.endswith('.pt')):
            continue
        step_text = file_name[len('model'):-len('.pt')]
        if not step_text.isdigit():
            continue
        candidates.append((int(step_text), os.path.join(save_dir, file_name)))
    if not candidates:
        return ''
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _apply_stage_override(stage_args, field_name, override_value, sentinel):
    if override_value == sentinel:
        return
    setattr(stage_args, field_name, override_value)


def _build_stage_args(base_args, stage_name, save_dir):
    stage_args = deepcopy(base_args)
    stage_args.save_dir = save_dir
    stage_args.load_optimizer_state = True

    _apply_stage_override(stage_args, 'num_steps', getattr(base_args, f'{stage_name}_num_steps'), -1)
    _apply_stage_override(stage_args, 'batch_size', getattr(base_args, f'{stage_name}_batch_size'), -1)
    _apply_stage_override(stage_args, 'lr', getattr(base_args, f'{stage_name}_lr'), -1.0)
    _apply_stage_override(stage_args, 'sample_limit', getattr(base_args, f'{stage_name}_sample_limit'), -1)

    if stage_name == 'stage1':
        stage_args.train_split = 'all'
        stage_args.disable_reference_branch = True
        stage_args.use_reference_conditioning = False
        stage_args.lambda_confidence_recon = 0.0
        stage_args.lambda_repair_recon = 0.0
        if base_args.stage1_resume_checkpoint:
            stage_args.resume_checkpoint = base_args.stage1_resume_checkpoint
    else:
        stage_args.train_split = 'train'
        stage_args.disable_reference_branch = False
        stage_args.use_reference_conditioning = True
        stage_args.load_optimizer_state = bool(base_args.stage2_load_optimizer_state)
        if base_args.stage2_resume_checkpoint:
            stage_args.resume_checkpoint = base_args.stage2_resume_checkpoint

    return stage_args


def _write_manifest(base_args, stage1_args, stage2_args, stage1_checkpoint, stage2_checkpoint):
    if not base_args.experiment_root:
        return
    os.makedirs(base_args.experiment_root, exist_ok=True)
    manifest = {
        'run_stage': base_args.run_stage,
        'stage1': {
            'save_dir': getattr(stage1_args, 'save_dir', ''),
            'resume_checkpoint': getattr(stage1_args, 'resume_checkpoint', ''),
            'final_checkpoint': stage1_checkpoint,
        },
        'stage2': {
            'save_dir': getattr(stage2_args, 'save_dir', ''),
            'resume_checkpoint': getattr(stage2_args, 'resume_checkpoint', ''),
            'final_checkpoint': stage2_checkpoint,
            'load_optimizer_state': getattr(stage2_args, 'load_optimizer_state', False),
        },
    }
    manifest_path = os.path.join(base_args.experiment_root, 'two_stage_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main():
    args = train_two_stage_args()
    stage1_args = None
    stage2_args = None
    stage1_checkpoint = ''
    stage2_checkpoint = ''

    if args.run_stage in ('stage1', 'both'):
        stage1_save_dir = _resolve_stage_save_dir(args, 'stage1', 'stage1_pretrain')
        stage1_args = _build_stage_args(args, 'stage1', stage1_save_dir)
        run_training(stage1_args)
        stage1_checkpoint = _find_latest_model_checkpoint(stage1_save_dir)

    if args.run_stage in ('stage2', 'both'):
        stage2_save_dir = _resolve_stage_save_dir(args, 'stage2', 'stage2_repair')
        stage2_args = _build_stage_args(args, 'stage2', stage2_save_dir)
        if not getattr(stage2_args, 'resume_checkpoint', ''):
            if args.run_stage == 'stage2':
                stage1_save_dir = _resolve_stage_save_dir(args, 'stage1', 'stage1_pretrain')
                stage2_args.resume_checkpoint = _find_latest_model_checkpoint(stage1_save_dir)
            else:
                stage2_args.resume_checkpoint = stage1_checkpoint
        if not stage2_args.resume_checkpoint:
            raise FileNotFoundError('Stage 2 requires a checkpoint. Provide --stage2_resume_checkpoint or point stage1_save_dir to an existing stage1 run.')
        run_training(stage2_args)
        stage2_checkpoint = _find_latest_model_checkpoint(stage2_save_dir)

    _write_manifest(args, stage1_args, stage2_args, stage1_checkpoint, stage2_checkpoint)


if __name__ == '__main__':
    main()