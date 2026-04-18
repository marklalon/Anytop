import functools
import os
import re
import time
import json
import numpy as np
from os.path import join as pjoin
from typing import Optional
import blobfile as bf
import torch
from torch.optim import AdamW
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer, format_nonfinite_stats, format_optimizer_slot_max, inspect_optimizer_slot_max, inspect_optimizer_state, sanitize_optimizer_state
from diffusion.nn import update_ema
from diffusion.resample import LossAwareSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from sample.generate import main as generate
import copy
from utils.model_util import load_model
import random
from data_loaders.get_data import get_dataset_loader

INITIAL_LOG_LOSS_SCALE = 20.0
EXP_AVG_SQ_CHECKPOINT_ALERT_THRESHOLD = 1e20

class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.model_avg = copy.deepcopy(model) if args.use_ema else None
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.eval_interval = getattr(args, 'eval_interval', 0)
        self.resume_checkpoint = args.resume_checkpoint
        self.amp_dtype = getattr(args, 'amp_dtype', 'fp32').lower()
        self.amp_enabled = self.amp_dtype in {'fp16', 'bf16'}
        self.use_fp16 = self.amp_dtype == 'fp16'
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        data_length = len(self.data)
        if data_length <= 0:
            dataset_length = None
            if hasattr(self.data, 'dataset'):
                dataset_length = len(self.data.dataset)
            raise ValueError(
                f"Training DataLoader is empty (loader_len={data_length}, dataset_len={dataset_length}, batch_size={self.batch_size}). "
                "This usually means the dataset has fewer effective samples than one full batch. "
                "For single-motion training on a long clip, enable --fixed_motion_random_crop so one fixed motion can provide many random windows per epoch."
            )
        self.num_epochs = self.num_steps // data_length + 1

        self.sync_cuda = torch.cuda.is_available()
        self.save_dir = args.save_dir
        self.auto_resume = getattr(args, 'auto_resume', False)

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())
        if self.amp_enabled and self.device.type != 'cuda':
            raise ValueError('AMP requires CUDA. Set --amp_dtype fp32 when training on CPU.')
        self.non_blocking = self.device.type == 'cuda'
        self.detect_anomaly = bool(getattr(self.args, 'detect_anomaly', False))
        self.load_optimizer_state = bool(getattr(self.args, 'load_optimizer_state', True))
        self.autocast_dtype = None
        if self.amp_dtype == 'fp16':
            self.autocast_dtype = torch.float16
        elif self.amp_dtype == 'bf16':
            self.autocast_dtype = torch.bfloat16

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False,
            amp_dtype=self.amp_dtype,
            amp_enabled=self.amp_enabled,
            device_type=self.device.type,
            fp16_scale_growth=self.fp16_scale_growth,
        )
        
        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_param_names = {id(param): name for name, param in self.model.named_parameters()}
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 
                                                step_size = 10000, 
                                                gamma = 0.99)
        
        if self.resume_step and bool(getattr(self.args, 'load_optimizer_state', True)):
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if self.args.eval_during_training:
            self.eval_data = get_dataset_loader(
                batch_size=self.args.eval_batch_size,
                num_frames=self.args.num_frames,
                split=self.args.eval_split,
                temporal_window=self.args.temporal_window,
                t5_name=self.args.t5_name,
                balanced=False,
                objects_subset=self.args.objects_subset,
                num_workers=self.args.num_workers,
                prefetch_factor=self.args.prefetch_factor,
                sample_limit=self.args.sample_limit,
                shuffle=False,
                drop_last=getattr(self.args, 'drop_last', False),
                action_tags=getattr(self.args, 'action_tags', ''),
                motion_cache_size=getattr(self.args, 'motion_cache_size', 0),
                main_process_prefetch_batches=getattr(self.args, 'main_process_prefetch_batches', 0),
                fixed_motion=getattr(self.args, 'fixed_motion', ''),
                fixed_window_start=getattr(self.args, 'fixed_window_start', 0),
            )
        self.use_ddp = False
        self.ddp_model = self.model
        self.forward_model = self.ddp_model
        self._interval_loss_sums = {}
        self._interval_loss_counts = {}

    def _load_and_sync_parameters(self):
        self.resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if self.resume_checkpoint:
            checkpoint_number = parse_checkpoint_number_from_filename(self.resume_checkpoint)
            numbering_mode = self._get_checkpoint_step_numbering(self.resume_checkpoint)
            if numbering_mode == 'completed_steps':
                self.resume_step = max(checkpoint_number - 1, 0)
            else:
                self.resume_step = checkpoint_number
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                self.resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model(self.model, state_dict)
                if self.model_avg is not None:
                    load_model(self.model_avg, state_dict_avg)
            else:
                load_model(self.model, state_dict)
                if self.model_avg is not None:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

    def _load_optimizer_state(self):
        opt_checkpoint = self.find_resume_opt_checkpoint()
        if not opt_checkpoint or not os.path.exists(opt_checkpoint):
            logger.log("optimizer checkpoint not found; skipping optimizer state restore")
            return
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        checkpoint_data = dist_util.load_state_dict(
            opt_checkpoint, map_location=dist_util.dev()
        )
        
        # Handle both new and old checkpoint formats
        if isinstance(checkpoint_data, dict) and 'opt' in checkpoint_data:
            state_dict = checkpoint_data['opt']
        else:
            state_dict = checkpoint_data
        
        # Load AMP scaler if available
        if self.amp_enabled and isinstance(checkpoint_data, dict) and 'scaler' in checkpoint_data:
            if self.mp_trainer.scaler.is_enabled():
                self.mp_trainer.scaler.load_state_dict(checkpoint_data['scaler'])
        elif self.use_fp16 and isinstance(checkpoint_data, dict) and 'scaler' in checkpoint_data:
            print("scaler state found, loading it.")
            self.mp_trainer.scaler.load_state_dict(checkpoint_data['scaler'])

        # Load optimizer state WITHOUT overriding LR
        logger.log(f"loading optimizer state from {opt_checkpoint}")
        try:
            self.opt.load_state_dict(state_dict)
        except ValueError as exc:
            logger.log(f"optimizer state restore skipped: {exc}")
            return
        logger.log("optimizer state restored successfully")
        optimizer_state_stats = sanitize_optimizer_state(self.opt)
        if optimizer_state_stats['found']:
            logger.log(
                "Sanitized non-finite optimizer state after restore "
                f"({format_nonfinite_stats(optimizer_state_stats)})"
            )
        
        # Restore LR scheduler state to continue from the correct step
        if isinstance(checkpoint_data, dict) and 'scheduler' in checkpoint_data:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_data['scheduler'])
                logger.log("LR scheduler state restored")
            except Exception as exc:
                logger.log(f"LR scheduler state restore skipped: {exc}")
        elif self.resume_checkpoint:
            try:
                checkpoint_number = parse_checkpoint_number_from_filename(self.resume_checkpoint)
                numbering_mode = self._get_checkpoint_step_numbering(self.resume_checkpoint)
                if numbering_mode == 'completed_steps':
                    inferred_last_epoch = checkpoint_number
                else:
                    inferred_last_epoch = max(checkpoint_number, 0)
                self.lr_scheduler.last_epoch = inferred_last_epoch
                self.lr_scheduler._step_count = inferred_last_epoch + 1
                self.lr_scheduler._last_lr = [group['lr'] for group in self.opt.param_groups]
                logger.log(f"LR scheduler state inferred from resume checkpoint step {inferred_last_epoch}")
            except Exception as exc:
                logger.log(f"LR scheduler inference skipped: {exc}")
        
        self._restore_rng_states(checkpoint_data)

    def run_loop(self):
        tqdm.write(f'train steps: {self.num_steps}')
        while self.total_step() < self.num_steps:
            tqdm.write(f'Starting a new epoch at step {self.total_step()}')
            data_iter = iter(tqdm(self.data))
            while True:
                try:
                    motion, cond = next(data_iter)
                except StopIteration:
                    break

                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                motion = self._move_batch_to_device(motion)
                cond = self._move_cond_to_device(cond)

                self.run_step(motion, cond)

                completed_step = self.total_step() + 1

                if completed_step % self.log_interval == 0:
                    self._assert_optimizer_state_finite(completed_step)
                    interval_loss_metrics = self._flush_interval_loss_metrics()
                    logger_metrics = logger.get_current().dumpkvs().items()
                    for k, v in [*interval_loss_metrics.items(), *logger_metrics]:
                        if k == 'loss':
                            tqdm.write('step[{}]: loss[{:0.5f}]'.format(completed_step, v))
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=completed_step, group_name='Loss')

                if self._should_validate(completed_step):
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                if self._should_save(completed_step):
                    self.save(completed_step)

                    self.model.eval()
                    self.generate_during_training(completed_step)
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.step += 1

                if completed_step == self.num_steps:
                    break

            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break

    def _move_batch_to_device(self, batch):
        return batch.to(self.device, non_blocking=self.non_blocking)



    def _move_cond_to_device(self, cond):
        return {
            'y': {
                key: val.to(self.device, non_blocking=self.non_blocking) if torch.is_tensor(val) else val
                for key, val in cond['y'].items()
            }
        }

    def _with_train_step(self, cond, train_step):
        updated = {'y': dict(cond['y'])}
        updated['train_step'] = int(train_step)
        return updated

    def _should_save(self, completed_step):
        return completed_step % self.save_interval == 0 or completed_step == self.num_steps

    def _should_validate(self, completed_step):
        if not self.args.eval_during_training or self.eval_data is None or self.eval_interval <= 0:
            return False
        return completed_step % self.eval_interval == 0 or completed_step == self.num_steps

    def _restore_rng_states(self, checkpoint_data):
        if not isinstance(checkpoint_data, dict):
            return

        restored = []
        errors = []

        torch_rng_state = checkpoint_data.get('torch_rng_state')
        if torch_rng_state is not None:
            try:
                if torch.is_tensor(torch_rng_state):
                    torch_rng_state = torch_rng_state.detach().to(device='cpu', dtype=torch.uint8)
                torch.set_rng_state(torch_rng_state)
                restored.append('torch')
            except Exception as exc:
                errors.append(f"torch={exc}")

        cuda_rng_state = checkpoint_data.get('cuda_rng_state')
        if torch.cuda.is_available() and cuda_rng_state is not None:
            try:
                normalized_cuda_rng_state = []
                for state in cuda_rng_state:
                    if torch.is_tensor(state):
                        state = state.detach().to(device='cpu', dtype=torch.uint8)
                    normalized_cuda_rng_state.append(state)
                torch.cuda.set_rng_state_all(normalized_cuda_rng_state)
                restored.append('cuda')
            except Exception as exc:
                errors.append(f"cuda={exc}")

        if 'python_rng_state' in checkpoint_data:
            try:
                random.setstate(checkpoint_data['python_rng_state'])
                restored.append('python')
            except Exception as exc:
                errors.append(f"python={exc}")

        if 'numpy_rng_state' in checkpoint_data:
            try:
                np.random.set_state(checkpoint_data['numpy_rng_state'])
                restored.append('numpy')
            except Exception as exc:
                errors.append(f"numpy={exc}")

        if restored:
            logger.log(
                'RNG states restored for reproducible data shuffling '
                f"({', '.join(restored)})"
            )
        if errors:
            logger.log(f"RNG state restore skipped for some entries: {'; '.join(errors)}")

    def _monitor_checkpoint_optimizer_state(self, completed_step):
        slot_stats = inspect_optimizer_slot_max(
            self.opt,
            'exp_avg_sq',
            param_name_lookup=self._optimizer_param_names,
        )
        if not slot_stats['found']:
            return None

        max_abs = float(slot_stats['max_abs'])
        self.train_platform.report_scalar(
            name='exp_avg_sq_absmax',
            value=max_abs,
            iteration=completed_step,
            group_name='Optimizer',
        )
        logger.log(
            f"Checkpoint optimizer monitor at step {completed_step}: "
            f"{format_optimizer_slot_max(slot_stats)}"
        )
        if max_abs > EXP_AVG_SQ_CHECKPOINT_ALERT_THRESHOLD:
            return (
                'Detected abnormal Adam exp_avg_sq growth at checkpoint step '
                f"{completed_step} ({format_optimizer_slot_max(slot_stats)}; "
                f"threshold={EXP_AVG_SQ_CHECKPOINT_ALERT_THRESHOLD:.1e})"
            )
        return None

    def _assert_optimizer_state_finite(self, completed_step):
        state_stats = inspect_optimizer_state(self.opt)
        if state_stats['found']:
            raise RuntimeError(
                'Detected non-finite optimizer state at '
                f'step {completed_step} ({format_nonfinite_stats(state_stats)})'
            )

    def _accumulate_interval_losses(self, losses):
        for key, value in losses.items():
            if not torch.is_tensor(value):
                continue
            mean_value = value.detach().float().mean()
            if key in self._interval_loss_sums:
                self._interval_loss_sums[key] = self._interval_loss_sums[key] + mean_value
                self._interval_loss_counts[key] += 1
            else:
                self._interval_loss_sums[key] = mean_value.clone()
                self._interval_loss_counts[key] = 1

    def _flush_interval_loss_metrics(self):
        metrics = {}
        for key, total in self._interval_loss_sums.items():
            count = max(self._interval_loss_counts.get(key, 1), 1)
            metrics[key] = float((total / count).item())
        self._interval_loss_sums.clear()
        self._interval_loss_counts.clear()
        return metrics

    def _compute_eval_losses(self, batch, cond):
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())
        with torch.no_grad(), self._autocast_context():
            losses = self.diffusion.training_losses(
                self.model,
                batch,
                t,
                model_kwargs=self._with_train_step(cond, self.total_step()),
            )

        reduced = {}
        for key, value in losses.items():
            if not torch.is_tensor(value):
                continue
            reduced[key] = float((value.detach() * weights).mean().item())
        return reduced

    def generate_during_training(self, completed_step):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        checkpoint_name = self.ckpt_file_name(completed_step)
        gen_args.model_path = os.path.join(self.save_dir, checkpoint_name)
        gen_args.output_dir = os.path.join(self.save_dir, f'{checkpoint_name}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.motion_length = 6.0 #None  # length is taken from the dataset
        gen_args.load_from_model_name = True
        all_objects = self.data.dataset.motion_dataset.cond_dict.keys() 
        selection_rng = random.Random(int(completed_step))
        gen_args.object_type = selection_rng.sample(list(all_objects), gen_args.num_samples)
        all_sample_save_path = generate(gen_args, self.data.dataset.motion_dataset.cond_dict)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=completed_step,
                                         local_path=all_sample_save_path)
        
    
    def total_step(self):
        total_step = self.step
        if self.resume_step:
            # we add 1 because self.resume_step has already been done and we don't want to run it again
            # in particular we don't want to run the evaluation and generation again
            total_step += self.resume_step + 1
        return total_step

    def evaluate(self):
        if not self.args.eval_during_training or self.eval_data is None:
            return
        totals = {}
        seen_samples = 0
        max_eval_samples = int(self.args.eval_num_samples)

        eval_iter = iter(self.eval_data)

        while True:
            try:
                motion, cond = next(eval_iter)
            except StopIteration:
                break

            motion = self._move_batch_to_device(motion)
            cond = self._move_cond_to_device(cond)

            batch_losses = self._compute_eval_losses(motion, cond)
            batch_size = motion.shape[0]

            for key, value in batch_losses.items():
                totals[key] = totals.get(key, 0.0) + (value * batch_size)

            seen_samples += batch_size
            if max_eval_samples > 0 and seen_samples >= max_eval_samples:
                break

        if seen_samples == 0:
            print('Validation skipped because the evaluation split is empty.')
            return

        averaged = {key: value / seen_samples for key, value in totals.items()}
        current_step = self.total_step()
        completed_step = current_step + 1
        if 'loss' in averaged:
            print('val_step[{}]: val_loss[{:0.5f}]'.format(completed_step, averaged['loss']))
        for key, value in averaged.items():
            self.train_platform.report_scalar(name=key, value=value, iteration=completed_step, group_name='Val')




    def run_step(self, batch, cond, epoch=-1):
        if self.detect_anomaly:
            with torch.autograd.detect_anomaly():
                self.forward_backward(batch, cond, epoch)
        else:
            self.forward_backward(batch, cond, epoch)
        #clip_grad_value_(self.model.parameters(), clip_value=1.5)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters())
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, epoch):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.forward_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=self._with_train_step(micro_cond, self.total_step()),
            )

            if last_batch or not self.use_ddp:
                with self._autocast_context():
                    losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    with self._autocast_context():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self._accumulate_interval_losses({k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def _autocast_context(self):
        if not self.amp_enabled:
            return torch.autocast(device_type=self.device.type, enabled=False)
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self, completed_step):
        return f"model{completed_step:09d}.pt"


    def save(self, completed_step):
            checkpoint_alert = self._monitor_checkpoint_optimizer_state(completed_step)

            def save_checkpoint():
                def del_clip(state_dict):
                    # Do not save CLIP weights
                    clip_weights = [
                        e for e in state_dict.keys() if e.startswith('clip_model.')
                    ]
                    for e in clip_weights:
                        del state_dict[e]

                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
                del_clip(state_dict)

                if self.args.use_ema and self.model_avg is not None:
                    # save both the model and the average model
                    state_dict_avg = self.model_avg.state_dict()
                    del_clip(state_dict_avg)
                    state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

                logger.log(f"saving model...")
                filename = self.ckpt_file_name(completed_step)
                checkpoint_path = pjoin(self.save_dir, filename)
                if '://' in self.save_dir:
                    file_ctx = bf.BlobFile(bf.join(self.save_dir, filename), "wb")
                else:
                    file_ctx = open(checkpoint_path, "wb")
                with file_ctx as f:
                    torch.save(state_dict, f)

            save_checkpoint()

            opt_filename = f"opt{completed_step:09d}.pt"
            opt_path = pjoin(self.save_dir, opt_filename)
            if '://' in self.save_dir:
                opt_ctx = bf.BlobFile(bf.join(self.save_dir, opt_filename), "wb")
            else:
                opt_ctx = open(opt_path, "wb")
            with opt_ctx as f:
                opt_state = self.opt.state_dict()
                if self.amp_enabled:
                    opt_state = {
                        'opt': opt_state,
                        'scaler': self.mp_trainer.scaler.state_dict(),
                    }
                else:
                    opt_state = {'opt': opt_state}
                
                # Save LR scheduler state for proper resumption
                opt_state['scheduler'] = self.lr_scheduler.state_dict()
                
                # Save RNG states to ensure reproducible data shuffling on resume
                opt_state['torch_rng_state'] = torch.get_rng_state()
                if torch.cuda.is_available():
                    opt_state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                opt_state['python_rng_state'] = random.getstate()
                opt_state['numpy_rng_state'] = np.random.get_state()

                torch.save(opt_state, f)

            if checkpoint_alert is not None:
                raise RuntimeError(checkpoint_alert)
                
    def find_resume_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        matches = {file: re.match(r'model(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None
    
    def find_resume_opt_checkpoint(self) -> Optional[str]:
        '''look for all file in save directory in the pattent of model{number}.pt
            and return the one with the highest step number.

        TODO: Implement this function (alredy existing in MDM), so that find model will call it in case a ckpt exist.
        TODO: Change call for find_resume_checkpoint and send save_dir as arg.
        TODO: This means ignoring the flag of resume_checkpoint in case some other ckpts exists in that dir!
        '''

        if self.resume_checkpoint:
            checkpoint_number = parse_checkpoint_number_from_filename(self.resume_checkpoint)
            resume_dir = os.path.dirname(self.resume_checkpoint)
            candidate = pjoin(resume_dir, f'opt{checkpoint_number:09d}.pt')
            if os.path.exists(candidate):
                return candidate

            legacy_candidate = pjoin(resume_dir, f'opt{max(checkpoint_number - 1, 0):09d}.pt')
            if os.path.exists(legacy_candidate):
                return legacy_candidate

        matches = {file: re.match(r'opt(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None




    def _get_checkpoint_step_numbering(self, checkpoint_path: str) -> str:
        args_path = pjoin(os.path.dirname(checkpoint_path), 'args.json')
        if not os.path.exists(args_path):
            return 'zero_based'
        try:
            with open(args_path, 'r', encoding='utf-8') as handle:
                saved_args = json.load(handle)
        except Exception:
            return 'zero_based'
        return saved_args.get('checkpoint_step_numbering', 'zero_based')


def parse_checkpoint_number_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()
            




