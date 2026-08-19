import functools
import os
import re
import time
import json
import copy as pycopy
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
import copy
from utils.model_util import load_model
from utils.model_util import create_model_and_diffusion_general_skeleton
import random
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.canonical_features import canonical_to_physical_hml
from eval.motion_quality import DistributionMotionQualityScorer

INITIAL_LOG_LOSS_SCALE = 20.0
EXP_AVG_SQ_CHECKPOINT_ALERT_THRESHOLD = 1e20


def _normalize_eval_action_tags(raw_action_tags):
    if raw_action_tags is None:
        return ()
    if isinstance(raw_action_tags, str):
        values = raw_action_tags.replace(';', ',').split(',')
    else:
        values = raw_action_tags
    normalized = {
        str(tag).strip().lower()
        for tag in values
        if str(tag).strip()
    }
    return tuple(sorted(normalized))

def _tile_eval_cond(cond, repeat):
    """Repeat each sample in a cond dict ``repeat`` times for batched DDIM sampling.

    All tensors in ``cond['y']`` are repeated along the batch axis;
    python lists (object_type, parents, action_tags, etc.) are
    element-replicated.
    """
    if repeat <= 1:
        return cond
    y = {}
    for key, val in cond['y'].items():
        if isinstance(val, torch.Tensor):
            y[key] = torch.cat([val] * repeat, dim=0)
        elif isinstance(val, list):
            y[key] = val * repeat
        else:
            y[key] = val
    return {'y': y}

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
                "This usually means the dataset has fewer effective samples than one full batch."
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
        # Spike-capture probe: when a step's pre-clip grad_norm exceeds a
        # threshold, dump the offending batch + top per-parameter grad norms.
        # See _maybe_capture_spike. Hardcoded always-on, and always serializes
        # the offending batch (.pt) alongside the JSON summary. Threshold / dump
        # cap stay tunable via args.
        self.spike_capture = True
        self.spike_save_batch = True
        self.spike_grad_threshold = float(getattr(self.args, 'spike_grad_threshold', 50.0))
        self.spike_max_dumps = int(getattr(self.args, 'spike_max_dumps', 10))
        self.spike_dumps_written = 0
        self._spike_ctx = None
        self.autocast_dtype = None
        if self.amp_dtype == 'fp16':
            self.autocast_dtype = torch.float16
        elif self.amp_dtype == 'bf16':
            self.autocast_dtype = torch.bfloat16
        # AMP runs under a single top-level torch.autocast context (see
        # _autocast_context), applied around every model forward. autocast's op
        # policy keeps softmax/layernorm in fp32 and runs linear/conv/matmul in
        # bf16, and inductor fuses it cleanly under --compile.
        self._load_and_sync_parameters()
        if self.amp_enabled:
            logger.log(
                f"{self.amp_dtype} autocast enabled via torch.autocast; softmax/layernorm stay fp32"
            )
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=False,
            amp_dtype=self.amp_dtype,
            amp_enabled=self.amp_enabled,
            device_type=self.device.type,
            fp16_scale_growth=self.fp16_scale_growth,
        )
        
        self.opt = AdamW(self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay, fused=True)
        self._optimizer_param_names = {id(param): name for name, param in self.model.named_parameters()}
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt,
                                                step_size=getattr(self.args, 'lr_scheduler_step_size', 10000),
                                                gamma=getattr(self.args, 'lr_scheduler_gamma', 0.99))
        
        if self.resume_step and bool(getattr(self.args, 'load_optimizer_state', True)):
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        self.inference_diffusion = None
        self.scorer = None
        if self.args.eval_during_training:
            eval_loop_cond_prob = getattr(self.args, 'loop_cond_prob', 1.0)
            self.eval_data = get_dataset_loader(
                cond_path=self.args.cond_path,
                batch_size=self.args.eval_batch_size,
                num_frames=self.args.num_frames,
                split=self.args.eval_split,
                temporal_window=self.args.temporal_window,
                balanced=False,
                objects_subset=self.args.objects_subset,
                sample_limit=self.args.sample_limit,
                shuffle=False,
                drop_last=True,
                action_tags=getattr(self.args, 'action_tags', ''),
                motion_cache_size=getattr(self.args, 'motion_cache_size', 0),
                min_length=getattr(self.args, 'min_length', 20),
                main_process_prefetch_batches=getattr(self.args, 'main_process_prefetch_batches', 0),
                loop_cond_prob=eval_loop_cond_prob,
            )
            sampling_steps = int(getattr(self.args, 'sampling_steps', 100))
            infer_args = pycopy.deepcopy(self.args)
            infer_args.timestep_respacing = f'ddim{sampling_steps}' if sampling_steps > 0 else ''
            _, self.inference_diffusion = create_model_and_diffusion_general_skeleton(infer_args)
            # The scorer's reference distribution must come from real clips, so
            # it reuses the same dataset sources the training cond derives.
            self.scorer = DistributionMotionQualityScorer(
                dataset_root=data.dataset.opt.sources
            )
        self.use_ddp = False
        self.ddp_model = self.model
        self.forward_model = self.ddp_model
        compile_mode = getattr(self.args, 'compile', None)
        if compile_mode and compile_mode != 'None':
            self._compile_forward_model(compile_mode)
        self._interval_loss_sums = {}
        self._interval_loss_counts = {}

    def _compile_forward_model(self, mode='default'):
        """Wrap the training forward path with torch.compile.

        Training is kernel-launch-bound: each step issues thousands of tiny
        per-layer kernels and the GPU sits idle between launches. torch.compile
        fuses the transformer decoder's ops, cutting launches.

        We compile ``self.forward_model`` (a thin wrapper over ``self.model``)
        and deliberately leave ``self.model`` itself untouched so the checkpoint
        path (``mp_trainer`` master params, ``state_dict``) and EMA are byte-for-
        byte identical to an uncompiled run -- the OptimizedModule never owns the
        parameters, so resume compatibility is preserved in both directions.

        Joint/frame dims are fixed across batches (joints pad to the global
        ``opt.max_joints``, frames are resampled to ``num_frames``), so
        ``dynamic=False`` is safe and avoids dynamic-shape tracing overhead; the
        only shape that can vary is a trailing partial batch (drop_last is
        always True, avoiding the one extra compile). The numpy/.item() conditioning in
        AnyTop.forward triggers graph breaks, but the heavy decoder region
        between breaks still compiles and fuses.
        """
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.specialize_int = False
            _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, 'cache_size_limit', 8), 32)
        except Exception:  # pragma: no cover - dynamo always present with compile
            pass
        torch.set_float32_matmul_precision('high')
        try:
            compile_kwargs = {'dynamic': False}
            if mode and mode != 'default':
                compile_kwargs['mode'] = mode
            compiled = torch.compile(self.forward_model, **compile_kwargs)
            self.forward_model = compiled
        except Exception as exc:  # pragma: no cover - depends on build toolchain
            logger.log(
                f"torch.compile unavailable ({exc}); falling back to eager. "
                "Ensure the MSVC build env is active (start_torch_compile_env.ps1)."
            )
            return
        logger.log(
            f"torch.compile enabled (mode={mode}, dynamic=False). The first step "
            "pays a one-time compilation cost before steady-state speedup."
        )

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
                        elif k.startswith('l_simple_'):
                            tqdm.write('step[{}]: {}[{:0.5f}]'.format(completed_step, k, v))
                        if k in ['step', 'samples']:
                            continue
                        self.train_platform.report_scalar(name=k, value=v, iteration=completed_step, group_name='Loss')

                if self._should_validate(completed_step):
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                if self._should_save(completed_step):
                    self.save(completed_step)
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

    def _accumulate_per_family_l_simple(self, losses, weights, cond):
        """Track l_simple broken down by topology family.

        Maps the per-family difficulty landscape (quad/biped/millipede/snake/
        fish/flying) and shows how negative transfer hits each family. Same
        weighting convention as the aggregate l_simple metric, so all of these
        are directly comparable to it and to a single-family run.
        """
        if "l_simple" not in losses:
            return
        object_types = cond.get('y', {}).get('object_type', None)
        if not object_types:
            return
        family_sets = getattr(self, '_family_species_sets', None)
        if family_sets is None:
            from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags
            members = dataset_tags().subset_members
            family_sets = {
                'quad': members['quadruped'],
                'biped': members['biped'],
                'milliped': members['multiped'],
                'snake': members['serpentine'],
                'fish': members['aquatic'],
                'flying': members['winged'],
            }
            self._family_species_sets = family_sets
        l_simple = (losses["l_simple"] * weights).detach().float()
        metrics = {}
        for family, species in family_sets.items():
            mask = torch.tensor(
                [ot in species for ot in object_types],
                device=l_simple.device, dtype=torch.bool,
            )
            if bool(mask.any()):
                metrics[f'l_simple_{family}'] = l_simple[mask].mean()
        if metrics:
            self._accumulate_interval_losses(metrics)

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
        cond_dict = self.data.dataset.motion_dataset.cond_dict
        infer_model = self.model  # use raw model (not EMA) to observe real val performance
        motion_groups = {}
        missing_action_tag_count = 0
        target_batch = int(self.args.eval_batch_size)

        infer_model.eval()
        with torch.no_grad(), self._autocast_context():
            # Iterate the whole eval split so every unique motion is sampled
            # at least once. The loader batches by eval_batch_size, so full
            # batches sample each motion once; a smaller trailing batch is
            # tiled up to fill eval_batch_size (each motion sampled repeat
            # times, may overshoot slightly when it doesn't divide evenly).
            for motion, cond in self.eval_data:
                motion = self._move_batch_to_device(motion)
                cond = self._move_cond_to_device(cond)
                native_batch = motion.shape[0]
                if native_batch < target_batch:
                    repeat = (target_batch + native_batch - 1) // native_batch
                    motion = torch.cat([motion] * repeat, dim=0)
                    cond = _tile_eval_cond(cond, repeat)

                batch_size = motion.shape[0]
                max_joints = motion.shape[1]
                n_frames = motion.shape[3]

                sample_shape = (batch_size, max_joints, infer_model.feature_len, n_frames)
                noise = torch.randn(sample_shape, device=dist_util.dev())
                sample = self.inference_diffusion.ddim_sample_loop(
                    model=infer_model,
                    shape=sample_shape,
                    noise=noise,
                    init_image=motion,
                    skip_timesteps=5,
                    clip_denoised=False,
                    model_kwargs=cond,
                    device=dist_util.dev(),
                    progress=False,
                    eta=0.0,
                )

                for i in range(batch_size):
                    object_type = cond['y']['object_type'][i]
                    action_tags = _normalize_eval_action_tags(
                        cond['y'].get('action_tags', [None] * batch_size)[i]
                    )
                    if not action_tags:
                        missing_action_tag_count += 1
                        continue
                    n_joints = cond['y']['n_joints'][i].item()
                    motion_sample = sample[i][:n_joints]
                    motion_physical = canonical_to_physical_hml(
                        motion_sample.unsqueeze(0),
                        {
                            'rest_pos_ric_hml': cond['y']['rest_pos_ric_hml'][i:i + 1, :n_joints],
                            'canonical_feature_mean': cond['y'].get('canonical_feature_mean'),
                            'canonical_feature_std': cond['y'].get('canonical_feature_std'),
                        },
                    )[0]
                    motion_np = motion_physical.cpu().permute(2, 0, 1).numpy()
                    group_key = (object_type, action_tags)
                    motion_groups.setdefault(group_key, []).append(motion_np.astype(np.float32))

        infer_model.train()

        if missing_action_tag_count:
            tqdm.write(f'Validation skipped {missing_action_tag_count} motion(s) without action_tags.')

        if not motion_groups:
            tqdm.write('Validation skipped: eval split returned no samples.')
            return

        scores = []
        score_weights = []
        jerk_scores = []
        snap_scores = []
        sf_scores = []
        bl_scores = []
        for (object_type, action_tags), motions in motion_groups.items():
            try:
                report = self.scorer.evaluate(
                    motions=motions,
                    object_type=object_type,
                    action_tags=','.join(action_tags),
                )
            except Exception as exc:
                tqdm.write(f"[eval] Scoring failed for {object_type} ({','.join(action_tags)}): {exc}")
                continue
            scores.append(report.overall_score)
            jerk_scores.append(report.jerk_score)
            snap_scores.append(report.snap_score)
            sf_scores.append(report.spectral_flatness_score)
            bl_scores.append(report.bone_length_score)
            score_weights.append(len(motions))

        if not scores:
            return

        w = np.array(score_weights)
        completed_step = self.total_step() + 1
        avg_score = float(np.average(scores, weights=w))
        avg_jerk = float(np.average(jerk_scores, weights=w))
        avg_snap = float(np.average(snap_scores, weights=w))
        avg_sf = float(np.average(sf_scores, weights=w))
        avg_bl = float(np.average(bl_scores, weights=w))
        tqdm.write('val_step[{}]: Score[{:.4f}] Jerk[{:.4f}] Snap[{:.4f}] SF[{:.4f}] BL[{:.4f}]'.format(
            completed_step, avg_score, avg_jerk, avg_snap, avg_sf, avg_bl))
        self.train_platform.report_scalar(name='Score', value=avg_score, iteration=completed_step, group_name='Val')
        self.train_platform.report_scalar(name='Jerk', value=avg_jerk, iteration=completed_step, group_name='Val')
        self.train_platform.report_scalar(name='Snap', value=avg_snap, iteration=completed_step, group_name='Val')
        self.train_platform.report_scalar(name='SpectralFlatness', value=avg_sf, iteration=completed_step, group_name='Val')
        self.train_platform.report_scalar(name='BoneLength', value=avg_bl, iteration=completed_step, group_name='Val')




    def _sync_ema_persistent_buffers(self):
        """Copy persistent buffers (e.g. global_energy_running_mean/var) from
        the live model into the EMA model.  update_ema only averages
        .parameters(), so these running stats would otherwise stay at their
        init values in the EMA checkpoint.  state_dict() keys select exactly
        parameters + persistent buffers, so subtracting the parameter names
        leaves the persistent buffers.  Non-persistent buffers (e.g.
        _cached_time_emb) are absent from state_dict and must NOT be copied:
        the EMA model is never forwarded, so its cache shape would mismatch the
        live model's and copy_ would raise.
        """
        param_names = {name for name, _ in self.model_avg.named_parameters()}
        avg_buffers = dict(self.model_avg.named_buffers())
        model_buffers = dict(self.model.named_buffers())
        for name in self.model_avg.state_dict():
            if name in param_names:
                continue
            avg_buffers[name].copy_(model_buffers[name])

    def run_step(self, batch, cond, epoch=-1):
        if self.detect_anomaly:
            with torch.autograd.detect_anomaly():
                self.forward_backward(batch, cond, epoch)
        else:
            self.forward_backward(batch, cond, epoch)
        #clip_grad_value_(self.model.parameters(), clip_value=1.5)
        took_step = self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        if self.spike_capture:
            self._maybe_capture_spike(batch, cond)
        if took_step and self.model_avg is not None:
            update_ema(self.model_avg.parameters(), self.model.parameters(),
                       rate=self.args.ema_rate)
            # EMA ignores registered buffers by default (it only iterates
            # .parameters()).  Sync persistent buffers (running statistics, e.g.
            # global_energy_running_mean/var) so they are available in the EMA
            # checkpoint at inference time.
            self._sync_ema_persistent_buffers()
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
            self._accumulate_per_family_l_simple(losses, weights, micro_cond)
            if self.spike_capture:
                self._stash_spike_ctx(losses, t)
            self.mp_trainer.backward(loss)

    def _stash_spike_ctx(self, losses, t):
        """Keep detached, on-device references to the just-computed per-sample
        losses and timesteps so _maybe_capture_spike can inspect them after the
        optimizer step. Cheap: no host sync here, tensors stay on the GPU."""
        reserved = ('loss', 'l_simple')
        l_simple = losses.get('l_simple', losses['loss'])
        self._spike_ctx = {
            'loss': losses['loss'].detach(),
            'l_simple': l_simple.detach(),
            't': t.detach(),
            'aux': {k: v.detach() for k, v in losses.items() if k not in reserved},
        }

    def _top_param_grad_norms(self, k=25):
        """Per-parameter L2 grad norms (top-k), to localize which layer's
        gradient dominates a spike. Called only on trigger. The grads here are
        post-clip (clip_grad_norm_ rescales them uniformly to total norm
        max_norm), so absolute values are scaled but the *ranking* across
        params is identical to pre-clip -- which is what localizes the layer.
        One host sync total (norms are stacked, then moved to CPU once)."""
        names, norms = [], []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            names.append(name)
            norms.append(p.grad.detach().norm())
        if not norms:
            return None, []
        norms_t = torch.stack(norms).float().cpu()
        total_postclip = float(norms_t.norm())
        order = torch.argsort(norms_t, descending=True)[:k].tolist()
        top = [{'param': names[i], 'grad_norm_postclip': float(norms_t[i])} for i in order]
        return total_postclip, top

    def _maybe_capture_spike(self, batch, cond):
        """If this step's pre-clip grad_norm exceeds spike_grad_threshold, dump
        the offending batch (clip names, per-sample t / loss / loss-components,
        augmentation flags) plus the top per-parameter grad norms to
        <save_dir>/spikes so the trigger AND the dominant layer of a spike can be
        identified post-hoc. Grad-norm is the sole trigger (it is already a host
        float from optimize(), so this probe adds no per-step sync)."""
        ctx = self._spike_ctx
        self._spike_ctx = None
        if ctx is None:
            return

        grad_norm = self.mp_trainer.last_grad_norm
        grad_trip = grad_norm is not None and (
            not np.isfinite(grad_norm) or grad_norm > self.spike_grad_threshold
        )
        if not grad_trip:
            return

        completed_step = self.total_step() + 1
        if self.spike_max_dumps and self.spike_dumps_written >= self.spike_max_dumps:
            if self.spike_dumps_written == self.spike_max_dumps:
                tqdm.write(
                    f'[spike] step {completed_step}: spike detected but --spike_max_dumps '
                    f'({self.spike_max_dumps}) reached; not writing further dumps.'
                )
                self.spike_dumps_written += 1  # advance once so the notice prints only once
            return

        y = cond['y']

        def field_list(key):
            v = y.get(key)
            if v is None:
                return None
            if torch.is_tensor(v):
                return v.detach().cpu().tolist()
            return list(v)

        names = field_list('motion_name')
        species = field_list('object_type')
        action_tags = y.get('action_tags')
        flag_keys = (
            'is_loop', 'loop_full_cycle', 'loop_data_aug_applied', 'loop_tile_count',
            'loop_phase_offset', 'motion_start_frame', 'playspeed_cond', 'n_joints',
        )
        flags = {k: field_list(k) for k in flag_keys}
        gec = y.get('global_energy_cond')
        flags['global_energy_cond'] = gec.detach().cpu().reshape(-1).tolist() if torch.is_tensor(gec) else None

        loss_np = ctx['loss'].float().cpu().numpy()
        lsimple_np = ctx['l_simple'].float().cpu().numpy()
        t_np = ctx['t'].cpu().numpy()
        bs = loss_np.shape[0]

        per_sample_aux, scalar_aux = {}, {}
        for k, v in ctx['aux'].items():
            if v.ndim == 0:
                scalar_aux[k] = float(v.item())
            else:
                per_sample_aux[k] = v.float().cpu().numpy().reshape(-1)

        samples = []
        for i in range(bs):
            rec = {
                'idx': i,
                'loss': float(loss_np[i]),
                'l_simple': float(lsimple_np[i]),
                't': int(t_np[i]),
                'name': names[i] if names else None,
                'species': species[i] if species else None,
                'action_tags': action_tags[i] if action_tags else None,
            }
            for k, arr in flags.items():
                rec[k] = arr[i] if (arr is not None and i < len(arr)) else None
            if per_sample_aux:
                rec['aux'] = {k: float(arr[i]) for k, arr in per_sample_aux.items() if i < len(arr)}
            samples.append(rec)
        samples.sort(key=lambda r: r['loss'], reverse=True)

        total_postclip, top_param_grads = self._top_param_grad_norms()

        record = {
            'completed_step': completed_step,
            'grad_norm_preclip': (None if grad_norm is None else float(grad_norm)),
            'grad_clip_max_norm': 1.0,
            'grad_norm_postclip_total': total_postclip,
            'amp_dtype': self.amp_dtype,
            'trigger': {'grad': bool(grad_trip)},
            'thresholds': {'grad': self.spike_grad_threshold},
            'batch_mean_loss': float(loss_np.mean()),
            'batch_max_loss': float(loss_np.max()),
            'scalar_losses': scalar_aux,
            'top_param_grad_norms': top_param_grads,
            'samples': samples,
        }

        spike_dir = pjoin(self.save_dir, 'spikes')
        os.makedirs(spike_dir, exist_ok=True)
        json_path = pjoin(spike_dir, f'spike_step{completed_step:09d}.json')
        with open(json_path, 'w') as f:
            json.dump(record, f, indent=2, default=str)

        if self.spike_save_batch:
            payload = {
                'completed_step': completed_step,
                'motion': batch.detach().cpu(),
                't': ctx['t'].cpu(),
                'motion_name': names,
                'object_type': species,
                'action_tags': action_tags,
            }
            for k in flag_keys:
                v = y.get(k)
                if torch.is_tensor(v):
                    payload[k] = v.detach().cpu()
            torch.save(payload, pjoin(spike_dir, f'spike_step{completed_step:09d}.pt'))

        self.spike_dumps_written += 1
        top_param = top_param_grads[0]['param'] if top_param_grads else '?'
        tqdm.write(
            f'[spike] step {completed_step}: grad_norm(pre-clip)='
            f'{"inf" if grad_norm is None or not np.isfinite(grad_norm) else f"{grad_norm:.1f}"}, '
            f'batch_mean_loss={record["batch_mean_loss"]:.3f}, '
            f'top-grad param={top_param} -> {json_path}'
        )

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
                    # save both the model and the average model.
                    # Ensure the EMA model's running-stat buffers are synced
                    # from the live model before serializing (belt-and-suspenders
                    # with run_step, in case the final save happens before the
                    # next optimizer step).
                    self._sync_ema_persistent_buffers()
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
            




