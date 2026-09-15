import functools
import os
import re
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
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    assert_bundle_matches_metadata,
    validate_action_conditioning_metadata,
)
from utils.model_util import load_model
from utils.model_util import (
    bind_checkpoint_action_conditioning,
    build_checkpoint_payload,
    create_model_and_diffusion_general_skeleton,
    load_checkpoint_weights,
    unwrap_anytop_model,
)
import random
from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.canonical_features import (
    REST_LENGTH_SCALE_KEY,
    canonical_to_physical_hml,
)
from eval.motion_quality import DistributionMotionQualityScorer
from eval.motion_quality.reference_bank import reference_prior_words
from train.sample_loss_limit import SampleLossLimiter

INITIAL_LOG_LOSS_SCALE = 20.0
EXP_AVG_SQ_CHECKPOINT_ALERT_THRESHOLD = 1e20
# How many fp16 loss-scale overflows to dump before only counting them. They are
# a property of the scaler, not of the batch, so a couple of examples is all the
# diagnosis anyone needs; the rate is what matters and it is logged every step as
# ``amp_overflow``.
AMP_OVERFLOW_MAX_DUMPS = 2


def classify_grad_event(grad_norm, scaler_enabled, spike_threshold):
    """Name what a step's pre-clip grad_norm represents.

    ``'overflow'``: the fp16 GradScaler found non-finite gradients, skipped the
    step and halved the scale. Routine -- it is how the scaler finds the top of
    the fp16 window -- so it must not be counted as a spike.
    ``'spike'``: a real gradient spike (finite and over the threshold), or
    non-finite gradients with no scaler to explain them, which is a genuine
    numerical failure.
    ``None``: an ordinary step.
    """
    if grad_norm is None:
        return None
    if not np.isfinite(grad_norm):
        return 'overflow' if scaler_enabled else 'spike'
    return 'spike' if grad_norm > spike_threshold else None


def _per_sample_decode_cond(y, index, n_joints):
    """Slice a collated ``y`` down to one sample's decode cond.

    ``rest_pos_ric_hml`` is cut to the sample's real joints (the collate pads it
    to max_joints), the per-sample canonical stats and rest length scale are
    taken at ``index``; a stat the collate did not emit stays absent so the
    decoder raises instead of silently skipping the de-standardization.
    """
    decode_cond = {
        'rest_pos_ric_hml': y['rest_pos_ric_hml'][index:index + 1, :n_joints],
    }
    for key in ('canonical_feature_mean', 'canonical_feature_std', REST_LENGTH_SCALE_KEY):
        value = y.get(key)
        if value is not None:
            decode_cond[key] = value[index]
    return decode_cond


def _tile_eval_cond(cond, repeat):
    """Repeat each sample in a cond dict ``repeat`` times for batched DDIM sampling.

    All tensors in ``cond['y']`` are repeated along the batch axis;
    python lists (object_type, parents, action_label, etc.) are
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

# Parameters AdamW must NOT weight-decay, by ``named_parameters()`` name
# suffix: the zero-init gates a residual or bias path is opened with
# (cross-limb ``reliability_bias`` / ``time_emb_scale`` /
# ``temporal_reliability_bias`` / ``cross_k_scale``, the decoder layer's
# ``temporal_phase_scale``, the global ``unreliable_embedding``) plus the
# cross-K LayerNorm gain/bias. Decay pulls each of them back toward its init,
# i.e. toward closing the path it was learned to open. A name rule, not
# ``param.ndim == 1``: those scalars are ``torch.zeros(1)``, the same rank as
# the LayerNorm affines that are NOT on this list.
NO_WEIGHT_DECAY_PARAM_SUFFIXES = (
    'unreliable_embedding',
    '.reliability_bias',
    '.time_emb_scale',
    '.temporal_reliability_bias',
    '.cross_k_scale',
    '.cross_k_norm.weight',
    '.cross_k_norm.bias',
    '.temporal_phase_scale',
)


def is_no_weight_decay_param(name: str) -> bool:
    return name == 'unreliable_embedding' or name.endswith(NO_WEIGHT_DECAY_PARAM_SUFFIXES)


def build_optimizer_param_groups(named_params, weight_decay: float):
    """Two AdamW groups (decay / no decay) from ``(name, param)`` pairs."""
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        (no_decay if is_no_weight_decay_param(name) else decay).append(param)
    groups = [{'params': decay, 'weight_decay': float(weight_decay)}]
    if no_decay:
        groups.append({'params': no_decay, 'weight_decay': 0.0})
    return groups


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
        # Built before the optimizer restore below, which reloads its reference.
        sample_loss_limit = float(getattr(self.args, 'sample_loss_limit', 0.0))
        self.sample_loss_limiter = (
            SampleLossLimiter(diffusion.num_timesteps, sample_loss_limit, self.device)
            if sample_loss_limit > 0.0 else None
        )
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
        # Ignore the warmup: early steps routinely exceed the threshold simply
        # because the optimizer has not settled yet, which drowns the real
        # spikes. Only steps with completed_step > spike_start_step are checked.
        self.spike_start_step = int(getattr(self.args, 'spike_start_step', 1000))
        self.spike_max_dumps = int(getattr(self.args, 'spike_max_dumps', 10))
        self.spike_dumps_written = 0
        # A non-finite grad_norm under the fp16 GradScaler is a loss-scale
        # overflow (the step is skipped and the scale halved), not a gradient
        # spike. It gets its own counter and a much smaller dump budget so it
        # cannot consume the spike budget -- in the first fp16 run every one of
        # the 10 dumps was an overflow and real spikes went unrecorded from step
        # 30k on. See diffusion/fp16_util.GRAD_SCALER_MAX_SCALE.
        self.amp_overflow_steps = 0
        self.amp_overflow_dumps_written = 0
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
        
        # Grouped by parameter name (see NO_WEIGHT_DECAY_PARAM_SUFFIXES); the
        # trainer's master params are the model params (use_fp16 is never on).
        self.opt = AdamW(
            build_optimizer_param_groups(self.model.named_parameters(), self.weight_decay),
            lr=self.lr, weight_decay=self.weight_decay, fused=True,
        )
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
                balanced=False,
                objects_subset=self.args.objects_subset,
                sample_limit=self.args.sample_limit,
                shuffle=False,
                drop_last=True,
                action_group=getattr(self.args, 'action_group', ''),
                action_label_cond=getattr(self.args, 'action_label_cond', False),
                action_conditioning=getattr(self.args, 'action_conditioning', None),
                motion_cache_size=getattr(self.args, 'motion_cache_size', 0),
                min_length=getattr(self.args, 'min_length', 20),
                main_process_prefetch_batches=getattr(self.args, 'main_process_prefetch_batches', 0),
                loop_cond_prob=eval_loop_cond_prob,
                # Evaluation sees the clips at their recorded tempo regardless
                # of --motion_speed_aug, so eval losses stay comparable across
                # runs that differ only in the augmentation.
                motion_speed_aug=1.0,
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
        self._ema_persistent_buffer_names = None

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

            payload = dist_util.load_state_dict(
                self.resume_checkpoint, map_location=dist_util.dev())
            # Two checks, on either side of the load, because they are about
            # different material. The metadata one comes FIRST: a resume across
            # word tables or conditioning contracts is refused before any weight
            # lands, since those weights load cleanly and then train under
            # semantics they were never fitted for.
            state_dict, state_dict_avg, metadata = load_checkpoint_weights(
                payload, self.resume_checkpoint, prefer_ema=False)
            self._assert_resume_action_conditioning(metadata)

            # The bind comes SECOND, on each set of weights, because the buffers
            # it certifies -- the word table and the role transform -- are the
            # CHECKPOINT's, and they only exist in the model once load_model has
            # overwritten this run's own. Binding first certified the material
            # the run started with and then let load_model replace it unchecked.
            load_model(self.model, state_dict)
            bind_checkpoint_action_conditioning(
                self.model, metadata, self.resume_checkpoint)
            if self.model_avg is not None:
                if state_dict_avg is not None:
                    print('loading both model and model_avg')
                    load_model(self.model_avg, state_dict_avg)
                    # The EMA copy carries its own buffers and is what sampling
                    # and every eval read, so it gets the same certification
                    # rather than inheriting the online model's.
                    bind_checkpoint_action_conditioning(
                        self.model_avg, metadata, self.resume_checkpoint)
                else:
                    # The run that wrote this checkpoint kept no EMA copy.
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())

    def _assert_resume_action_conditioning(self, metadata):
        """Refuse a resume whose contract or word table is not this run's.

        Pure metadata, and deliberately BEFORE the weights land: a checkpoint
        under a conditioning contract this code no longer implements, or fitted
        on a different word table than the loader is emitting ids into, must be
        refused while the model still holds this run's own material. The
        buffer-level half is bind_checkpoint_action_conditioning, which can only
        run once those buffers ARE the checkpoint's.
        """
        if not getattr(unwrap_anytop_model(self.model), 'action_label_cond', False):
            return
        action_conditioning = (metadata or {}).get('action_conditioning') or {}
        bundle = getattr(self.args, 'action_conditioning', None)
        if bundle is None:
            validate_action_conditioning_metadata(
                action_conditioning, source=self.resume_checkpoint)
            return
        assert_bundle_matches_metadata(
            bundle, action_conditioning, source=self.resume_checkpoint,
        )

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

        limiter_state = checkpoint_data.get('sample_loss_limiter') if isinstance(checkpoint_data, dict) else None
        if self.sample_loss_limiter is not None and limiter_state is not None:
            if self.sample_loss_limiter.load_state_dict(limiter_state):
                logger.log("sample loss limiter reference restored")
            else:
                logger.log("sample loss limiter reference restore skipped: timestep count changed")

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



    # Read only by the host-side joint-mask sampler (AnyTop.sample_subtree_joint_mask_train):
    # moving it to the device just to read it back cost a stream sync per step.
    HOST_ONLY_COND_KEYS = ('joint_mask_candidate_roots',)

    def _move_cond_to_device(self, cond):
        y = {
            key: val if key in self.HOST_ONLY_COND_KEYS or not torch.is_tensor(val)
            else val.to(self.device, non_blocking=self.non_blocking)
            for key, val in cond['y'].items()
        }
        # The same sampler needs the joint counts on the host; keep the
        # collate's copy next to the device one instead of reading it back.
        n_joints = cond['y'].get('n_joints')
        if torch.is_tensor(n_joints):
            y['n_joints_cpu'] = n_joints
        return {'y': y}

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
        infer_model = self.model  # use raw model (not EMA) to observe real val performance
        motion_groups = {}
        missing_action_label_count = 0
        target_batch = int(self.args.eval_batch_size)

        infer_model.eval()
        # Sampling is fp32 whatever --amp_dtype trains with: bf16 rounding of the
        # x0 prediction is frame-independent noise that inflates the Jerk / Snap /
        # SpectralFlatness terms scored below (docs/bf16_precision_issues.md), so
        # bf16 scores would not be comparable across checkpoints or with
        # eval_checkpoint.py. TF32 stays whatever training set (on under --compile).
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
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
                    # Grouped by the label's prior words, not its spelling: the
                    # two directions of one transition share a reference prior.
                    #
                    # This call RAISES on a label that breaks the vocabulary
                    # contract (unknown token, no head word, repeats, too many
                    # words) and takes validation down with it -- deliberately,
                    # and unlike the empty label handled just below: a typo must
                    # fail loudly rather than silently narrow the prior to the
                    # words it happened to hit and report a confident score for
                    # it. An empty label is legal (no condition), so it only
                    # skips this clip.
                    prior_words = reference_prior_words(
                        cond['y'].get('action_label', [None] * batch_size)[i]
                    )
                    if not prior_words:
                        missing_action_label_count += 1
                        continue
                    n_joints = cond['y']['n_joints'][i].item()
                    motion_sample = sample[i][:n_joints]
                    # Every per-sample field is sliced to row i: the canonical stats
                    # are per object_subset, and the whole [B, F] stack does not fit
                    # a one-sample feature (the decoder rejects it).
                    motion_physical = canonical_to_physical_hml(
                        motion_sample.unsqueeze(0),
                        _per_sample_decode_cond(cond['y'], i, n_joints),
                    )[0]
                    motion_np = motion_physical.cpu().permute(2, 0, 1).numpy()
                    group_key = (object_type, prior_words)
                    motion_groups.setdefault(group_key, []).append(motion_np.astype(np.float32))

        infer_model.train()

        if missing_action_label_count:
            tqdm.write(f'Validation skipped {missing_action_label_count} motion(s) with no action_label.')

        if not motion_groups:
            tqdm.write('Validation skipped: eval split returned no samples.')
            return

        scores = []
        score_weights = []
        jerk_scores = []
        snap_scores = []
        sf_scores = []
        bl_scores = []
        for (object_type, prior_words), motions in motion_groups.items():
            action_label = ', '.join(prior_words)
            try:
                report = self.scorer.evaluate(
                    motions=motions,
                    object_type=object_type,
                    action_label=action_label,
                )
            except Exception as exc:
                tqdm.write(f"[eval] Scoring failed for {object_type} ({action_label}): {exc}")
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
        """Copy persistent buffers from the live model into the EMA model.

        update_ema only averages
        .parameters(), so these running stats would otherwise stay at their
        init values in the EMA checkpoint.  state_dict() keys select exactly
        parameters + persistent buffers, so subtracting the parameter names
        leaves the persistent buffers.  Non-persistent buffers (e.g.
        _cached_time_emb) are absent from state_dict and must NOT be copied:
        the EMA model is never forwarded, so its cache shape would mismatch the
        live model's and copy_ would raise.
        """
        # The name list is fixed once the model is built; rebuilding a full
        # state_dict() every step just to find it cost ~2 ms of host time.
        if self._ema_persistent_buffer_names is None:
            param_names = {name for name, _ in self.model_avg.named_parameters()}
            self._ema_persistent_buffer_names = [
                name for name in self.model_avg.state_dict() if name not in param_names
            ]
        for name in self._ema_persistent_buffer_names:
            self.model_avg.get_buffer(name).copy_(self.model.get_buffer(name))

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
            # .parameters()).  Sync persistent buffers (running statistics) so
            # they are available in the EMA checkpoint at inference time.
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

            if self.sample_loss_limiter is not None:
                # Only l_simple's gradient is rescaled; the logged l_simple stays raw.
                limit_weight = self.sample_loss_limiter.weights(t, losses["l_simple"])
                losses["loss"] = losses["loss"] + (limit_weight - 1.0) * losses["l_simple"]

            loss = (losses["loss"] * weights).mean()
            self._accumulate_interval_losses({k: v * weights for k, v in losses.items()})
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
        float from optimize(), so this probe adds no per-step sync). Steps at or
        below spike_start_step are skipped as warmup noise.

        An infinite grad_norm while the fp16 GradScaler is on is classified as a
        loss-scale overflow instead, with its own counter, budget and file name
        (see AMP_OVERFLOW_MAX_DUMPS)."""
        ctx = self._spike_ctx
        self._spike_ctx = None
        if ctx is None:
            return

        completed_step = self.total_step() + 1
        if completed_step <= self.spike_start_step:
            return

        grad_norm = self.mp_trainer.last_grad_norm
        # An fp16 loss-scale overflow is a different event from a gradient spike:
        # separate counter, separate (small) dump budget, separate file name.
        kind = classify_grad_event(
            grad_norm, self.mp_trainer.scaler.is_enabled(), self.spike_grad_threshold
        )
        if kind is None:
            return
        nonfinite = not np.isfinite(grad_norm)
        amp_overflow = kind == 'overflow'
        if amp_overflow:
            self.amp_overflow_steps += 1
            if self.amp_overflow_dumps_written >= AMP_OVERFLOW_MAX_DUMPS:
                return
        elif self.spike_max_dumps and self.spike_dumps_written >= self.spike_max_dumps:
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
        action_labels = y.get('action_label')
        action_groups = y.get('action_group')
        flag_keys = (
            'is_loop', 'loop_data_aug_applied', 'loop_tile_count',
            'loop_phase_offset', 'resample_speed_cond', 'motion_speed_applied',
            'n_joints',
        )
        flags = {k: field_list(k) for k in flag_keys}

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
                'action_label': action_labels[i] if action_labels else None,
                'action_group': action_groups[i] if action_groups else None,
            }
            for k, arr in flags.items():
                rec[k] = arr[i] if (arr is not None and i < len(arr)) else None
            if per_sample_aux:
                rec['aux'] = {k: float(arr[i]) for k, arr in per_sample_aux.items() if i < len(arr)}
            samples.append(rec)
        samples.sort(key=lambda r: r['loss'], reverse=True)

        # An overflow is a skipped step: the grads are non-finite, so per-param
        # grad norms are meaningless and the host sync is pure waste. Only real
        # spikes need top_param_grads to localize the layer (doc 9.5).
        if amp_overflow:
            total_postclip, top_param_grads = None, []
        else:
            total_postclip, top_param_grads = self._top_param_grad_norms()

        record = {
            'completed_step': completed_step,
            'kind': kind,
            'grad_norm_preclip': float(grad_norm),
            'grad_clip_max_norm': 1.0,
            'grad_norm_postclip_total': total_postclip,
            'amp_dtype': self.amp_dtype,
            'loss_scale': (self.mp_trainer.scaler.get_scale()
                           if self.mp_trainer.scaler.is_enabled() else None),
            'amp_overflow_steps_so_far': self.amp_overflow_steps,
            'trigger': {'grad': not amp_overflow, 'amp_overflow': amp_overflow},
            'thresholds': {'grad': self.spike_grad_threshold},
            'batch_mean_loss': float(loss_np.mean()),
            'batch_max_loss': float(loss_np.max()),
            'scalar_losses': scalar_aux,
            'top_param_grad_norms': top_param_grads,
            'samples': samples,
        }

        spike_dir = pjoin(self.save_dir, 'spikes')
        os.makedirs(spike_dir, exist_ok=True)
        json_path = pjoin(spike_dir, f'{kind}_step{completed_step:09d}.json')
        with open(json_path, 'w') as f:
            json.dump(record, f, indent=2, default=str)

        if self.spike_save_batch:
            payload = {
                'completed_step': completed_step,
                'motion': batch.detach().cpu(),
                't': ctx['t'].cpu(),
                'motion_name': names,
                'object_type': species,
                'action_label': action_labels,
                'action_group': action_groups,
            }
            for k in flag_keys:
                v = y.get(k)
                if torch.is_tensor(v):
                    payload[k] = v.detach().cpu()
            torch.save(payload, pjoin(spike_dir, f'{kind}_step{completed_step:09d}.pt'))

        if amp_overflow:
            self.amp_overflow_dumps_written += 1
            tqdm.write(
                f'[overflow] step {completed_step}: fp16 loss-scale overflow #'
                f'{self.amp_overflow_steps} (step skipped, scale halved to '
                f'{record["loss_scale"]}), batch_mean_loss='
                f'{record["batch_mean_loss"]:.3f} -> {json_path}'
            )
            return

        self.spike_dumps_written += 1
        top_param = top_param_grads[0]['param'] if top_param_grads else '?'
        tqdm.write(
            f'[spike] step {completed_step}: grad_norm(pre-clip)='
            f'{"inf" if nonfinite else f"{grad_norm:.1f}"}, '
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

                state_dict_avg = None
                if self.args.use_ema and self.model_avg is not None:
                    # save both the model and the average model.
                    # Ensure the EMA model's running-stat buffers are synced
                    # from the live model before serializing (belt-and-suspenders
                    # with run_step, in case the final save happens before the
                    # next optimizer step).
                    self._sync_ema_persistent_buffers()
                    state_dict_avg = self.model_avg.state_dict()
                    del_clip(state_dict_avg)
                # Every checkpoint carries the conditioning contract it was
                # trained under, so neither a resume nor generation has to guess
                # (and neither needs the dataset's word sidecar to find out).
                state_dict = build_checkpoint_payload(
                    state_dict, state_dict_avg, self.model)

                logger.log("saving model...")
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
                if self.sample_loss_limiter is not None:
                    opt_state['sample_loss_limiter'] = self.sample_loss_limiter.state_dict()
                
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
            


