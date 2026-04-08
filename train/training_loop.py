import functools
import os
import re
import time
from os.path import join as pjoin
from typing import Optional
import blobfile as bf
import torch
from torch.optim import AdamW
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
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
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )
        

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 
                                                step_size = 10000, 
                                                gamma = 0.99)
        
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())
        self.non_blocking = self.device.type == 'cuda'
        self.detect_anomaly = bool(getattr(self.args, 'detect_anomaly', False))

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
                drop_last=False,
            )
        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        self.resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
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
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
         print('train steps:', self.num_steps)
         while self.total_step() < self.num_steps:
            print(f'Starting a new epoch at step {self.total_step()}')
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

                current_step = self.total_step()

                if current_step % self.log_interval == 0:
                    print()
                    print(cond['y']['object_type'])
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(current_step, v))
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=current_step, group_name='Loss')

                if self._should_validate(current_step):
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                if self._should_save(current_step):
                    self.save()

                    self.model.eval()
                    self.generate_during_training()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.step += 1

                if self.total_step() == self.num_steps:
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

    def _should_save(self, current_step):
        return (current_step % self.save_interval == 0 and current_step != 0) or current_step == self.num_steps - 1

    def _should_validate(self, current_step):
        if not self.args.eval_during_training or self.eval_data is None or self.eval_interval <= 0:
            return False
        return (current_step % self.eval_interval == 0 and current_step != 0) or current_step == self.num_steps - 1

    def _compute_eval_losses(self, batch, cond):
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())
        with torch.no_grad():
            losses = self.diffusion.training_losses(
                self.model,
                batch,
                t,
                model_kwargs=cond,
            )

        reduced = {}
        for key, value in losses.items():
            if not torch.is_tensor(value):
                continue
            reduced[key] = float((value.detach() * weights).mean().item())
        return reduced

    def generate_during_training(self):
        if not self.args.gen_during_training:
            return
        gen_args = copy.deepcopy(self.args)
        gen_args.model_path = os.path.join(self.save_dir, self.ckpt_file_name())
        gen_args.output_dir = os.path.join(self.save_dir, f'{self.ckpt_file_name()}.samples')
        gen_args.num_samples = self.args.gen_num_samples
        gen_args.num_repetitions = self.args.gen_num_repetitions
        gen_args.motion_length = 6.0 #None  # length is taken from the dataset
        gen_args.load_from_model_name = True
        all_objects = self.data.dataset.motion_dataset.cond_dict.keys() 
        random.seed(self.step)
        gen_args.object_type = random.sample(all_objects, gen_args.num_samples)
        random.seed(self.args.seed)
        all_sample_save_path = generate(gen_args, self.data.dataset.motion_dataset.cond_dict)
        self.train_platform.report_media(title='Motion', series='Predicted Motion', iteration=self.total_step(),
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
        if 'loss' in averaged:
            print('val_step[{}]: val_loss[{:0.5f}]'.format(current_step, averaged['loss']))
        for key, value in averaged.items():
            self.train_platform.report_scalar(name=key, value=value, iteration=current_step, group_name='Val')




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
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


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


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"


    def save(self):
            def save_checkpoint():
                def del_clip(state_dict):
                    # Do not save CLIP weights
                    clip_weights = [
                        e for e in state_dict.keys() if e.startswith('clip_model.')
                    ]
                    for e in clip_weights:
                        del state_dict[e]

                if self.use_fp16:
                    state_dict = self.model.state_dict()
                else:
                    state_dict = self.mp_trainer.master_params_to_state_dict(
                        self.mp_trainer.master_params)
                del_clip(state_dict)

                if self.args.use_ema and self.model_avg is not None:
                    # save both the model and the average model
                    state_dict_avg = self.model_avg.state_dict()
                    del_clip(state_dict_avg)
                    state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

                logger.log(f"saving model...")
                filename = self.ckpt_file_name()
                checkpoint_path = pjoin(self.save_dir, filename)
                if '://' in self.save_dir:
                    file_ctx = bf.BlobFile(bf.join(self.save_dir, filename), "wb")
                else:
                    file_ctx = open(checkpoint_path, "wb")
                with file_ctx as f:
                    torch.save(state_dict, f)

            save_checkpoint()

            opt_filename = f"opt{(self.total_step()):09d}.pt"
            opt_path = pjoin(self.save_dir, opt_filename)
            if '://' in self.save_dir:
                opt_ctx = bf.BlobFile(bf.join(self.save_dir, opt_filename), "wb")
            else:
                opt_ctx = open(opt_path, "wb")
            with opt_ctx as f:
                opt_state = self.opt.state_dict()
                if self.use_fp16:
                    # with fp16 we also save the state dict
                    opt_state = {
                        'opt': opt_state,
                        'scaler': self.scaler.state_dict(),
                    }

                torch.save(opt_state, f)
                
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

        matches = {file: re.match(r'opt(\d+).pt$', file) for file in os.listdir(self.args.save_dir)}
        models = {int(match.group(1)): file for file, match in matches.items() if match}

        return pjoin(self.args.save_dir, models[max(models)]) if models else None




def parse_resume_step_from_filename(filename):
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


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            




