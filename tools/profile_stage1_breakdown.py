from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from types import MethodType, SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion import logger
from model.motion_transformer import GraphMultiHeadAttention
from train.train_anytop import create_training_data_loader
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.ml_platforms import NoPlatform
from utils.model_util import create_model_and_diffusion_general_skeleton


def _synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimerStore:
    def __init__(self) -> None:
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def timed(self, key: str, fn, *args, **kwargs):
        _synchronize_if_needed()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        _synchronize_if_needed()
        self.totals[key] += time.perf_counter() - start
        self.counts[key] += 1
        return result


def _wrap_bound_method(obj, method_name: str, timer_store: TimerStore, key: str) -> None:
    original = getattr(obj, method_name)

    def wrapped(self, *args, **kwargs):
        return timer_store.timed(key, original, *args, **kwargs)

    setattr(obj, method_name, MethodType(wrapped, obj))


def _wrap_attention_modules(model: torch.nn.Module, timer_store: TimerStore) -> None:
    for module in model.modules():
        if isinstance(module, GraphMultiHeadAttention):
            _wrap_bound_method(module, "forward", timer_store, "attention_graph_s")
        elif isinstance(module, torch.nn.MultiheadAttention):
            _wrap_bound_method(module, "forward", timer_store, "attention_mha_s")


def _load_args(run_dir: str) -> SimpleNamespace:
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return SimpleNamespace(**data)


def _prepare_args(args: SimpleNamespace, output_dir: str, warmup_steps: int, profile_steps: int) -> SimpleNamespace:
    args = SimpleNamespace(**vars(args))
    args.save_dir = output_dir
    args.output_dir = os.path.dirname(output_dir)
    args.auto_resume = False
    args.resume_checkpoint = ""
    args.load_optimizer_state = False
    args.eval_during_training = False
    args.gen_during_training = False
    args.ml_platform_type = "NoPlatform"
    args.use_torch_compile = False
    args.num_steps = warmup_steps + profile_steps + 5
    args.log_interval = max(warmup_steps + profile_steps + 1, int(getattr(args, "log_interval", 100)))
    args.save_interval = max(args.num_steps + 1, int(getattr(args, "save_interval", 1000)))
    return args


def _build_trainer(args: SimpleNamespace) -> TrainLoop:
    fixseed(
        args.seed,
        cudnn_benchmark=getattr(args, "cudnn_benchmark", True),
        allow_tf32=getattr(args, "allow_tf32", True),
    )
    os.makedirs(args.save_dir, exist_ok=True)
    dist_util.setup_dist(args.device)
    data = create_training_data_loader(args)
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())
    trainer = TrainLoop(args, NoPlatform(), model, diffusion, data)
    logger.set_level(logger.WARN)
    return trainer


def _run_profile(trainer: TrainLoop, warmup_steps: int, profile_steps: int) -> dict[str, object]:
    timer_store = TimerStore()
    _wrap_attention_modules(trainer.model, timer_store)
    data_iter = iter(trainer.data)

    total_profiled_steps = 0
    total_data_wait_s = 0.0
    total_step_s = 0.0
    total_loop_s = 0.0

    for step_index in range(warmup_steps + profile_steps):
        loop_start = time.perf_counter()
        _synchronize_if_needed()
        fetch_start = time.perf_counter()
        try:
            motion, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.data)
            motion, cond = next(data_iter)
        _synchronize_if_needed()
        data_wait_s = time.perf_counter() - fetch_start

        motion = trainer._move_batch_to_device(motion)
        cond = trainer._move_cond_to_device(cond)

        _synchronize_if_needed()
        step_start = time.perf_counter()
        trainer.run_step(motion, cond)
        _synchronize_if_needed()
        step_s = time.perf_counter() - step_start
        loop_s = time.perf_counter() - loop_start

        trainer.step += 1

        if step_index + 1 == warmup_steps:
            timer_store.totals.clear()
            timer_store.counts.clear()
            continue

        if step_index >= warmup_steps:
            total_profiled_steps += 1
            total_data_wait_s += data_wait_s
            total_step_s += step_s
            total_loop_s += loop_s

    attention_s = timer_store.totals.get("attention_graph_s", 0.0) + timer_store.totals.get("attention_mha_s", 0.0)
    other_step_s = max(total_step_s - attention_s, 0.0)

    denom = total_loop_s if total_loop_s > 0 else 1.0
    summary = {
        "profiled_steps": total_profiled_steps,
        "timings_s": {
            "loop_total": total_loop_s,
            "step_total": total_step_s,
            "data_wait": total_data_wait_s,
            "attention_total": attention_s,
            "other_step": other_step_s,
        },
        "percent_of_loop": {
            "data_wait": 100.0 * total_data_wait_s / denom,
            "attention_total": 100.0 * attention_s / denom,
            "other_step": 100.0 * other_step_s / denom,
        },
        "attention_breakdown": {
            "graph_attention_s": timer_store.totals.get("attention_graph_s", 0.0),
            "mha_attention_s": timer_store.totals.get("attention_mha_s", 0.0),
        },
        "call_counts": dict(timer_store.counts),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile AnyTop stage1 training time breakdown.")
    parser.add_argument(
        "--run-dir",
        default="save/stage1_tiny_overfit_locomotion_teacher_v4/stage1_pretrain",
        help="Existing stage1 run directory containing args.json.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument(
        "--output-json",
        default="outputs/profile_stage1_breakdown/summary.json",
        help="Where to write the profiling summary.",
    )
    args = parser.parse_args()

    run_args = _load_args(args.run_dir)
    output_json = os.path.abspath(args.output_json)
    output_dir = os.path.dirname(output_json)
    prepared_args = _prepare_args(run_args, output_dir, args.warmup_steps, args.profile_steps)
    trainer = _build_trainer(prepared_args)
    summary = _run_profile(trainer, args.warmup_steps, args.profile_steps)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()