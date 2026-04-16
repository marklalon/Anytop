from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.fp16_util import count_nonfinite_gradients, format_nonfinite_stats, inspect_optimizer_state
from train.train_anytop import create_training_data_loader
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.ml_platforms import MLPlatform
from utils.model_util import create_model_and_diffusion_general_skeleton


def _load_args(run_dir: Path) -> SimpleNamespace:
    args_path = run_dir / "args.json"
    with open(args_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SimpleNamespace(**payload)


def _build_opt_checkpoint(model_checkpoint: Path) -> Path:
    step_text = model_checkpoint.stem.replace("model", "")
    return model_checkpoint.with_name(f"opt{step_text}.pt")


def _restore_rng_state(opt_checkpoint: Path) -> None:
    payload = torch.load(opt_checkpoint, map_location="cpu", weights_only=False)
    torch_rng_state = payload.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu().to(dtype=torch.uint8))
    cuda_rng_state = payload.get("cuda_rng_state")
    if cuda_rng_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu().to(dtype=torch.uint8) for state in cuda_rng_state])
    if "python_rng_state" in payload:
        random.setstate(payload["python_rng_state"])
    if "numpy_rng_state" in payload:
        np.random.set_state(payload["numpy_rng_state"])


def _compute_grad_summary(model: torch.nn.Module) -> dict[str, object]:
    grad_sq_sum = 0.0
    top_params: list[tuple[float, str]] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        grad_sq_sum += float(grad.pow(2).sum().item())
        max_abs = float(grad.abs().max().item())
        if len(top_params) < 5:
            top_params.append((max_abs, name))
            top_params.sort(reverse=True)
        elif max_abs > top_params[-1][0]:
            top_params[-1] = (max_abs, name)
            top_params.sort(reverse=True)
    return {
        "grad_norm": math.sqrt(max(grad_sq_sum, 0.0)),
        "top_abs_grad_params": [{"name": name, "max_abs_grad": value} for value, name in top_params],
    }


def _advance_to_step(loop: TrainLoop, target_step: int, print_interval: int) -> None:
    data_iter = iter(loop.data)
    while loop.total_step() < target_step - 1:
        try:
            motion, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(loop.data)
            motion, cond = next(data_iter)

        motion = loop._move_batch_to_device(motion)
        cond = loop._move_cond_to_device(cond)
        loop.run_step(motion, cond)
        loop.step += 1

        completed_step = loop.total_step()
        if print_interval > 0 and completed_step % print_interval == 0:
            print(f"advanced_to_step={completed_step}")

    loop._debug_data_iter = data_iter


def _next_batch(loop: TrainLoop):
    data_iter = getattr(loop, "_debug_data_iter", None)
    if data_iter is None:
        data_iter = iter(loop.data)
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(loop.data)
        batch = next(data_iter)
    loop._debug_data_iter = data_iter
    return batch


def _analyze_target_step(loop: TrainLoop) -> None:
    batch, cond = _next_batch(loop)
    batch = loop._move_batch_to_device(batch)
    cond = loop._move_cond_to_device(cond)

    current_step = loop.total_step()
    target_step = current_step + 1
    motion_names = cond["y"].get("motion_name", [])
    object_types = cond["y"].get("object_type", [])

    print(f"target_step={target_step}")
    print("motion_names=", motion_names)
    print("object_types=", object_types)

    t, weights = loop.schedule_sampler.sample(batch.shape[0], dist_util.dev())
    noise = torch.randn_like(batch)
    model_kwargs = loop._with_train_step(cond, current_step)

    def compute_losses():
        loop._maybe_mark_compile_step_begin()
        with loop._autocast_context():
            return loop.diffusion.training_losses(
                loop.forward_model,
                batch,
                t,
                model_kwargs=model_kwargs,
                noise=noise,
            )

    losses = compute_losses()

    scalar_losses = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            scalar_losses[key] = float((value.detach().float() * weights.float()).mean().item())
    print("weighted_losses=", json.dumps(scalar_losses, indent=2, sort_keys=True))
    print("sampled_t=", t.detach().cpu().tolist())

    components: list[tuple[str, torch.Tensor]] = [
        ("total_loss", (losses["loss"] * weights).mean()),
        ("l_simple", (losses["l_simple"] * weights).mean()),
    ]
    if "geodesic_loss" in losses:
        components.append(("geodesic", (loop.diffusion.lambda_geo * losses["geodesic_loss"] * weights).mean()))

    for name, _ in components:
        losses = compute_losses()
        component_lookup = {
            "total_loss": (losses["loss"] * weights).mean(),
            "l_simple": (losses["l_simple"] * weights).mean(),
        }
        if "geodesic_loss" in losses:
            component_lookup["geodesic"] = (loop.diffusion.lambda_geo * losses["geodesic_loss"] * weights).mean()

        scalar = component_lookup[name]
        loop.mp_trainer.zero_grad()
        scalar.backward()
        grad_stats = count_nonfinite_gradients(loop.mp_trainer.model_params)
        summary = _compute_grad_summary(loop.model)
        print(
            json.dumps(
                {
                    "component": name,
                    "scalar": float(scalar.detach().float().item()),
                    "nonfinite_grads": format_nonfinite_stats(grad_stats),
                    **summary,
                },
                indent=2,
                sort_keys=True,
            )
        )

    opt_state = inspect_optimizer_state(loop.opt)
    print("optimizer_state_before_step=", format_nonfinite_stats(opt_state))


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a specific resumed stage1 training step.")
    parser.add_argument("--run-dir", required=True, help="Directory containing args.json and checkpoints.")
    parser.add_argument("--resume-step", type=int, required=True, help="Checkpoint step to resume from.")
    parser.add_argument("--target-step", type=int, required=True, help="Step to analyze without applying the optimizer step.")
    parser.add_argument("--output-dir", required=True, help="Temporary debug save dir for this probe run.")
    parser.add_argument("--print-interval", type=int, default=50, help="Progress print interval while advancing.")
    parser.add_argument("--no-use-torch-compile", action="store_true", help="Disable torch.compile during probing.")
    args_cli = parser.parse_args()

    run_dir = Path(args_cli.run_dir).resolve()
    args = _load_args(run_dir)
    args.output_dir = str(Path(args_cli.output_dir).resolve().parent)
    args.save_dir = str(Path(args_cli.output_dir).resolve())
    args.auto_resume = False
    args.detect_anomaly = False
    args.resume_checkpoint = str((run_dir / f"model{args_cli.resume_step:09d}.pt").resolve())
    args.num_steps = int(args_cli.target_step)
    args.log_interval = max(int(args.log_interval), 10**9)
    args.save_interval = max(int(args.save_interval), 10**9)
    if args_cli.no_use_torch_compile:
        args.use_torch_compile = False

    os.makedirs(args.save_dir, exist_ok=True)

    fixseed(
        args.seed,
        cudnn_benchmark=getattr(args, "cudnn_benchmark", True),
        allow_tf32=getattr(args, "allow_tf32", True),
    )
    dist_util.setup_dist(args.device)

    data = create_training_data_loader(args)
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())
    loop = TrainLoop(args, MLPlatform(save_dir=args.save_dir), model, diffusion, data)

    opt_checkpoint = _build_opt_checkpoint(Path(args.resume_checkpoint))
    _restore_rng_state(opt_checkpoint)
    print(f"restored_exact_rng_from={opt_checkpoint}")
    print(f"initial_total_step={loop.total_step()}")

    _advance_to_step(loop, int(args_cli.target_step), int(args_cli.print_interval))
    _analyze_target_step(loop)


if __name__ == "__main__":
    main()