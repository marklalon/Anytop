import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.get_data import get_dataset_loader
from data_loaders.truebones.truebones_utils.get_opt import get_opt
import BVH
from data_loaders.truebones.truebones_utils.motion_process import recover_animation_from_motion_np
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run restoration inference on offline corrupted-reference samples.")
    parser.add_argument("--model-path", required=True, help="Path to model####.pt checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Directory to write restoration outputs into.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for inference.")
    parser.add_argument("--num-repetitions", default=1, type=int, help="How many restoration samples to draw per input.")
    parser.add_argument("--sample-limit", default=0, type=int, help="Limit number of sample subdirectories. 0 keeps all.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to sample from.")
    parser.add_argument("--objects-subset", default="", help="Override object subset. Defaults to checkpoint args.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers for loading stored corrupted references.")
    return parser.parse_args()


def load_model_args(model_path: Path) -> SimpleNamespace:
    args_path = model_path.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Arguments json file was not found next to checkpoint: {args_path}")
    with open(args_path, "r", encoding="utf-8") as handle:
        return SimpleNamespace(**json.load(handle))


def move_cond_to_device(cond: dict, device: torch.device) -> dict:
    moved = {"y": {}}
    for key, value in cond["y"].items():
        moved["y"][key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main() -> int:
    args = parse_args()
    fixseed(args.seed)
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = load_model_args(model_path)
    model_args.model_path = str(model_path)
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    if args.objects_subset:
        model_args.objects_subset = args.objects_subset

    dist_util.setup_dist(args.device)
    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(model_args)

    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location="cpu")
    if "model_avg" in state_dict:
        print("EMA checkpoint detected, loading model_avg weights.")
        state_dict = state_dict["model_avg"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    load_model(model, state_dict)

    model.to(dist_util.dev())
    model.eval()

    data = get_dataset_loader(
        batch_size=args.batch_size,
        num_frames=model_args.num_frames,
        split=args.split,
        temporal_window=model_args.temporal_window,
        t5_name=model_args.t5_name,
        balanced=False,
        objects_subset=model_args.objects_subset,
        num_workers=args.num_workers,
        prefetch_factor=getattr(model_args, "prefetch_factor", 2),
        sample_limit=args.sample_limit,
        shuffle=False,
        drop_last=False,
    )
    manifest = []
    produced = 0

    for batch_index, (_, model_kwargs) in enumerate(data):
        model_kwargs = move_cond_to_device(model_kwargs, dist_util.dev())
        max_frames = int(model_kwargs["y"]["lengths"].max().item())
        batch_size = int(model_kwargs["y"]["lengths"].shape[0])

        for rep_index in range(args.num_repetitions):
            print(f"### Restoring batch {batch_index} [repetition #{rep_index}]")
            restored = diffusion.p_sample_loop(
                model,
                (batch_size, opt.max_joints, model.feature_len, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )

            for item_index in range(batch_size):
                object_type = model_kwargs["y"]["object_type"][item_index]
                motion_name = model_kwargs["y"]["motion_name"][item_index]
                n_joints = int(model_kwargs["y"]["n_joints"][item_index].item())
                length = int(model_kwargs["y"]["lengths"][item_index].item())
                parents = [int(parent) for parent in cond_dict[object_type]["parents"]]
                mean = cond_dict[object_type]["mean"].astype(np.float32)
                std = cond_dict[object_type]["std"].astype(np.float32)

                restored_norm = restored[item_index, :n_joints, :, :length].detach().cpu().permute(2, 0, 1).numpy()
                restored_motion = restored_norm * std[None, :n_joints, :] + mean[None, :n_joints, :]
                sample_output_dir = output_dir / Path(motion_name).stem
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_output_dir / f"restored_rep{rep_index}.npy", restored_motion.astype(np.float32))
                offsets = cond_dict[object_type]["offsets"]
                joints_names = cond_dict[object_type]["joints_names"]
                out_anim, has_animated_pos = recover_animation_from_motion_np(restored_motion, parents, offsets)
                if out_anim is not None:
                    BVH.save(str(sample_output_dir / f"restored_rep{rep_index}.bvh"), out_anim, joints_names,
                             positions=has_animated_pos)

                manifest.append({
                    "motion_file": motion_name,
                    "output_dir": str(sample_output_dir),
                    "object_type": object_type,
                    "length": length,
                    "repetition": rep_index,
                    "source_batch_index": batch_index,
                })
                produced += 1

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved {produced} restored outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())