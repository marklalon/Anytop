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

from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from model.conditioners import T5Conditioner
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run restoration inference on offline corrupted-reference samples.")
    parser.add_argument("--model-path", required=True, help="Path to model####.pt checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory containing exported corrupted sample subdirectories.")
    parser.add_argument("--output-dir", required=True, help="Directory to write restoration outputs into.")
    parser.add_argument("--device", default=0, type=int, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed.")
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for inference.")
    parser.add_argument("--num-repetitions", default=1, type=int, help="How many restoration samples to draw per input.")
    parser.add_argument("--cond-path", default="", help="Optional cond.npy override.")
    parser.add_argument("--sample-limit", default=0, type=int, help="Limit number of sample subdirectories. 0 keeps all.")
    return parser.parse_args()


def load_model_args(model_path: Path) -> SimpleNamespace:
    args_path = model_path.parent / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Arguments json file was not found next to checkpoint: {args_path}")
    with open(args_path, "r", encoding="utf-8") as handle:
        return SimpleNamespace(**json.load(handle))


def encode_joints_names(joints_names, t5_conditioner):
    names_tokens = t5_conditioner.tokenize(joints_names)
    return t5_conditioner(names_tokens)


def infer_sample_dirs(input_dir: Path, sample_limit: int) -> list[Path]:
    sample_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if sample_limit > 0:
        sample_dirs = sample_dirs[:sample_limit]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found under {input_dir}")
    return sample_dirs


def load_sample(sample_dir: Path) -> dict[str, object]:
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {sample_dir}")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    reference_motion = np.load(sample_dir / "corrupted_reference.npy").astype(np.float32)
    confidence = np.load(sample_dir / "soft_confidence_mask.npy").astype(np.float32)
    clean_target_path = sample_dir / "clean_target.npy"
    clean_target = np.load(clean_target_path).astype(np.float32) if clean_target_path.exists() else None
    return {
        "sample_dir": sample_dir,
        "object_type": metadata["object_type"],
        "reference_motion": reference_motion,
        "soft_confidence_mask": confidence,
        "clean_target": clean_target,
    }


def build_condition_batch(samples, cond_dict, temporal_window, t5_conditioner, max_joints, feature_len):
    batches = []
    for sample in samples:
        object_type = sample["object_type"]
        reference_motion = sample["reference_motion"]
        confidence = sample["soft_confidence_mask"]
        object_cond = cond_dict[object_type]
        n_frames = reference_motion.shape[0]
        parents = object_cond["parents"]
        n_joints = len(parents)
        mean = object_cond["mean"].astype(np.float32)
        std = object_cond["std"].astype(np.float32)
        tpos_first_frame = np.nan_to_num((object_cond["tpos_first_frame"] - mean) / (std + 1e-6)).astype(np.float32)
        joints_names_embs = encode_joints_names(object_cond["joints_names"], t5_conditioner).detach().cpu().numpy()
        reference_norm = np.nan_to_num((reference_motion - mean[None, :]) / (std[None, :] + 1e-6)).astype(np.float32)
        batches.append([
            np.zeros((n_frames, n_joints, feature_len), dtype=np.float32),
            n_frames,
            parents,
            tpos_first_frame,
            object_cond["offsets"],
            create_temporal_mask_for_window(temporal_window, n_frames),
            object_cond["joints_graph_dist"],
            object_cond["joint_relations"],
            object_type,
            joints_names_embs,
            0,
            mean,
            std,
            max_joints,
            reference_norm,
            confidence,
            None,
        ])
    return truebones_batch_collate(batches)


def move_cond_to_device(cond: dict, device: torch.device) -> dict:
    moved = {"y": {}}
    for key, value in cond["y"].items():
        moved["y"][key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def main() -> int:
    args = parse_args()
    fixseed(args.seed)
    model_path = Path(args.model_path).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = load_model_args(model_path)
    model_args.model_path = str(model_path)
    model_args.device = args.device
    model_args.batch_size = args.batch_size

    dist_util.setup_dist(args.device)
    opt = get_opt(args.device)
    cond_dict = np.load(args.cond_path, allow_pickle=True).item() if args.cond_path else np.load(opt.cond_file, allow_pickle=True).item()

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

    print("Loading T5 model")
    t5_device = "cuda" if args.device >= 0 else "cpu"
    t5_conditioner = T5Conditioner(name=model_args.t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device=t5_device)
    model.to(dist_util.dev())
    model.eval()

    sample_dirs = infer_sample_dirs(input_dir, args.sample_limit)
    samples = [load_sample(sample_dir) for sample_dir in sample_dirs]
    manifest = []

    for start_index in range(0, len(samples), args.batch_size):
        batch_samples = samples[start_index:start_index + args.batch_size]
        _, model_kwargs = build_condition_batch(
            batch_samples,
            cond_dict,
            model_args.temporal_window,
            t5_conditioner,
            max_joints=opt.max_joints,
            feature_len=opt.feature_len,
        )
        model_kwargs = move_cond_to_device(model_kwargs, dist_util.dev())
        max_frames = int(model_kwargs["y"]["lengths"].max().item())

        for rep_index in range(args.num_repetitions):
            print(f"### Restoring batch starting at {start_index} [repetition #{rep_index}]")
            restored = diffusion.p_sample_loop(
                model,
                (len(batch_samples), opt.max_joints, model.feature_len, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )

            for item_index, sample in enumerate(batch_samples):
                object_type = sample["object_type"]
                n_joints = int(model_kwargs["y"]["n_joints"][item_index].item())
                length = int(model_kwargs["y"]["lengths"][item_index].item())
                parents = [int(parent) for parent in cond_dict[object_type]["parents"]]
                mean = cond_dict[object_type]["mean"].astype(np.float32)
                std = cond_dict[object_type]["std"].astype(np.float32)

                restored_norm = restored[item_index, :n_joints, :, :length].detach().cpu().permute(2, 0, 1).numpy()
                restored_motion = restored_norm * std[None, :n_joints, :] + mean[None, :n_joints, :]
                restored_positions = recover_from_bvh_ric_np(restored_motion)

                sample_output_dir = output_dir / sample["sample_dir"].name
                sample_output_dir.mkdir(parents=True, exist_ok=True)
                np.save(sample_output_dir / f"restored_rep{rep_index}.npy", restored_motion.astype(np.float32))
                plot_general_skeleton_3d_motion(
                    str(sample_output_dir / f"restored_rep{rep_index}.mp4"),
                    parents,
                    restored_positions,
                    title=f"restored_rep{rep_index}",
                    fps=opt.fps,
                )

                manifest.append({
                    "input_sample_dir": str(sample["sample_dir"]),
                    "output_dir": str(sample_output_dir),
                    "object_type": object_type,
                    "length": length,
                    "repetition": rep_index,
                    "clean_target_available": sample["clean_target"] is not None,
                })

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved {len(manifest)} restored outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())