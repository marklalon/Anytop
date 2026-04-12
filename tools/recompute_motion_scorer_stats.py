from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from model.motion_autoencoder import MotionAutoencoder
from train.train_motion_scorer import compute_and_save_train_stats, find_latest_checkpoint, select_model_state_dict


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="Checkpoint directory containing args.json, model checkpoints, and train_stats.npy.")
    parser.add_argument("--checkpoint_path", default="", type=str,
                        help="Optional explicit checkpoint path. Empty means latest model*.pt in checkpoint_dir.")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device for stat recomputation. Falls back to CPU if CUDA is unavailable.")
    return parser


def dict_to_namespace(values: dict) -> Namespace:
    return Namespace(**values)


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json was not found in {checkpoint_dir}")

    with open(args_path, "r", encoding="utf-8") as handle:
        saved_args = json.load(handle)
    train_args = dict_to_namespace(saved_args)
    train_args.save_dir = str(checkpoint_dir)

    checkpoint_path = args.checkpoint_path or find_latest_checkpoint(str(checkpoint_dir), prefix="model")
    if not checkpoint_path:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = MotionAutoencoder(
        feature_dim=int(saved_args.get("feature_dim", 13)),
        d_model=int(saved_args.get("d_model", 128)),
        latent_dim=int(saved_args.get("latent_dim", 64)),
        num_conv_layers=int(saved_args.get("num_conv_layers", 3)),
        decoder_num_conv_layers=int(saved_args.get("decoder_num_conv_layers", 0)),
        kernel_size=int(saved_args.get("kernel_size", 5)),
        max_joints=int(saved_args.get("max_joints", 143)),
        max_frames=int(saved_args.get("num_frames", 120)),
    ).to(device)
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(select_model_state_dict(payload, prefer_ema=True), strict=True)

    amp_dtype = str(saved_args.get("amp_dtype", "fp32")).lower()
    autocast_dtype = None
    amp_enabled = False
    if device.type == "cuda":
        if amp_dtype == "fp16":
            autocast_dtype = torch.float16
            amp_enabled = True
        elif amp_dtype == "bf16":
            autocast_dtype = torch.bfloat16
            amp_enabled = True

    compute_and_save_train_stats(train_args, model, device, autocast_dtype, amp_enabled)
    print(f"recomputed train_stats.npy for {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
