from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from train.train_motion_scorer import compute_and_save_train_stats, find_latest_checkpoint


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="Checkpoint directory containing args.json/model*.pt/train_stats.npy, or a specific model*.pt checkpoint file.")
    parser.add_argument("--checkpoint_path", default="", type=str,
                        help="Optional explicit checkpoint path. Empty means latest model*.pt in checkpoint_dir.")
    return parser


def dict_to_namespace(values: dict) -> Namespace:
    return Namespace(**values)


def resolve_checkpoint_inputs(checkpoint_dir_arg: str, checkpoint_path_arg: str) -> tuple[Path, str]:
    checkpoint_dir = Path(checkpoint_dir_arg)
    checkpoint_path = checkpoint_path_arg

    if checkpoint_dir.is_file():
        if checkpoint_path:
            raise ValueError("--checkpoint_dir cannot point to a checkpoint file when --checkpoint_path is also set")
        checkpoint_path = str(checkpoint_dir)
        checkpoint_dir = checkpoint_dir.parent
    elif not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")

    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(str(checkpoint_dir), prefix="model")
    if not checkpoint_path:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir}")

    return checkpoint_dir, checkpoint_path


def main() -> int:
    args = build_parser().parse_args()
    checkpoint_dir, checkpoint_path = resolve_checkpoint_inputs(args.checkpoint_dir, args.checkpoint_path)
    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json was not found in {checkpoint_dir}")

    with open(args_path, "r", encoding="utf-8") as handle:
        saved_args = json.load(handle)
    train_args = dict_to_namespace(saved_args)
    train_args.save_dir = str(checkpoint_dir)
    train_args.checkpoint_path = checkpoint_path

    compute_and_save_train_stats(train_args)
    print(f"recomputed train_stats.npy for {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
