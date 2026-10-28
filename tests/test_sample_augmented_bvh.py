"""CLI exclusivity and isolated random augmentations."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.sample_augmented_bvh import PreviewMotionDataset, parse_args
from data_loaders.truebones.data.dataset import MotionDataset


@pytest.mark.parametrize("argv", [
    [], ["--mode", "unknown"], ["--mode", "loop", "leaf-drop"],
    ["--mode", "loop", "--mode", "leaf-drop"],
    ["--mode", "motion-speed", "--motion-speed-aug", "0.9"],
    ["--mode", "loop", "--loop-only"],
    ["--mode", "bone-length", "--bone-length-aug-max", "0"],
    ["--mode", "bone-length", "--bone-length-aug-max", "1"],
    ["--mode", "bone-length", "--bone-length-aug-max", "nan"],
])
def test_cli_rejects_missing_mixed_or_disabled_mode(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["preview", *argv])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["loop", "motion-speed", "leaf-drop", "bone-length"])
def test_cli_accepts_one_mode(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", mode])
    assert parse_args().mode == mode


@pytest.mark.parametrize("mode", ["motion-speed", "leaf-drop", "bone-length"])
def test_other_modes_bypass_loop_stages_without_mutating_source(monkeypatch, mode):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=mode)
    data = {"motion_metadata": {"is_loop": True, "action_label": "walk"}}
    monkeypatch.setattr(MotionDataset, "_prepare_sample", lambda self, name, sample, **kw: sample)
    sample = dataset._prepare_sample("clip", data)
    assert sample["motion_metadata"]["is_loop"] is False
    assert data["motion_metadata"]["is_loop"] is True
    assert sample["motion_metadata"]["action_label"] == "walk"


def test_speed_mode_preserves_training_slow_only_flag(monkeypatch):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode="motion-speed", motion_speed_aug=1.3)
    captured = {}

    def sampler(self, length, budget, *, slow_only):
        captured["slow_only"] = slow_only
        return 250

    monkeypatch.setattr(MotionDataset, "_sample_motion_speed_target_length", sampler)
    assert dataset._sample_motion_speed_target_length(300, 120) == 250
    assert captured == {"slow_only": False}
    assert dataset._sample_motion_speed_target_length(30, 60, slow_only=True) == 250
    assert captured == {"slow_only": True}


def test_speed_preview_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", "motion-speed"])
    args = parse_args()
    assert args.motion_speed_aug == 1.3


def test_explicit_unit_speed_disables_random_speed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", "motion-speed", "--motion-speed-aug", "1.0"])
    args = parse_args()
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=args.mode, motion_speed_aug=args.motion_speed_aug)
    dataset.min_length = 20
    assert dataset._sample_motion_speed_target_length(300, 120) == 300


@pytest.mark.parametrize("mode", ["loop", "leaf-drop"])
def test_non_speed_modes_keep_the_recorded_length(mode):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=mode, motion_speed_aug=1.0)
    dataset.min_length = 20
    assert dataset._sample_motion_speed_target_length(300, 120) == 300
