"""CLI exclusivity, isolated random augmentations and shared speed fitting."""

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
])
def test_cli_rejects_missing_mixed_or_disabled_mode(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["preview", *argv])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["loop", "motion-speed", "leaf-drop"])
def test_cli_accepts_one_mode(monkeypatch, mode):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", mode])
    assert parse_args().mode == mode


@pytest.mark.parametrize("mode", ["motion-speed", "leaf-drop"])
def test_other_modes_bypass_loop_stages_without_mutating_source(monkeypatch, mode):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=mode)
    data = {"motion_metadata": {"is_loop": True, "action_label": "walk"}}
    monkeypatch.setattr(MotionDataset, "_prepare_sample", lambda self, name, sample, **kw: sample)
    sample = dataset._prepare_sample("clip", data)
    assert sample["motion_metadata"]["is_loop"] is False
    assert data["motion_metadata"]["is_loop"] is True
    assert sample["motion_metadata"]["action_label"] == "walk"


@pytest.mark.parametrize("mode", ["loop", "motion-speed", "leaf-drop"])
@pytest.mark.parametrize("length, expected", [(180, 120), (300, 200)])
def test_all_modes_speed_fit_overlong_anchored_actions(mode, length, expected):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(
        preview_mode=mode, motion_speed_aug=1.3 if mode == "motion-speed" else 1.0,
        motion_speed_aug_prob=float(mode == "motion-speed"),
    )
    dataset.min_length = 20
    assert dataset._sample_motion_speed_target_length(length, False, 120, fit_budget=True) == expected


def test_speed_mode_preserves_training_fit_flag(monkeypatch):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode="motion-speed", motion_speed_aug=1.3)
    captured = {}

    def sampler(self, length, is_loop, budget, *, fit_budget):
        captured["fit_budget"] = fit_budget
        return 250

    monkeypatch.setattr(MotionDataset, "_sample_motion_speed_target_length", sampler)
    assert dataset._sample_motion_speed_target_length(300, False, 120, fit_budget=True) == 250
    assert captured["fit_budget"] is True


def test_speed_preview_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", "motion-speed"])
    args = parse_args()
    assert args.motion_speed_aug == 1.3


def test_explicit_unit_speed_keeps_fitting_but_disables_random_speed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["preview", "--mode", "motion-speed", "--motion-speed-aug", "1.0"])
    args = parse_args()
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=args.mode, motion_speed_aug=args.motion_speed_aug)
    dataset.min_length = 20
    assert dataset._sample_motion_speed_target_length(300, False, 120, fit_budget=True) == 200
    assert dataset._sample_motion_speed_target_length(300, False, 120, fit_budget=False) == 300


@pytest.mark.parametrize("mode", ["loop", "leaf-drop"])
def test_non_speed_modes_do_not_speed_fit_phase_free_actions(mode):
    dataset = PreviewMotionDataset.__new__(PreviewMotionDataset)
    dataset.opt = SimpleNamespace(preview_mode=mode, motion_speed_aug=1.0)
    dataset.min_length = 20
    assert dataset._sample_motion_speed_target_length(300, False, 120, fit_budget=False) == 300
