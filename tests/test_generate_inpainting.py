from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from sample.generate import (  # noqa: E402
    _parse_frame_ranges,
    _resolve_inpaint_joint_indices,
    _sample_batch,
    build_inpaint_mask,
)


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1)


class _CaptureDiffusion:
    def __init__(self, return_values: list[object] | None = None) -> None:
        self.last_call = None
        self.last_kwargs = None
        self.calls = []
        self.return_values = list(return_values or [])

    def _record_call(self, name: str, kwargs: dict) -> object:
        self.last_call = name
        self.last_kwargs = kwargs
        self.calls.append((name, kwargs))
        if self.return_values:
            return self.return_values.pop(0)
        return name

    def p_sample_loop(self, **kwargs):
        return self._record_call("ddpm", kwargs)

    def ddim_sample_loop(self, **kwargs):
        return self._record_call("ddim", kwargs)

    def plms_sample_loop(self, **kwargs):
        return self._record_call("plms", kwargs)


def _make_cond_entry() -> dict:
    return {
        "joints_names": ["Root", "Hip", "Knee"],
        "canonical_joint_names": ["Root", "Left Hip", "Left Knee"],
        "canonical_bvh_joint_names": ["Root", "LeftHip", "LeftKnee"],
        "parents": np.array([-1, 0, 1], dtype=np.int64),
    }


def _make_diffusion(num_steps: int = 2) -> GaussianDiffusion:
    betas = np.linspace(0.001, 0.002, num_steps, dtype=np.float64)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def test_parse_frame_ranges_clips_and_keeps_inclusive_bounds() -> None:
    assert _parse_frame_ranges("2-9", 4) == {2, 3}
    assert _parse_frame_ranges("3-1", 5) == {1, 2, 3}


def test_build_inpaint_mask_uses_all_real_joints_for_selected_frames() -> None:
    mask = build_inpaint_mask(
        _make_cond_entry(),
        inpaint_joints_arg="",
        inpaint_include_subtree=True,
        inpaint_frames_arg="1-2",
        batch_size=2,
        max_joints=5,
        n_frames=4,
    )

    assert tuple(mask.shape) == (2, 5, 1, 4)
    assert torch.all(mask[:, 3:, :, :] == 0.0)
    assert int(mask[0, :3, 0, :].sum().item()) == 6
    assert torch.all(mask[:, :3, 0, 0] == 0.0)
    assert torch.all(mask[:, :3, 0, 1:3] == 1.0)
    assert torch.all(mask[:, :3, 0, 3] == 0.0)


def test_build_inpaint_mask_accepts_aliases_and_expands_subtree() -> None:
    mask = build_inpaint_mask(
        _make_cond_entry(),
        inpaint_joints_arg="LeftHip",
        inpaint_include_subtree=True,
        inpaint_frames_arg="0",
        batch_size=1,
        max_joints=3,
        n_frames=2,
    )

    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 1, 0, 0].item() == 1.0
    assert mask[0, 2, 0, 0].item() == 1.0


def test_resolve_inpaint_joint_indices_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown joint name"):
        _resolve_inpaint_joint_indices(_make_cond_entry(), "MissingJoint", True)


def test_sample_batch_routes_inpainting_through_ddpm_from_pure_noise() -> None:
    diffusion = _CaptureDiffusion()
    model = _DummyModel()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    inpaint_mask = torch.zeros((1, 3, 1, 4), dtype=torch.float32)

    result = _sample_batch(
        diffusion=diffusion,
        model=model,
        model_kwargs={},
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        skip_timesteps=0,
        inpaint_mask=inpaint_mask,
    )

    assert result == "ddpm"
    assert diffusion.last_call == "ddpm"
    assert len(diffusion.calls) == 1
    assert diffusion.last_kwargs["init_image"] is None
    assert diffusion.last_kwargs["skip_timesteps"] == 0
    assert torch.equal(diffusion.last_kwargs["inpaint_reference"], reference_motion)
    assert torch.equal(diffusion.last_kwargs["inpaint_mask"], inpaint_mask)
    assert tuple(diffusion.last_kwargs["noise"].shape) == sample_shape


def test_sample_batch_forwards_repaint_resampling_args_to_ddim() -> None:
    diffusion = _CaptureDiffusion()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    inpaint_mask = torch.zeros((1, 3, 1, 4), dtype=torch.float32)

    result = _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(),
        model_kwargs={},
        sampling_method="ddim",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        skip_timesteps=0,
        inpaint_mask=inpaint_mask,
        repaint_jump_length=2,
        repaint_jump_n_sample=3,
    )

    assert result == "ddim"
    assert diffusion.last_kwargs["repaint_jump_length"] == 2
    assert diffusion.last_kwargs["repaint_jump_n_sample"] == 3


def test_sample_batch_injects_cross_limb_unreliable_mask_for_single_inpaint_pass() -> None:
    diffusion = _CaptureDiffusion()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    inpaint_mask = torch.tensor(
        [[[[0.0, 1.0, 0.0, 0.0]],
          [[1.0, 0.0, 1.0, 0.0]],
          [[0.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    model_kwargs = {"y": {"existing": torch.tensor([1.0])}}

    _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(),
        model_kwargs=model_kwargs,
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        skip_timesteps=0,
        inpaint_mask=inpaint_mask,
    )

    routed_y = diffusion.last_kwargs["model_kwargs"]["y"]
    expected = torch.cat(
        [
            torch.zeros(1, 1, 3, dtype=inpaint_mask.dtype),
            inpaint_mask.squeeze(2).permute(0, 2, 1).contiguous(),
        ],
        dim=1,
    ).transpose(0, 1).contiguous()
    assert torch.equal(routed_y["cross_limb_unreliable_mask"], expected)
    assert torch.equal(routed_y["existing"], model_kwargs["y"]["existing"])
    assert "cross_limb_unreliable_mask" not in model_kwargs["y"]


def test_sample_batch_uses_two_pass_ddpm_for_inpaint_with_skip_timesteps() -> None:
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    varied_motion = torch.full(sample_shape, 7.0, dtype=torch.float32)
    inpaint_mask = torch.tensor(
        [[[[0.0, 1.0, 0.0, 0.0]],
          [[1.0, 0.0, 1.0, 0.0]],
          [[0.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    diffusion = _CaptureDiffusion(return_values=[varied_motion, "ddpm-final"])
    model_kwargs = {"y": {"existing": torch.tensor([1.0])}}

    result = _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(),
        model_kwargs=model_kwargs,
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        skip_timesteps=80,
        inpaint_mask=inpaint_mask,
    )

    assert result == "ddpm-final"
    assert [name for name, _ in diffusion.calls] == ["ddpm", "ddpm"]

    first_kwargs = diffusion.calls[0][1]
    second_kwargs = diffusion.calls[1][1]

    assert torch.equal(first_kwargs["init_image"], reference_motion)
    assert first_kwargs["skip_timesteps"] == 80
    assert first_kwargs["inpaint_reference"] is None
    assert first_kwargs["inpaint_mask"] is None
    assert tuple(first_kwargs["noise"].shape) == sample_shape
    assert "cross_limb_unreliable_mask" not in first_kwargs["model_kwargs"]["y"]

    assert second_kwargs["init_image"] is None
    assert second_kwargs["skip_timesteps"] == 0
    assert torch.equal(second_kwargs["inpaint_reference"], varied_motion)
    assert torch.equal(second_kwargs["inpaint_mask"], inpaint_mask)
    assert tuple(second_kwargs["noise"].shape) == sample_shape
    expected = torch.cat(
        [
            torch.zeros(1, 1, 3, dtype=inpaint_mask.dtype),
            inpaint_mask.squeeze(2).permute(0, 2, 1).contiguous(),
        ],
        dim=1,
    ).transpose(0, 1).contiguous()
    assert torch.equal(second_kwargs["model_kwargs"]["y"]["cross_limb_unreliable_mask"], expected)
    assert torch.equal(second_kwargs["model_kwargs"]["y"]["existing"], model_kwargs["y"]["existing"])
    assert "cross_limb_unreliable_mask" not in model_kwargs["y"]


def test_sample_batch_requires_reference_for_inpainting() -> None:
    with pytest.raises(ValueError, match="reference_motion"):
        _sample_batch(
            diffusion=_CaptureDiffusion(),
            model=_DummyModel(),
            model_kwargs={},
            sampling_method="ddpm",
            sample_shape=(1, 3, 13, 4),
            ddim_eta=0.0,
            seed=123,
            device=torch.device("cpu"),
            reference_motion=None,
            skip_timesteps=0,
            inpaint_mask=torch.zeros((1, 3, 1, 4), dtype=torch.float32),
        )


def test_sample_batch_rejects_plms_for_inpainting() -> None:
    with pytest.raises(ValueError, match="PLMS does not support motion inpainting"):
        _sample_batch(
            diffusion=_CaptureDiffusion(),
            model=_DummyModel(),
            model_kwargs={},
            sampling_method="plms",
            sample_shape=(1, 3, 13, 4),
            ddim_eta=0.0,
            seed=123,
            device=torch.device("cpu"),
            reference_motion=torch.zeros((1, 3, 13, 4), dtype=torch.float32),
            skip_timesteps=0,
            inpaint_mask=torch.zeros((1, 3, 1, 4), dtype=torch.float32),
        )


def test_p_sample_loop_returns_projected_final_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=2)
    sample = torch.tensor([[[[10.0, 20.0]]]], dtype=torch.float32)
    reference = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)

    def fake_p_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "p_sample", fake_p_sample)

    result = diffusion.p_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        inpaint_mask=mask,
        inpaint_reference=reference,
    )

    assert torch.equal(result, torch.tensor([[[[10.0, 2.0]]]], dtype=torch.float32))


def test_p_sample_loop_visits_final_timestep_without_inpaint(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    reverse_ts: list[int] = []
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    def fake_p_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "p_sample", fake_p_sample)

    diffusion.p_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
    )

    assert reverse_ts == [3, 2, 1, 0]


def test_build_repaint_schedule_revisits_anchor_timesteps_and_keeps_final_step() -> None:
    diffusion = _make_diffusion(num_steps=4)

    schedule = diffusion._build_repaint_schedule(
        start_t=3,
        jump_length=1,
        jump_n_sample=2,
    )

    assert schedule == [3, 2, 3, 2, 1, 2, 1, 0, -1]


def test_p_sample_loop_uses_repaint_time_travel(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    reverse_ts: list[int] = []
    forward_ts: list[int] = []

    def fake_p_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, const_noise=False):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    def fake_repaint_time_travel(sample_in, t, const_noise=False):
        forward_ts.append(int(t.item()))
        return sample_in

    monkeypatch.setattr(diffusion, "p_sample", fake_p_sample)
    monkeypatch.setattr(diffusion, "_repaint_time_travel", fake_repaint_time_travel)

    diffusion.p_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        inpaint_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
        inpaint_reference=torch.zeros_like(sample),
        repaint_jump_length=1,
        repaint_jump_n_sample=2,
    )

    assert reverse_ts == [3, 3, 2, 2, 1, 0]
    assert forward_ts == [3, 2]


def test_ddim_sample_loop_returns_projected_final_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=2)
    sample = torch.tensor([[[[7.0, 9.0]]]], dtype=torch.float32)
    reference = torch.tensor([[[[3.0, 4.0]]]], dtype=torch.float32)
    mask = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

    def fake_ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "ddim_sample", fake_ddim_sample)

    result = diffusion.ddim_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        eta=0.0,
        inpaint_mask=mask,
        inpaint_reference=reference,
    )

    assert torch.equal(result, torch.tensor([[[[3.0, 9.0]]]], dtype=torch.float32))


def test_ddim_sample_loop_visits_final_timestep_without_inpaint(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    reverse_ts: list[int] = []
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    def fake_ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    monkeypatch.setattr(diffusion, "ddim_sample", fake_ddim_sample)

    diffusion.ddim_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        eta=0.0,
    )

    assert reverse_ts == [3, 2, 1, 0]


def test_ddim_sample_loop_rejects_const_noise() -> None:
    diffusion = _make_diffusion(num_steps=2)

    with pytest.raises(NotImplementedError):
        diffusion.ddim_sample_loop(
            _DummyModel(),
            shape=(1, 1, 1, 1),
            const_noise=True,
        )


def test_ddim_sample_loop_uses_repaint_time_travel(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusion = _make_diffusion(num_steps=4)
    sample = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    reverse_ts: list[int] = []
    forward_ts: list[int] = []

    def fake_ddim_sample(model, x, t, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        reverse_ts.append(int(t.item()))
        return {"sample": sample.clone(), "pred_xstart": torch.zeros_like(sample)}

    def fake_repaint_time_travel(sample_in, t, const_noise=False):
        forward_ts.append(int(t.item()))
        return sample_in

    monkeypatch.setattr(diffusion, "ddim_sample", fake_ddim_sample)
    monkeypatch.setattr(diffusion, "_repaint_time_travel", fake_repaint_time_travel)

    diffusion.ddim_sample_loop(
        _DummyModel(),
        shape=tuple(sample.shape),
        noise=torch.zeros_like(sample),
        device=torch.device("cpu"),
        progress=False,
        eta=0.0,
        inpaint_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
        inpaint_reference=torch.zeros_like(sample),
        repaint_jump_length=1,
        repaint_jump_n_sample=2,
    )

    assert reverse_ts == [3, 3, 2, 2, 1, 0]
    assert forward_ts == [3, 2]