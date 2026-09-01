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
    _close_loop_root_xz_via_velocity,
    _contiguous_frame_runs,
    _finalize_output_lengths,
    _map_frame_ranges_to_internal,
    _parse_frame_ranges,
    _prepare_img2img_reference_bundle,
    _reanchor_inpaint_root_y_via_velocity,
    _reground_inpaint_joint_y,
    _resolve_inpaint_joint_indices,
    _sample_batch,
    _validate_reference_motion_path,
    build_inpaint_mask,
    create_condition,
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


def _make_cond_entry() -> dict:
    return {
        "joints_names": ["Root", "Hip", "Knee"],
        "canonical_joint_names": ["Root", "Left Hip", "Left Knee"],
        "canonical_bvh_joint_names": ["Root", "LeftHip", "LeftKnee"],
        "parents": np.array([-1, 0, 1], dtype=np.int64),
    }


def _make_full_cond_entry(n_joints: int, feature_len: int = 13) -> dict:
    parents = np.array([-1] + list(range(n_joints - 1)), dtype=np.int64)
    return {
        "joints_names": [f"Joint{i}" for i in range(n_joints)],
        "canonical_joint_names": [f"Joint{i}" for i in range(n_joints)],
        "canonical_bvh_joint_names": [f"Joint{i}" for i in range(n_joints)],
        "parents": parents,
        "canonical_feature_mean": np.zeros((feature_len,), dtype=np.float32),
        "canonical_feature_std": np.ones((feature_len,), dtype=np.float32),
        "rest_pose": np.zeros((n_joints, feature_len), dtype=np.float32),
        "joint_relations": np.zeros((n_joints, n_joints), dtype=np.int64),
        "joints_graph_dist": np.zeros((n_joints, n_joints), dtype=np.int64),
        "offsets": np.zeros((n_joints, 3), dtype=np.float32),
        "joints_names_embs": np.zeros((n_joints, 4), dtype=np.float32),
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


def test_map_frame_ranges_to_internal_preserves_contiguous_spans() -> None:
    assert _map_frame_ranges_to_internal("0-19", 20, 60) == "0-59"
    assert _map_frame_ranges_to_internal("10-20", 30, 60) == "20-41"
    assert _map_frame_ranges_to_internal("0-119", 120, 60) == "0-59"


def test_finalize_output_lengths_returns_frames_and_playspeed() -> None:
    requested, target, playspeed = _finalize_output_lengths(
        requested_frames=90, min_length=20, internal_num_frames=60
    )
    assert requested == 90
    assert target == 90
    assert playspeed == pytest.approx(90.0 / 60.0)


def test_finalize_output_lengths_rejects_out_of_window() -> None:
    with pytest.raises(SystemExit):
        _finalize_output_lengths(requested_frames=10, min_length=20, internal_num_frames=60)
    with pytest.raises(SystemExit):
        _finalize_output_lengths(requested_frames=121, min_length=20, internal_num_frames=60)


def test_validate_reference_motion_path_accepts_supported_suffixes() -> None:
    assert _validate_reference_motion_path("clip.npy") == ".npy"
    assert _validate_reference_motion_path("clip.fbx") == ".fbx"
    assert _validate_reference_motion_path("clip.glb") == ".glb"
    assert _validate_reference_motion_path("clip.gltf") == ".gltf"


def test_validate_reference_motion_path_rejects_unsupported_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported reference motion format"):
        _validate_reference_motion_path("clip.txt")


def test_prepare_reference_bundle_uses_preloaded_cropped_features() -> None:
    # Crop path: feed exactly M=40 frames (as main() does for R > M). The bundle
    # must consume the preloaded array verbatim (no disk load) and not re-trim it.
    n_joints, feat = 3, 13
    cond = _make_full_cond_entry(n_joints, feature_len=feat)
    preloaded = np.random.default_rng(0).normal(
        size=(40, n_joints, feat)
    ).astype(np.float32)

    bundle = _prepare_img2img_reference_bundle(
        reference_motion_path="/nonexistent/should_not_be_loaded.npy",
        target_type="Horse",
        target_cond=cond,
        max_joints=n_joints,
        target_feature_len=feat,
        batch_size=2,
        requested_output_frame_count=60,
        requested_visible_frame_count=40,
        preloaded_features=preloaded,
    )

    assert bundle["loaded_reference_frame_count"] == 40
    assert bundle["loaded_reference_joint_count"] == n_joints
    # The model is a fixed-window model trained only at num_frames. The bundle
    # always runs at that native window (requested_output_frame_count=60) and
    # resamples the shorter reference up to it; the requested output length is
    # honored later by resampling the sampled motion. reference_source_frame_count
    # records the pre-resample reference length (40) for playspeed.
    assert bundle["output_frame_count"] == 60
    assert bundle["reference_source_frame_count"] == 40
    assert tuple(bundle["reference_motion"].shape) == (2, n_joints, feat, 60)


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


def test_create_condition_can_sample_at_target_joint_count() -> None:
    cond_dict = {"Horse": _make_full_cond_entry(3)}

    motion_batch, model_kwargs = create_condition(
        ["Horse", "Horse"],
        cond_dict,
        n_frames=4,
        max_joints=3,
        feature_len=13,
    )

    y = model_kwargs["y"]
    assert tuple(motion_batch.shape) == (2, 3, 13, 4)
    assert tuple(y["joints_padding_mask"].shape) == (2, 1, 1, 4, 4)
    assert tuple(y["graph_dist"].shape) == (2, 3, 3)
    assert torch.equal(y["n_joints"], torch.tensor([3, 3]))
    # The output coordinate frame is an unconditional model input: AnyTop.forward
    # reads it every step, so the generation path has to stack it per sample too.
    assert tuple(y["canonical_feature_mean"].shape) == (2, 13)
    assert tuple(y["canonical_feature_std"].shape) == (2, 13)


def test_create_condition_rejects_cond_entry_without_canonical_stats() -> None:
    # A stats-free cond can only produce a y the model cannot read, so generation
    # must fail loudly at the boundary rather than hand forward an incomplete y.
    entry = _make_full_cond_entry(3)
    del entry["canonical_feature_mean"]
    cond_dict = {"Horse": entry}

    with pytest.raises(KeyError, match="canonical_feature_mean"):
        create_condition(
            ["Horse"],
            cond_dict,
            n_frames=4,
            max_joints=3,
            feature_len=13,
        )


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


def test_sample_batch_applies_skip_timesteps_only_inside_inpaint_mask() -> None:
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    inpaint_mask = torch.tensor(
        [[[[0.0, 1.0, 0.0, 0.0]],
          [[1.0, 0.0, 1.0, 0.0]],
          [[0.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    diffusion = _CaptureDiffusion(return_values=["ddpm-final"])
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
    assert [name for name, _ in diffusion.calls] == ["ddpm"]

    kwargs = diffusion.calls[0][1]

    assert torch.equal(kwargs["init_image"], reference_motion)
    assert kwargs["skip_timesteps"] == 80
    assert torch.equal(kwargs["inpaint_reference"], reference_motion)
    assert torch.equal(kwargs["inpaint_mask"], inpaint_mask)
    assert tuple(kwargs["noise"].shape) == sample_shape
    expected = torch.cat(
        [
            torch.zeros(1, 1, 3, dtype=inpaint_mask.dtype),
            inpaint_mask.squeeze(2).permute(0, 2, 1).contiguous(),
        ],
        dim=1,
    ).transpose(0, 1).contiguous()
    assert torch.equal(kwargs["model_kwargs"]["y"]["cross_limb_unreliable_mask"], expected)
    assert torch.equal(kwargs["model_kwargs"]["y"]["existing"], model_kwargs["y"]["existing"])
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


def _make_root_y_motion(pos_y, vel_y, root_idx=0, n_joints=2, n_feat=13):
    """Build a (F, J, C) motion_np tensor with the translation-root's
    pos_y / vel_y channels set, other channels zeroed.
    """
    pos_y = np.asarray(pos_y, dtype=np.float32)
    vel_y = np.asarray(vel_y, dtype=np.float32)
    F = pos_y.shape[0]
    motion = np.zeros((F, n_joints, n_feat), dtype=np.float32)
    motion[:, root_idx, 1] = pos_y
    motion[:, root_idx, 10] = vel_y
    return motion


def test_contiguous_frame_runs_basic():
    assert _contiguous_frame_runs(set()) == []
    assert _contiguous_frame_runs({7}) == [(7, 7)]
    assert _contiguous_frame_runs({3, 4, 5}) == [(3, 5)]
    assert _contiguous_frame_runs({1, 2, 5, 6, 7, 10}) == [(1, 2), (5, 7), (10, 10)]


def test_reanchor_root_y_closes_both_seams():
    # Reference Y trajectory: flat 1.0 outside, model produced a downward
    # bias (~0.4) inside frames 3..6 with a low-amplitude vel_y wiggle. After
    # fix, pos_y at frame a-1 / b+1 must remain untouched and the integrated
    # trajectory must bridge them exactly (in the integral sense — the per-
    # step adjustment is adjust/(L+1)).
    pos_y_ref = np.array(
        [1.0, 1.0, 1.0, 0.4, 0.42, 0.41, 0.43, 1.0, 1.0, 1.0], dtype=np.float32
    )
    vel_y_model = np.array(
        [0.0, 0.0, 0.0, 0.02, -0.01, 0.015, -0.02, 0.0, 0.0, 0.0], dtype=np.float32
    )
    motion = _make_root_y_motion(pos_y_ref, vel_y_model)
    _reanchor_inpaint_root_y_via_velocity(motion, spans=[(3, 6)])

    fixed_pos_y = motion[:, 0, 1]
    np.testing.assert_array_equal(fixed_pos_y[:3], pos_y_ref[:3])
    np.testing.assert_array_equal(fixed_pos_y[7:], pos_y_ref[7:])
    assert np.all(fixed_pos_y[3:7] > 0.9), f"Y not lifted: {fixed_pos_y[3:7]}"

    # Recompute the expected trajectory from first principles so the test
    # also serves as documentation of the formula.
    a, b = 3, 6
    L = b - a + 1
    integrated = pos_y_ref[a - 1] + np.cumsum(vel_y_model[a - 1:b], dtype=np.float64)
    integrated_at_b_plus_1 = integrated[-1] + float(vel_y_model[b])
    adjust = float(pos_y_ref[b + 1]) - integrated_at_b_plus_1
    ramp = np.arange(1, L + 1, dtype=np.float64) / float(L + 1)
    expected = integrated + adjust * ramp
    np.testing.assert_allclose(fixed_pos_y[a:b + 1], expected, atol=1e-6)

    # Seam closure happens in the integral sense: the total Y change from
    # the left to the right anchor across the span matches the reference,
    # and each per-step deviation is bounded by |adjust|/(L+1).
    per_step_bound = abs(adjust) / (L + 1) + 1e-6
    assert abs(
        (fixed_pos_y[a] - pos_y_ref[a - 1]) - vel_y_model[a - 1]
    ) <= per_step_bound
    assert abs(
        (pos_y_ref[b + 1] - fixed_pos_y[b]) - vel_y_model[b]
    ) <= per_step_bound


def test_reanchor_root_y_preserves_consistent_trajectory():
    # When pos_y[k+1] == pos_y[k] + vel_y[k] everywhere (clean reference),
    # the helper should be a no-op within float precision.
    rng = np.random.default_rng(0)
    vel = rng.normal(0.0, 0.05, size=12).astype(np.float32)
    pos = np.concatenate([[2.5], 2.5 + np.cumsum(vel[:-1])]).astype(np.float32)
    motion = _make_root_y_motion(pos, vel)
    original_pos = pos.copy()
    _reanchor_inpaint_root_y_via_velocity(motion, spans=[(4, 8)])
    np.testing.assert_allclose(motion[:, 0, 1], original_pos, atol=1e-5)


def test_reanchor_root_y_noop_when_span_touches_boundary():
    pos = np.array([1.0, 0.2, 0.3, 0.4, 1.0, 1.0], dtype=np.float32)
    vel = np.array([0.0, 0.01, 0.02, 0.03, 0.0, 0.0], dtype=np.float32)
    # Span starts at frame 0 — no left anchor available.
    motion = _make_root_y_motion(pos, vel)
    _reanchor_inpaint_root_y_via_velocity(motion, spans=[(0, 3)])
    np.testing.assert_array_equal(motion[:, 0, 1], pos)
    # Span ends at last frame — no right anchor available.
    motion2 = _make_root_y_motion(pos, vel)
    _reanchor_inpaint_root_y_via_velocity(motion2, spans=[(2, 5)])
    np.testing.assert_array_equal(motion2[:, 0, 1], pos)


def test_reanchor_root_y_corrects_all_joints():
    # Multi-joint motion: every joint's pos_y is independently biased
    # inside the inpaint span. The helper must correct each joint
    # against its own boundary anchors.
    F = 6
    n_joints = 3
    motion = np.zeros((F, n_joints, 13), dtype=np.float32)
    # Joint 0 (translation_root style): outside ~1.0, inside biased to 0.4
    motion[:, 0, 1] = [1.0, 1.0, 0.4, 0.42, 1.0, 1.0]
    # Joint 1: outside ~2.5, inside biased to 1.0
    motion[:, 1, 1] = [2.5, 2.5, 1.0, 1.05, 2.5, 2.5]
    # Joint 2: outside ~0.3, inside biased to -0.5
    motion[:, 2, 1] = [0.3, 0.3, -0.5, -0.45, 0.3, 0.3]
    # vel_y all zero (model's locomotion prior approximates this)
    _reanchor_inpaint_root_y_via_velocity(motion, spans=[(2, 3)])
    # Every joint's Y should now bridge its two boundary anchors at 1.0,
    # 2.5, and 0.3 respectively (with zero vel, both span frames land on
    # the constant anchor value).
    np.testing.assert_allclose(motion[2:4, 0, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(motion[2:4, 1, 1], 2.5, atol=1e-6)
    np.testing.assert_allclose(motion[2:4, 2, 1], 0.3, atol=1e-6)
    # Outside frames unchanged (float32 exact compare via array equality).
    for j, expected in enumerate([1.0, 2.5, 0.3]):
        np.testing.assert_array_equal(
            motion[[0, 1, 4, 5], j, 1],
            np.full(4, expected, dtype=np.float32),
        )


def test_reanchor_root_y_multiple_spans_independent():
    # Two disjoint inpaint spans; each must be corrected against its own
    # anchors and not perturb the gap between them.
    pos_y_ref = np.array(
        [1.0, 1.0, 0.4, 0.42, 1.0, 1.0, 1.0, 0.5, 0.48, 1.0, 1.0],
        dtype=np.float32,
    )
    vel_y_model = np.zeros_like(pos_y_ref)
    motion = _make_root_y_motion(pos_y_ref, vel_y_model)
    _reanchor_inpaint_root_y_via_velocity(motion, spans=[(2, 3), (7, 8)])
    fixed = motion[:, 0, 1]
    np.testing.assert_array_equal(fixed[:2], pos_y_ref[:2])
    np.testing.assert_array_equal(fixed[4:7], pos_y_ref[4:7])
    np.testing.assert_array_equal(fixed[9:], pos_y_ref[9:])
    # With all-zero vel_y, both spans ramp linearly from 1.0 → 1.0.
    np.testing.assert_allclose(fixed[2:4], 1.0, atol=1e-6)
    np.testing.assert_allclose(fixed[7:9], 1.0, atol=1e-6)


def test_close_loop_root_xz_distributes_velocity_residual():
    motion = np.zeros((5, 2, 13), dtype=np.float32)
    motion[:, 1, 0] = np.linspace(-0.1, 0.1, num=5, dtype=np.float32)
    motion[:, 1, 2] = np.linspace(0.2, -0.2, num=5, dtype=np.float32)
    motion[:-1, 1, 9] = np.array([1.0, 2.0, -1.0, 0.0], dtype=np.float32)
    motion[:-1, 1, 11] = np.array([0.5, -0.25, 0.25, 1.5], dtype=np.float32)
    motion[-1, 1, [9, 11]] = 100.0
    original_nonroot = motion[:, 0].copy()

    _close_loop_root_xz_via_velocity(motion, translation_root_index=1)

    np.testing.assert_allclose(motion[:-1, 1, [9, 11]].sum(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(motion[:, 1, [0, 2]], 0.0, atol=1e-6)
    np.testing.assert_allclose(motion[-1, 1, [9, 11]], 0.0, atol=1e-6)
    np.testing.assert_array_equal(motion[:, 0], original_nonroot)


def test_close_loop_root_xz_noop_for_invalid_root():
    motion = np.zeros((4, 1, 13), dtype=np.float32)
    motion[:-1, 0, 9] = 1.0
    original = motion.copy()

    _close_loop_root_xz_via_velocity(motion, translation_root_index=5)

    np.testing.assert_array_equal(motion, original)


def _make_pos_y_motion(pos_y_by_joint, n_feat=13):
    """Build a (F, J, C) motion tensor with only the pos_y channel (index 1)
    populated from a dict {joint_index: [per-frame Y]}.
    """
    joints = sorted(pos_y_by_joint)
    F = len(next(iter(pos_y_by_joint.values())))
    J = max(joints) + 1
    motion = np.zeros((F, J, n_feat), dtype=np.float32)
    for j, ys in pos_y_by_joint.items():
        motion[:, j, 1] = np.asarray(ys, dtype=np.float32)
    return motion


def test_reground_inpaint_joint_y_reseats_floating_subtree():
    # Chain Root(-1) -> A(0) -> B(1) -> C(2); inpaint B with subtree => free {2, 3}.
    parents = np.array([-1, 0, 1, 2], dtype=np.int64)
    ref = _make_pos_y_motion({
        0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0],
        2: [1.0, 1.1, 0.9], 3: [0.5, 0.6, 0.4],
    })
    gen = _make_pos_y_motion({
        0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0],
        # Grounded articulation lifted by a constant +5 float offset.
        2: [6.0, 6.2, 5.8], 3: [5.5, 5.9, 5.1],
    })
    clamped_before = gen[:, [0, 1], 1].copy()
    articulation_before = (gen[:, 3, 1] - gen[:, 2, 1]).copy()

    delta = _reground_inpaint_joint_y(gen, ref, {2, 3}, parents)

    # Boundary joint is joint 2 (parent 1 is clamped); delta grounds its DC.
    assert delta == pytest.approx(-5.0, abs=1e-5)
    np.testing.assert_allclose(gen[:, 2, 1], [1.0, 1.2, 0.8], atol=1e-5)
    np.testing.assert_allclose(gen[:, 3, 1], [0.5, 0.9, 0.1], atol=1e-5)
    # Generated internal articulation is preserved (only a constant removed).
    np.testing.assert_allclose(gen[:, 3, 1] - gen[:, 2, 1], articulation_before, atol=1e-5)
    # Clamped joints untouched.
    np.testing.assert_array_equal(gen[:, [0, 1], 1], clamped_before)


def test_reground_inpaint_joint_y_anchors_only_on_boundary():
    # free = {1, 2, 3}; boundary is joint 1 (parent 0 clamped). Interior joints
    # 2/3 carry a huge offset that must NOT influence the estimated delta.
    parents = np.array([-1, 0, 1, 2], dtype=np.int64)
    ref = _make_pos_y_motion({
        0: [0.0, 0.0], 1: [2.0, 2.0], 2: [1.0, 1.0], 3: [0.0, 0.0],
    })
    gen = _make_pos_y_motion({
        0: [0.0, 0.0], 1: [4.0, 4.0], 2: [101.0, 101.0], 3: [100.0, 100.0],
    })

    delta = _reground_inpaint_joint_y(gen, ref, {1, 2, 3}, parents)

    # Delta comes from boundary joint 1 only: mean(2.0 - 4.0) = -2.0.
    assert delta == pytest.approx(-2.0, abs=1e-5)
    np.testing.assert_allclose(gen[:, 1, 1], [2.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(gen[:, 2, 1], [99.0, 99.0], atol=1e-5)
    np.testing.assert_allclose(gen[:, 3, 1], [98.0, 98.0], atol=1e-5)


def test_reground_inpaint_joint_y_noops_without_reference_or_free_set():
    parents = np.array([-1, 0], dtype=np.int64)
    gen = _make_pos_y_motion({0: [0.0], 1: [7.0]})
    original = gen.copy()

    assert _reground_inpaint_joint_y(gen, None, {1}, parents) == 0.0
    np.testing.assert_array_equal(gen, original)
    assert _reground_inpaint_joint_y(gen, gen.copy(), set(), parents) == 0.0
    np.testing.assert_array_equal(gen, original)
