from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from sample.generate import (  # noqa: E402
    _prepare_reference_for_mode,
    _prepare_reference_prior_bundle,
    _sample_batch,
    _should_retarget_reference,
    _validate_reference_sampling_request,
    resolve_global_energy_condition,
    resolve_reference_scale,
    validate_reference_mode_configuration,
)
from model.anytop import AnyTop, ReferencePriorEncoder  # noqa: E402
from model.motion_transformer import GraphMotionDecoderLayer, ReferenceCrossAttnBlock  # noqa: E402
from utils.model_util import (  # noqa: E402
    ClassifierFreeReferenceModel,
    model_supports_global_energy_conditioning,
    model_supports_reference_conditioning,
)


class _CaptureDiffusion:
    def __init__(self) -> None:
        self.calls = []
        self.last_call = None
        self.last_kwargs = None

    def _record_call(self, name: str, kwargs: dict):
        self.calls.append((name, kwargs))
        self.last_call = name
        self.last_kwargs = kwargs
        return name

    def p_sample_loop(self, **kwargs):
        return self._record_call("ddpm", kwargs)

    def ddim_sample_loop(self, **kwargs):
        return self._record_call("ddim", kwargs)


class _ReferenceAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reference_cond = True
        self.reference_encoder = object()
        self.num_layers = 2
        self.forward_ys = []

    def forward(self, x, timesteps, y=None, **unused_kwargs):
        self.forward_ys.append(None if y is None else dict(y))
        bias = 5.0 if y is not None and y.get('reference_motion') is not None else 1.0
        output = torch.full_like(x, bias)
        return output


class _AxisAwareReferenceModel(nn.Module):
    def __init__(self, root_joint: int = 1) -> None:
        super().__init__()
        self.reference_cond = True
        self.reference_encoder = object()
        self.num_layers = 1
        self.root_joint = root_joint

    def forward(self, x, timesteps, y=None, **unused_kwargs):
        if y is not None and y.get('reference_motion') is not None:
            output = torch.full_like(x, 5.0)
            output[:, self.root_joint, [2, 11], :] = 7.0
        else:
            output = torch.full_like(x, 1.0)
        return output


class _DummyModel(nn.Module):
    def __init__(self, *, reference_cond: bool = False, global_energy_cond: bool = False) -> None:
        super().__init__()
        self.num_layers = 1
        self.reference_cond = reference_cond
        self.reference_encoder = object() if reference_cond else None
        self.global_energy_cond = global_energy_cond
        self.global_energy_projection = object() if global_energy_cond else None
        self.global_energy_running_mean = torch.tensor([0.25, 0.05], dtype=torch.float32)
        self.global_energy_running_var = torch.ones(2, dtype=torch.float32)

    def forward(self, x, timesteps, y=None, **unused_kwargs):
        return x


class _CaptureDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["tgt"]


class _CaptureReferenceEncoder(nn.Module):
    def __init__(self, latent_dim: int, num_tokens: int = 8) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.last_kwargs = None

    def forward(
        self,
        reference_motion,
        n_joints,
        translation_root_index,
        joints_embedded_names,
    ):
        self.last_kwargs = {
            "reference_motion": reference_motion,
            "n_joints": n_joints,
            "translation_root_index": translation_root_index,
            "joints_embedded_names": joints_embedded_names,
        }
        batch_size = reference_motion.shape[0]
        return torch.zeros(
            self.num_tokens,
            batch_size,
            self.latent_dim,
            device=reference_motion.device,
            dtype=reference_motion.dtype,
        )


def test_cfg_reference_wrapper_matches_cond_and_uncond_extremes() -> None:
    base_model = _ReferenceAwareModel()
    wrapped_model = ClassifierFreeReferenceModel(base_model)
    x = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    t = torch.tensor([1], dtype=torch.int64)
    reference_motion = torch.ones_like(x)
    y = {"reference_motion": reference_motion, "reference_scale": 1.0, "existing": torch.tensor([1.0])}

    cond_output = wrapped_model(x, t, y=y)
    assert torch.equal(cond_output, torch.full_like(x, 5.0))
    assert "reference_motion" in y

    y["reference_scale"] = 0.0
    uncond_output = wrapped_model(x, t, y=y)
    assert torch.equal(uncond_output, torch.full_like(x, 1.0))

    y["reference_scale"] = 2.0
    guided_output = wrapped_model(x, t, y=y)
    assert torch.equal(guided_output, torch.full_like(x, 9.0))
    assert base_model.forward_ys[-1].get("reference_motion") is None


def test_cfg_reference_wrapper_always_falls_back_to_uncond_on_root_xz_axes() -> None:
    base_model = _AxisAwareReferenceModel(root_joint=1)
    wrapped_model = ClassifierFreeReferenceModel(base_model)
    x = torch.zeros((1, 3, 13, 4), dtype=torch.float32)
    t = torch.tensor([1], dtype=torch.int64)
    reference_motion = torch.zeros_like(x)
    reference_motion[:, 1, [2, 11], :] = 0.5
    y = {
        "reference_motion": reference_motion,
        "reference_scale": 2.0,
        "reference_translation_root_index": torch.tensor([1]),
    }

    guided_output = wrapped_model(x, t, y=y)

    assert torch.equal(guided_output[:, 1, 0, :], torch.ones((1, 4), dtype=torch.float32))
    assert torch.equal(guided_output[:, 1, 9, :], torch.ones((1, 4), dtype=torch.float32))
    assert torch.equal(guided_output[:, 1, 2, :], torch.ones((1, 4), dtype=torch.float32))
    assert torch.equal(guided_output[:, 1, 11, :], torch.ones((1, 4), dtype=torch.float32))
    assert torch.equal(guided_output[:, 0, 0, :], torch.full((1, 4), 9.0, dtype=torch.float32))


def test_root_axis_guidance_guard_per_sample_root_indices() -> None:
    """Each batch entry's root xz/x feature channels (0, 2, 9, 11) should be
    replaced by the uncond tensor's values at THAT sample's own root joint —
    not the first sample's joint. Locks in the vectorized rewrite against
    accidental cross-sample bleeding.
    """
    B, J, F, T = 3, 4, 13, 5
    guided = torch.full((B, J, F, T), 9.0)
    uncond = torch.full((B, J, F, T), 1.0)
    # Distinct root joint per batch entry.
    y = {"reference_translation_root_index": torch.tensor([0, 2, 1])}

    out = ClassifierFreeReferenceModel._apply_reference_root_axis_guidance_guard(
        guided, uncond, y,
    )

    # Each sample b: only joint root_idx[b], features {0,2,9,11} → 1.0; else 9.0.
    for b, root_idx in enumerate([0, 2, 1]):
        for j in range(J):
            for f in range(F):
                expected = 1.0 if (j == root_idx and f in {0, 2, 9, 11}) else 9.0
                assert torch.all(out[b, j, f] == expected), (
                    f"sample={b} joint={j} feature={f}: expected {expected}"
                )


def test_root_axis_guidance_guard_skips_invalid_root_index() -> None:
    """Negative or out-of-range root_idx entries leave the corresponding
    batch sample untouched (matches old loop's `0 <= joint_idx < J` skip).
    """
    B, J, F, T = 3, 4, 13, 2
    guided = torch.full((B, J, F, T), 9.0)
    uncond = torch.full((B, J, F, T), 1.0)
    y = {"reference_translation_root_index": torch.tensor([-1, 99, 2])}

    out = ClassifierFreeReferenceModel._apply_reference_root_axis_guidance_guard(
        guided, uncond, y,
    )

    # Samples 0 (-1) and 1 (99) untouched.
    assert torch.all(out[0] == 9.0)
    assert torch.all(out[1] == 9.0)
    # Sample 2: joint 2, features {0, 2, 9, 11} → 1.0; others unchanged.
    for f in range(F):
        expected = 1.0 if f in {0, 2, 9, 11} else 9.0
        assert torch.all(out[2, 2, f] == expected)
    # Non-root joints of sample 2 untouched.
    for j in (0, 1, 3):
        assert torch.all(out[2, j] == 9.0)


def test_root_axis_guidance_guard_handles_root_index_length_mismatch() -> None:
    B, J, F, T = 3, 4, 13, 1
    guided = torch.full((B, J, F, T), 9.0)
    uncond = torch.full((B, J, F, T), 1.0)
    # Length 2 (< B): the third sample should be skipped, not crash.
    y_short = {"reference_translation_root_index": torch.tensor([1, 2])}
    out_short = ClassifierFreeReferenceModel._apply_reference_root_axis_guidance_guard(
        guided, uncond, y_short,
    )
    assert torch.all(out_short[2] == 9.0)
    assert out_short[0, 1, 0, 0] == 1.0
    assert out_short[1, 2, 0, 0] == 1.0

    # Length 5 (> B): only first B entries are used.
    y_long = {"reference_translation_root_index": torch.tensor([0, 1, 2, 3, 0])}
    out_long = ClassifierFreeReferenceModel._apply_reference_root_axis_guidance_guard(
        guided, uncond, y_long,
    )
    assert out_long[0, 0, 0, 0] == 1.0
    assert out_long[1, 1, 0, 0] == 1.0
    assert out_long[2, 2, 0, 0] == 1.0


def test_cfg_reference_wrapper_strips_reference_prior_metadata_from_uncond_path() -> None:
    base_model = _ReferenceAwareModel()
    wrapped_model = ClassifierFreeReferenceModel(base_model)
    x = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    t = torch.tensor([1], dtype=torch.int64)
    y = {
        "reference_motion": torch.ones_like(x),
        "reference_scale": 2.0,
        "reference_translation_root_index": torch.tensor([2]),
        "reference_n_joints": torch.tensor([3]),
    }

    wrapped_model(x, t, y=y)

    assert "reference_translation_root_index" not in base_model.forward_ys[-1]
    assert "reference_n_joints" not in base_model.forward_ys[-1]


def test_model_supports_reference_conditioning_detects_capability() -> None:
    assert model_supports_reference_conditioning(_DummyModel(reference_cond=True))
    assert not model_supports_reference_conditioning(_DummyModel(reference_cond=False))


def test_model_supports_global_energy_conditioning_detects_capability() -> None:
    assert model_supports_global_energy_conditioning(_DummyModel(global_energy_cond=True))
    assert not model_supports_global_energy_conditioning(_DummyModel(global_energy_cond=False))


def test_build_reference_conditioning_reuses_detached_x_start_without_clone() -> None:
    diffusion = GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _DummyModel(reference_cond=True)
    model.reference_cond_prob = 1.0
    x_start = torch.randn((2, 3, 13, 4), dtype=torch.float32, requires_grad=True)
    model_kwargs = {"y": {}}

    diffusion._build_reference_conditioning(model, x_start, model_kwargs)

    reference_motion = model_kwargs["y"]["reference_motion"]
    assert reference_motion is not None
    assert reference_motion.requires_grad is False
    assert reference_motion.data_ptr() == x_start.data_ptr()
    assert torch.equal(model_kwargs["y"]["reference_cond_mask"], torch.ones(2, dtype=torch.bool))


def test_build_global_energy_conditioning_sets_clip_condition() -> None:
    diffusion = GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _DummyModel(global_energy_cond=True)
    x_start = torch.zeros((2, 3, 13, 4), dtype=torch.float32)
    x_start[0, :, 9, :] = 0.3
    x_start[0, :, 10, :] = 0.4
    x_start[1, :, 3, 1:] = 0.5
    model_kwargs = {
        "y": {
            "lengths": torch.tensor([4, 4], dtype=torch.int64),
            "n_joints": torch.tensor([3, 3], dtype=torch.int64),
        }
    }

    diffusion._build_global_energy_conditioning(model, x_start, model_kwargs)

    expected = ReferencePriorEncoder.compute_global_energy_condition(
        x_start,
        n_joints=model_kwargs["y"]["n_joints"],
    )
    assert torch.allclose(model_kwargs["y"]["global_energy_cond"], expected)


def test_build_global_energy_conditioning_respects_existing_precomputed_value() -> None:
    diffusion = GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _DummyModel(global_energy_cond=True)
    x_start = torch.zeros((1, 3, 13, 7), dtype=torch.float32)
    x_start[:, :, 3, :] = torch.linspace(0.0, 3.0, steps=7, dtype=torch.float32).view(1, 1, 7)
    precomputed = torch.tensor([[0.75, 0.01]], dtype=torch.float32)
    model_kwargs = {
        "y": {
            "lengths": torch.tensor([7], dtype=torch.int64),
            "n_joints": torch.tensor([3], dtype=torch.int64),
            "playspeed_cond": torch.tensor([4.0 / 7.0], dtype=torch.float32),
            "global_energy_cond": precomputed.clone(),
        }
    }

    diffusion._build_global_energy_conditioning(model, x_start, model_kwargs)

    assert torch.equal(model_kwargs["y"]["global_energy_cond"], precomputed)


def test_build_global_energy_conditioning_uses_playspeed_for_rotation_energy() -> None:
    diffusion = GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _DummyModel(global_energy_cond=True)

    physical_motion = torch.zeros((1, 3, 13, 4), dtype=torch.float32)
    physical_motion[:, :, 3, :] = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    stretched_motion = torch.zeros((1, 3, 13, 7), dtype=torch.float32)
    stretched_motion[:, :, 3, :] = torch.linspace(0.0, 3.0, steps=7, dtype=torch.float32).view(1, 1, 7)
    model_kwargs = {
        "y": {
            "lengths": torch.tensor([7], dtype=torch.int64),
            "n_joints": torch.tensor([3], dtype=torch.int64),
            "playspeed_cond": torch.tensor([4.0 / 7.0], dtype=torch.float32),
        }
    }

    diffusion._build_global_energy_conditioning(model, stretched_motion, model_kwargs)

    expected = ReferencePriorEncoder.compute_global_energy_condition(
        physical_motion,
        n_joints=model_kwargs["y"]["n_joints"],
    )
    assert torch.allclose(model_kwargs["y"]["global_energy_cond"], expected, atol=1e-5)


def test_anytop_forward_accepts_reference_motion_with_independent_frame_count() -> None:
    model = AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        t5_out_dim=8,
        cross_limb=False,
        reference_cond=True,
    )
    capture_decoder = _CaptureDecoder()
    capture_reference_encoder = _CaptureReferenceEncoder(model.latent_dim)
    model.seqTransDecoder = capture_decoder
    model.reference_encoder = capture_reference_encoder
    model.eval()

    x = torch.randn(1, 4, 13, 7, dtype=torch.float32)
    reference_motion = torch.randn(1, 4, 13, 3, dtype=torch.float32)
    y = {
        "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
        "mask": torch.ones(1, 1, 1, 8, 8, dtype=torch.float32),
        "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
        "n_joints": torch.tensor([4], dtype=torch.int64),
        "lengths": torch.tensor([7], dtype=torch.int64),
        "translation_root_index": torch.tensor([0], dtype=torch.int64),
        "joints_names_embs": torch.zeros(1, 4, 8, dtype=torch.float32),
        "parents": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64),
        "reference_motion": reference_motion,
        "reference_n_joints": torch.tensor([4], dtype=torch.int64),
        "reference_translation_root_index": torch.tensor([0], dtype=torch.int64),
        "reference_parents": [np.asarray([-1, 0, 1, 2], dtype=np.int64)],
        "reference_joints_names_embs": torch.zeros(1, 4, 8, dtype=torch.float32),
    }

    output = model(x, torch.tensor([1], dtype=torch.int64), y=y)

    assert output.shape == (1, 4, 13, 7)
    assert capture_reference_encoder.last_kwargs is not None
    assert capture_reference_encoder.last_kwargs["reference_motion"].shape[-1] == 3
    assert capture_decoder.last_kwargs is not None
    assert capture_decoder.last_kwargs["reference_memory"].shape == (8, 1, 8)


def test_anytop_forward_uses_cached_reference_memory_and_skips_encoder() -> None:
    model = AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        t5_out_dim=8,
        cross_limb=False,
        reference_cond=True,
    )
    capture_decoder = _CaptureDecoder()
    capture_reference_encoder = _CaptureReferenceEncoder(model.latent_dim)
    model.seqTransDecoder = capture_decoder
    model.reference_encoder = capture_reference_encoder
    model.eval()

    x = torch.randn(1, 4, 13, 7, dtype=torch.float32)
    # Cached memory has the same shape ReferencePriorEncoder would produce:
    # (num_tokens, batch, latent_dim).
    cached_memory = torch.randn(
        capture_reference_encoder.num_tokens, 1, model.latent_dim, dtype=torch.float32,
    )
    y = {
        "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
        "mask": torch.ones(1, 1, 1, 8, 8, dtype=torch.float32),
        "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
        "n_joints": torch.tensor([4], dtype=torch.int64),
        "lengths": torch.tensor([7], dtype=torch.int64),
        "translation_root_index": torch.tensor([0], dtype=torch.int64),
        "joints_names_embs": torch.zeros(1, 4, 8, dtype=torch.float32),
        "parents": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64),
        # reference_motion is intentionally absent: cache should be enough.
        "reference_memory": cached_memory,
    }

    output = model(x, torch.tensor([1], dtype=torch.int64), y=y)

    assert output.shape == (1, 4, 13, 7)
    assert capture_reference_encoder.last_kwargs is None, (
        "reference_encoder must not be called when y['reference_memory'] is supplied"
    )
    assert capture_decoder.last_kwargs is not None
    assert torch.equal(capture_decoder.last_kwargs["reference_memory"], cached_memory)


def test_anytop_forward_accepts_global_energy_condition_without_reference_motion() -> None:
    model = AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        t5_out_dim=8,
        cross_limb=False,
        reference_cond=False,
        global_energy_cond=True,
    )
    capture_decoder = _CaptureDecoder()
    model.seqTransDecoder = capture_decoder
    model.eval()
    with torch.no_grad():
        model.global_energy_running_mean.copy_(torch.tensor([0.25, 0.05], dtype=torch.float32))
        model.global_energy_running_var.copy_(torch.tensor([0.04, 0.01], dtype=torch.float32))

    x = torch.randn(1, 4, 13, 7, dtype=torch.float32)
    y = {
        "joints_padding_mask": torch.ones(1, 1, 1, 5, 5, dtype=torch.float32),
        "mask": torch.ones(1, 1, 1, 8, 8, dtype=torch.float32),
        "tpos_first_frame": torch.randn(1, 4, 13, dtype=torch.float32),
        "n_joints": torch.tensor([4], dtype=torch.int64),
        "lengths": torch.tensor([7], dtype=torch.int64),
        "translation_root_index": torch.tensor([0], dtype=torch.int64),
        "joints_names_embs": torch.zeros(1, 4, 8, dtype=torch.float32),
        "parents": torch.tensor([[-1, 0, 1, 2]], dtype=torch.int64),
        "global_energy_cond": torch.tensor([[0.45, 0.15]], dtype=torch.float32),
    }

    output = model(x, torch.tensor([1], dtype=torch.int64), y=y)

    assert output.shape == (1, 4, 13, 7)
    assert capture_decoder.last_kwargs is not None
    assert capture_decoder.last_kwargs["reference_memory"] is None
    assert capture_decoder.last_kwargs["global_energy_condition"].shape == (1, model.latent_dim)


def test_decoder_layer_keeps_global_energy_condition_when_reference_gate_is_zero() -> None:
    layer = GraphMotionDecoderLayer(
        d_model=4,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        reference_residual_gate=0.0,
    )
    layer.embed_timesteps = nn.Identity()
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.norm3 = nn.Identity()
    layer.norm_ref = nn.Identity()

    def _zero_block(self, x, *args, **kwargs):
        return torch.zeros_like(x)

    def _global_energy_cond(self, x, global_energy_condition):
        return x + global_energy_condition[:, :1].to(device=x.device, dtype=x.dtype).view(1, x.shape[1], 1, 1)

    layer._spatial_mha_block = types.MethodType(_zero_block, layer)
    layer._temporal_mha_block_sin_joint = types.MethodType(_zero_block, layer)
    layer._ff_block = types.MethodType(_zero_block, layer)
    layer._apply_global_energy_cond = types.MethodType(_global_energy_cond, layer)

    tgt = torch.zeros((2, 1, 3, 4), dtype=torch.float32)
    timesteps_emb = torch.zeros((1, 4), dtype=torch.float32)
    reference_memory = torch.full((1, 1, 4), 7.0, dtype=torch.float32)
    global_energy_condition = torch.full((1, 4), 3.0, dtype=torch.float32)

    output = layer(
        tgt=tgt,
        timesteps_emb=timesteps_emb,
        topology_rel=None,
        edge_rel=None,
        edge_key_emb=None,
        edge_query_emb=None,
        edge_value_emb=None,
        topo_key_emb=None,
        topo_query_emb=None,
        topo_value_emb=None,
        reference_memory=reference_memory,
        global_energy_condition=global_energy_condition,
        reference_batch_mask=torch.tensor([True], dtype=torch.bool),
    )

    assert torch.allclose(output, torch.full_like(tgt, 3.0))


def test_reference_prior_encoder_rejects_feature_schemas_shorter_than_13_dims() -> None:
    with pytest.raises(ValueError, match="13-dim"):
        ReferencePriorEncoder(
            max_joints=4,
            input_feats=12,
            latent_dim=32,
            ff_size=64,
            num_heads=4,
            dropout=0.0,
            t5_out_dim=8,
num_layers=1,
        )


def test_prepare_reference_prior_bundle_preserves_reference_length_and_zero_feature_padding(tmp_path: Path) -> None:
    reference_motion_path = tmp_path / "reference_prior.npy"
    ref_raw = np.ones((5, 2, 11), dtype=np.float32)
    np.save(reference_motion_path, ref_raw)

    source_cond = {
        "parents": np.asarray([-1, 0], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "mean": np.asarray(
            [
                [0.0] * 11 + [3.0, 4.0],
                [0.0] * 11 + [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        "std": np.ones((2, 13), dtype=np.float32),
        "joints_names_embs": np.zeros((2, 8), dtype=np.float32),
    }

    ref_tensor, reference_kwargs, reference_frame_count, reference_source_frame_count, output_frame_count = _prepare_reference_prior_bundle(
        str(reference_motion_path),
        "TestObject",
        source_cond,
        max_joints=2,
        target_feature_len=13,
        batch_size=1,
    )

    assert reference_frame_count == 5
    assert reference_source_frame_count == 5
    assert output_frame_count == 5
    assert ref_tensor.shape == (1, 2, 13, 5)
    assert torch.equal(reference_kwargs["reference_n_joints"], torch.tensor([2], dtype=torch.long))
    assert torch.allclose(ref_tensor[0, :, 11:13, :], torch.zeros_like(ref_tensor[0, :, 11:13, :]))


def test_prepare_reference_prior_bundle_rejects_incompatible_stat_schema(tmp_path: Path) -> None:
    reference_motion_path = tmp_path / "reference_prior_bad_stats.npy"
    ref_raw = np.ones((5, 2, 11), dtype=np.float32)
    np.save(reference_motion_path, ref_raw)

    source_cond = {
        "parents": np.asarray([-1, 0], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "mean": np.zeros((2, 10), dtype=np.float32),
        "std": np.ones((2, 10), dtype=np.float32),
        "joints_names_embs": np.zeros((2, 8), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="need at least 11 feature channels"):
        _prepare_reference_prior_bundle(
            str(reference_motion_path),
            "TestObject",
            source_cond,
            max_joints=2,
            target_feature_len=13,
            batch_size=1,
        )


def test_prepare_reference_for_mode_controlnet_matches_img2img_when_reference_is_shorter(tmp_path: Path) -> None:
    reference_motion_path = tmp_path / "reference_prior_controlnet.npy"
    ref_raw = np.ones((5, 2, 11), dtype=np.float32)
    np.save(reference_motion_path, ref_raw)

    source_cond = {
        "parents": np.asarray([-1, 0], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "mean": np.zeros((2, 13), dtype=np.float32),
        "std": np.ones((2, 13), dtype=np.float32),
        "joints_names_embs": np.zeros((2, 8), dtype=np.float32),
    }

    bundle = _prepare_reference_for_mode(
        str(reference_motion_path),
        reference_mode="controlnet",
        source_type="TestObject",
        source_cond=source_cond,
        target_type="OtherObject",
        target_cond=source_cond,
        max_joints=4,
        target_feature_len=13,
        batch_size=1,
        requested_output_frame_count=7,
    )

    assert bundle["loaded_reference_frame_count"] == 5
    assert bundle["reference_source_frame_count"] == 5
    assert bundle["loaded_reference_joint_count"] == 2
    assert bundle["output_frame_count"] == 5
    assert bundle["reference_motion"].shape == (1, 4, 13, 5)


def test_prepare_reference_for_mode_controlnet_truncates_longer_reference_to_requested_length(tmp_path: Path) -> None:
    reference_motion_path = tmp_path / "reference_prior_controlnet_long.npy"
    ref_raw = np.ones((6, 2, 11), dtype=np.float32)
    np.save(reference_motion_path, ref_raw)

    cond = {
        "parents": np.asarray([-1, 0], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "mean": np.zeros((2, 13), dtype=np.float32),
        "std": np.ones((2, 13), dtype=np.float32),
        "joints_names_embs": np.zeros((2, 8), dtype=np.float32),
    }

    bundle = _prepare_reference_for_mode(
        str(reference_motion_path),
        reference_mode="controlnet",
        source_type="TestObject",
        source_cond=cond,
        target_type="OtherObject",
        target_cond=cond,
        max_joints=4,
        target_feature_len=13,
        batch_size=1,
        requested_output_frame_count=4,
    )

    assert bundle["loaded_reference_frame_count"] == 6
    assert bundle["reference_source_frame_count"] == 6
    assert bundle["output_frame_count"] == 4
    assert bundle["reference_motion"].shape == (1, 4, 13, 4)


def test_prepare_reference_for_mode_controlnet_uses_target_metadata(tmp_path: Path) -> None:
    reference_motion_path = tmp_path / "reference_prior_controlnet_target.npy"
    ref_raw = np.ones((5, 3, 11), dtype=np.float32)
    np.save(reference_motion_path, ref_raw)

    source_cond = {
        "parents": np.asarray([-1, 0], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "mean": np.zeros((2, 13), dtype=np.float32),
        "std": np.ones((2, 13), dtype=np.float32),
        "joints_names_embs": np.zeros((2, 8), dtype=np.float32),
    }
    target_cond = {
        "parents": np.asarray([-1, 0, 1], dtype=np.int64),
        "offsets": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        "mean": np.zeros((3, 13), dtype=np.float32),
        "std": np.ones((3, 13), dtype=np.float32),
        "joints_names_embs": np.zeros((3, 8), dtype=np.float32),
    }

    bundle = _prepare_reference_for_mode(
        str(reference_motion_path),
        reference_mode="controlnet",
        source_type="Horse",
        source_cond=source_cond,
        target_type="Buffalo",
        target_cond=target_cond,
        max_joints=4,
        target_feature_len=13,
        batch_size=1,
        requested_output_frame_count=7,
    )

    assert bundle["loaded_reference_joint_count"] == 3
    assert bundle["reference_source_frame_count"] == 5
    assert bundle["output_frame_count"] == 5
    assert torch.equal(bundle["reference_conditioning_kwargs"]["reference_n_joints"], torch.tensor([3]))
    assert bundle["reference_motion"].shape == (1, 4, 13, 5)


def test_should_retarget_reference_forces_cross_skeleton_controlnet_retarget() -> None:
    assert _should_retarget_reference("Horse", "Buffalo", "img2img")
    assert _should_retarget_reference("Horse", "Buffalo", "controlnet")
    assert not _should_retarget_reference("Horse", "Horse", "img2img")


def test_reference_prior_encoder_uses_effective_translation_root() -> None:
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=1,
    )
    encoder.eval()

    motion = torch.zeros((1, 4, 13, 5), dtype=torch.float32)
    motion[0, 0, 9, 1:] = 0.25
    motion[0, 2, 9, 1:] = 2.0
    motion[0, 2, 10, 1:] = 1.0
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)

    tokens_root0 = encoder(
        motion,
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([0]),
        joints_embedded_names=joints_names_embs,
    )
    tokens_root2 = encoder(
        motion,
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([2]),
        joints_embedded_names=joints_names_embs,
    )

    assert tokens_root0.shape == (8, 1, 32)
    assert not torch.allclose(tokens_root0, tokens_root2)


def test_reference_prior_encoder_accepts_padded_motion_with_unpadded_parents() -> None:
    encoder = ReferencePriorEncoder(
        max_joints=8,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=1,
    )
    encoder.eval()

    motion = torch.zeros((1, 8, 13, 5), dtype=torch.float32)
    joints_names_embs = torch.zeros((1, 8, 8), dtype=torch.float32)

    tokens = encoder(
        motion,
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([2]),
        joints_embedded_names=joints_names_embs,
    )

    assert tokens.shape == (8, 1, 32)


def test_reference_prior_encoder_respects_zero_temporal_layers() -> None:
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )

    assert len(encoder.conv_blocks) == 0
    assert len(encoder.temporal_layers) == 0


def test_reference_prior_encoder_honours_per_sample_metadata_in_batch() -> None:
    encoder = ReferencePriorEncoder(
        max_joints=6,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=1,
    )
    encoder.eval()

    torch.manual_seed(0)
    # Both batch entries share identical motion, so any token difference can
    # only come from the per-sample skeleton metadata (never silently broadcast).
    shared_motion = torch.randn((1, 6, 13, 7), dtype=torch.float32)
    motion = shared_motion.expand(2, -1, -1, -1).contiguous()
    joints_names_embs = torch.zeros((2, 6, 8), dtype=torch.float32)

    tokens = encoder(
        motion,
        n_joints=torch.tensor([4, 5]),
        translation_root_index=torch.tensor([0, 0]),
        joints_embedded_names=joints_names_embs,
    )

    assert tokens.shape == (8, 2, 32)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_reference_prior_encoder_preserves_velocity_direction_sign() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )
    encoder.eval()

    motion_pos = torch.zeros((1, 4, 13, 5), dtype=torch.float32)
    motion_neg = torch.zeros((1, 4, 13, 5), dtype=torch.float32)
    motion_pos[0, 1, 9:12, 1:] = torch.tensor([[1.0], [-0.5], [0.25]], dtype=torch.float32)
    motion_neg[0, 1, 9:12, 1:] = torch.tensor([[-1.0], [0.5], [-0.25]], dtype=torch.float32)
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)

    common_kwargs = dict(
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([0]),
        joints_embedded_names=joints_names_embs,
    )

    tokens_pos = encoder(motion_pos, **common_kwargs)
    tokens_neg = encoder(motion_neg, **common_kwargs)

    assert tokens_pos.shape == (8, 1, 32)
    assert not torch.allclose(tokens_pos, tokens_neg)


def test_reference_prior_encoder_joint_motion_features_ignore_velocity_sign() -> None:
    vel_pos = torch.tensor([[[[1.0, -0.5, 0.25]]]], dtype=torch.float32)
    vel_neg = -vel_pos
    rot_delta_norm = torch.tensor([[[[0.75]]]], dtype=torch.float32)
    contact = torch.tensor([[[[1.0]]]], dtype=torch.float32)

    features_pos, _, _ = ReferencePriorEncoder._build_joint_motion_frame_features(
        vel_pos,
        rot_delta_norm,
        contact,
    )
    features_neg, _, _ = ReferencePriorEncoder._build_joint_motion_frame_features(
        vel_neg,
        rot_delta_norm,
        contact,
    )

    assert torch.allclose(features_pos, features_neg)


def test_reference_prior_encoder_phase_joint_features_keep_velocity_sign() -> None:
    vel_pos = torch.tensor([[[[1.0, -0.5, 0.25]]]], dtype=torch.float32)
    vel_neg = -vel_pos
    contact = torch.tensor([[[[1.0]]]], dtype=torch.float32)

    features_pos = ReferencePriorEncoder._build_phase_joint_features(vel_pos, contact)
    features_neg = ReferencePriorEncoder._build_phase_joint_features(vel_neg, contact)

    assert not torch.allclose(features_pos, features_neg)


def test_reference_prior_encoder_keeps_joint_phase_identity_outside_group_pooling() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )
    encoder.eval()
    with torch.no_grad():
        encoder.group_queries.zero_()

    signal_a = torch.tensor([1.0, 0.0, -1.0, 0.0, 1.0], dtype=torch.float32)
    signal_b = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0], dtype=torch.float32)
    motion_a = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    motion_b = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    motion_a[0, 1, 9, 1:] = signal_a
    motion_a[0, 2, 9, 1:] = signal_b
    motion_b[0, 1, 9, 1:] = signal_b
    motion_b[0, 2, 9, 1:] = signal_a
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)

    captured = {}

    def capture_prior_input(module, inputs):
        captured['prior_sequence'] = inputs[0].detach().clone()

    hook = encoder.sequence_projection.register_forward_pre_hook(capture_prior_input)
    try:
        encoder(motion_a, n_joints=torch.tensor([4]), translation_root_index=torch.tensor([0]), joints_embedded_names=joints_names_embs)
        prior_a = captured['prior_sequence']
        captured.clear()
        encoder(motion_b, n_joints=torch.tensor([4]), translation_root_index=torch.tensor([0]), joints_embedded_names=joints_names_embs)
        prior_b = captured['prior_sequence']
    finally:
        hook.remove()

    group_end = encoder.global_motion_feature_dim + encoder.num_groups * encoder.group_feature_dim
    assert torch.allclose(prior_a[..., :group_end], prior_b[..., :group_end])
    assert not torch.allclose(prior_a[..., group_end:], prior_b[..., group_end:])


def test_reference_prior_encoder_includes_joint_semantics_in_prior_sequence() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )
    encoder.eval()
    with torch.no_grad():
        encoder.group_queries.zero_()

    motion = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    motion[0, 1, 9, 1:] = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    names_a = torch.zeros((1, 4, 8), dtype=torch.float32)
    names_b = names_a.clone()
    names_b[0, 2, 3] = 1.0

    captured = {}

    def capture_prior_input(module, inputs):
        captured['prior_sequence'] = inputs[0].detach().clone()

    hook = encoder.sequence_projection.register_forward_pre_hook(capture_prior_input)
    try:
        encoder(motion, n_joints=torch.tensor([4]), translation_root_index=torch.tensor([0]), joints_embedded_names=names_a)
        prior_a = captured['prior_sequence']
        captured.clear()
        encoder(motion, n_joints=torch.tensor([4]), translation_root_index=torch.tensor([0]), joints_embedded_names=names_b)
        prior_b = captured['prior_sequence']
    finally:
        hook.remove()

    group_end = encoder.global_motion_feature_dim + encoder.num_groups * encoder.group_feature_dim
    assert torch.allclose(prior_a[..., :group_end], prior_b[..., :group_end])
    assert not torch.allclose(prior_a[..., group_end:], prior_b[..., group_end:])


def test_reference_prior_encoder_phase_tokens_ignore_root_position() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )
    encoder.eval()

    motion_a = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    motion_b = motion_a.clone()
    motion_a[0, 0, 0, :] = torch.linspace(0.0, 1.0, 6)
    motion_b[0, 0, 0, :] = torch.linspace(3.0, -2.0, 6)
    motion_a[0, 1, 9, 1:] = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
    motion_b[0, 1, 9, 1:] = motion_a[0, 1, 9, 1:]
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)

    common_kwargs = dict(
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([0]),
        joints_embedded_names=joints_names_embs,
    )

    tokens_a = encoder(motion_a, **common_kwargs)
    tokens_b = encoder(motion_b, **common_kwargs)

    assert torch.allclose(tokens_a, tokens_b)


def test_reference_prior_encoder_tracks_limb_phase_without_root_motion() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        max_joints=4,
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
num_layers=0,
    )
    encoder.eval()

    motion_in_phase = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    motion_antiphase = torch.zeros((1, 4, 13, 6), dtype=torch.float32)
    in_phase_signal = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
    anti_phase_signal = -in_phase_signal
    motion_in_phase[0, 1, 9, 1:] = in_phase_signal
    motion_in_phase[0, 2, 9, 1:] = in_phase_signal
    motion_antiphase[0, 1, 9, 1:] = in_phase_signal
    motion_antiphase[0, 2, 9, 1:] = anti_phase_signal
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)

    common_kwargs = dict(
        n_joints=torch.tensor([4]),
        translation_root_index=torch.tensor([0]),
        joints_embedded_names=joints_names_embs,
    )

    tokens_in_phase = encoder(motion_in_phase, **common_kwargs)
    tokens_antiphase = encoder(motion_antiphase, **common_kwargs)

    assert not torch.allclose(tokens_in_phase, tokens_antiphase)


def test_decoder_reference_mask_keeps_unconditioned_samples_on_baseline_path() -> None:
    torch.manual_seed(0)
    layer = GraphMotionDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    tsteps, batch_size, njoints, d_model = 3, 2, 1, 8
    x = torch.randn(tsteps, batch_size, njoints, d_model)

    class _FakeReferenceBlock(nn.Module):
        def forward(self, x, reference_memory, key_padding_mask, reference_batch_mask):
            delta = torch.full_like(x, 0.5)
            if reference_batch_mask is not None:
                delta = delta * reference_batch_mask.to(device=x.device, dtype=x.dtype).view(1, x.shape[1], 1, 1)
            return delta

    fake_reference_block = _FakeReferenceBlock()

    common_kwargs = dict(
        tgt=x,
        timesteps_emb=torch.zeros(batch_size, d_model),
        topology_rel=torch.zeros(batch_size, layer.heads, njoints, njoints, dtype=torch.long),
        edge_rel=torch.zeros(batch_size, layer.heads, njoints, njoints, dtype=torch.long),
        edge_key_emb=nn.Embedding(6, d_model),
        edge_query_emb=nn.Embedding(6, d_model),
        edge_value_emb=None,
        topo_key_emb=nn.Embedding(6, d_model),
        topo_query_emb=nn.Embedding(6, d_model),
        topo_value_emb=None,
        spatial_mask=torch.zeros(batch_size, layer.heads, njoints, njoints),
        temporal_mask=torch.zeros(batch_size * njoints * layer.heads, tsteps, tsteps),
        tgt_key_padding_mask=torch.zeros(batch_size, njoints, dtype=torch.bool),
        y={"joints_key_padding_mask": torch.zeros(batch_size, njoints, dtype=torch.bool)},
        reference_key_padding_mask=None,
        temporal_template=torch.zeros(batch_size * layer.heads, tsteps, tsteps),
        cross_limb_block=None,
        cross_limb_unreliable_mask=None,
    )

    baseline = layer.forward(
        reference_memory=None, reference_block=None, reference_batch_mask=None, **common_kwargs,
    )
    mixed = layer.forward(
        reference_memory=torch.zeros(tsteps, batch_size, njoints, d_model),
        reference_block=fake_reference_block,
        reference_batch_mask=torch.tensor([True, False]),
        **common_kwargs,
    )

    assert torch.allclose(mixed[:, 1], baseline[:, 1])


def test_decoder_reference_block_accepts_shared_prior_tokens() -> None:
    torch.manual_seed(0)
    layer = GraphMotionDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    reference_block = ReferenceCrossAttnBlock(d_model=8, nhead=2, dropout=0.0)
    tsteps, batch_size, njoints, d_model = 3, 2, 2, 8
    x = torch.randn(tsteps, batch_size, njoints, d_model)

    output = layer.forward(
        tgt=x,
        timesteps_emb=torch.zeros(batch_size, d_model),
        topology_rel=torch.zeros(batch_size, layer.heads, njoints, njoints, dtype=torch.long),
        edge_rel=torch.zeros(batch_size, layer.heads, njoints, njoints, dtype=torch.long),
        edge_key_emb=nn.Embedding(6, d_model),
        edge_query_emb=nn.Embedding(6, d_model),
        edge_value_emb=None,
        topo_key_emb=nn.Embedding(6, d_model),
        topo_query_emb=nn.Embedding(6, d_model),
        topo_value_emb=None,
        spatial_mask=torch.zeros(batch_size, layer.heads, njoints, njoints),
        temporal_mask=torch.zeros(batch_size * njoints * layer.heads, tsteps, tsteps),
        tgt_key_padding_mask=torch.zeros(batch_size, njoints, dtype=torch.bool),
        y={"joints_key_padding_mask": torch.zeros(batch_size, njoints, dtype=torch.bool)},
        reference_memory=torch.randn(5, batch_size, d_model),
        reference_key_padding_mask=torch.zeros(batch_size, 5, dtype=torch.bool),
        temporal_template=torch.zeros(batch_size * layer.heads, tsteps, tsteps),
        cross_limb_block=None,
        reference_block=reference_block,
        cross_limb_unreliable_mask=None,
        reference_batch_mask=torch.ones(batch_size, dtype=torch.bool),
    )

    assert output.shape == x.shape


def test_validate_reference_mode_configuration_rejects_invalid_controlnet_setup() -> None:
    with pytest.raises(ValueError, match="requires --reference_motion"):
        validate_reference_mode_configuration("controlnet", reference_motion_path=None, skip_timesteps=0)
    # As of the "soft skip" change, non-zero skip_timesteps with controlnet
    # no longer raises — it prints a warning and forces 0.
    mode, skip_ts = validate_reference_mode_configuration(
        "controlnet", reference_motion_path="ref.npy", skip_timesteps=3,
    )
    assert mode == "controlnet"
    assert skip_ts == 0
    with pytest.raises(ValueError, match="does not support --reference_mode controlnet"):
        validate_reference_mode_configuration(
            "controlnet",
            reference_motion_path="ref.npy",
            skip_timesteps=0,
            model=_DummyModel(reference_cond=False),
        )


def test_resolve_reference_scale_preserves_explicit_zero() -> None:
    assert resolve_reference_scale(None) == 1.0
    assert resolve_reference_scale(0.0) == 0.0
    assert resolve_reference_scale(1.5) == 1.5


def test_resolve_global_energy_condition_uses_running_defaults_for_missing_components() -> None:
    model = _DummyModel(global_energy_cond=True)

    resolved = resolve_global_energy_condition(
        model,
        global_energy_mean=0.4,
        global_energy_std=None,
        batch_size=2,
    )

    # With running_mean=[0.25, 0.05] and running_var=ones(2) (std=1.0):
    # raw[0] = 0.4 * 1.0 + 0.25 = 0.65
    # raw[1] = running_mean[1] = 0.05 (unchanged because global_energy_std=None)
    assert resolved.shape == (2, 2)
    assert torch.allclose(resolved[:, 0], torch.full((2,), 0.65))
    assert torch.allclose(resolved[:, 1], torch.full((2,), 0.05))


def test_cfg_reference_wrapper_preserves_global_energy_condition_on_uncond_path() -> None:
    base_model = _ReferenceAwareModel()
    wrapped_model = ClassifierFreeReferenceModel(base_model)
    x = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    t = torch.tensor([1], dtype=torch.int64)
    y = {
        "reference_motion": torch.ones_like(x),
        "reference_scale": 0.0,
        "global_energy_cond": torch.tensor([[0.3, 0.1]], dtype=torch.float32),
    }

    wrapped_model(x, t, y=y)

    assert torch.equal(base_model.forward_ys[-1]["global_energy_cond"], y["global_energy_cond"])
    assert "reference_motion" not in base_model.forward_ys[-1]


def test_sample_batch_routes_controlnet_through_cfg_wrapper_without_mutating_y() -> None:
    diffusion = _CaptureDiffusion()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    model_kwargs = {"y": {"existing": torch.tensor([1.0])}}

    result = _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(reference_cond=True),
        model_kwargs=model_kwargs,
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        reference_mode="controlnet",
        reference_scale=2.5,
        skip_timesteps=0,
    )

    assert result == "ddpm"
    assert diffusion.last_kwargs["init_image"] is None
    assert diffusion.last_kwargs["skip_timesteps"] == 0
    assert isinstance(diffusion.last_kwargs["model"], ClassifierFreeReferenceModel)
    routed_y = diffusion.last_kwargs["model_kwargs"]["y"]
    assert torch.equal(routed_y["reference_motion"], reference_motion)
    assert routed_y["reference_scale"] == 2.5
    assert torch.equal(routed_y["existing"], model_kwargs["y"]["existing"])
    assert "reference_motion" not in model_kwargs["y"]


def test_sample_batch_routes_reference_prior_metadata_without_mutating_y() -> None:
    diffusion = _CaptureDiffusion()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.ones(sample_shape, dtype=torch.float32)
    reference_conditioning_kwargs = {
        "reference_translation_root_index": torch.tensor([2]),
        "reference_n_joints": torch.tensor([3]),
    }
    model_kwargs = {"y": {"existing": torch.tensor([1.0])}}

    _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(reference_cond=True),
        model_kwargs=model_kwargs,
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        reference_conditioning_kwargs=reference_conditioning_kwargs,
        reference_mode="controlnet",
        reference_scale=2.5,
        skip_timesteps=0,
    )

    routed_y = diffusion.last_kwargs["model_kwargs"]["y"]
    assert torch.equal(routed_y["reference_translation_root_index"], torch.tensor([2]))
    assert torch.equal(routed_y["reference_n_joints"], torch.tensor([3]))
    assert "reference_translation_root_index" not in model_kwargs["y"]


def test_sample_batch_supports_controlnet_plus_inpaint_in_single_pass() -> None:
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

    result = _sample_batch(
        diffusion=diffusion,
        model=_DummyModel(reference_cond=True),
        model_kwargs=model_kwargs,
        sampling_method="ddpm",
        sample_shape=sample_shape,
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=reference_motion,
        reference_mode="controlnet",
        reference_scale=1.5,
        skip_timesteps=0,
        inpaint_mask=inpaint_mask,
    )

    assert result == "ddpm"
    assert len(diffusion.calls) == 1
    routed_kwargs = diffusion.last_kwargs
    assert routed_kwargs["init_image"] is None
    assert torch.equal(routed_kwargs["inpaint_reference"], reference_motion)
    routed_y = routed_kwargs["model_kwargs"]["y"]
    assert torch.equal(routed_y["reference_motion"], reference_motion)
    assert routed_y["reference_scale"] == 1.5
    assert "cross_limb_unreliable_mask" in routed_y
    assert "reference_motion" not in model_kwargs["y"]


def test_sample_batch_rejects_controlnet_inpaint_when_reference_length_differs_from_target() -> None:
    diffusion = _CaptureDiffusion()
    sample_shape = (1, 3, 13, 4)
    reference_motion = torch.arange(1, 1 + (1 * 3 * 13 * 6), dtype=torch.float32).reshape(1, 3, 13, 6)
    inpaint_mask = torch.ones((1, 3, 1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="reference_motion frame count to match target sample length"):
        _sample_batch(
            diffusion=diffusion,
            model=_DummyModel(reference_cond=True),
            model_kwargs={"y": {"existing": torch.tensor([1.0])}},
            sampling_method="ddpm",
            sample_shape=sample_shape,
            ddim_eta=0.0,
            seed=123,
            device=torch.device("cpu"),
            reference_motion=reference_motion,
            reference_mode="controlnet",
            reference_scale=1.5,
            skip_timesteps=0,
            inpaint_mask=inpaint_mask,
        )


def test_sample_batch_tolerates_controlnet_skip_timesteps() -> None:
    # Non-zero skip_timesteps with controlnet no longer raises — it prints a
    # warning and forces 0 internally via validate_reference_mode_configuration.
    _sample_batch(
        diffusion=_CaptureDiffusion(),
        model=_DummyModel(reference_cond=True),
        model_kwargs={"y": {}},
        sampling_method="ddpm",
        sample_shape=(1, 3, 13, 4),
        ddim_eta=0.0,
        seed=123,
        device=torch.device("cpu"),
        reference_motion=torch.zeros((1, 3, 13, 4), dtype=torch.float32),
        reference_mode="controlnet",
        reference_scale=2.0,
        skip_timesteps=4,
    )


def test_validate_reference_sampling_request_allows_cross_species_controlnet_inpaint() -> None:
    _validate_reference_sampling_request(
        inpaint_enabled=True,
        reference_mode="controlnet",
        cross_species_reference=True,
    )

    _validate_reference_sampling_request(
        inpaint_enabled=True,
        reference_mode="img2img",
        cross_species_reference=True,
    )

