from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from sample.generate import _sample_batch, validate_reference_mode_configuration  # noqa: E402
from model.motion_transformer import GraphMotionDecoderLayer  # noqa: E402
from utils.model_util import ClassifierFreeReferenceModel, model_supports_reference_conditioning  # noqa: E402


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

    def plms_sample_loop(self, **kwargs):
        return self._record_call("plms", kwargs)


class _ReferenceAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reference_cond = True
        self.reference_encoder = object()
        self.num_layers = 2
        self.forward_ys = []

    def forward(self, x, timesteps, get_layer_activation=-1, y=None, **unused_kwargs):
        self.forward_ys.append(None if y is None else dict(y))
        bias = 5.0 if y is not None and y.get('reference_motion') is not None else 1.0
        output = torch.full_like(x, bias)
        if get_layer_activation > -1:
            return output, {0: output.clone()}
        return output


class _DummyModel(nn.Module):
    def __init__(self, *, reference_cond: bool = False) -> None:
        super().__init__()
        self.num_layers = 1
        self.reference_cond = reference_cond
        self.reference_encoder = object() if reference_cond else None

    def forward(self, x, timesteps, get_layer_activation=-1, y=None, **unused_kwargs):
        return x


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


def test_cfg_reference_wrapper_guides_activations_when_requested() -> None:
    base_model = _ReferenceAwareModel()
    wrapped_model = ClassifierFreeReferenceModel(base_model)
    x = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    t = torch.tensor([1], dtype=torch.int64)
    y = {
        "reference_motion": torch.ones_like(x),
        "reference_scale": 2.0,
        "existing": torch.tensor([1.0]),
    }

    guided_output, activations = wrapped_model(x, t, get_layer_activation=0, y=y)

    expected = torch.full_like(x, 9.0)
    assert torch.equal(guided_output, expected)
    assert torch.equal(activations[0], expected)


def test_model_supports_reference_conditioning_detects_capability() -> None:
    assert model_supports_reference_conditioning(_DummyModel(reference_cond=True))
    assert not model_supports_reference_conditioning(_DummyModel(reference_cond=False))


def test_decoder_reference_mask_keeps_unconditioned_samples_on_baseline_path() -> None:
    torch.manual_seed(0)
    layer = GraphMotionDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
    tsteps, batch_size, njoints, d_model = 3, 2, 1, 8
    x = torch.randn(tsteps, batch_size, njoints, d_model)

    def fake_reference_mha_block(self, x, reference_memory, key_padding_mask, reference_batch_mask):
        return torch.full_like(x, 0.5)

    layer._reference_mha_block = types.MethodType(fake_reference_mha_block, layer)

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

    baseline = layer.forward(reference_memory=None, reference_batch_mask=None, **common_kwargs)
    mixed = layer.forward(
        reference_memory=torch.zeros(tsteps, batch_size, njoints, d_model),
        reference_batch_mask=torch.tensor([True, False]),
        **common_kwargs,
    )

    assert torch.allclose(mixed[:, 1], baseline[:, 1])


def test_validate_reference_mode_configuration_rejects_invalid_controlnet_setup() -> None:
    with pytest.raises(ValueError, match="requires --reference_motion"):
        validate_reference_mode_configuration("controlnet", reference_motion_path=None, skip_timesteps=0)
    with pytest.raises(ValueError, match="requires --skip_timesteps 0"):
        validate_reference_mode_configuration("controlnet", reference_motion_path="ref.npy", skip_timesteps=3)
    with pytest.raises(ValueError, match="does not support --reference_mode controlnet"):
        validate_reference_mode_configuration(
            "controlnet",
            reference_motion_path="ref.npy",
            skip_timesteps=0,
            model=_DummyModel(reference_cond=False),
        )


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


def test_sample_batch_rejects_controlnet_skip_timesteps() -> None:
    with pytest.raises(ValueError, match="requires --skip_timesteps 0"):
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