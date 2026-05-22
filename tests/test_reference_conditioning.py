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
    validate_reference_mode_configuration,
)
from model.anytop import AnyTop, ReferencePriorEncoder  # noqa: E402
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
        lengths,
        translation_root_index,
        parents_batch,
        joints_embedded_names,
    ):
        self.last_kwargs = {
            "reference_motion": reference_motion,
            "n_joints": n_joints,
            "lengths": lengths,
            "translation_root_index": translation_root_index,
            "parents_batch": parents_batch,
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


def test_build_reference_conditioning_reuses_detached_x_start_without_clone() -> None:
    diffusion = GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model = _DummyModel(reference_cond=True)
    model.reference_uncond_prob = 0.0
    x_start = torch.randn((2, 3, 13, 4), dtype=torch.float32, requires_grad=True)
    model_kwargs = {"y": {}}

    diffusion._build_reference_conditioning(model, x_start, model_kwargs)

    reference_motion = model_kwargs["y"]["reference_motion"]
    assert reference_motion is not None
    assert reference_motion.requires_grad is False
    assert reference_motion.data_ptr() == x_start.data_ptr()
    assert torch.equal(model_kwargs["y"]["reference_cond_mask"], torch.ones(2, dtype=torch.bool))


def test_anytop_forward_accepts_reference_motion_with_independent_frame_count() -> None:
    model = AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        skip_t5=True,
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
        "reference_lengths": torch.tensor([3], dtype=torch.int64),
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


def test_reference_prior_encoder_rejects_feature_schemas_shorter_than_13_dims() -> None:
    with pytest.raises(ValueError, match="13-dim"):
        ReferencePriorEncoder(
            input_feats=12,
            latent_dim=32,
            ff_size=64,
            num_heads=4,
            dropout=0.0,
            t5_out_dim=8,
            skip_t5=True,
            num_layers=1,
        )


def test_reference_prior_encoder_ignores_padded_tail_frames_after_conv() -> None:
    torch.manual_seed(0)
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=True,
        num_layers=1,
    )
    encoder.eval()

    valid_motion = torch.randn((1, 4, 13, 3), dtype=torch.float32)
    padded_motion = torch.zeros((1, 4, 13, 5), dtype=torch.float32)
    padded_motion[..., :3] = valid_motion
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)
    parents = [np.asarray([-1, 0, 1, 1], dtype=np.int64)]

    tokens_short = encoder(
        valid_motion,
        n_joints=torch.tensor([4]),
        lengths=torch.tensor([3]),
        translation_root_index=torch.tensor([0]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )
    tokens_padded = encoder(
        padded_motion,
        n_joints=torch.tensor([4]),
        lengths=torch.tensor([3]),
        translation_root_index=torch.tensor([0]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )

    assert torch.allclose(tokens_short, tokens_padded, atol=1e-5, rtol=1e-5)


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

    ref_tensor, reference_kwargs, reference_frame_count = _prepare_reference_prior_bundle(
        str(reference_motion_path),
        "TestObject",
        source_cond,
        max_joints=2,
        target_feature_len=13,
        batch_size=1,
    )

    assert reference_frame_count == 5
    assert ref_tensor.shape == (1, 2, 13, 5)
    assert torch.equal(reference_kwargs["reference_n_joints"], torch.tensor([2], dtype=torch.long))
    assert torch.equal(reference_kwargs["reference_lengths"], torch.tensor([5], dtype=torch.long))
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


def test_prepare_reference_for_mode_controlnet_keeps_reference_and_output_lengths_separate(tmp_path: Path) -> None:
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
    assert bundle["loaded_reference_joint_count"] == 2
    assert bundle["output_frame_count"] == 7
    assert bundle["reference_motion"].shape == (1, 4, 13, 5)


def test_should_retarget_reference_skips_controlnet_source_space_path() -> None:
    assert _should_retarget_reference("Horse", "Buffalo", "img2img")
    assert not _should_retarget_reference("Horse", "Buffalo", "controlnet")
    assert not _should_retarget_reference("Horse", "Horse", "img2img")


def test_reference_prior_encoder_uses_effective_translation_root() -> None:
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=True,
        num_layers=1,
    )
    encoder.eval()

    motion = torch.zeros((1, 4, 13, 5), dtype=torch.float32)
    motion[0, 0, 9, 1:] = 0.25
    motion[0, 2, 9, 1:] = 2.0
    motion[0, 2, 10, 1:] = 1.0
    joints_names_embs = torch.zeros((1, 4, 8), dtype=torch.float32)
    parents = [np.asarray([-1, 0, 1, 1], dtype=np.int64)]

    tokens_root0 = encoder(
        motion,
        n_joints=torch.tensor([4]),
        lengths=torch.tensor([5]),
        translation_root_index=torch.tensor([0]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )
    tokens_root2 = encoder(
        motion,
        n_joints=torch.tensor([4]),
        lengths=torch.tensor([5]),
        translation_root_index=torch.tensor([2]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )

    assert tokens_root0.shape == (8, 1, 32)
    assert not torch.allclose(tokens_root0, tokens_root2)


def test_reference_prior_encoder_accepts_padded_motion_with_unpadded_parents() -> None:
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=False,
        num_layers=1,
    )
    encoder.eval()

    motion = torch.zeros((1, 8, 13, 5), dtype=torch.float32)
    joints_names_embs = torch.zeros((1, 8, 8), dtype=torch.float32)
    parents = [np.asarray([-1, 0, 1, 1], dtype=np.int64)]

    tokens = encoder(
        motion,
        n_joints=torch.tensor([4]),
        lengths=torch.tensor([5]),
        translation_root_index=torch.tensor([2]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )

    assert tokens.shape == (8, 1, 32)


def test_reference_prior_encoder_respects_zero_temporal_layers() -> None:
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=True,
        num_layers=0,
    )

    assert len(encoder.conv_blocks) == 0
    assert len(encoder.temporal_layers) == 0


def test_depth_to_root_matches_iterative_relaxation() -> None:
    """Pointer-doubling depth must equal the O(J) relaxation it replaced."""

    def _relaxation_depth(parents: torch.Tensor) -> torch.Tensor:
        max_joints = parents.shape[1]
        safe_parents = parents.clamp_min(0)
        has_parent = parents >= 0
        depth = torch.zeros_like(parents, dtype=torch.float32)
        for _ in range(max_joints - 1):
            parent_depth = depth.gather(1, safe_parents)
            depth = torch.maximum(depth, torch.where(has_parent, parent_depth + 1.0, depth))
        return depth

    torch.manual_seed(0)
    for _ in range(64):
        max_joints = int(torch.randint(2, 48, (1,)))
        n_joints = int(torch.randint(1, max_joints + 1, (1,)))
        parents = torch.full((1, max_joints), -1, dtype=torch.long)
        for joint in range(1, n_joints):
            parents[0, joint] = int(torch.randint(0, joint, (1,)))
        assert torch.equal(
            ReferencePriorEncoder._depth_to_root(parents, torch.float32),
            _relaxation_depth(parents),
        )

    # Worst case: a single chain is the deepest tree a J-joint skeleton allows.
    chain = torch.arange(-1, 31, dtype=torch.long).unsqueeze(0)
    chain_depth = ReferencePriorEncoder._depth_to_root(chain, torch.float32)
    assert torch.equal(chain_depth, _relaxation_depth(chain))
    assert int(chain_depth.max()) == 31


def test_reference_prior_encoder_honours_per_sample_metadata_in_batch() -> None:
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=True,
        num_layers=1,
    )
    encoder.eval()

    torch.manual_seed(0)
    # Both batch entries share identical motion, so any token difference can
    # only come from the per-sample skeleton metadata (never silently broadcast).
    shared_motion = torch.randn((1, 6, 13, 7), dtype=torch.float32)
    motion = shared_motion.expand(2, -1, -1, -1).contiguous()
    joints_names_embs = torch.zeros((2, 6, 8), dtype=torch.float32)
    parents = [
        np.asarray([-1, 0, 1, 2], dtype=np.int64),       # chain of 4
        np.asarray([-1, 0, 0, 0, 0], dtype=np.int64),     # star of 5
    ]

    tokens = encoder(
        motion,
        n_joints=torch.tensor([4, 5]),
        lengths=torch.tensor([7, 7]),
        translation_root_index=torch.tensor([0, 0]),
        parents_batch=parents,
        joints_embedded_names=joints_names_embs,
    )

    assert tokens.shape == (8, 2, 32)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])


def test_reference_prior_encoder_rejects_batch_size_mismatch() -> None:
    encoder = ReferencePriorEncoder(
        input_feats=13,
        latent_dim=32,
        ff_size=64,
        num_heads=4,
        dropout=0.0,
        t5_out_dim=8,
        skip_t5=True,
        num_layers=1,
    )
    encoder.eval()

    motion = torch.zeros((2, 4, 13, 5), dtype=torch.float32)
    joints_names_embs = torch.zeros((2, 4, 8), dtype=torch.float32)

    # A single parents entry must not be silently broadcast onto a 2-sample batch.
    with pytest.raises(ValueError, match="does not match batch size 2"):
        encoder(
            motion,
            n_joints=torch.tensor([4, 4]),
            lengths=torch.tensor([5, 5]),
            translation_root_index=torch.tensor([0, 0]),
            parents_batch=[np.asarray([-1, 0, 1, 1], dtype=np.int64)],
            joints_embedded_names=joints_names_embs,
        )


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


def test_decoder_reference_block_accepts_shared_prior_tokens() -> None:
    torch.manual_seed(0)
    layer = GraphMotionDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, dropout=0.0)
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
        cross_limb_unreliable_mask=None,
        reference_batch_mask=torch.ones(batch_size, dtype=torch.bool),
    )

    assert output.shape == x.shape


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
    sample_shape = (1, 3, 13, 6)
    reference_motion = torch.ones((1, 3, 13, 4), dtype=torch.float32)
    inpaint_mask = torch.ones((1, 3, 1, 6), dtype=torch.float32)

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


def test_validate_reference_sampling_request_rejects_cross_species_controlnet_inpaint() -> None:
    with pytest.raises(ValueError, match="cross-species inpainting is not supported"):
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