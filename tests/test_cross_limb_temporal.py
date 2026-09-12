"""Unit tests for the cross-limb temporal block (architecture fix for
inter-limb frequency/phase coupling in inpainting).

These test the block in isolation -- it is self-contained (x,
joints_key_padding_mask in; x out) -- which targets the highest-risk part of
the change: the (T,B,J,d) <-> (J,T*B,d) / (K,T,B,d) <-> (T,B*K,d) reshapes. A
silent batch-dim transpose there would not change shapes but would corrupt
results, so we assert full-batch == per-sample-sliced equivalence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import model.motion_transformer as motion_transformer_module

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.motion_transformer import (  # noqa: E402
    CrossLimbTemporalBlock,
    GraphMotionDecoder,
    GraphMotionDecoderLayer,
)


D, H, K = 16, 4, 3
T, B, J = 5, 3, 6


def _block(
    dropout: float = 0.0,
    latent_width: int = D,
) -> CrossLimbTemporalBlock:
    torch.manual_seed(0)
    blk = CrossLimbTemporalBlock(
        D,
        H,
        num_latents=K,
        dropout=dropout,
        latent_width=latent_width,
    )
    blk.eval()  # deterministic (no dropout sampling) regardless of dropout arg
    return blk


def _kpm(b_count: int, valid_counts: list[int]) -> torch.Tensor:
    """(b_count, J) bool, True == padded joint."""
    idx = torch.arange(J)[None, :]
    n = torch.tensor(valid_counts)[:, None]
    return idx >= n


def _unreliable_mask(b_count: int, *, per_batch_pattern: bool = False) -> torch.Tensor:
    mask = torch.zeros(T, b_count, J)
    if per_batch_pattern:
        for b in range(b_count):
            mask[:, b, b % J] = 1.0
            mask[1::2, b, (b + 2) % J] = 1.0
    return mask


def test_block_preserves_shape_is_finite_and_trains():
    blk = _block()
    x = torch.randn(T, B, J, D, requires_grad=True)
    kpm = _kpm(B, [J, J, J])

    out = blk(x, kpm)

    assert out.shape == (T, B, J, D)
    assert torch.isfinite(out).all()

    out.sum().backward()
    # Every sub-path must receive gradient (no dead branch / no detach).
    for name, p in [
        ("latents", blk.latents),
        ("cross_in", blk.cross_in_attn.in_proj_weight),
        ("temporal", blk.temporal_attn.in_proj_weight),
        ("cross_out", blk.cross_out_attn.in_proj_weight),
        ("norm_cl", blk.norm_cl.weight),
    ]:
        assert p.grad is not None, f"{name} got no grad"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("latent_width", [D, 8])  # full-width (Identity) + bottleneck
def test_full_batch_equals_per_sample_sliced(latent_width):
    """Catches any batch-dim transpose in the flatten/unflatten: each sample
    gets distinct x, padding and reliability, so a wrong ordering makes the
    sliced result diverge from the full-batch result."""
    blk = _block(latent_width=latent_width)
    blk.reliability_bias.data.fill_(-2.0)
    blk.temporal_reliability_bias.data.fill_(-1.5)
    blk.cross_k_scale.data.fill_(0.7)
    x = torch.randn(T, B, J, D)
    kpm = _kpm(B, [J, J - 1, J - 3])
    unreliable = _unreliable_mask(B, per_batch_pattern=True)

    out_full = blk(x, kpm, unreliable)

    for b in range(B):
        out_b = blk(
            x[:, b : b + 1],
            kpm[b : b + 1],
            unreliable[:, b : b + 1],
        )
        assert torch.allclose(out_full[:, b], out_b[:, 0], atol=1e-5), (
            f"batch {b} diverges between full and sliced run -> batch-dim "
            f"ordering bug"
        )


def test_reliability_path_is_exact_noop_at_init():
    blk = _block()
    x = torch.randn(T, B, J, D)
    kpm = _kpm(B, [J, J - 1, J - 2])
    unreliable = _unreliable_mask(B, per_batch_pattern=True)

    out_without_mask = blk(x, kpm, None)
    out_with_mask = blk(x, kpm, unreliable)

    assert blk.time_emb_scale.item() == 0.0
    assert blk.reliability_bias.item() == 0.0
    assert blk.temporal_reliability_bias.item() == 0.0
    assert blk.cross_k_scale.item() == 0.0
    assert torch.allclose(out_without_mask, out_with_mask, atol=1e-6)


def test_unreliable_mask_none_matches_zero_mask_even_with_nonzero_bias():
    blk = _block()
    blk.reliability_bias.data.fill_(-7.0)
    blk.temporal_reliability_bias.data.fill_(-3.0)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J - 1])
    zero_mask = torch.zeros(T, 1, J)

    out_none = blk(x, kpm, None)
    out_zero = blk(x, kpm, zero_mask)

    assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_negative_reliability_bias_downweights_flagged_joint_influence():
    blk = _block()
    blk.reliability_bias.data.fill_(-20.0)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])
    unreliable = torch.zeros(T, 1, J)
    unreliable[:, 0, 0] = 1.0

    baseline = blk(x, kpm, unreliable)

    x_unreliable = x.clone()
    x_unreliable[:, 0, 0, :] += 8.0
    x_reliable = x.clone()
    x_reliable[:, 0, 1, :] += 8.0

    probe_joint = 2
    unreliable_delta = torch.linalg.norm(
        blk(x_unreliable, kpm, unreliable)[:, 0, probe_joint] - baseline[:, 0, probe_joint]
    )
    reliable_delta = torch.linalg.norm(
        blk(x_reliable, kpm, unreliable)[:, 0, probe_joint] - baseline[:, 0, probe_joint]
    )

    assert unreliable_delta < reliable_delta



# --- Frame-level reliability (temporal key bias) ---------------------------


def _whole_frame_mask(b_count: int, frames: list[int]) -> torch.Tensor:
    """Every joint of the listed frames flagged: the case a per-joint logit
    bias alone cannot express (softmax shift invariance cancels it)."""
    mask = torch.zeros(T, b_count, J)
    mask[frames] = 1.0
    return mask


def test_whole_frame_mask_is_invisible_to_the_cross_in_bias_alone():
    """The defect being fixed: with only the per-joint cross-in bias (the
    frame-level bias at 0), an all-joints-of-a-frame mask is exactly
    cancelled by softmax and changes nothing."""
    blk = _block()
    blk.reliability_bias.data.fill_(-5.0)
    blk.temporal_reliability_bias.data.zero_()
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])

    out_none = blk(x, kpm, None)
    out_frame = blk(x, kpm, _whole_frame_mask(1, [1, 2]))

    assert torch.allclose(out_none, out_frame, atol=1e-5)


def test_temporal_reliability_bias_makes_whole_frame_mask_change_output():
    blk = _block()
    blk.reliability_bias.data.fill_(-5.0)
    blk.temporal_reliability_bias.data.fill_(-4.0)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])

    out_none = blk(x, kpm, None)
    out_frame = blk(x, kpm, _whole_frame_mask(1, [1, 2]))

    assert not torch.allclose(out_none, out_frame, atol=1e-5)


def test_temporal_reliability_bias_downweights_flagged_frames_as_sources():
    """Perturbing a frame that is flagged unreliable must move the other
    frames' outputs less than perturbing an equally-placed reliable frame."""
    blk = _block()
    blk.temporal_reliability_bias.data.fill_(-20.0)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])
    unreliable = _whole_frame_mask(1, [1])

    baseline = blk(x, kpm, unreliable)
    x_unreliable = x.clone()
    x_unreliable[1] += 8.0            # flagged frame
    x_reliable = x.clone()
    x_reliable[2] += 8.0              # reliable frame

    probe = 4
    unreliable_delta = torch.linalg.norm(
        blk(x_unreliable, kpm, unreliable)[probe] - baseline[probe]
    )
    reliable_delta = torch.linalg.norm(
        blk(x_reliable, kpm, unreliable)[probe] - baseline[probe]
    )

    assert unreliable_delta < reliable_delta


def test_frame_unreliability_ignores_padded_joints():
    """A flag on a padded joint is not a flag on the frame: the per-frame
    fraction counts valid joints only, so it stays 0 and the temporal bias
    (even when large) has nothing to act on."""
    blk = _block()
    blk.temporal_reliability_bias.data.fill_(-9.0)
    valid = J - 2
    kpm = _kpm(1, [valid])
    x = torch.randn(T, 1, J, D)
    mask = torch.zeros(T, 1, J)
    mask[:, 0, valid:] = 1.0          # padded joints only

    out_none = blk(x, kpm, None)
    out_mask = blk(x, kpm, mask)

    assert torch.allclose(out_none[:, :, :valid], out_mask[:, :, :valid], atol=1e-5)


def test_uniformly_unreliable_window_cancels_the_temporal_bias():
    """Documented, expected: if every frame is equally unreliable there is
    no reliable source to prefer, and the uniform key bias cancels."""
    blk = _block()
    blk.temporal_reliability_bias.data.fill_(-6.0)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])

    out_none = blk(x, kpm, None)
    out_all = blk(x, kpm, torch.ones(T, 1, J))

    assert torch.allclose(out_none, out_all, atol=1e-5)


# --- Cross-K latent communication ------------------------------------------


def _cross_out_kv_after_perturbing_one_latent(
    blk: CrossLimbTemporalBlock, x, kpm, *, latent: int, delta: float
) -> torch.Tensor:
    """Run the block, adding ``delta`` to latent ``latent`` (batch 0) right
    after temporal attention, and return the (K, T*B, d) key/value tensor
    that reaches cross-out -- i.e. the latents AFTER cross-K."""
    orig_temporal = blk.temporal_attn.forward
    orig_cross_out = blk.cross_out_attn.forward
    captured = {}

    # Non-uniform across channels on purpose: cross-K is Pre-Norm, and a
    # constant shift of every channel is exactly what LayerNorm removes.
    bump = delta * torch.linspace(-1.0, 1.0, blk.latent_dim)

    def temporal_with_bump(*a, **kw):
        out, w = orig_temporal(*a, **kw)
        if delta != 0.0:
            out = out.clone()
            out[:, 0 * blk.num_latents + latent, :] += bump   # (T, B*K, d), b=0
        return out, w

    def cross_out_capture(query, key, value, *a, **kw):
        captured["kv"] = key.detach().clone()
        return orig_cross_out(query, key, value, *a, **kw)

    blk.temporal_attn.forward = temporal_with_bump
    blk.cross_out_attn.forward = cross_out_capture
    try:
        blk(x, kpm)
    finally:
        blk.temporal_attn.forward = orig_temporal
        blk.cross_out_attn.forward = orig_cross_out
    return captured["kv"]


@pytest.mark.parametrize("scale, expect_talk", [(0.0, False), (1.0, True)])
def test_cross_k_lets_latents_of_a_frame_communicate_only_when_gated_open(scale, expect_talk):
    blk = _block()
    blk.cross_k_scale.data.fill_(scale)
    x = torch.randn(T, 1, J, D)
    kpm = _kpm(1, [J])
    bumped = 1

    kv_base = _cross_out_kv_after_perturbing_one_latent(blk, x, kpm, latent=bumped, delta=0.0)
    kv_bump = _cross_out_kv_after_perturbing_one_latent(blk, x, kpm, latent=bumped, delta=5.0)

    others = [k for k in range(K) if k != bumped]
    moved = not torch.allclose(kv_base[others], kv_bump[others], atol=1e-6)
    assert moved == expect_talk
    # The bumped latent itself always moves (its own residual carries the bump).
    assert not torch.allclose(kv_base[bumped], kv_bump[bumped], atol=1e-6)


def test_cross_k_is_exact_noop_at_zero_scale_but_trains():
    blk = _block()
    x = torch.randn(T, B, J, D, requires_grad=True)
    kpm = _kpm(B, [J, J, J])

    out = blk(x, kpm)
    out.sum().backward()
    # Gate closed: the cross-K weights sit behind a zero scale and get no
    # gradient, the scale itself does (it is how the path opens).
    assert blk.cross_k_scale.grad is not None
    assert blk.cross_k_scale.grad.abs().sum() > 0
    w_grad = blk.cross_k_attn.in_proj_weight.grad
    assert w_grad is None or w_grad.abs().sum() == 0


def test_cross_k_attention_inherits_the_block_dropout():
    blk = CrossLimbTemporalBlock(D, H, num_latents=K, dropout=0.1, latent_width=D)
    assert blk.cross_k_attn.dropout == blk.temporal_attn.dropout == 0.1


def test_new_gates_are_scalar_and_zero_init():
    blk = _block()
    assert blk.temporal_reliability_bias.shape == (1,)
    assert blk.cross_k_scale.shape == (1,)
    assert blk.temporal_reliability_bias.item() == 0.0
    assert blk.cross_k_scale.item() == 0.0


def test_padded_joints_do_not_leak_into_valid_outputs():
    """key_padding_mask semantics (True == padded, excluded from cross-in):
    perturbing padded-joint inputs must not change valid-joint outputs."""
    blk = _block()
    valid = J - 2
    kpm = _kpm(1, [valid])
    x = torch.randn(T, 1, J, D)

    out_a = blk(x, kpm)
    x2 = x.clone()
    x2[:, :, valid:, :] += 5.0  # perturb only padded joints
    out_b = blk(x2, kpm)

    assert torch.allclose(out_a[:, :, :valid], out_b[:, :, :valid], atol=1e-5)


def test_bottleneck_width_is_clamped_and_multiple_of_heads():
    # <= d_model and rounded down to a multiple of nhead.
    assert _block(latent_width=8).latent_dim == 8       # 8 % 4 == 0
    assert _block(latent_width=10).latent_dim == 8      # 10 -> 8
    assert _block(latent_width=1000).latent_dim == D    # clamped to d_model
    # No-bottleneck case uses Identity projections (zero extra params).
    full = _block(latent_width=D)
    assert isinstance(full.proj_in, torch.nn.Identity)
    assert isinstance(full.proj_out, torch.nn.Identity)
    # Bottleneck case wires real projections d_model <-> d_cl.
    bn = _block(latent_width=8)
    assert (bn.proj_in.in_features, bn.proj_in.out_features) == (D, 8)
    assert (bn.proj_out.in_features, bn.proj_out.out_features) == (8, D)


def test_cross_limb_blocks_are_per_layer_and_dead_attn_removed():
    num_layers = 3
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer, num_layers=num_layers, cross_limb=True,
        cross_limb_latents=K, cross_limb_dim=8,
    )
    assert isinstance(dec.cross_limb_blocks, torch.nn.ModuleList)
    assert len(dec.cross_limb_blocks) == num_layers  # last_n=0 -> all layers
    for blk in dec.cross_limb_blocks:
        assert isinstance(blk, CrossLimbTemporalBlock)
        assert blk.num_latents == K
        assert blk.latent_dim == 8  # bottleneck threaded through
    # Independent instances, not aliases of the same block.
    ids = {id(b) for b in dec.cross_limb_blocks}
    assert len(ids) == num_layers

    for lyr in dec.layers:
        # Cross-limb pathway is owned by the decoder, not the layers.
        assert not hasattr(lyr, "cross_limb_block")
        assert not hasattr(lyr, "cross_limb_blocks")
        # Dead nn.TransformerDecoderLayer attention modules were removed so
        # they no longer bloat the checkpoint.
        assert not hasattr(lyr, "self_attn")
        assert not hasattr(lyr, "multihead_attn")


def test_cross_limb_last_n_sizes_the_module_list():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer, num_layers=5, cross_limb=True,
        cross_limb_latents=K, cross_limb_dim=8, cross_limb_last_n=2,
    )
    # Only the last N layers get a block, so the list has exactly N entries.
    assert len(dec.cross_limb_blocks) == 2


def test_cross_limb_can_be_disabled():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=2, cross_limb=False)
    assert dec.cross_limb_blocks is None


def _run_decoder_recording_block(num_layers: int, last_n: int) -> list:
    """Drive GraphMotionDecoder.forward with stub layers that record which
    cross-limb block (if any) each layer received. Exercises the real loop +
    cross_limb_last_n gating without AnyTop's mask algebra."""
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer, num_layers=num_layers, cross_limb=True,
        cross_limb_latents=K, cross_limb_dim=8, cross_limb_last_n=last_n,
    )
    got: list = []

    def make_stub():
        def stub(output, *a, cross_limb_block=None, **kw):
            got.append(cross_limb_block)
            return output
        return stub

    dec.layers = torch.nn.ModuleList(dec.layers)  # keep len; replace __call__
    for i in range(num_layers):
        dec.layers[i].forward = make_stub()

    y = {"graph_dist": torch.zeros(1, 1, 1), "joints_relations": torch.zeros(1, 1, 1)}
    dec.forward(tgt=torch.zeros(1, 1, 1, D), timesteps_embs=None, memory=None, y=y)
    return got


def test_cross_limb_last_n_gates_which_layers_get_a_block():
    # last_n=0 -> every layer gets its own block, in order.
    blocks_all = _run_decoder_recording_block(num_layers=4, last_n=0)
    assert all(b is not None for b in blocks_all)
    assert len({id(b) for b in blocks_all}) == 4  # all distinct

    # last_n=N -> only the last N layers get a block; the first num_layers-N
    # see None. Distinct instances for each active layer.
    blocks_tail = _run_decoder_recording_block(num_layers=5, last_n=2)
    assert [b is None for b in blocks_tail] == [True, True, True, False, False]
    active = [b for b in blocks_tail if b is not None]
    assert len({id(b) for b in active}) == 2


def test_decoder_expands_graph_relations_once_per_batch_not_per_frame():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=1, cross_limb=False)
    captured = {}

    def stub(output, *a, **kw):
        captured["topology_shape"] = a[1].shape
        captured["edge_shape"] = a[2].shape
        return output

    dec.layers[0].forward = stub
    y = {
        "graph_dist": torch.zeros(2, 3, 3, dtype=torch.int64),
        "joints_relations": torch.zeros(2, 3, 3, dtype=torch.int64),
    }

    dec.forward(
        tgt=torch.zeros(5, 2, 3, D),
        timesteps_embs=torch.zeros(2, D),
        memory=None,
        y=y,
    )

    assert captured["topology_shape"] == (2, H, 3, 3)
    assert captured["edge_shape"] == (2, H, 3, 3)


def test_decoder_reuses_precomputed_loop_phase_embeddings(monkeypatch):
    calls: list[tuple[int, int, int]] = []
    original = motion_transformer_module.circular_phase_embedding

    def wrapped(length, dim, device, dtype):
        calls.append((length, dim))
        return original(length, dim, device, dtype)

    monkeypatch.setattr(motion_transformer_module, "circular_phase_embedding", wrapped)

    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer,
        num_layers=3,
        cross_limb=True,
        cross_limb_latents=K,
        cross_limb_dim=8,
    )
    seen = []

    def stub(output, *a, **kw):
        seen.append((kw["loop_phase_embedding"], kw["cross_limb_time_embedding"]))
        return output

    for decoder_layer in dec.layers:
        decoder_layer.forward = stub

    y = {
        "graph_dist": torch.zeros(2, J, J, dtype=torch.int64),
        "joints_relations": torch.zeros(2, J, J, dtype=torch.int64),
    }
    dec.forward(
        tgt=torch.zeros(T, 2, J, D),
        timesteps_embs=torch.zeros(2, D),
        memory=None,
        y=y,
        loop_phase_mask=torch.tensor([True, False]),
    )

    assert calls == [(T, D), (T, 8)]
    assert len({id(pair[0]) for pair in seen}) == 1
    assert len({id(pair[1]) for pair in seen}) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
