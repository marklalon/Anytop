"""Unit tests for the cross-limb temporal block (architecture fix for
inter-limb frequency/phase coupling in inpainting).

These test the block in isolation -- it is self-contained (x,
temporal_template, joints_key_padding_mask in; x out) -- which targets the
highest-risk part of the change: the (T,B,J,d) <-> (J,T*B,d) /
(K,T,B,d) <-> (T,B*K,d) reshapes and the per-latent mask expansion. A
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


def _template(b_count: int, *, per_batch_pattern: bool = False) -> torch.Tensor:
    """(b_count*H, T, T) additive float mask: 0.0 == attend, -1e4 == block.

    col 0 (T-pose token) and the diagonal are always attendable so softmax
    never sees an all -inf row.
    """
    tt = torch.zeros(b_count, H, T, T)
    if per_batch_pattern:
        g = torch.Generator().manual_seed(123)
        for b in range(b_count):
            blocked = torch.rand(T, T, generator=g) < 0.5
            blocked[:, 0] = False
            blocked[torch.arange(T), torch.arange(T)] = False
            tt[b, :, blocked] = -1e4
    return tt.reshape(b_count * H, T, T)


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
    tt = _template(B)
    kpm = _kpm(B, [J, J, J])

    out = blk(x, tt, kpm)

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
    """Catches any batch-dim transpose in the flatten/unflatten + mask
    expansion: each sample gets distinct x, mask and padding, so a wrong
    ordering makes the sliced result diverge from the full-batch result."""
    blk = _block(latent_width=latent_width)
    blk.reliability_bias.data.fill_(-2.0)
    x = torch.randn(T, B, J, D)
    tt = _template(B, per_batch_pattern=True)
    kpm = _kpm(B, [J, J - 1, J - 3])
    unreliable = _unreliable_mask(B, per_batch_pattern=True)

    out_full = blk(x, tt, kpm, unreliable)

    tt_bh = tt.reshape(B, H, T, T)
    for b in range(B):
        out_b = blk(
            x[:, b : b + 1],
            tt_bh[b : b + 1].reshape(H, T, T),
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
    tt = _template(B, per_batch_pattern=True)
    kpm = _kpm(B, [J, J - 1, J - 2])
    unreliable = _unreliable_mask(B, per_batch_pattern=True)

    out_without_mask = blk(x, tt, kpm, None)
    out_with_mask = blk(x, tt, kpm, unreliable)

    assert blk.time_emb_scale.item() == 0.0
    assert blk.reliability_bias.item() == 0.0
    assert torch.allclose(out_without_mask, out_with_mask, atol=1e-6)


def test_unreliable_mask_none_matches_zero_mask_even_with_nonzero_bias():
    blk = _block()
    blk.reliability_bias.data.fill_(-7.0)
    x = torch.randn(T, 1, J, D)
    tt = _template(1, per_batch_pattern=True)
    kpm = _kpm(1, [J - 1])
    zero_mask = torch.zeros(T, 1, J)

    out_none = blk(x, tt, kpm, None)
    out_zero = blk(x, tt, kpm, zero_mask)

    assert torch.allclose(out_none, out_zero, atol=1e-6)


def test_negative_reliability_bias_downweights_flagged_joint_influence():
    blk = _block()
    blk.reliability_bias.data.fill_(-20.0)
    x = torch.randn(T, 1, J, D)
    tt = _template(1, per_batch_pattern=True)
    kpm = _kpm(1, [J])
    unreliable = torch.zeros(T, 1, J)
    unreliable[:, 0, 0] = 1.0

    baseline = blk(x, tt, kpm, unreliable)

    x_unreliable = x.clone()
    x_unreliable[:, 0, 0, :] += 8.0
    x_reliable = x.clone()
    x_reliable[:, 0, 1, :] += 8.0

    probe_joint = 2
    unreliable_delta = torch.linalg.norm(
        blk(x_unreliable, tt, kpm, unreliable)[:, 0, probe_joint] - baseline[:, 0, probe_joint]
    )
    reliable_delta = torch.linalg.norm(
        blk(x_reliable, tt, kpm, unreliable)[:, 0, probe_joint] - baseline[:, 0, probe_joint]
    )

    assert unreliable_delta < reliable_delta



def test_padded_joints_do_not_leak_into_valid_outputs():
    """key_padding_mask semantics (True == padded, excluded from cross-in):
    perturbing padded-joint inputs must not change valid-joint outputs."""
    blk = _block()
    valid = J - 2
    kpm = _kpm(1, [valid])
    tt = _template(1)
    x = torch.randn(T, 1, J, D)

    out_a = blk(x, tt, kpm)
    x2 = x.clone()
    x2[:, :, valid:, :] += 5.0  # perturb only padded joints
    out_b = blk(x2, tt, kpm)

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
    original = motion_transformer_module._circular_phase_embedding

    def wrapped(length, dim, batch_size, device, dtype, lengths=None):
        calls.append((length, dim, batch_size))
        return original(length, dim, batch_size, device, dtype, lengths)

    monkeypatch.setattr(motion_transformer_module, "_circular_phase_embedding", wrapped)

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
        lengths=torch.tensor([T - 1, T - 1]),
    )

    assert calls == [(T, D, 2), (T, 8, 2)]
    assert len({id(pair[0]) for pair in seen}) == 1
    assert len({id(pair[1]) for pair in seen}) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
