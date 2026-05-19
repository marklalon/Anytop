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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.motion_transformer import (  # noqa: E402
    CrossLimbTemporalBlock,
    GraphMotionDecoder,
    GraphMotionDecoderLayer,
)


D, H, K = 16, 4, 3
T, B, J = 5, 3, 6


def _block(dropout: float = 0.0, latent_width: int = D) -> CrossLimbTemporalBlock:
    torch.manual_seed(0)
    blk = CrossLimbTemporalBlock(
        D, H, num_latents=K, dropout=dropout, latent_width=latent_width
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
    x = torch.randn(T, B, J, D)
    tt = _template(B, per_batch_pattern=True)
    kpm = _kpm(B, [J, J - 1, J - 3])

    out_full = blk(x, tt, kpm)

    tt_bh = tt.reshape(B, H, T, T)
    for b in range(B):
        out_b = blk(
            x[:, b : b + 1],
            tt_bh[b : b + 1].reshape(H, T, T),
            kpm[b : b + 1],
        )
        assert torch.allclose(out_full[:, b], out_b[:, 0], atol=1e-5), (
            f"batch {b} diverges between full and sliced run -> batch-dim "
            f"ordering bug"
        )


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


def test_cross_limb_block_is_shared_across_layers_and_dead_attn_removed():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer, num_layers=3, cross_limb=True, cross_limb_latents=K, cross_limb_dim=8
    )
    assert isinstance(dec.cross_limb_block, CrossLimbTemporalBlock)
    assert dec.cross_limb_block.num_latents == K
    assert dec.cross_limb_block.latent_dim == 8  # bottleneck threaded through

    for lyr in dec.layers:
        # Cross-layer weight sharing: layers do not own a per-layer block.
        assert not hasattr(lyr, "cross_limb_block")
        # Dead nn.TransformerDecoderLayer attention modules were removed so
        # they no longer bloat the checkpoint.
        assert not hasattr(lyr, "self_attn")
        assert not hasattr(lyr, "multihead_attn")


def test_cross_limb_can_be_disabled():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=2, cross_limb=False)
    assert dec.cross_limb_block is None


def _run_decoder_recording_block(num_layers: int, last_n: int) -> list[bool]:
    """Drive GraphMotionDecoder.forward with stub layers that only record
    whether they received the shared cross_limb_block. Exercises the real
    loop + cross_limb_last_n gating without AnyTop's mask algebra."""
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(
        layer, num_layers=num_layers, cross_limb=True,
        cross_limb_latents=K, cross_limb_dim=8, cross_limb_last_n=last_n,
    )
    shared = dec.cross_limb_block
    got: list[bool] = []

    def make_stub():
        def stub(output, *a, cross_limb_block=None, **kw):
            got.append(cross_limb_block is shared)
            return output
        return stub

    dec.layers = torch.nn.ModuleList(dec.layers)  # keep len; replace __call__
    for i in range(num_layers):
        dec.layers[i].forward = make_stub()

    y = {"graph_dist": torch.zeros(1, 1, 1), "joints_relations": torch.zeros(1, 1, 1)}
    dec.forward(tgt=torch.zeros(1, 1, 1, D), timesteps_embs=None, memory=None, y=y)
    return got


def test_cross_limb_last_n_gates_which_layers_get_the_block():
    # 0 -> every layer gets the shared block.
    assert _run_decoder_recording_block(num_layers=4, last_n=0) == [True] * 4
    # N>0 -> only the last N layers.
    assert _run_decoder_recording_block(num_layers=5, last_n=2) == [
        False, False, False, True, True
    ]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
