import torch

from model.motion_transformer import (
    SelectiveMultiheadAttention,
    apply_temporal_rope,
    temporal_rope_tables,
)


def test_rope_score_depends_only_on_frame_offset():
    frames, head_dim = 24, 16
    cos, sin = temporal_rope_tables(frames, head_dim, torch.device('cpu'))
    q = apply_temporal_rope(torch.randn(head_dim).expand(frames, head_dim).clone(), cos, sin)
    k = apply_temporal_rope(torch.randn(head_dim).expand(frames, head_dim).clone(), cos, sin)
    scores = q @ k.T
    for offset in (0, 3, -5):
        diagonal = scores.diagonal(offset)
        assert torch.allclose(diagonal, diagonal[0].expand_as(diagonal), atol=1e-4)


def test_rope_changes_attention_only_when_enabled():
    torch.manual_seed(0)
    attn = SelectiveMultiheadAttention(32, 4)
    x = torch.randn(10, 3, 32)
    plain, _ = attn(x, x, x, need_weights=False)
    attn.enable_rope()
    rotated, _ = attn(x, x, x, need_weights=False)
    attn.rope = False
    again, _ = attn(x, x, x, need_weights=False)
    assert torch.equal(plain, again)
    assert not torch.allclose(plain, rotated)
