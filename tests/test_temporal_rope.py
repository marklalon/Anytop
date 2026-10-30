import torch

from model.motion_transformer import (
    SelectiveMultiheadAttention,
    apply_temporal_rope,
    temporal_rope_tables,
)


def _rope_scores(frames, head_dim, loop):
    cos, sin = temporal_rope_tables(frames, head_dim, torch.device('cpu'), loop=loop)
    q = apply_temporal_rope(torch.randn(head_dim).expand(frames, head_dim).clone(), cos, sin)
    k = apply_temporal_rope(torch.randn(head_dim).expand(frames, head_dim).clone(), cos, sin)
    return q @ k.T


def test_rope_score_depends_only_on_frame_offset():
    scores = _rope_scores(24, 16, loop=False)
    for offset in (0, 3, -5):
        diagonal = scores.diagonal(offset)
        assert torch.allclose(diagonal, diagonal[0].expand_as(diagonal), atol=1e-4)


def test_loop_rope_score_depends_on_offset_modulo_window():
    frames = 30
    scores = _rope_scores(frames, 64, loop=True)
    # Circulant: shifting both frames by s (mod T) keeps the score, so the
    # seam pair (T - 1, 0) scores like any one-frame step.
    for shift in (1, 7, frames - 1):
        rolled = torch.roll(scores, shifts=(shift, shift), dims=(0, 1))
        assert torch.allclose(rolled, scores, atol=1e-4)
    assert torch.allclose(scores[frames - 1, 0], scores[0, 1], atol=1e-4)


def test_loop_rope_keeps_nyquist_and_one_turn():
    frames, head_dim = 30, 64
    cos, _ = temporal_rope_tables(frames, head_dim, torch.device('cpu'), loop=True)
    # Fastest pair alternates sign each frame; slowest turns once per window.
    assert torch.allclose(cos[:, 0], torch.tensor([(-1.0) ** t for t in range(frames)]), atol=1e-5)
    one_turn = torch.cos(torch.arange(frames) * (2 * torch.pi / frames))
    assert torch.allclose(cos[:, -1], one_turn, atol=1e-5)


def test_rope_changes_attention_only_when_passed():
    torch.manual_seed(0)
    attn = SelectiveMultiheadAttention(32, 4)
    x = torch.randn(10, 3, 32)
    cos, sin = temporal_rope_tables(10, attn.head_dim, torch.device('cpu'))
    rope = (cos.expand(3, -1, -1), sin.expand(3, -1, -1))
    plain, _ = attn(x, x, x, need_weights=False)
    rotated, _ = attn(x, x, x, need_weights=False, rope=rope)
    again, _ = attn(x, x, x, need_weights=False)
    assert torch.equal(plain, again)
    assert not torch.allclose(plain, rotated)


def test_decoder_loop_rows_are_circular_shift_equivariant():
    from model.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer

    torch.manual_seed(0)
    frames, batch, joints, width, heads = 12, 2, 3, 16, 2
    layer = GraphMotionDecoderLayer(width, heads, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=2, cross_limb=False, temporal_rope=True).eval()
    y = {
        "graph_dist": torch.zeros(batch, joints, joints, dtype=torch.int64),
        "joints_relations": torch.zeros(batch, joints, joints, dtype=torch.int64),
    }
    is_loop = torch.tensor([True, False])

    def run(tgt):
        return dec(tgt=tgt, timesteps_embs=torch.zeros(batch, width), memory=None, y=y,
                   loop_phase_mask=is_loop)

    x = torch.randn(frames, batch, joints, width)
    with torch.no_grad():
        out = run(x)
        shifted = run(torch.roll(x, 5, dims=0))
    # The per-layer phase gate is zero-init, so only RoPE sees the frame index:
    # the loop row has no seam, the open row does.
    assert torch.allclose(torch.roll(out[:, 0], 5, dims=0), shifted[:, 0], atol=1e-4)
    assert not torch.allclose(torch.roll(out[:, 1], 5, dims=0), shifted[:, 1], atol=1e-3)


def test_decoder_rope_off_rows_match_the_model_without_rope():
    from model.motion_transformer import GraphMotionDecoder, GraphMotionDecoderLayer

    torch.manual_seed(0)
    frames, batch, joints, width, heads = 12, 2, 3, 16, 2
    layer = GraphMotionDecoderLayer(width, heads, dim_feedforward=32, dropout=0.0)
    with_rope = GraphMotionDecoder(layer, num_layers=2, cross_limb=False, temporal_rope=True).eval()
    without = GraphMotionDecoder(layer, num_layers=2, cross_limb=False, temporal_rope=False).eval()
    without.load_state_dict(with_rope.state_dict())
    y = {
        "graph_dist": torch.zeros(batch, joints, joints, dtype=torch.int64),
        "joints_relations": torch.zeros(batch, joints, joints, dtype=torch.int64),
    }
    x = torch.randn(frames, batch, joints, width)
    kw = dict(timesteps_embs=torch.zeros(batch, width), memory=None, y=y)
    with torch.no_grad():
        off = with_rope(tgt=x, temporal_rope_off=torch.tensor([True, False]), **kw)
        rope = with_rope(tgt=x, **kw)
        plain = without(tgt=x, **kw)
    assert torch.allclose(off[:, 0], plain[:, 0], atol=1e-5)
    assert torch.allclose(off[:, 1], rope[:, 1], atol=1e-5)
    assert not torch.allclose(rope[:, 0], plain[:, 0], atol=1e-3)
