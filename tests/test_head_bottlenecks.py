"""Conditioning heads use latent_dim; last-layer FFN width stays configurable."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim  # noqa: E402
from model.motion_diffusion import MotionDiffusion  # noqa: E402
from model.motion_transformer import (  # noqa: E402
    ACTION_ADALN_PARAMS_PER_LAYER,
    GraphMotionDecoder,
    GraphMotionDecoderLayer,
)
from utils.model_util import get_gmdm_args  # noqa: E402

D, FF, L, T5 = 32, 64, 3, 24


def _args(**overrides):
    a = SimpleNamespace(
        t5_out_dim=T5, latent_dim=D, ff_size=FF, layers=L, dropout_prob=0.0,
        value_emb=False, cross_limb_latents=2, cross_limb_dim=8, cross_limb_last_n=1,
        species_cond=True,
        joint_name_drop_prob=0.0, action_label_cond=True,
        action_conditioning=None,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _shapes(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def test_get_gmdm_args_only_exposes_last_layer_width():
    a = _args()
    for k in ('action_adaln_bottleneck', 'species_film_bottleneck', 'last_layer_ff'):
        assert not hasattr(a, k)
    kw = get_gmdm_args(a)
    assert 'action_adaln_bottleneck' not in kw
    assert 'species_film_bottleneck' not in kw
    assert kw['last_layer_ff'] == 0


def test_explicit_defaults_match_implicit_layout():
    base = MotionDiffusion(**get_gmdm_args(_args()))
    explicit = MotionDiffusion(**get_gmdm_args(_args(
        last_layer_ff=0)))
    assert _shapes(base) == _shapes(explicit)
    dec = base.seqTransDecoder
    assert dec.action_adaln[0].out_features == D
    assert base.input_process.species_film_j[0].out_features == D
    assert [layer.linear1.out_features for layer in dec.layers] == [FF] * L


def test_full_width_conditioning_heads_and_narrow_last_layer():
    m = MotionDiffusion(**get_gmdm_args(_args(
        last_layer_ff=16)))
    dec = m.seqTransDecoder
    assert tuple(dec.action_adaln[0].weight.shape) == (D, D)
    assert tuple(dec.action_adaln[2].weight.shape) == (L * ACTION_ADALN_PARAMS_PER_LAYER * D, D)
    # Output layer stays zero-init: gamma = beta = 0 on a fresh model.
    assert torch.count_nonzero(dec.action_adaln[2].weight) == 0
    assert torch.count_nonzero(dec.action_adaln[2].bias) == 0
    assert tuple(m.input_process.species_film_j[0].weight.shape) == (D, T5 + species_code_dim(T5))
    assert tuple(m.input_process.species_film_j[2].weight.shape) == (2 * T5, D)
    assert torch.count_nonzero(m.input_process.species_film_j[2].weight) == 0
    # Only the LAST layer is narrower; the others keep ff_size.
    assert [layer.linear1.out_features for layer in dec.layers] == [FF] * (L - 1) + [16]
    assert dec.layers[-1].linear2.in_features == 16
    assert isinstance(dec.layers[-1], GraphMotionDecoderLayer)
    assert dec.layers[-1].d_model == D and dec.layers[-1].heads == dec.nheads


def test_last_layer_ff_equal_to_ff_size_keeps_the_clone():
    layer = GraphMotionDecoderLayer(D, 4, dim_feedforward=FF, dropout=0.0, activation='gelu')
    dec = GraphMotionDecoder(layer, num_layers=L, cross_limb=False, last_layer_ff=FF)
    # Same width -> no replacement, so the clone's weights are the prototype's.
    assert torch.equal(dec.layers[-1].linear1.weight, layer.linear1.weight)


def test_last_layer_ff_replacement_keeps_dropout_and_activation():
    layer = GraphMotionDecoderLayer(D, 4, dim_feedforward=FF, dropout=0.25, activation='gelu')
    dec = GraphMotionDecoder(layer, num_layers=L, cross_limb=False, last_layer_ff=16)
    last = dec.layers[-1]
    assert last.dropout1.p == 0.25 and last.dropout.p == 0.25
    assert last.activation is layer.activation


@pytest.mark.parametrize('flag', ['last_layer_ff'])
def test_negative_widths_are_refused(flag):
    with pytest.raises(ValueError, match=flag):
        MotionDiffusion(**get_gmdm_args(_args(**{flag: -1})))


def test_forward_runs_with_full_width_heads():
    torch.manual_seed(0)
    m = MotionDiffusion(**get_gmdm_args(_args(
        last_layer_ff=16))).eval()
    # Exercise action modulation through the decoder with a minimal condition.
    dec = m.seqTransDecoder
    B, J, frames = 2, 4, 5
    y = {
        'graph_dist': torch.zeros(B, J, J, dtype=torch.int64),
        'joints_relations': torch.zeros(B, J, J, dtype=torch.int64),
    }
    out = dec(
        tgt=torch.randn(frames, B, J, D),
        timesteps_embs=torch.randn(B, D),
        memory=None,
        y=y,
        tgt_key_padding_mask=torch.zeros(B, J, dtype=torch.bool),
        action_adaln_cond=torch.randn(B, D),
    )
    assert tuple(out.shape) == (frames, B, J, D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("flag", ["action_adaln_bottleneck", "species_film_bottleneck"])
def test_parser_rejects_removed_head_width_options(flag):
    from argparse import ArgumentParser
    from utils.parser_util import add_model_options

    parser = ArgumentParser()
    add_model_options(parser)
    assert flag not in vars(parser.parse_args([]))
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{flag}", "16"])
