"""Diffusion has no dynamic reference condition; AE keeps transfer ownership."""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from data_loaders.tensors import truebones_batch_collate
from sample.conditioning import create_condition
from tests.test_loop_cycles_condition import _model, _y
from tests.test_rest_geom import _plain_dataset
from utils import parser_util


def test_loader_and_generation_emit_only_target_conditions():
    dataset, data = _plain_dataset()
    sample = dataset._prepare_sample('test', data)
    _, loaded = truebones_batch_collate([sample])
    _, generated = create_condition(
        ['test'], dataset.cond_dict, 30, dataset.opt.max_joints, 12)
    assert 'reference' not in sample[12]
    for batch in (loaded, generated):
        assert not any(key.startswith('ref_') for key in batch['y'])


def test_diffusion_forward_needs_no_reference_and_has_no_reference_weights():
    model = _model()
    with torch.no_grad():
        result = model(torch.randn(2, 4, 12, 5), torch.tensor([1, 2]), y=_y(
            2, graph_dist=torch.zeros(2, 4, 4, dtype=torch.long),
            joints_relations=torch.zeros(2, 4, 4, dtype=torch.long)))
    assert result.shape == (2, 4, 12, 5)
    assert torch.isfinite(result).all()
    assert not any(
        'reference' in key or '.ref_' in key or 'unreliable' in key or 'reliability_bias' in key
        for key in model.state_dict()
    )


@pytest.mark.parametrize('options', [
    {}, {'skip_timesteps': 0}, {'skip_timesteps': 5},
])
def test_edit_source_requires_explicit_edit_options(options):
    assert parser_util.reference_requires_edit_options('motion.npy', **options) == (not options)
    assert not parser_util.reference_requires_edit_options(None, **options)


def test_generation_rejects_reference_without_edit_before_model_load(monkeypatch):
    monkeypatch.setattr(parser_util, 'parse_and_load_from_model',
                        lambda parser, **kw: parser.parse_args(kw['argv']))
    with pytest.raises(SystemExit):
        parser_util.generate_args(['--model_path', 'unused.pt', '--reference_motion', 'motion.npy'])
    args = parser_util.generate_args([
        '--model_path', 'unused.pt', '--reference_motion', 'motion.npy', '--skip_timesteps', '5',
        '--action_label', 'walk, forward'])
    assert args.action_label == 'walk, forward'


def test_training_parser_has_no_reference_options():
    parser = ArgumentParser()
    parser_util.add_model_options(parser)
    assert not any(key.startswith('ref_') for key in vars(parser.parse_args([])))


def test_generation_parser_rejects_removed_reference_options():
    parser = ArgumentParser()
    parser_util.add_generate_options(parser)
    assert 'reference_action_group' not in vars(parser.parse_args([]))
    for option, value in (
            ('--reference_action_group', 'locomotion'),
            ('--reference_source_type', 'Horse'),
            ('--ref_cfg_scale', '3')):
        with pytest.raises(SystemExit):
            parser.parse_args([option, value])


@pytest.mark.parametrize("option", [
    "joint_mask_prob", "unreliable_mask_drop_prob", "temporal_span_mask_prob",
    "temporal_span_mask_min_frames", "temporal_span_mask_max_frames",
    "temporal_span_seam_loss_weight", "temporal_span_seam_width", "renoise_same_level_prob",
])
def test_training_rejects_removed_completion_options(option):
    parser = ArgumentParser()
    parser_util.add_training_options(parser)
    assert option not in vars(parser.parse_args(["--cond_path", "unused.npy"]))
    with pytest.raises(SystemExit):
        parser.parse_args(["--cond_path", "unused.npy", f"--{option}", "0"])


@pytest.mark.parametrize("option", ["inpaint_joints", "inpaint_frames", "inpaint_include_subtree", "outpaint"])
def test_generation_rejects_removed_completion_options(option):
    parser = ArgumentParser()
    parser_util.add_generate_options(parser)
    assert option not in vars(parser.parse_args([]))
    with pytest.raises(SystemExit):
        parser.parse_args([f"--{option}"])
