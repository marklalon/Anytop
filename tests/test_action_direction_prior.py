from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.action_direction_prior import (
    build_direction_prior, complete_direction, load_direction_prior,
    prior_from_dataset, save_direction_prior,
)
from sample.conditioning import _resolve_action_condition


def prior_for(*groups):
    labels = [label for label, count in groups for _ in range(count)]
    return build_direction_prior((str(i), label) for i, label in enumerate(labels))


def test_coverage_boundary_and_no_minimum_sample_count():
    prior = prior_for(('run, forward', 4), ('run', 1), ('climb, up', 1))
    assert complete_direction('run', prior)[0] == 'run, forward'
    assert complete_direction('climb', prior)[0] == 'climb, up'
    below = prior_for(('run, forward', 3), ('run', 1))
    assert complete_direction('run', below) == ('run', None)


def test_complete_head_combination_and_modifiers():
    prior = prior_for(('roll, forward', 10), ('roll, backward, hover', 2),
                      ('roll, jump, backward', 16))
    assert prior['by_heads']['roll']['backward'] == 2
    assert complete_direction('roll', prior)[0] == 'roll, forward'
    assert complete_direction('roll, hover', prior)[0] == 'roll, forward, hover'
    assert complete_direction('roll, jump', prior)[0] == 'roll, jump, backward'
    assert complete_direction('jump, roll', prior) == ('jump, roll', None)


def test_near_top_candidates_use_relative_counts_and_include_boundary():
    prior = prior_for(('turn, left', 45), ('turn, right', 40), ('turn, forward', 15))
    _, decision = complete_direction('turn', prior)
    assert decision['candidates'] == pytest.approx({'left': 45 / 85, 'right': 40 / 85})
    draws = [complete_direction('turn', prior, seed)[0] for seed in range(1000)]
    assert set(draws) == {'turn, left', 'turn, right'}
    assert abs(draws.count('turn, left') / len(draws) - 45 / 85) < 0.05
    boundary = prior_for(('turn, left', 55), ('turn, right', 45))
    assert len(complete_direction('turn', boundary)[1]['candidates']) == 2


def test_direction_combinations_remain_intact_and_seed_does_not_touch_global_rng():
    prior = prior_for(('run, forward, left', 1), ('run, backward, right', 1))
    np.random.seed(42)
    state = np.random.get_state()
    result = complete_direction('run', prior, 123)
    assert result == complete_direction('run', prior, 123)
    assert result[0] in ('run, forward, left', 'run, backward, right')
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    assert complete_direction('run, up', prior) == ('run, up', None)


def test_training_snapshot_uses_only_selected_clips(tmp_path):
    dataset = SimpleNamespace(motion_dataset=SimpleNamespace(
        name_list=['kept'], data_dict={
            'kept': {'motion_metadata': {'action_label': 'run, forward'}},
            'excluded': {'motion_metadata': {'action_label': 'run, backward'}},
        }))
    prior = prior_from_dataset(dataset)
    assert prior['by_heads'] == {'run': {'forward': 1}}
    save_direction_prior(prior, tmp_path)
    assert load_direction_prior(tmp_path / 'model.pt') == prior
    assert prior['source_fingerprint'] == prior_from_dataset(dataset)['source_fingerprint']


def test_legacy_selection_and_missing_statistics(tmp_path):
    model_path = tmp_path / 'model.pt'
    assert load_direction_prior(model_path) is None
    (tmp_path / 'sample_selection.json').write_text(json.dumps({'clips': [
        {'clip_id': 'a', 'action_label': 'run, forward'},
    ]}), encoding='utf-8')
    assert complete_direction('run', load_direction_prior(model_path))[0] == 'run, forward'


def test_inference_slots_follow_completed_label_and_off_preserves_prompt(tmp_path):
    save_direction_prior(prior_for(('run, forward', 1)), tmp_path)
    args = SimpleNamespace(action_label='run', model_path=str(tmp_path / 'model.pt'), seed=5)
    completed = _resolve_action_condition(args)
    assert completed['action_label'] == 'run, forward'
    assert completed['action_slots']['word_mask'].sum() == 2
    args.auto_direction = 'off'
    original = _resolve_action_condition(args)
    assert original['action_label'] == 'run'
    assert original['action_slots']['word_mask'].sum() == 1
    args.auto_direction = 'auto'
    args.action_label = 'run, backward'
    assert _resolve_action_condition(args)['action_label'] == 'run, backward'
    args.action_label = ''
    assert _resolve_action_condition(args) is None


def test_backfill_tool_uses_saved_selection(tmp_path):
    from tools.build_action_direction_prior import main
    (tmp_path / 'sample_selection.json').write_text(json.dumps({'clips': [
        {'clip_id': 'a', 'action_label': 'climb, up'},
    ]}), encoding='utf-8')
    main(['--checkpoint-dir', str(tmp_path)])
    assert load_direction_prior(tmp_path / 'model.pt')['by_heads'] == {'climb': {'up': 1}}


def test_auto_direction_is_a_generation_override(tmp_path):
    from utils.parser_util import generate_args, CKPT_VERSION, joint_condition_schema_versions
    (tmp_path / 'args.json').write_text(json.dumps({
        'version': CKPT_VERSION, **joint_condition_schema_versions(),
        'auto_direction': 'off',
        'direction_slot_drop_prob': 0.15, 'modifier_slot_drop_prob': 0.15,
    }), encoding='utf-8')
    argv = ['--model_path', str(tmp_path / 'model.pt')]
    assert generate_args(argv).auto_direction == 'auto'
    assert generate_args(argv + ['--auto_direction', 'off']).auto_direction == 'off'


def test_missing_prior_does_not_break_legacy_inference(tmp_path):
    args = SimpleNamespace(action_label='run', model_path=str(tmp_path / 'model.pt'))
    assert _resolve_action_condition(args)['action_label'] == 'run'
