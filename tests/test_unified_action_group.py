"""One model over every action group (docs/unified_action_group_training.md).

Covers the pieces an ``--action_group all`` run adds: the group sampling weights,
the weighted sampler, the collate's group ids, the model's group token and its
independent CFG drop, generation's condition resolution, and the eval harness's
group skip.
"""
from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.tensors import truebones_collate  # noqa: E402
from data_loaders.truebones.data.dataset import (  # noqa: E402
    TruebonesSampler,
    resolve_action_group_weights,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.truebones.truebones_utils.motion_labels import ACTION_GROUPS  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.cfg_sampler import ClassifierFreeActionModel  # noqa: E402
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
    sample_action_slots,
)


class ActionGroupWeightsTest(unittest.TestCase):

    def test_all_defaults_to_equal_weights(self):
        for raw in (None, ''):
            with self.subTest(raw=raw):
                self.assertEqual(
                    resolve_action_group_weights(raw, 'all'),
                    {group: 1.0 for group in ACTION_GROUPS},
                )

    def test_values_follow_the_group_order(self):
        self.assertEqual(
            resolve_action_group_weights('2, 1,0.5', 'all'),
            dict(zip(ACTION_GROUPS, (2.0, 1.0, 0.5))),
        )

    def test_single_group_run_has_no_weights_and_refuses_them(self):
        self.assertIsNone(resolve_action_group_weights(None, 'locomotion'))
        with self.assertRaises(ValueError):
            resolve_action_group_weights('1,1,1', 'locomotion')

    def test_malformed_weights_are_refused(self):
        for raw in ('1,1', '1,1,1,1', '1,x,1', '1,-1,1', '1,nan,1', '0,0,0'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    resolve_action_group_weights(raw, 'all')

    def test_training_validates_before_touching_save_dir(self):
        from train.train_anytop import validate_action_group_options
        self.assertEqual(
            validate_action_group_options(Namespace(action_group='all', action_group_cond=True,
                                                    action_group_weights='1,2,1')),
            dict(zip(ACTION_GROUPS, (1.0, 2.0, 1.0))),
        )
        with self.assertRaises(SystemExit):
            validate_action_group_options(Namespace(action_group='locomotion',
                                                    action_group_cond=True,
                                                    action_group_weights=None))
        with self.assertRaises(SystemExit):
            validate_action_group_options(Namespace(action_group='locomotion',
                                                    action_group_cond=False,
                                                    action_group_weights='1,1,1'))


def _fake_motion_dataset(clips, balanced, group_weights, pointer=0):
    """``clips``: (object_type, group) per name_list entry."""
    name_list = [f'clip{i}' for i in range(len(clips))]
    data_dict = {
        name: {'object_type': object_type, 'motion_metadata': {'action_group': group}}
        for name, (object_type, group) in zip(name_list, clips)
    }
    cond_dict = {object_type: {} for object_type, _ in clips}
    return types.SimpleNamespace(
        name_list=name_list, data_dict=data_dict, cond_dict=cond_dict,
        pointer=pointer, balanced=balanced, action_group_weights=group_weights,
    )


class GroupWeightedSamplerTest(unittest.TestCase):
    # 6 locomotion clips (one species), 3 stationary (A:2, B:1), 1 transition.
    CLIPS = (
        [('A', 'locomotion')] * 6
        + [('A', 'stationary')] * 2 + [('B', 'stationary')]
        + [('B', 'transition')]
    )

    @staticmethod
    def _group_mass(weights, clips):
        mass = {}
        for weight, (_, group) in zip(weights, clips):
            mass[group] = mass.get(group, 0.0) + weight
        return mass

    def test_equal_weights_give_every_group_the_same_mass(self):
        weights = TruebonesSampler.compute_weights(
            _fake_motion_dataset(self.CLIPS, False, resolve_action_group_weights(None, 'all')))
        mass = self._group_mass(weights, self.CLIPS)
        for group in ACTION_GROUPS:
            self.assertAlmostEqual(mass[group], 1.0 / 3.0)
        # Uniform per clip inside a group.
        self.assertTrue(np.allclose(weights[:6], 1.0 / 18.0))
        self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_weights_are_relative_and_zero_leaves_a_group_out(self):
        weights = TruebonesSampler.compute_weights(
            _fake_motion_dataset(self.CLIPS, False, resolve_action_group_weights('2,1,0', 'all')))
        mass = self._group_mass(weights, self.CLIPS)
        self.assertAlmostEqual(mass['locomotion'], 2.0 / 3.0)
        self.assertAlmostEqual(mass['stationary'], 1.0 / 3.0)
        self.assertEqual(mass['transition'], 0.0)

    def test_an_empty_group_does_not_absorb_mass(self):
        clips = [c for c in self.CLIPS if c[1] != 'transition']
        weights = TruebonesSampler.compute_weights(
            _fake_motion_dataset(clips, False, resolve_action_group_weights(None, 'all')))
        mass = self._group_mass(weights, clips)
        self.assertAlmostEqual(mass['locomotion'], 0.5)
        self.assertAlmostEqual(mass['stationary'], 0.5)

    def test_balanced_splits_each_group_by_sqrt_species(self):
        weights = TruebonesSampler.compute_weights(
            _fake_motion_dataset(self.CLIPS, True, resolve_action_group_weights(None, 'all')))
        # stationary: A has 2 clips, B 1 -> species shares sqrt(2) : 1.
        share_a = np.sqrt(2.0) / (np.sqrt(2.0) + 1.0) / 3.0
        self.assertAlmostEqual(float(weights[6:8].sum()), share_a)
        self.assertAlmostEqual(float(weights[8]), 1.0 / 3.0 - share_a)

    def test_balanced_without_groups_matches_the_species_sampler(self):
        weights = TruebonesSampler.compute_weights(_fake_motion_dataset(self.CLIPS, True, None))
        # A: 8 clips, B: 2 clips over the whole subset.
        total = np.sqrt(8.0) + np.sqrt(2.0)
        self.assertTrue(np.allclose(weights[:6], (np.sqrt(8.0) / total) / 8.0))
        self.assertTrue(np.allclose(weights[8:], (np.sqrt(2.0) / total) / 2.0))

    def test_clips_below_the_pointer_are_never_drawn(self):
        weights = TruebonesSampler.compute_weights(
            _fake_motion_dataset(self.CLIPS, False, resolve_action_group_weights(None, 'all'),
                                 pointer=2))
        self.assertTrue(np.all(weights[:2] == 0.0))
        self.assertAlmostEqual(float(weights[2:6].sum()), 1.0 / 3.0)


def _collate_item(label, group):
    return {
        'inp': torch.zeros(4, 12, 3, dtype=torch.float32),
        'n_joints': 4,
        'graph_dist': torch.zeros(4, 4, dtype=torch.float32),
        'joints_relations': torch.zeros(4, 4, dtype=torch.float32),
        'joints_names_embs': torch.zeros(4, TEST_T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'rest_pose': torch.zeros(4, 12, dtype=torch.float32),
        'action_label': label,
        'action_group': group,
        'action_slots': sample_action_slots(label, group) if group else None,
    }


class CollateGroupIdTest(unittest.TestCase):

    def test_group_ids_follow_action_groups_and_missing_is_minus_one(self):
        _, cond = truebones_collate([
            _collate_item('run, forward', 'locomotion'),
            _collate_item('die', 'transition'),
            _collate_item('', None),
        ])
        self.assertEqual(cond['y']['action_group_id'].dtype, torch.long)
        self.assertEqual(
            cond['y']['action_group_id'].tolist(),
            [ACTION_GROUPS.index('locomotion'), ACTION_GROUPS.index('transition'), -1],
        )

    def test_an_unknown_group_is_an_error(self):
        with self.assertRaises(ValueError):
            truebones_collate([_collate_item('', 'dance')])


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt'] + kwargs['timesteps_embs'].sum()


def _make_model(action_group_cond=True, group_drop=0.15, label_drop=0.3):
    return AnyTop(
        max_joints=4, feature_len=12, latent_dim=TEST_LATENT_DIM, ff_size=32,
        num_layers=1, num_heads=2, dropout=0.0, cross_limb=True, t5_out_dim=TEST_T5_DIM,
        action_label_cond=True, action_label_cfg_drop_prob=label_drop,
        action_conditioning=make_test_bundle(),
        action_group_cond=action_group_cond, action_group_cfg_drop_prob=group_drop,
    )


def _make_eval_model(**kwargs):
    # AnyTop.train() returns None, so .eval() cannot be chained.
    model = _make_model(**kwargs)
    model.eval()
    return model


def _make_y(**extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 12, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, TEST_T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    y.update(extra)
    return y


def _ids(*groups):
    return torch.tensor([ACTION_GROUPS.index(g) if g else -1 for g in groups], dtype=torch.long)


class ActionGroupTokenTest(unittest.TestCase):
    CPU = torch.device('cpu')

    def _randomize(self, model):
        with torch.no_grad():
            model.action_group_embedding.weight.normal_()

    def test_off_by_default_adds_no_parameters(self):
        model = _make_model(action_group_cond=False)
        self.assertIsNone(model.action_group_embedding)
        self.assertFalse(any('action_group' in name for name, _ in model.named_parameters()))
        self.assertIsNone(model._build_action_group_token(
            _make_y(action_group_id=_ids('locomotion', 'stationary')), 2, self.CPU, torch.float32))

    def test_zero_init_starts_as_the_null_condition(self):
        model = _make_eval_model()
        token = model._build_action_group_token(
            _make_y(action_group_id=_ids('locomotion', 'transition')), 2, self.CPU, torch.float32)
        self.assertTrue(torch.equal(token, torch.zeros_like(token)))
        self.assertEqual(model.action_group_embedding.num_embeddings, len(ACTION_GROUPS) + 1)

    def test_groups_select_their_rows_and_missing_reads_null(self):
        model = _make_eval_model()
        self._randomize(model)
        table = model.action_group_embedding.weight
        null_row = table[len(ACTION_GROUPS)]
        token = model._build_action_group_token(
            _make_y(action_group_id=_ids('stationary', '')), 2, self.CPU, torch.float32)
        self.assertTrue(torch.equal(token[0], table[ACTION_GROUPS.index('stationary')]))
        self.assertTrue(torch.equal(token[1], null_row))
        # No ids at all: every row is null.
        token = model._build_action_group_token(_make_y(), 2, self.CPU, torch.float32)
        self.assertTrue(torch.equal(token, null_row.expand(2, -1)))

    def test_explicit_inactive_and_full_training_drop_read_null(self):
        model = _make_model(group_drop=1.0)
        self._randomize(model)
        null_row = model.action_group_embedding.weight[len(ACTION_GROUPS)]
        y = _make_y(action_group_id=_ids('locomotion', 'stationary'))
        model.train()
        token = model._build_action_group_token(y, 2, self.CPU, torch.float32)
        self.assertTrue(torch.equal(token, null_row.expand(2, -1)))
        model.eval()
        token = model._build_action_group_token(
            dict(y, action_group_active=torch.tensor([False, True])), 2, self.CPU, torch.float32)
        self.assertTrue(torch.equal(token[0], null_row))
        self.assertFalse(torch.equal(token[1], null_row))

    def test_out_of_range_ids_are_refused(self):
        model = _make_eval_model()
        for bad in ([-2, 0], [0, len(ACTION_GROUPS)]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    model._build_action_group_token(
                        _make_y(action_group_id=torch.tensor(bad)), 2, self.CPU, torch.float32)

    def test_forward_adds_the_group_token_to_the_timestep_embedding(self):
        model = _make_eval_model()
        self._randomize(model)
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x = torch.randn(2, 4, 12, 3)
        ts = torch.tensor([1, 2])
        labels = action_cond_fields(['idle', 'idle'], ['stationary', 'stationary'])
        model(x, ts, y=_make_y(action_group_id=_ids('stationary', 'stationary'), **labels))
        stationary = capture.last_kwargs['timesteps_embs'].clone()
        model(x, ts, y=_make_y(action_group_id=_ids('transition', 'stationary'), **labels))
        mixed = capture.last_kwargs['timesteps_embs'].clone()
        self.assertFalse(torch.allclose(stationary[0], mixed[0]))
        self.assertTrue(torch.allclose(stationary[1], mixed[1]))

    def test_label_cfg_keeps_the_group_in_its_unconditional_pass(self):
        model = _make_eval_model()
        self._randomize(model)
        seen = []

        def record(module, args, kwargs):
            seen.append(module._build_action_group_token(kwargs['y'], 2, self.CPU, torch.float32))

        model.register_forward_pre_hook(record, with_kwargs=True)
        model.seqTransDecoder = _CaptureDecoder()
        y = _make_y(action_group_id=_ids('transition', 'transition'),
                    **action_cond_fields(['die', 'getup'], ['transition', 'transition']))
        ClassifierFreeActionModel(model, 2.0)(torch.randn(2, 4, 12, 3), torch.tensor([1, 2]), y=y)
        self.assertEqual(len(seen), 2)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertFalse(torch.equal(seen[0], model.action_group_embedding.weight[-1].expand(2, -1)))


class GenerationConditionTest(unittest.TestCase):

    @staticmethod
    def _resolve(label='', group='', checkpoint_group='', group_cond=False, label_cond=True):
        from sample.generate import _resolve_action_condition
        args = Namespace(action_label=label, action_group=group,
                         checkpoint_action_group=checkpoint_group)
        model = types.SimpleNamespace(action_group_cond=group_cond, action_label_cond=label_cond)
        return _resolve_action_condition(args, model)

    def test_group_only_request_on_a_group_conditioned_model(self):
        condition = self._resolve(group='stationary', checkpoint_group='all', group_cond=True)
        self.assertEqual(condition,
                         {'action_group': 'stationary', 'action_label': '', 'action_slots': None})

    def test_no_label_on_a_model_without_group_cond_is_unconditional(self):
        self.assertIsNone(self._resolve(group='locomotion', checkpoint_group='locomotion'))
        self.assertIsNone(self._resolve(group='stationary', checkpoint_group='all'))
        self.assertIsNone(self._resolve(checkpoint_group='all', group_cond=True))

    def test_label_on_an_all_checkpoint_needs_a_group(self):
        with self.assertRaises(SystemExit):
            self._resolve(label='die', checkpoint_group='all', group_cond=True)

    def test_label_uses_the_requested_group_for_its_roles(self):
        condition = self._resolve(label='idle, crouch', group='transition',
                                  checkpoint_group='all', group_cond=True)
        expected = sample_action_slots('idle, crouch', 'transition')
        self.assertEqual(condition['action_group'], 'transition')
        self.assertEqual(condition['action_slots']['role_ids'].tolist(),
                         expected['role_ids'].tolist())
        # The same words in stationary carry no ordered role.
        stationary = self._resolve(label='idle, crouch', group='stationary',
                                   checkpoint_group='all', group_cond=True)
        self.assertNotEqual(stationary['action_slots']['role_ids'].tolist(),
                            expected['role_ids'].tolist())

    def test_guidance_needs_a_label_even_with_a_group_condition(self):
        from sample.generate import _wrap_action_label_cfg
        args = Namespace(action_label_cfg_scale=2.0, action_label_cfg_drop_prob=0.3)
        with self.assertRaises(SystemExit):
            _wrap_action_label_cfg(torch.nn.Identity(), args,
                                   {'action_group': 'stationary', 'action_label': '',
                                    'action_slots': None})


class EvalGroupSkipTest(unittest.TestCase):

    def test_single_group_checkpoint_skips_other_groups_only(self):
        from eval.eval_checkpoint import _task_group_mismatch
        stationary_task = ['--object_type', 'Buffalo', '--action_group', 'stationary']
        self.assertIsNotNone(_task_group_mismatch('locomotion', stationary_task))
        self.assertIsNone(_task_group_mismatch('stationary', stationary_task))
        self.assertIsNone(_task_group_mismatch('all', stationary_task))
        self.assertIsNone(_task_group_mismatch('locomotion', ['--object_type', 'Buffalo']))

    def test_task_action_words_pick_the_scorer_prior(self):
        from eval.eval_checkpoint import _SCORE_ACTION_WORDS, _score_action_words
        self.assertEqual(_score_action_words(['--action_words', 'die']), 'die')
        self.assertEqual(_score_action_words(['--object_type', 'Buffalo']), _SCORE_ACTION_WORDS)


if __name__ == '__main__':
    unittest.main()
