from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.tensors import truebones_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_VOCAB_CORE,
    GROUP_MULTIHOT_MASK,
    action_multihot_words,
    coarse_label_from_words,
    group_multihot_mask,
    vocab_words_in,
)
from model.anytop import AnyTop  # noqa: E402


T5_DIM = 512


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_model(action_label_cond=True, action_label_cfg_drop_prob=0.3):
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        action_label_cond=action_label_cond,
        action_label_cfg_drop_prob=action_label_cfg_drop_prob,
    )


def _make_y(**extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'mask': torch.ones(2, 1, 1, 4, 4, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 13, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
    }
    y.update(extra)
    return y


def _label_cond(labels, groups, seed=0):
    """A conditioned y: label text, group, a deterministic stand-in T5 emb, valid mask."""
    generator = torch.Generator().manual_seed(seed)
    embs = torch.zeros(len(labels), T5_DIM)
    valid = torch.zeros(len(labels), dtype=torch.bool)
    for i, label in enumerate(labels):
        if not label:
            continue
        # Deterministic per label text, so identical labels get identical vectors.
        embs[i] = torch.rand(
            T5_DIM, generator=torch.Generator().manual_seed(abs(hash(label)) % (2 ** 31))
        )
        valid[i] = True
    del generator
    return {
        'action_label': list(labels),
        'action_group': list(groups),
        'action_label_emb': embs,
        'action_label_valid': valid,
    }


class ActionLabelVocabularyTest(unittest.TestCase):
    def test_group_mask_is_frozen_and_well_formed(self):
        for group, mask in GROUP_MULTIHOT_MASK.items():
            self.assertEqual(len(mask), len(ACTION_VOCAB_CORE), group)
            self.assertTrue(set(mask) <= {0, 1}, group)
        # Spot-check the two documented cases: 'walk' is a locomotion slot but is
        # held at zero in stationary; 'die' is a transition slot only.
        walk = ACTION_VOCAB_CORE.index('walk')
        die = ACTION_VOCAB_CORE.index('die')
        self.assertEqual(GROUP_MULTIHOT_MASK['locomotion'][walk], 1)
        self.assertEqual(GROUP_MULTIHOT_MASK['stationary'][walk], 0)
        self.assertEqual(GROUP_MULTIHOT_MASK['transition'][die], 1)
        self.assertEqual(GROUP_MULTIHOT_MASK['locomotion'][die], 0)

    def test_unknown_group_is_rejected(self):
        with self.assertRaises(ValueError):
            group_multihot_mask('emote')

    def test_multihot_words_are_masked_per_group(self):
        # 'roar' is a stationary slot but is masked out in locomotion.
        self.assertIn('roar', action_multihot_words('idle, growls', 'stationary'))
        self.assertNotIn('roar', action_multihot_words('runs and growls', 'locomotion'))
        # Unmasked (group=None) keeps everything the text hits.
        self.assertIn('roar', action_multihot_words('runs and growls'))

    def test_coarse_synthesis_ignores_the_group_mask(self):
        # The synthesized string is T5 text, not a multi-hot: a word the group
        # masks out must still appear, or it never learns a short-query response.
        label = 'runs forward and growls'
        coarse = coarse_label_from_words(vocab_words_in(label))
        self.assertIn('roar', coarse)
        self.assertNotIn('roar', action_multihot_words(label, 'locomotion'))

    def test_detail_only_label_falls_back_to_detail_words(self):
        # Returning '' here would hand the augmentation the null condition, so
        # these actions would train as "unconditional" and never be queryable.
        coarse = coarse_label_from_words(vocab_words_in('sneaks forward low to the ground'))
        self.assertTrue(coarse)
        self.assertIn('sneak', coarse)


class ActionLabelConditioningTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(
            max_joints=4, feature_len=13, latent_dim=8, ff_size=32,
            num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
            t5_out_dim=T5_DIM,
        )
        self.assertFalse(model.action_label_cond)
        self.assertIsNone(model.action_label_projection)
        self.assertIsNone(model.action_multihot_projection)
        # Forward must run and ignore any provided action condition.
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        model.eval()
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        model(x, torch.tensor([1, 2], dtype=torch.int64),
              y=_make_y(**_label_cond(['attack, bites', 'idle'], ['stationary'] * 2)))
        self.assertIsNotNone(capture.last_kwargs)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(action_label_cfg_drop_prob=1.5)

    def test_multihot_derived_from_label_and_group(self):
        model = _make_model()
        multihot = model._build_action_multihot(
            {'action_label': ['attack, lunges and bites', '', 'idle'],
             'action_group': ['stationary', 'stationary', 'stationary']},
            batch_size=3,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.assertEqual(multihot.shape, (3, len(ACTION_VOCAB_CORE)))
        idx = model.action_word_to_index
        self.assertEqual(float(multihot[0, idx['attack']]), 1.0)
        self.assertEqual(float(multihot[0, idx['bite']]), 1.0)
        self.assertEqual(float(multihot[1].sum()), 0.0)  # empty label -> all zero
        self.assertEqual(float(multihot[2, idx['idle']]), 1.0)

    def test_multihot_respects_the_group_mask(self):
        model = _make_model()
        idx = model.action_word_to_index
        stationary = model._build_action_multihot(
            {'action_label': ['idle, growls'], 'action_group': ['stationary']},
            1, torch.device('cpu'), torch.float32)
        locomotion = model._build_action_multihot(
            {'action_label': ['runs and growls'], 'action_group': ['locomotion']},
            1, torch.device('cpu'), torch.float32)
        self.assertEqual(float(stationary[0, idx['roar']]), 1.0)
        # 'roar' is downgraded in locomotion, so its column stays zero even though
        # the label names it.
        self.assertEqual(float(locomotion[0, idx['roar']]), 0.0)
        self.assertEqual(float(locomotion[0, idx['run']]), 1.0)

    def test_distinct_labels_produce_distinct_tokens(self):
        model = _make_model()
        model.eval()  # no random dropout in eval
        a = model._build_action_label_token(
            _make_y(**_label_cond(['attack, bites', 'idle'], ['stationary'] * 2)), 2,
            torch.device('cpu'), torch.float32)
        b = model._build_action_label_token(
            _make_y(**_label_cond(['idle', 'idle'], ['stationary'] * 2)), 2,
            torch.device('cpu'), torch.float32)
        # Row 0 differs (attack vs idle); row 1 identical (idle vs idle).
        self.assertFalse(torch.allclose(a[0], b[0]))
        self.assertTrue(torch.allclose(a[1], b[1]))

    def test_hard_drop_uses_null_embedding(self):
        model = _make_model()
        model.eval()
        token = model._build_action_label_token(
            _make_y(action_label_active=torch.tensor([True, False]),
                    **_label_cond(['attack, bites', 'idle'], ['stationary'] * 2)),
            2, torch.device('cpu'), torch.float32)
        # Dropped row == null embedding (zero-init by default).
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))
        # Kept row != null embedding.
        self.assertFalse(torch.allclose(token[0], model.action_label_null_emb))

    def test_training_dropout_is_stochastic_but_full_drop_is_unconditional(self):
        # drop_prob=1.0 in training => every sample dropped => null embedding.
        model = _make_model(action_label_cfg_drop_prob=1.0)
        model.train()
        token = model._build_action_label_token(
            _make_y(**_label_cond(['attack, bites', 'idle'], ['stationary'] * 2)),
            2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token[0], model.action_label_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))

    def test_missing_action_condition_is_unconditional(self):
        # No action fields in y at all -> every row is the learned null embedding,
        # NOT the untrained all-zero projection bias.
        model = _make_model()
        model.eval()
        token = model._build_action_label_token(
            _make_y(), 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token[0], model.action_label_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))

    def test_empty_label_is_unconditional_even_when_active(self):
        # An empty label carries no condition and must route to null even when it
        # is nominally active -- encoding '' through T5 would teach the model that
        # empty text means "any motion".
        model = _make_model()
        model.eval()
        token = model._build_action_label_token(
            _make_y(action_label_active=torch.tensor([True, True]),
                    **_label_cond(['attack, bites', ''], ['stationary'] * 2)),
            2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], model.action_label_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))

    def test_all_zero_multihot_label_is_still_conditioned(self):
        # A label whose only core words are masked out in this group keeps its T5
        # condition: the all-zero multi-hot is the projection's bias, a defined
        # state, and must NOT collapse to the hard-dropped null.
        model = _make_model(action_label_cfg_drop_prob=0.0)
        model.eval()
        y = _make_y(**_label_cond(['runs and growls', 'runs and growls'],
                                  ['locomotion', 'locomotion']))
        multihot = model._build_action_multihot(y, 2, torch.device('cpu'), torch.float32)
        self.assertEqual(float(multihot[0].sum()), 1.0)  # only 'run' survives the mask
        token = model._build_action_label_token(y, 2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], model.action_label_null_emb))

    def test_prebatched_multihot_matches_the_text_derived_path(self):
        model = _make_model(action_label_cfg_drop_prob=0.0)
        model.eval()
        y = _make_y(**_label_cond(['attack, bites', 'idle'], ['stationary'] * 2))
        token_from_text = model._build_action_label_token(
            y, 2, torch.device('cpu'), torch.float32)
        prebatched_y = dict(y)
        prebatched_y['action_multihot'] = model._build_action_multihot(
            y, 2, torch.device('cpu'), torch.float32)
        token_from_prebatched = model._build_action_label_token(
            prebatched_y, 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token_from_text, token_from_prebatched))

    def test_collate_precomputes_masked_multihot_and_label_emb(self):
        def _item(label, group, emb):
            item = {
                'inp': torch.zeros(4, 13, 3, dtype=torch.float32),
                'n_joints': 4,
                'temporal_mask': torch.ones(4, 4, dtype=torch.float32),
                'graph_dist': torch.zeros(4, 4, dtype=torch.float32),
                'joints_relations': torch.zeros(4, 4, dtype=torch.float32),
                'joints_names_embs': torch.zeros(4, T5_DIM, dtype=torch.float32),
                'rest_pose': torch.zeros(4, 13, dtype=torch.float32),
                'mean': torch.zeros(4, 13, dtype=torch.float32),
                'std': torch.ones(4, 13, dtype=torch.float32),
                'action_label': label,
                'action_group': group,
                'action_label_emb': emb,
            }
            return item

        _, cond = truebones_collate([
            _item('runs and growls', 'locomotion', torch.ones(T5_DIM)),
            _item('', 'stationary', None),
        ])
        multihot = cond['y']['action_multihot']
        idx = {word: i for i, word in enumerate(ACTION_VOCAB_CORE)}
        self.assertEqual(multihot.shape, (2, len(ACTION_VOCAB_CORE)))
        self.assertEqual(float(multihot[0, idx['run']]), 1.0)
        # 'roar' is masked out in locomotion, so the collate must not light it up.
        self.assertEqual(float(multihot[0, idx['roar']]), 0.0)
        self.assertEqual(float(multihot[1].sum()), 0.0)
        self.assertEqual(cond['y']['action_label_emb'].shape, (2, T5_DIM))
        self.assertTrue(bool(cond['y']['action_label_valid'][0]))
        self.assertFalse(bool(cond['y']['action_label_valid'][1]))

    def test_forward_adds_action_token_to_timestep_embedding(self):
        model = _make_model(action_label_cfg_drop_prob=0.0)
        model.eval()
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)

        model(x, ts, y=_make_y(**_label_cond(['attack, bites', 'idle'], ['stationary'] * 2)))
        with_attack = capture.last_kwargs['timesteps_embs'].clone()

        model(x, ts, y=_make_y(**_label_cond(['idle', 'idle'], ['stationary'] * 2)))
        with_idle = capture.last_kwargs['timesteps_embs'].clone()

        # Row 0 token differs between attack and idle -> timestep emb differs.
        self.assertFalse(torch.allclose(with_attack[0], with_idle[0]))


if __name__ == '__main__':
    unittest.main()
