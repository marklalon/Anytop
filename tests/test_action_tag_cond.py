from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.tensors import truebones_collate  # noqa: E402
from model.anytop import AnyTop  # noqa: E402


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_model(action_tag_cond=True, action_tag_cfg_drop_prob=0.3):
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        action_tag_cond=action_tag_cond,
        action_tag_cfg_drop_prob=action_tag_cfg_drop_prob,
    )


def _make_y(action_tags=None, **extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'mask': torch.ones(2, 1, 1, 4, 4, dtype=torch.float32),
        'tpos_first_frame': torch.randn(2, 4, 13, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, 512, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
    }
    if action_tags is not None:
        y['action_tags'] = action_tags
    y.update(extra)
    return y


class ActionTagConditioningTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(
            max_joints=4, feature_len=13, latent_dim=8, ff_size=32,
            num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
        )
        self.assertFalse(model.action_tag_cond)
        self.assertIsNone(model.action_tag_projection)
        # Forward must run and ignore any provided action_tags.
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        model.eval()
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        model(x, torch.tensor([1, 2], dtype=torch.int64),
              y=_make_y(action_tags=[['attack'], ['idle']]))
        self.assertIsNotNone(capture.last_kwargs)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(action_tag_cfg_drop_prob=1.5)

    def test_multihot_encoding(self):
        model = _make_model()
        multihot = model._build_action_tag_multihot(
            [['attack', 'locomotion'], None, 'idle'],
            batch_size=3,
            device=torch.device('cpu'),
            dtype=torch.float32,
        )
        self.assertEqual(multihot.shape, (3, len(model.action_tag_vocab)))
        idx = model.action_tag_to_index
        self.assertEqual(float(multihot[0, idx['attack']]), 1.0)
        self.assertEqual(float(multihot[0, idx['locomotion']]), 1.0)
        self.assertEqual(float(multihot[1].sum()), 0.0)  # None row -> all zero
        self.assertEqual(float(multihot[2, idx['idle']]), 1.0)
        # Unknown vocabulary entries are dropped, not crashed on.
        zero = model._build_action_tag_multihot(
            [['not_a_real_tag']], 1, torch.device('cpu'), torch.float32)
        self.assertEqual(float(zero.sum()), 0.0)

    def test_distinct_tags_produce_distinct_tokens(self):
        model = _make_model()
        model.eval()  # no random dropout in eval
        a = model._build_action_tag_token(
            _make_y(action_tags=[['attack'], ['idle']]), 2,
            torch.device('cpu'), torch.float32)
        b = model._build_action_tag_token(
            _make_y(action_tags=[['idle'], ['idle']]), 2,
            torch.device('cpu'), torch.float32)
        # Row 0 differs (attack vs idle); row 1 identical (idle vs idle).
        self.assertFalse(torch.allclose(a[0], b[0]))
        self.assertTrue(torch.allclose(a[1], b[1]))

    def test_hard_drop_uses_null_embedding(self):
        model = _make_model()
        model.eval()
        token = model._build_action_tag_token(
            _make_y(action_tags=[['attack'], ['idle']],
                    action_tag_active=torch.tensor([True, False])),
            2, torch.device('cpu'), torch.float32)
        # Dropped row == null embedding (zero-init by default).
        self.assertTrue(torch.allclose(token[1], model.action_tag_null_emb))
        # Kept row != null embedding.
        self.assertFalse(torch.allclose(token[0], model.action_tag_null_emb))

    def test_training_dropout_is_stochastic_but_full_drop_is_unconditional(self):
        # drop_prob=1.0 in training => every sample dropped => null embedding.
        model = _make_model(action_tag_cfg_drop_prob=1.0)
        model.train()
        token = model._build_action_tag_token(
            _make_y(action_tags=[['attack'], ['idle']]),
            2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token[0], model.action_tag_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_tag_null_emb))

    def test_missing_action_tags_key_is_unconditional(self):
        # No action_tags in y at all -> every row is the learned null embedding
        # (unconditional), NOT the untrained all-zero multi-hot projection bias.
        model = _make_model()
        model.eval()
        token = model._build_action_tag_token(
            _make_y(), 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token[0], model.action_tag_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_tag_null_emb))

    def test_empty_or_none_row_is_unconditional_even_when_active(self):
        # A None / empty / all-out-of-vocab row routes to null even if it is
        # nominally active (explicit active=True must not resurrect the
        # all-zero multi-hot bias).
        model = _make_model()
        model.eval()
        token = model._build_action_tag_token(
            _make_y(action_tags=[['attack'], None],
                    action_tag_active=torch.tensor([True, True])),
            2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], model.action_tag_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_tag_null_emb))

    def test_prebatched_action_tag_multihot_matches_python_tags_path(self):
        model = _make_model(action_tag_cfg_drop_prob=0.0)
        model.eval()
        y = _make_y(action_tags=[['attack'], ['idle']])
        token_from_python_tags = model._build_action_tag_token(
            y, 2, torch.device('cpu'), torch.float32)
        prebatched_y = dict(y)
        prebatched_y['action_tag_multihot'] = model._build_action_tag_multihot(
            y['action_tags'], 2, torch.device('cpu'), torch.float32)
        token_from_prebatched_tensor = model._build_action_tag_token(
            prebatched_y, 2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token_from_python_tags, token_from_prebatched_tensor))

    def test_collate_precomputes_action_tag_multihot(self):
        _, cond = truebones_collate([
            {
                'inp': torch.zeros(4, 13, 3, dtype=torch.float32),
                'n_joints': 4,
                'temporal_mask': torch.ones(4, 4, dtype=torch.float32),
                'graph_dist': torch.zeros(4, 4, dtype=torch.float32),
                'joints_relations': torch.zeros(4, 4, dtype=torch.float32),
                'joints_names_embs': torch.zeros(4, 512, dtype=torch.float32),
                'tpos_first_frame': torch.zeros(4, 13, dtype=torch.float32),
                'mean': torch.zeros(4, 13, dtype=torch.float32),
                'std': torch.ones(4, 13, dtype=torch.float32),
                'action_tags': ['attack', 'locomotion'],
            },
            {
                'inp': torch.zeros(4, 13, 3, dtype=torch.float32),
                'n_joints': 4,
                'temporal_mask': torch.ones(4, 4, dtype=torch.float32),
                'graph_dist': torch.zeros(4, 4, dtype=torch.float32),
                'joints_relations': torch.zeros(4, 4, dtype=torch.float32),
                'joints_names_embs': torch.zeros(4, 512, dtype=torch.float32),
                'tpos_first_frame': torch.zeros(4, 13, dtype=torch.float32),
                'mean': torch.zeros(4, 13, dtype=torch.float32),
                'std': torch.ones(4, 13, dtype=torch.float32),
                'action_tags': None,
            },
        ])
        multihot = cond['y']['action_tag_multihot']
        model = _make_model()
        idx = model.action_tag_to_index
        self.assertEqual(multihot.shape, (2, len(model.action_tag_vocab)))
        self.assertEqual(float(multihot[0, idx['attack']]), 1.0)
        self.assertEqual(float(multihot[0, idx['locomotion']]), 1.0)
        self.assertEqual(float(multihot[1].sum()), 0.0)

    def test_forward_adds_action_token_to_timestep_embedding(self):
        model = _make_model(action_tag_cfg_drop_prob=0.0)
        model.eval()
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)

        model(x, ts, y=_make_y(action_tags=[['attack'], ['idle']]))
        with_tags = capture.last_kwargs['timesteps_embs'].clone()

        model(x, ts, y=_make_y(action_tags=[['idle'], ['idle']]))
        other_tags = capture.last_kwargs['timesteps_embs'].clone()

        # Row 0 token differs between attack and idle -> timestep emb differs.
        self.assertFalse(torch.allclose(with_tags[0], other_tags[0]))


if __name__ == '__main__':
    unittest.main()
