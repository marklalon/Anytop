from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.tensors import truebones_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_VOCAB,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    action_words_in,
    canonical_action_label,
    direction_words_in,
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
        # The output coordinate frame is an unconditional model input: every
        # forward reads it, so a hand-built y has to carry it.
        'canonical_feature_mean': torch.zeros(13, dtype=torch.float32),
        'canonical_feature_std': torch.ones(13, dtype=torch.float32),
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
    def test_vocabulary_is_flat_and_direction_comes_last(self):
        # One flat action vocabulary plus a separate direction axis: no core /
        # detail split survives the removal of the multi-hot.
        self.assertEqual(CONTROLLED_VOCAB, ACTION_VOCAB + DIRECTION_VOCAB)
        self.assertEqual(len(set(CONTROLLED_VOCAB)), len(CONTROLLED_VOCAB))
        self.assertEqual(DIRECTION_VOCAB, ("forward", "backward", "left", "right"))
        # Derived adjectives are deliberately absent -- T5 presses "leftward" and
        # "rightward" to near-synonyms -- and so is the mushy "sideways".
        for absent in ("leftward", "rightward", "sideways", "up", "down"):
            self.assertNotIn(absent, CONTROLLED_VOCAB)

    def test_gait_modifiers_survive_alongside_their_base_word(self):
        # The equal-length rule keeps both, which is the whole point: a Sprint
        # clip and a Run clip must not collapse onto the same label.
        self.assertEqual(vocab_words_in('sprints forward'), ['run', 'sprint', 'forward'])
        self.assertEqual(vocab_words_in('shuffles left'), ['walk', 'shuffle', 'left'])
        self.assertEqual(vocab_words_in('strafe right'), ['walk', 'strafe', 'right'])

    def test_strafe_no_longer_lights_turn(self):
        # Strafing is pure translation and does not change facing.
        self.assertNotIn('turn', vocab_words_in('walk, strafe left with arms swinging'))
        # Circling and banking do change facing, so they stay.
        self.assertIn('turn', vocab_words_in('banks left with beak open'))

    def test_backward_no_longer_lights_retreat(self):
        # A direction must not conjure an action now that direction has its own
        # axis: being knocked backward is not a retreat.
        self.assertEqual(vocab_words_in('knocked backward tumbling'), ['fall', 'backward'])
        # The retreat word itself still works.
        self.assertIn('retreat', vocab_words_in('backs away slowly'))

    def test_bare_back_is_not_a_direction(self):
        # Every corpus use of bare "back" is anatomy or recovery.
        for text in ('collapses onto back', 'stands back up', 'arches back'):
            self.assertNotIn('backward', vocab_words_in(text), text)

    def test_action_and_direction_views_partition_the_hits(self):
        text = 'run, sprint, forward, left'
        self.assertEqual(action_words_in(text), ['run', 'sprint'])
        self.assertEqual(direction_words_in(text), ['forward', 'left'])
        self.assertEqual(
            action_words_in(text) + direction_words_in(text), vocab_words_in(text)
        )

    def test_canonical_label_is_order_and_duplicate_free(self):
        # One word combination must have exactly one spelling, or its training
        # mass splits across several T5 vectors.
        self.assertEqual(
            canonical_action_label(['left', 'forward', 'walk', 'crouch']),
            'walk, crouch, forward, left',
        )
        self.assertEqual(canonical_action_label(['run', 'run']), 'run')
        self.assertEqual(canonical_action_label(['run', 'nonsense']), 'run')
        # Round trip: a canonical label re-parses to itself.
        for label in ('walk, forward', 'run, sprint, forward, left', 'attack, bite'):
            self.assertEqual(canonical_action_label(vocab_words_in(label)), label)

    def test_validator_rejects_prose_and_wrong_order(self):
        from data_loaders.truebones.truebones_utils import motion_labels

        def _validate(label):
            motion_labels._validate_action_label_entry('locomotion', label, 'c.npy', 1)

        # Legal: canonical keywords, and the empty (unconditional) label.
        _validate('walk, forward')
        _validate('')
        # Prose is rejected even though it hits controlled words.
        with self.assertRaises(SystemExit):
            _validate('walk, strides forward with arms swinging')
        # Right words, wrong order.
        with self.assertRaises(SystemExit):
            _validate('forward, walk')
        # Repeats.
        with self.assertRaises(SystemExit):
            _validate('walk, walk')

    def test_validator_rejects_an_unknown_group(self):
        from data_loaders.truebones.truebones_utils import motion_labels

        with self.assertRaises(SystemExit):
            motion_labels._validate_action_label_entry('emote', 'idle', 'c.npy', 1)


class ActionLabelConditioningTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(
            max_joints=4, feature_len=13, latent_dim=8, ff_size=32,
            num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
            t5_out_dim=T5_DIM,
        )
        self.assertFalse(model.action_label_cond)
        self.assertIsNone(model.action_label_projection)
        self.assertIsNone(model.action_label_null_emb)
        # Forward must run and ignore any provided action condition.
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        model.eval()
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        model(x, torch.tensor([1, 2], dtype=torch.int64),
              y=_make_y(**_label_cond(['attack, bite', 'idle'], ['stationary'] * 2)))
        self.assertIsNotNone(capture.last_kwargs)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(action_label_cfg_drop_prob=1.5)

    def test_one_pathway_and_one_injection(self):
        model = _make_model()
        self.assertIsNotNone(model.action_label_projection)
        self.assertIsNotNone(model.action_label_null_emb)
        # No multi-hot pathway survives: nothing in y named 'action_multihot' is
        # read, and the model exposes no vocabulary index layout at all.
        self.assertFalse(hasattr(model, 'action_multihot_projection'))
        self.assertFalse(hasattr(model, 'action_word_to_index'))

    def test_distinct_labels_produce_distinct_tokens(self):
        model = _make_model()
        model.eval()  # no random dropout in eval
        a = model._build_action_label_token(
            _make_y(**_label_cond(['attack, bite', 'idle'], ['stationary'] * 2)), 2,
            torch.device('cpu'), torch.float32)
        b = model._build_action_label_token(
            _make_y(**_label_cond(['idle', 'idle'], ['stationary'] * 2)), 2,
            torch.device('cpu'), torch.float32)
        # Row 0 differs (attack vs idle); row 1 identical (idle vs idle).
        self.assertFalse(torch.allclose(a[0], b[0]))
        self.assertTrue(torch.allclose(a[1], b[1]))

    def test_direction_alone_changes_the_token(self):
        # The point of the refactor: two labels differing only in their direction
        # word must reach the model as different conditions.
        model = _make_model(action_label_cfg_drop_prob=0.0)
        model.eval()
        token = model._build_action_label_token(
            _make_y(**_label_cond(['walk, forward', 'walk, left'], ['locomotion'] * 2)),
            2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], token[1]))

    def test_hard_drop_uses_null_embedding(self):
        model = _make_model()
        model.eval()
        token = model._build_action_label_token(
            _make_y(action_label_active=torch.tensor([True, False]),
                    **_label_cond(['attack, bite', 'idle'], ['stationary'] * 2)),
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
            _make_y(**_label_cond(['attack, bite', 'idle'], ['stationary'] * 2)),
            2, torch.device('cpu'), torch.float32)
        self.assertTrue(torch.allclose(token[0], model.action_label_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))

    def test_missing_action_condition_is_unconditional(self):
        # No action fields in y at all -> every row is the learned null embedding,
        # NOT the untrained all-zero projection output.
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
                    **_label_cond(['attack, bite', ''], ['stationary'] * 2)),
            2, torch.device('cpu'), torch.float32)
        self.assertFalse(torch.allclose(token[0], model.action_label_null_emb))
        self.assertTrue(torch.allclose(token[1], model.action_label_null_emb))

    def test_collate_precomputes_the_label_emb_and_valid_mask(self):
        def _item(label, group, emb):
            return {
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

        _, cond = truebones_collate([
            _item('run, forward', 'locomotion', torch.ones(T5_DIM)),
            _item('', 'stationary', None),
        ])
        self.assertNotIn('action_multihot', cond['y'])
        self.assertEqual(cond['y']['action_label'], ['run, forward', ''])
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

        model(x, ts, y=_make_y(**_label_cond(['attack, bite', 'idle'], ['stationary'] * 2)))
        with_attack = capture.last_kwargs['timesteps_embs'].clone()

        model(x, ts, y=_make_y(**_label_cond(['idle', 'idle'], ['stationary'] * 2)))
        with_idle = capture.last_kwargs['timesteps_embs'].clone()

        # Row 0 token differs between attack and idle -> timestep emb differs.
        self.assertFalse(torch.allclose(with_attack[0], with_idle[0]))


if __name__ == '__main__':
    unittest.main()
