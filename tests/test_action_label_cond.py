from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.tensors import truebones_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    SLOT_PAD_ID,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    ACTION_VOCAB,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    STATE_VOCAB,
    ActionLabelError,
    action_words_in,
    canonical_action_label,
    direction_words_in,
    head_words_in,
    parse_action_label,
    vocab_t5_text,
    vocab_words_in,
)
from model.anytop import AnyTop  # noqa: E402
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
    reference_channels,
    sample_action_slots,
)


T5_DIM = TEST_T5_DIM


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_model(action_label_cond=True, action_label_cfg_drop_prob=0.3, bundle=None):
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=TEST_LATENT_DIM,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        action_label_cond=action_label_cond,
        action_label_cfg_drop_prob=action_label_cfg_drop_prob,
        action_conditioning=(
            (bundle or make_test_bundle()) if action_label_cond else None
        ),
    )


def _make_y(**extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
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


def _label_cond(labels, groups):
    """A conditioned y: label text, group, the collate's word ids and valid mask."""
    return action_cond_fields(labels, groups)


class ActionLabelVocabularyTest(unittest.TestCase):
    def test_vocabulary_is_flat_and_direction_comes_last(self):
        # One flat action vocabulary plus a separate direction axis: no core /
        # detail split survives the removal of the multi-hot.
        self.assertEqual(CONTROLLED_VOCAB, ACTION_VOCAB + DIRECTION_VOCAB)
        self.assertEqual(len(set(CONTROLLED_VOCAB)), len(CONTROLLED_VOCAB))
        self.assertEqual(DIRECTION_VOCAB, ("forward", "backward", "left", "right", "up", "down"))
        # Derived adjectives are deliberately absent -- T5 presses "leftward" and
        # "rightward" to near-synonyms -- and so is the mushy "sideways".
        for absent in ("leftward", "rightward", "sideways"):
            self.assertNotIn(absent, CONTROLLED_VOCAB)
        # A token is an ID, so it never carries whitespace: multi-word text lives
        # on the T5 side only.
        for word in CONTROLLED_VOCAB:
            self.assertEqual(word, word.strip())
            self.assertNotIn(' ', word)

    def test_zero_use_words_are_gone_from_the_closed_table(self):
        # The vocabulary is closed now: a word nobody annotates is a token the
        # autocomplete would offer, the model never trained, and whose weight
        # scalar would sit at its prior forever.
        for absent in ('climb', 'gallop', 'shuffle', 'sneak', 'flap', 'stand',
                       'stretch', 'dig', 'peck', 'drag', 'drink', 'graze',
                       'haste'):
            self.assertNotIn(absent, CONTROLLED_VOCAB, absent)
        # ...and the words the corpus actually uses are all in.
        for present in ('weapon', 'cast', 'projectile', 'swat', 'spawn', '2hand',
                        'spin', 'headbutt', 'hover', 'work', 'dead', 'clean',
                        'fast', 'fishing'):
            self.assertIn(present, CONTROLLED_VOCAB, present)

    def test_state_vocab_is_a_closed_subset(self):
        self.assertLessEqual(set(STATE_VOCAB), set(CONTROLLED_VOCAB))
        self.assertEqual(len(set(STATE_VOCAB)), len(STATE_VOCAB))
        # Equipment, direction and manner words can never be head words: the
        # head slot is "a state the body is in", not "the important word".
        for absent in ('weapon', '1hand', 'forward', 'cast', 'spin', 'block'):
            self.assertNotIn(absent, STATE_VOCAB, absent)
        for present in ('idle', 'attack', 'crouch', 'rear', 'hover', 'sleep', 'sit'):
            self.assertIn(present, STATE_VOCAB, present)

    def test_t5_text_map_is_one_to_one_on_the_expanded_table(self):
        # The constraint is on the EXPANDED table, not on the override dict:
        # an override colliding with an identity token would share its vector.
        effective = [vocab_t5_text(word) for word in CONTROLLED_VOCAB]
        self.assertEqual(len(set(effective)), len(CONTROLLED_VOCAB))
        # A token with no override encodes as itself.
        self.assertEqual(vocab_t5_text('walk'), 'walk')
        # The measured overrides: the bare token lands on a different referent.
        self.assertEqual(vocab_t5_text('land'), 'touching down')
        self.assertEqual(vocab_t5_text('bow'), 'archery bow')
        self.assertEqual(vocab_t5_text('fishing'), 'fishing')
        self.assertNotEqual(vocab_t5_text('1hand'), vocab_t5_text('2hand'))

    def test_vocab_words_in_is_exact_token_matching(self):
        # Synonym translation is gone: a label is exact tokens, and free text
        # matches only where it happens to BE a token.
        self.assertEqual(vocab_words_in('run, forward, fast'), ['run', 'fast', 'forward'])
        self.assertEqual(vocab_words_in('run, haste, forward'), ['run', 'forward'])
        self.assertEqual(vocab_words_in('hurries forward'), ['forward'])
        self.assertEqual(vocab_words_in('jogging'), [])
        self.assertEqual(vocab_words_in('stands still'), [])
        # Order is VOCABULARY order, which is why this is the word SET and must
        # not be fed to canonical_action_label.
        self.assertEqual(vocab_words_in('forward, walk'), ['walk', 'forward'])

    def test_action_and_direction_views_partition_the_hits(self):
        text = 'run, forward, left, fast'
        self.assertEqual(action_words_in(text), ['run', 'fast'])
        self.assertEqual(direction_words_in(text), ['forward', 'left'])
        self.assertEqual(
            action_words_in(text) + direction_words_in(text), vocab_words_in(text)
        )

    def test_head_words_keep_the_written_order(self):
        self.assertEqual(head_words_in(['idle', 'attack']), ['idle', 'attack'])
        self.assertEqual(head_words_in(['attack', 'idle']), ['attack', 'idle'])
        self.assertEqual(head_words_in(['walk', 'weapon', 'forward']), ['walk'])

    def test_canonical_label_sorts_modifiers_but_never_heads(self):
        # Modifiers are sorted, so one combination has exactly one spelling.
        self.assertEqual(
            canonical_action_label(['walk', 'forward', 'weapon', '1hand']),
            'walk, forward, weapon, 1hand',
        )
        self.assertEqual(
            canonical_action_label(['walk', '1hand', 'forward', 'weapon']),
            'walk, forward, weapon, 1hand',
        )
        self.assertEqual(
            canonical_action_label(['walk', 'weapon', 'right']),
            'walk, right, weapon',
        )
        # Directions qualify the head sequence, so unrelated qualifiers and
        # equipment must not split them from it.
        self.assertEqual(
            canonical_action_label(['run', 'turn', 'weapon', '1hand', 'right']),
            'run, turn, right, weapon, 1hand',
        )
        self.assertEqual(
            canonical_action_label(['run', 'right', '1hand', 'turn', 'weapon']),
            'run, turn, right, weapon, 1hand',
        )
        # Filtering the result back to head words still preserves their order.
        self.assertEqual(
            canonical_action_label(['turn', 'hover', 'left']),
            'turn, left, hover',
        )
        # Head order is NOT sorted: it is the clip's time order and the only
        # record of which way a transition runs. This is the load-bearing case.
        self.assertEqual(canonical_action_label(['idle', 'attack']), 'idle, attack')
        self.assertEqual(canonical_action_label(['attack', 'idle']), 'attack, idle')
        self.assertEqual(canonical_action_label(['run', 'run']), 'run')
        with self.assertRaises(ActionLabelError):
            canonical_action_label(['run', 'nonsense'])

    def test_canonical_label_round_trips_through_the_parser(self):
        for label in ('walk, forward', 'run, forward, left, fast', 'attack, bite',
                      'run, strafe', 'idle, attack', 'attack, idle',
                      'walk, forward, weapon, 1hand', 'idle, rear, roar',
                      'run, turn, right, weapon, 1hand', 'turn, left, hover'):
            self.assertEqual(canonical_action_label(parse_action_label(label)), label)

    def test_parser_enforces_the_spelling_contract(self):
        self.assertEqual(parse_action_label(''), [])
        self.assertEqual(parse_action_label('attack, idle'), ['attack', 'idle'])
        for bad, why in (
            ('walk, jogging', 'out-of-vocabulary token'),
            ('walk, , forward', 'empty comma segment'),
            ('walk, walk', 'repeated token'),
            ('weapon, 1hand', 'no head word'),
            ('idle, hover, crouch', 'three head words'),
            (', '.join(['idle'] + list(DIRECTION_VOCAB) + ['weapon', 'bow', 'gun']),
             'over the token cap'),
        ):
            with self.assertRaises(ActionLabelError, msg=why):
                parse_action_label(bad)

    def test_max_words_and_max_heads_are_the_single_source_of_truth(self):
        self.assertEqual(ACTION_LABEL_MAX_WORDS, 8)
        self.assertEqual(ACTION_LABEL_MAX_HEADS, 2)

    def test_eight_total_words_are_accepted_and_nine_are_rejected(self):
        eight = 'idle, fast, bite, roar, eat, look, shake, throw'
        self.assertEqual(len(parse_action_label(eight)), 8)
        with self.assertRaises(ActionLabelError):
            parse_action_label(eight + ', taunt')

    def test_validator_hard_fails_on_a_non_canonical_or_unknown_label(self):
        from data_loaders.truebones.truebones_utils import motion_labels

        def _validate(label, group='locomotion'):
            motion_labels._validate_action_label_entry(group, label, 'c.npy', 1)

        # Legal: canonical keywords, and the empty (unconditional) label.
        _validate('walk, forward')
        _validate('')
        _validate('attack, idle', group='transition')
        # The vocabulary is closed and the corpus is spelled to match, so what
        # used to be advisory is now a gate -- a warning here could only buy a
        # silent regression, and reordering heads is a silent direction flip.
        for bad in ('walk, strides forward with arms swinging',
                    'walk, weapon, forward',   # modifier before direction
                    'walk, walk'):
            with self.assertRaises(SystemExit):
                _validate(bad)

    def test_validator_hard_fails_on_an_unknown_group(self):
        from data_loaders.truebones.truebones_utils import motion_labels

        with self.assertRaises(SystemExit):
            motion_labels._validate_action_label_entry('emote', 'idle', 'c.npy', 1)

    def test_head_order_may_only_diverge_in_transition(self):
        from data_loaders.truebones.truebones_utils import motion_labels

        # Two spellings of one word set: the clip's time order in a transition...
        motion_labels._validate_head_order_consistency([
            (1, 'transition', 'a.npy', ['idle', 'attack']),
            (2, 'transition', 'b.npy', ['attack', 'idle']),
        ])
        # ...and an inconsistent annotation anywhere else.
        with self.assertRaises(SystemExit):
            motion_labels._validate_head_order_consistency([
                (1, 'stationary', 'a.npy', ['idle', 'rear']),
                (2, 'stationary', 'b.npy', ['rear', 'idle']),
            ])


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

    def test_collate_emits_padded_word_ids_and_a_valid_mask(self):
        def _item(label, group):
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
                'action_slots': sample_action_slots(label, group),
            }

        _, cond = truebones_collate([
            _item('run, forward', 'locomotion'),
            _item('', 'stationary'),
        ])
        self.assertNotIn('action_multihot', cond['y'])
        # No assembled vector reaches the model from the data path; the loader
        # emits ids and the frozen table lives in the checkpoint.
        self.assertNotIn('action_label_emb', cond['y'])
        self.assertEqual(cond['y']['action_label'], ['run, forward', ''])
        # One fixed width for every batch, not the batch's longest label.
        self.assertEqual(cond['y']['action_word_ids'].shape, (2, ACTION_LABEL_MAX_WORDS))
        self.assertEqual(
            cond['y']['action_word_mask'][0].tolist(),
            [True, True] + [False] * (ACTION_LABEL_MAX_WORDS - 2),
        )
        # An empty label carries no words at all.
        self.assertFalse(bool(cond['y']['action_word_mask'][1].any()))
        # Padding matches no slot even without the mask.
        self.assertEqual(
            cond['y']['action_slot_ids'][0, 2:].tolist(),
            [SLOT_PAD_ID] * (ACTION_LABEL_MAX_WORDS - 2),
        )
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
