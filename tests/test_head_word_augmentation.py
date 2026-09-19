"""Head-word augmentation: a training-only ``slot_ids`` swap, plus aux routing.

The head slot takes a label's FIRST head word only, so a clip that spells its
main body event second ("attack, jump, charge") gives the jump channel nothing.
``--head_aug_words`` promotes such a word during training. These tests pin the
three properties that make the promotion safe to ship: it produces exactly the
channels the contract would produce for the promoted spelling, it never fires
outside training, and it cannot be triggered by padding.

See docs/aux_group_and_head_word_augmentation.md §5, §6.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    SLOT_HEAD,
    SLOT_MODIFIER,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    CONTROLLED_VOCAB,
)
from data_loaders.tensors import truebones_batch_collate  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from tests.action_label_test_utils import (  # noqa: E402
    TEST_LATENT_DIM,
    TEST_T5_DIM,
    action_cond_fields,
    make_test_bundle,
    reference_channels,
    sample_action_slots,
)


def _model(bundle, *, head_aug_words='', head_aug_prob=0.0, aux_label_mode='aug',
           eval_mode=False):
    model = AnyTop(
        max_joints=4,
        feature_len=12,
        latent_dim=TEST_LATENT_DIM,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=TEST_T5_DIM,
        action_label_cond=True,
        action_label_cfg_drop_prob=0.0,
        head_aug_words=head_aug_words,
        head_aug_prob=head_aug_prob,
        aux_label_mode=aux_label_mode,
        action_conditioning=bundle,
    )
    if eval_mode:
        model.eval()
    return model


def _promote(model, fields, force=None):
    batch = fields['action_word_ids'].shape[0]
    return model._promote_head_slot(
        fields['action_word_ids'],
        fields['action_slot_ids'],
        fields['action_word_mask'],
        batch,
        torch.device('cpu'),
        force=force,
    )


LABELS = ['attack, jump, charge', 'land, jump', 'jump', 'attack, swat']
GROUPS = ['stationary', 'transition', 'transition', 'stationary']


# --------------------------------------------------------------------------
# What the swap produces
# --------------------------------------------------------------------------
def test_promotion_swaps_head_and_modifier_slot_ids():
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=1.0)
    fields = action_cond_fields(LABELS, GROUPS)
    before = fields['action_slot_ids']
    after = _promote(model, fields)

    jump_id = list(CONTROLLED_VOCAB).index('jump')
    jump_at = fields['action_word_ids'] == jump_id

    # 'attack, jump, charge' and 'land, jump' spell jump second -> promoted.
    for row in (0, 1):
        assert before[row][jump_at[row]].tolist() == [SLOT_MODIFIER]
        assert after[row][jump_at[row]].tolist() == [SLOT_HEAD]
        # and the word that HELD the head slot is now a modifier
        head_before = before[row] == SLOT_HEAD
        assert (after[row][head_before] == SLOT_MODIFIER).all()
    # 'jump' already leads its label; 'attack, swat' has no jump at all.
    assert torch.equal(after[2], before[2])
    assert torch.equal(after[3], before[3])


def test_promoted_row_equals_the_contract_for_the_promoted_spelling():
    """The swap is not an approximation of re-spelling -- it IS re-spelling.

    Written order never reaches the model beyond the slot ids, so promoting
    jump in 'attack, jump, charge' must land on exactly the channels the numpy
    contract computes for 'jump, attack, charge'.
    """
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=1.0)
    fields = action_cond_fields(['attack, jump, charge'], ['stationary'])
    promoted_slots = _promote(model, fields)
    got = model._assemble_action_slot_channels(
        fields['action_word_ids'], promoted_slots, fields['action_word_mask'], torch.float64,
    )
    expected = reference_channels(bundle, ['jump, attack, charge'], ['transition'])
    assert torch.allclose(got, expected, atol=1e-12)


# --------------------------------------------------------------------------
# When it must not fire
# --------------------------------------------------------------------------
def test_no_promotion_when_the_word_list_is_empty():
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='', head_aug_prob=1.0)
    fields = action_cond_fields(LABELS, GROUPS)
    assert torch.equal(_promote(model, fields), fields['action_slot_ids'])


def test_no_promotion_at_probability_zero():
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=0.0)
    fields = action_cond_fields(LABELS, GROUPS)
    assert torch.equal(_promote(model, fields), fields['action_slot_ids'])


def test_no_promotion_in_eval_mode():
    """Inference must see the corpus spelling, whatever the flags say."""
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=1.0, eval_mode=True)
    fields = action_cond_fields(LABELS, GROUPS)
    assert torch.equal(_promote(model, fields), fields['action_slot_ids'])


def test_padding_columns_are_never_promoted():
    """Padding carries word id 0, which is a REAL vocabulary word.

    Without the word_mask gate the lookup fires on every padded column, which
    would rewrite the slot ids of rows that spell nothing of the kind.
    """
    bundle = make_test_bundle()
    pad_word = CONTROLLED_VOCAB[0]
    model = _model(bundle, head_aug_words=pad_word, head_aug_prob=1.0)
    # A short label, so most columns are padding holding word id 0 == pad_word.
    fields = action_cond_fields(['die'], ['transition'])
    after = _promote(model, fields)
    padded = ~fields['action_word_mask']
    assert torch.equal(after[padded], fields['action_slot_ids'][padded])


def test_only_head_vocab_words_may_be_listed():
    bundle = make_test_bundle()
    with pytest.raises(ValueError, match='HEAD_VOCAB'):
        _model(bundle, head_aug_words='charge')
    with pytest.raises(ValueError, match='controlled-vocabulary'):
        _model(bundle, head_aug_words='pirouette')


# --------------------------------------------------------------------------
# Auxiliary routing (--aux_label_mode)
# --------------------------------------------------------------------------
def _keep(model, labels, groups, is_aux, slots=None, word_ids=None):
    fields = action_cond_fields(labels, groups)
    batch = fields['action_word_ids'].shape[0]
    y = dict(fields)
    y['is_aux'] = torch.tensor(is_aux, dtype=torch.bool)
    return model._keep_aux_label(
        y,
        fields['action_word_ids'] if word_ids is None else word_ids,
        fields['action_slot_ids'] if slots is None else slots,
        batch,
        torch.device('cpu'),
    )


def test_aux_mode_null_sends_every_borrowed_row_to_the_unconditional_branch():
    bundle = make_test_bundle()
    model = _model(bundle, aux_label_mode='null')
    keep = _keep(model, ['attack, jump, charge', 'die'], ['stationary', 'transition'],
                 [True, False])
    assert keep.tolist() == [False, True]


def test_aux_mode_label_keeps_everything():
    bundle = make_test_bundle()
    model = _model(bundle, aux_label_mode='label')
    keep = _keep(model, ['attack, jump, charge', 'die'], ['stationary', 'transition'],
                 [True, False])
    assert keep.tolist() == [True, True]


def test_aux_mode_aug_keeps_promoted_rows_and_nulls_the_rest():
    """The point of 'aug': a borrowed clip either arrives as a word this group
    is queried for, or contributes species prior only."""
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=0.0, aux_label_mode='aug')
    labels = ['attack, jump, charge', 'walk, forward', 'die']
    groups = ['stationary', 'locomotion', 'transition']
    fields = action_cond_fields(labels, groups)
    is_aux = torch.tensor([True, True, False])
    promoted = _promote(model, fields, force=is_aux)
    keep = _keep(model, labels, groups, [True, True, False], slots=promoted)
    # jump was promotable -> the row keeps its (now head=jump) label.
    # 'walk, forward' has no promotable word -> unconditional.
    # the non-aux row is untouched.
    assert keep.tolist() == [True, False, True]


def test_aux_force_promotes_regardless_of_head_aug_prob():
    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=0.0)
    fields = action_cond_fields(['attack, jump, charge'], ['stationary'])
    unforced = _promote(model, fields)
    forced = _promote(model, fields, force=torch.tensor([True]))
    assert torch.equal(unforced, fields['action_slot_ids'])
    assert not torch.equal(forced, fields['action_slot_ids'])


def test_aux_flags_survive_full_collate_and_control_label_routing():
    def raw_sample(label, group, is_aux):
        return (
            np.zeros((3, 2, 12), dtype=np.float32),
            3,
            np.array([-1, 0], dtype=np.int64),
            np.zeros((2, 12), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            'Buffalo',
            np.zeros((2, TEST_T5_DIM), dtype=np.float32),
            2,
            {
                'action_group': group,
                'action_label': label,
                'action_slots': sample_action_slots(label, group),
                'is_aux': is_aux,
            },
            'Buffalo_clip.npy',
        )

    _, cond = truebones_batch_collate([
        raw_sample('attack, jump, charge', 'stationary', True),
        raw_sample('walk, forward', 'locomotion', True),
        raw_sample('die', 'transition', False),
    ])
    assert cond['y']['is_aux'].tolist() == [True, True, False]

    bundle = make_test_bundle()
    model = _model(bundle, head_aug_words='jump', head_aug_prob=0.0,
                   aux_label_mode='aug')
    channels, active = model._action_condition(
        cond['y'], 3, torch.device('cpu'), torch.float64
    )
    assert active.tolist() == [True, False, True]
    assert torch.allclose(
        channels[:1],
        reference_channels(bundle, ['jump, attack, charge'], ['transition']),
        atol=1e-12,
    )


def test_head_promotion_compiles_without_a_graph_break():
    model = _model(make_test_bundle(), head_aug_words='jump', head_aug_prob=1.0)
    fields = action_cond_fields(['attack, jump, charge'], ['stationary'])
    torch._dynamo.reset()
    compiled = torch.compile(model._promote_head_slot, backend='eager',
                             dynamic=False, fullgraph=True)
    got = compiled(
        fields['action_word_ids'], fields['action_slot_ids'],
        fields['action_word_mask'], 1, torch.device('cpu'),
    )
    assert torch.equal(got, _promote(model, fields))
