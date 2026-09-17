from pathlib import Path

import numpy as np
import pytest

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_LABEL_SLOTS,
    SLOT_DIRECTION,
    SLOT_HANDS,
    SLOT_HEAD,
    SLOT_MODIFIER,
    action_label_slots,
    assemble_slot_channels,
    conditioning_contract_payload,
    embedding_contract_payload,
    fingerprint,
    slot_channel_representation,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    CONTROLLED_VOCAB,
    HANDS_VOCAB,
    vocab_t5_text,
)
from tools.evaluate_action_label_geometry import _slot_source_rank_report


ROOT = Path(__file__).resolve().parents[1]


def test_slot_source_rank_covers_the_full_domain_and_projection_width():
    table = _word_table()
    report = _slot_source_rank_report(table, latent_dim=256)
    assert report["full_rank"]
    assert report["fits_projection"]
    # A head word is a modifier source too (any head word after the first), so
    # the modifier slot draws on the whole action vocabulary: 65 + 32.
    assert report["total_rank"] == 138
    assert {name: item["rank"] for name, item in report["slots"].items()} == {
        "head": 32,
        "direction": 6,
        "modifier": 97,
        "hands": 3,
    }
    assert not _slot_source_rank_report(table, latent_dim=137)["fits_projection"]


def test_slot_assignment_carries_ids_masks_and_slots_only():
    """No role ids, no order mask: a label's condition is its (word, slot) set."""
    slots = action_label_slots(("idle", "attack"))
    assert set(slots) == {"word_ids", "word_mask", "slot_ids"}
    assert slots["word_mask"] == (True, True)
    # Only the first head word is the head; the second is a modifier-slot member.
    assert slots["slot_ids"] == (SLOT_HEAD, SLOT_MODIFIER)
    assert action_label_slots(("attack", "idle"))["slot_ids"] == (SLOT_HEAD, SLOT_MODIFIER)
    assert action_label_slots(("run", "turn", "left", "fast", "hand1"))["slot_ids"] == (
        SLOT_HEAD, SLOT_MODIFIER, SLOT_DIRECTION, SLOT_MODIFIER, SLOT_HANDS,
    )


def _word_table(dim=768):
    """A deterministic stand-in for the frozen T5 table."""
    generator = np.random.default_rng(20260905)
    return generator.standard_normal((len(CONTROLLED_VOCAB), dim))


def _channels(tokens, table):
    return assemble_slot_channels(table, action_label_slots(tokens))


def test_slot_ids_partition_the_label_by_slot():
    slots = action_label_slots(("walk", "forward", "fast", "hand1"))
    assert slots["slot_ids"] == (SLOT_HEAD, SLOT_DIRECTION, SLOT_MODIFIER, SLOT_HANDS)
    assert ACTION_LABEL_SLOTS == ("head", "direction", "modifier", "hands")


def test_hands_axis_admits_one_member():
    with pytest.raises(ValueError, match="hand-state words"):
        action_label_slots(("idle", "hand0", "hand2"))


def test_hands_channel_is_the_token_vector_and_leaves_the_modifier_channel_alone():
    """The reason the axis has its own channel: annotating hand state on nearly
    every clip of a species must not dilute that species' real modifiers."""
    table = _word_table()
    bare, bare_present = _channels(("attack", "slash"), table)
    armed, armed_present = _channels(("attack", "slash", "hand2"), table)
    assert np.array_equal(bare[SLOT_MODIFIER], armed[SLOT_MODIFIER])
    assert np.array_equal(bare[SLOT_HEAD], armed[SLOT_HEAD])
    assert not bare_present[SLOT_HANDS] and armed_present[SLOT_HANDS]
    hand2 = table[CONTROLLED_VOCAB.index("hand2")]
    assert np.allclose(armed[SLOT_HANDS], hand2 / np.linalg.norm(hand2))
    # Three exclusive members, so the channel's domain is exactly three points.
    assert all(word in CONTROLLED_VOCAB for word in HANDS_VOCAB)


def test_head_and_direction_channels_ignore_added_modifiers():
    """The property the one-vector weighted mean could not have at any weight.

    A slot channel reads its own slot only, so appending modifiers moves it by
    exactly zero -- long-label axis retention is an equality here, not a tuned
    number.
    """
    table = _word_table()
    short, short_present = _channels(("walk", "forward"), table)
    long, long_present = _channels(
        ("walk", "forward", "fast", "bow", "shield", "hand2"), table
    )
    assert np.array_equal(short[SLOT_HEAD], long[SLOT_HEAD])
    assert np.array_equal(short[SLOT_DIRECTION], long[SLOT_DIRECTION])
    assert not short_present[SLOT_MODIFIER] and long_present[SLOT_MODIFIER]
    assert not short_present[SLOT_HANDS] and long_present[SLOT_HANDS]
    assert np.array_equal(short[SLOT_MODIFIER], np.zeros(table.shape[1]))
    assert np.array_equal(short[SLOT_HANDS], np.zeros(table.shape[1]))


def test_absent_slot_does_not_renormalise_the_others():
    table = _word_table()
    with_direction, _ = _channels(("walk", "forward"), table)
    without, present = _channels(("walk",), table)
    assert np.array_equal(with_direction[SLOT_HEAD], without[SLOT_HEAD])
    assert not present[SLOT_DIRECTION]
    for slot in range(len(ACTION_LABEL_SLOTS)):
        if present[slot]:
            assert np.linalg.norm(without[slot]) == pytest.approx(1.0)


def test_first_head_word_is_the_head_and_later_ones_are_modifiers():
    """"a, b" and "b, a" are two conditions: the first head word is the head.

    The head channel is that one word's vector, undiluted by a second head
    word, which lands in the modifier channel instead. The data contract's
    one-head-order-per-word-set rule (motion_labels._validate_head_order_consistency)
    is what keeps the corpus from spelling one kind of clip as two conditions.
    """
    table = _word_table()
    first, first_present = _channels(("attack", "hover"), table)
    second, _ = _channels(("hover", "attack"), table)
    assert not np.allclose(first[SLOT_HEAD], second[SLOT_HEAD])
    # The head channel of "attack, hover" is exactly the head channel of "attack".
    alone, alone_present = _channels(("attack",), table)
    assert np.array_equal(first[SLOT_HEAD], alone[SLOT_HEAD])
    assert not alone_present[SLOT_MODIFIER] and first_present[SLOT_MODIFIER]
    # ...and the second head word pools with the modifiers: "attack, hover, bite"
    # has the modifier channel of the {hover, bite} set, while its head channel
    # is still "attack" alone.
    with_bite, _ = _channels(("attack", "hover", "bite"), table)
    assert np.array_equal(with_bite[SLOT_HEAD], alone[SLOT_HEAD])
    hover_id = CONTROLLED_VOCAB.index("hover")
    bite_id = CONTROLLED_VOCAB.index("bite")
    expected = (table[hover_id] + table[bite_id]) / 2.0
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(with_bite[SLOT_MODIFIER], expected)


def test_slot_channel_representation_is_pinned_in_the_conditioning_fingerprint():
    common = dict(embedding_fingerprint=fingerprint(_embedding_payload()))
    approved = conditioning_contract_payload(representation=slot_channel_representation(), **common)
    altered = dict(slot_channel_representation(), slot_aggregation="mean of member word vectors")
    assert fingerprint(approved) != fingerprint(
        conditioning_contract_payload(representation=altered, **common)
    )


def _embedding_payload():
    return embedding_contract_payload(
        token_to_text={token: vocab_t5_text(token) for token in CONTROLLED_VOCAB},
        t5_name="t5-base",
        t5_artifact_sha256="a" * 64,
        tokenizer_class="T5Tokenizer",
        tokenizer_version="5.5.4",
        pooling="masked_mean",
        eos_policy="keep",
        vector_postprocess="raw",
        embedding_dim=768,
        dtype="float32",
        word_table_sha256="b" * 64,
    )


def test_conditioning_contract_carries_no_role_material():
    """The contract names what the model does with a word: its slot, nothing else."""
    payload = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(_embedding_payload()),
        representation=slot_channel_representation(),
    )
    assert payload["slot_fields"] == ["word_ids", "word_mask", "slot_ids"]
    assert not any("role" in key for key in payload)
    assert not any("role" in key for key in payload["representation"])
    # 4: the first head word is the head slot, later head words are modifiers.
    assert payload["parser_contract_version"] == 4


def test_embedding_change_propagates_into_conditioning_fingerprint():
    first_embedding = _embedding_payload()
    second_embedding = dict(first_embedding, eos_policy="drop")
    common = dict(representation=slot_channel_representation())
    first = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(first_embedding), **common
    )
    second = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(second_embedding), **common
    )
    assert fingerprint(first) != fingerprint(second)


def test_representation_layout_is_part_of_conditioning_fingerprint():
    common = dict(embedding_fingerprint=fingerprint(_embedding_payload()))
    slots = conditioning_contract_payload(
        representation=slot_channel_representation(), **common
    )
    # The live alternative: adopting it must not be able to look like
    # the same contract to a checkpoint trained on slot channels.
    tokenized = conditioning_contract_payload(
        representation={"kind": "k_token", "max_tokens": 8}, **common
    )
    assert fingerprint(slots) != fingerprint(tokenized)
