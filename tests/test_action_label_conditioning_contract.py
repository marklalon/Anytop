from pathlib import Path

import numpy as np
import pytest

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_LABEL_SLOTS,
    SLOT_DIRECTION,
    SLOT_HANDS,
    HEAD_SLOT_PRIMARY_WEIGHT,
    SLOT_HEAD,
    SLOT_MODIFIER,
    action_label_slots,
    assemble_slot_channels,
    slot_member_weights,
    conditioning_contract_payload,
    embedding_contract_payload,
    ordered_token_sources,
    fingerprint,
    slot_channel_representation,
    slot_source_rank_report,
    ActionConditioningError,
    SYNTHETIC_CODE_SCHEME,
    build_action_conditioning_bundle,
    scatter_synthetic_code_rows,
    synthetic_code_axis,
    synthetic_code_rows,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    HANDS_VOCAB,
    SYNTHETIC_CODE_VOCAB,
    T5_ENCODED_VOCAB,
    vocab_t5_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_slot_source_rank_covers_the_full_domain_and_projection_width():
    table = _word_table()
    report = slot_source_rank_report(table, latent_dim=256)
    assert report["full_rank"]
    assert report["fits_projection"]
    # The slots partition the vocabulary -- a head word is a head source and
    # nothing else, however late in the label it is spelled -- so the ranks add
    # to the vocabulary size. The hands slot has two members since hand0 was
    # retired (empty hands is the zero row, not a vector).
    assert report["total_rank"] == 104
    assert {name: item["rank"] for name, item in report["slots"].items()} == {
        "head": 32,
        "direction": 6,
        "modifier": 64,
        "hands": 2,
    }
    assert sum(report["slots"][name]["rank"] for name in report["slots"]) == len(
        CONTROLLED_VOCAB
    )
    assert not slot_source_rank_report(table, latent_dim=103)["fits_projection"]


def test_slot_assignment_carries_ids_masks_and_slots_only():
    """No role ids, no order mask: a label's condition is its (word, slot) set."""
    slots = action_label_slots(("idle", "attack"))
    assert set(slots) == {"word_ids", "word_mask", "slot_ids"}
    assert slots["word_mask"] == (True, True)
    # Both head words feed the head slot; which of them leads is positional,
    # and reaches the condition as a weight (slot_member_weights).
    assert slots["slot_ids"] == (SLOT_HEAD, SLOT_HEAD)
    assert slot_member_weights(slots["slot_ids"]) == (HEAD_SLOT_PRIMARY_WEIGHT, 1.0)
    assert action_label_slots(("attack", "idle"))["slot_ids"] == (SLOT_HEAD, SLOT_HEAD)
    assert action_label_slots(("run", "turn", "left", "fast", "hand1"))["slot_ids"] == (
        SLOT_HEAD, SLOT_HEAD, SLOT_DIRECTION, SLOT_MODIFIER, SLOT_HANDS,
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
        action_label_slots(("idle", "hand1", "hand2"))


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
    # Two exclusive members plus the zero row for empty hands, so the channel's
    # domain is exactly three points.
    assert all(word in CONTROLLED_VOCAB for word in HANDS_VOCAB)
    assert "hand0" not in CONTROLLED_VOCAB


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


def test_every_head_word_pools_into_the_head_channel_first_one_weighted():
    """The head channel is every head word, the first one weighted above the rest.

    A later head word ("attack, JUMP") names a body event the head channel has
    to carry, so it pools there rather than with the modifiers -- at weight 1.0
    against HEAD_SLOT_PRIMARY_WEIGHT for the word that leads.
    """
    table = _word_table()
    attack = table[CONTROLLED_VOCAB.index("attack")]
    hover = table[CONTROLLED_VOCAB.index("hover")]
    expected = HEAD_SLOT_PRIMARY_WEIGHT * attack + hover
    expected = expected / np.linalg.norm(expected)
    channels, present = _channels(("attack", "hover"), table)
    assert np.allclose(channels[SLOT_HEAD], expected)
    # The later head word left the modifier slot entirely: a word feeds exactly
    # one channel, which is what keeps the rank report a sum over disjoint blocks.
    assert not present[SLOT_MODIFIER]
    # A true modifier still pools on its own, unweighted, and does not touch the
    # head channel.
    with_bite, bite_present = _channels(("attack", "hover", "bite"), table)
    assert np.allclose(with_bite[SLOT_HEAD], expected)
    assert bite_present[SLOT_MODIFIER]
    bite = table[CONTROLLED_VOCAB.index("bite")]
    assert np.allclose(with_bite[SLOT_MODIFIER], bite / np.linalg.norm(bite))


def test_written_head_order_still_decides_the_condition():
    """"a, b" and "b, a" are two conditions -- by weight now, not by channel.

    This is the property that makes the weighted pool admissible at all: at
    weight 1:1 the two would be the same vector and written order would stop
    reaching the model. The data contract's one-head-order-per-word-set rule
    (motion_labels._validate_head_order_consistency) keeps the corpus from
    spelling one kind of clip as two conditions.
    """
    assert HEAD_SLOT_PRIMARY_WEIGHT != 1.0
    table = _word_table()
    first, _ = _channels(("attack", "hover"), table)
    second, _ = _channels(("hover", "attack"), table)
    assert not np.allclose(first[SLOT_HEAD], second[SLOT_HEAD])
    # A single-head label is unchanged by the weight: one member, and the
    # channel is L2-normalised, so only the ratio between members can reach it.
    alone, _ = _channels(("attack",), table)
    attack = table[CONTROLLED_VOCAB.index("attack")]
    assert np.allclose(alone[SLOT_HEAD], attack / np.linalg.norm(attack))


def test_only_the_weight_ratio_reaches_the_channel():
    """Scaling every weight of a slot is a no-op, so the constant is a ratio."""
    table = _word_table()
    slots = action_label_slots(("attack", "hover"))
    weights = slot_member_weights(slots["slot_ids"])
    assert weights == (HEAD_SLOT_PRIMARY_WEIGHT, 1.0)
    channels, _ = assemble_slot_channels(table, slots)
    scaled = (table * 1.0)
    doubled = np.stack(
        [HEAD_SLOT_PRIMARY_WEIGHT * 2 * scaled[CONTROLLED_VOCAB.index("attack")]
         + 2 * scaled[CONTROLLED_VOCAB.index("hover")]]
    )[0]
    assert np.allclose(channels[SLOT_HEAD], doubled / np.linalg.norm(doubled))


def test_slot_channel_representation_is_pinned_in_the_conditioning_fingerprint():
    common = dict(embedding_fingerprint=fingerprint(_embedding_payload()))
    approved = conditioning_contract_payload(representation=slot_channel_representation(), **common)
    altered = dict(slot_channel_representation(), slot_aggregation="mean of member word vectors")
    assert fingerprint(approved) != fingerprint(
        conditioning_contract_payload(representation=altered, **common)
    )


def _embedding_payload():
    return embedding_contract_payload(
        token_sources=ordered_token_sources(),
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
    # 5: hand0 retired (an empty hands slot is empty hands), direction dropout.
    # 6: later head words rejoined the head slot, weighted below the first.
    assert payload["parser_contract_version"] == 6


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


# --------------------------------------------------------------------------
# The synthetic code rows
# --------------------------------------------------------------------------
def _encoded_rows(dim=768, seed=20260919):
    """Stand-in rows for the T5-encoded half, unit norm like the real ones."""
    rows = np.random.default_rng(seed).standard_normal((len(T5_ENCODED_VOCAB), dim))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_the_code_rows_are_orthonormal_within_every_slot_they_feed():
    """The whole point of the code: no member correlates with another.

    T5 put every one of these next to its own antonym (left/right +0.461,
    hand1/hand2 +0.529, against a vocabulary-wide |cos| p95 of 0.19), which is
    the correlation the first Linear had to spend capacity undoing.
    """
    table = scatter_synthetic_code_rows(_encoded_rows())
    index = {word: position for position, word in enumerate(CONTROLLED_VOCAB)}
    for axis_words in (DIRECTION_VOCAB, HANDS_VOCAB):
        block = table[[index[word] for word in axis_words]]
        gram = block @ block.T
        assert np.allclose(gram, np.eye(len(axis_words)), atol=1e-6), axis_words


def test_the_code_blocks_are_exactly_conditioned():
    """Full rank is not enough -- these blocks are perfectly conditioned."""
    report = slot_source_rank_report(scatter_synthetic_code_rows(_encoded_rows()), 256)
    for name in ("direction", "hands"):
        assert report["slots"][name]["full_rank"]
        assert report["slots"][name]["relative_min_singular"] == pytest.approx(1.0)


def test_scatter_puts_every_row_where_its_word_id_points():
    """A word id is a position in CONTROLLED_VOCAB, for both halves of the table."""
    encoded = _encoded_rows()
    table = scatter_synthetic_code_rows(encoded)
    assert table.shape == (len(CONTROLLED_VOCAB), encoded.shape[1])
    code = synthetic_code_rows(encoded.shape[1])
    for position, word in enumerate(CONTROLLED_VOCAB):
        if word in SYNTHETIC_CODE_VOCAB:
            expected = code[synthetic_code_axis(word)]
        else:
            expected = encoded[T5_ENCODED_VOCAB.index(word)]
        assert np.array_equal(table[position], expected), word


def _code_bundle_contract(table):
    return embedding_contract_payload(
        token_sources=ordered_token_sources(),
        t5_name="t5-base",
        t5_artifact_sha256="a" * 64,
        tokenizer_class="T5Tokenizer",
        tokenizer_version="5.5.4",
        pooling="masked_mean",
        eos_policy="keep",
        vector_postprocess="center_l2",
        embedding_dim=int(table.shape[1]),
        dtype="float32",
        word_table_sha256=word_table_sha256(table),
    )


def test_an_edited_code_row_is_refused_and_named():
    """word_table_sha256 cannot catch this: it hashes whatever table it was given.

    A builder that scattered the code block wrong writes a contract from that
    same table, so the hash agrees with itself. The code rows are therefore
    checked against what this version writes, by value.
    """
    table = scatter_synthetic_code_rows(_encoded_rows())
    tampered = table.copy()
    tampered[CONTROLLED_VOCAB.index("left")] = tampered[CONTROLLED_VOCAB.index("right")]
    # A contract that fully describes the tampered table, so the code check is
    # the only thing left that can refuse it.
    contract = _code_bundle_contract(tampered)
    with pytest.raises(ActionConditioningError, match="left"):
        build_action_conditioning_bundle(tampered, contract, source="tampered")
    # The untampered table of the same shape and recipe passes.
    build_action_conditioning_bundle(table, _code_bundle_contract(table), source="ok")


def test_a_checkpoint_table_is_not_held_to_the_current_code_layout():
    """Old weights were fitted against their own rows, not against this version's."""
    table = scatter_synthetic_code_rows(_encoded_rows())
    table[CONTROLLED_VOCAB.index("up")] = table[CONTROLLED_VOCAB.index("down")]
    build_action_conditioning_bundle(
        table, _code_bundle_contract(table), source="ckpt", check_token_sources=False,
    )


def test_the_code_axis_is_part_of_the_embedding_fingerprint():
    """Moving a token between axes has to invalidate every bound checkpoint."""
    table = scatter_synthetic_code_rows(_encoded_rows())
    approved = _code_bundle_contract(table)
    swapped = [dict(entry) for entry in ordered_token_sources()]
    for entry in swapped:
        if entry.get("token") == "left":
            entry["axis"] = 99
    altered = dict(approved, ordered_token_sources=swapped)
    assert fingerprint(approved) != fingerprint(altered)


def test_a_synthetic_token_records_its_code_not_a_text():
    sources = {entry["token"]: entry for entry in ordered_token_sources()}
    assert [entry["token"] for entry in ordered_token_sources()] == list(CONTROLLED_VOCAB)
    for word in SYNTHETIC_CODE_VOCAB:
        assert sources[word] == {
            "token": word,
            "code": SYNTHETIC_CODE_SCHEME,
            "axis": SYNTHETIC_CODE_VOCAB.index(word),
        }
        assert "text" not in sources[word]
    for word in T5_ENCODED_VOCAB:
        assert sources[word] == {"token": word, "text": vocab_t5_text(word)}
