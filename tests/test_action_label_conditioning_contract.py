from pathlib import Path

import numpy as np
import pytest

from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (
    ACTION_LABEL_SLOTS,
    ROLE_B_EMBEDDING_DIM,
    ROLE_B_MATERIAL_SHA256,
    ROLE_B_NAMESPACE,
    ROLE_HEAD_1,
    ROLE_NONE,
    SLOT_DIRECTION,
    SLOT_HEAD,
    SLOT_MODIFIER,
    action_label_slots,
    assemble_slot_channels,
    conditioning_contract_payload,
    embedding_contract_payload,
    fingerprint,
    role_b_material,
    role_b_payload_hash,
    slot_channel_representation,
    validate_role_b_payload,
)
from data_loaders.truebones.truebones_utils.motion_labels import (
    CONTROLLED_VOCAB,
    vocab_t5_text,
)
from tools.evaluate_action_label_geometry import _slot_source_rank_report


ROOT = Path(__file__).resolve().parents[1]


def test_role_b_is_a_material_orthogonal_signed_permutation():
    payload = role_b_material(expected_dim=ROLE_B_EMBEDDING_DIM)
    perm = np.asarray(payload["perm"], dtype=np.int64)
    sign = np.asarray(payload["sign"], dtype=np.float32)
    probe = np.arange(ROLE_B_EMBEDDING_DIM, dtype=np.float32) - 100.0
    transformed = sign * probe[perm]
    assert np.dot(transformed, transformed) == pytest.approx(np.dot(probe, probe))


def test_role_b_derivation_is_stable_and_matches_the_committed_hash():
    """The committed hash, not a checked-in file, is what freezes R_B."""
    first = role_b_material()
    second = role_b_material()
    assert first == second
    assert first["namespace"] == ROLE_B_NAMESPACE
    assert first["material_sha256"] == ROLE_B_MATERIAL_SHA256
    assert role_b_payload_hash(first) == ROLE_B_MATERIAL_SHA256
    # The committed material itself, so a re-derivation cannot quietly move it.
    assert first["perm"][:4] == [434, 266, 440, 623]
    assert first["sign"][:4] == [-1, -1, 1, -1]
    assert sum(first["sign"]) == -52


def test_role_b_rejects_material_tampering():
    """Material that did not come from the derivation still gets checked."""
    payload = role_b_material()
    payload["sign"][0] *= -1
    with pytest.raises(ValueError, match="material hash mismatch"):
        validate_role_b_payload(payload, expected_dim=ROLE_B_EMBEDDING_DIM)


def test_role_b_rejects_a_foreign_permutation():
    payload = role_b_material()
    payload["perm"][0] = payload["perm"][1]
    payload["material_sha256"] = role_b_payload_hash(payload)
    with pytest.raises(ValueError, match="not a permutation"):
        validate_role_b_payload(payload, expected_dim=ROLE_B_EMBEDDING_DIM)


def test_slot_source_rank_covers_the_full_domain_and_projection_width():
    table = _word_table()
    payload = role_b_material(expected_dim=table.shape[1])
    report = _slot_source_rank_report(table, payload, latent_dim=256)
    assert report["full_rank"]
    assert report["fits_projection"]
    assert report["total_rank"] == 135
    assert {name: item["rank"] for name, item in report["slots"].items()} == {
        "head": 64,
        "direction": 6,
        "modifier": 65,
    }
    assert not _slot_source_rank_report(table, payload, latent_dim=134)[
        "fits_projection"
    ]


@pytest.mark.parametrize(
    ("group", "tokens", "roles", "order_mask"),
    [
        ("transition", ("idle",), (ROLE_NONE,), (False,)),
        ("transition", ("turn", "hover", "left"), (ROLE_NONE,) * 3, (False,) * 3),
        ("transition", ("idle", "attack"), (ROLE_NONE, ROLE_HEAD_1), (True, True)),
        ("stationary", ("idle", "attack"), (ROLE_NONE, ROLE_NONE), (False, False)),
    ],
)
def test_slot_contract_is_contextual(group, tokens, roles, order_mask):
    slots = action_label_slots(group, tokens)
    assert slots["role_ids"] == roles
    assert slots["order_head_mask"] == order_mask


def _word_table(dim=768):
    """A deterministic stand-in for the frozen T5 table."""
    generator = np.random.default_rng(20260905)
    return generator.standard_normal((len(CONTROLLED_VOCAB), dim))


def _channels(group, tokens, table):
    payload = role_b_material(expected_dim=table.shape[1])
    return assemble_slot_channels(
        table, action_label_slots(group, tokens), payload["perm"], payload["sign"]
    )


def test_slot_ids_partition_the_label_by_role():
    slots = action_label_slots("locomotion", ("walk", "forward", "weapon", "1hand"))
    assert slots["slot_ids"] == (SLOT_HEAD, SLOT_DIRECTION, SLOT_MODIFIER, SLOT_MODIFIER)


def test_head_and_direction_channels_ignore_added_modifiers():
    """The property the one-vector weighted mean could not have at any weight.

    A slot channel reads its own slot only, so appending modifiers moves it by
    exactly zero -- long-label axis retention is an equality here, not a tuned
    number.
    """
    table = _word_table()
    short, short_present = _channels("locomotion", ("walk", "forward"), table)
    long, long_present = _channels(
        "locomotion", ("walk", "forward", "weapon", "bow", "shield"), table
    )
    assert np.array_equal(short[SLOT_HEAD], long[SLOT_HEAD])
    assert np.array_equal(short[SLOT_DIRECTION], long[SLOT_DIRECTION])
    assert not short_present[SLOT_MODIFIER] and long_present[SLOT_MODIFIER]
    assert np.array_equal(short[SLOT_MODIFIER], np.zeros(table.shape[1]))


def test_absent_slot_does_not_renormalise_the_others():
    table = _word_table()
    with_direction, _ = _channels("locomotion", ("walk", "forward"), table)
    without, present = _channels("stationary", ("walk",), table)
    assert np.array_equal(with_direction[SLOT_HEAD], without[SLOT_HEAD])
    assert not present[SLOT_DIRECTION]
    for slot in range(len(ACTION_LABEL_SLOTS)):
        if present[slot]:
            assert np.linalg.norm(without[slot]) == pytest.approx(1.0)


def test_role_transform_separates_a_transition_from_its_reverse():
    table = _word_table()
    forward, _ = _channels("transition", ("idle", "attack"), table)
    backward, _ = _channels("transition", ("attack", "idle"), table)
    cosine = float(
        forward[SLOT_HEAD] @ backward[SLOT_HEAD]
        / (np.linalg.norm(forward[SLOT_HEAD]) * np.linalg.norm(backward[SLOT_HEAD]))
    )
    assert cosine < 0.5


def test_head_slot_is_order_invariant_where_the_role_gate_is_closed():
    """Outside transition the head slot pools without a role transform.

    That is safe only because the data contract lets one word set have one head
    order there (motion_labels._validate_head_order_consistency). This test
    records the dependency, so removing that rule fails here.
    """
    table = _word_table()
    first, _ = _channels("stationary", ("idle", "sit"), table)
    second, _ = _channels("stationary", ("sit", "idle"), table)
    assert np.allclose(first[SLOT_HEAD], second[SLOT_HEAD])


def test_slot_channel_representation_is_pinned_in_the_conditioning_fingerprint():
    common = dict(
        embedding_fingerprint=fingerprint(_embedding_payload()),
        role_b_material_sha256="1" * 64,
    )
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


def test_role_change_invalidates_conditioning_but_not_embedding_fingerprint():
    embedding_payload = _embedding_payload()
    embedding_fp = fingerprint(embedding_payload)
    common = dict(
        embedding_fingerprint=embedding_fp,
        representation=slot_channel_representation(),
    )
    first = conditioning_contract_payload(
        role_b_material_sha256="1" * 64,
        **common,
    )
    second = conditioning_contract_payload(
        role_b_material_sha256="2" * 64,
        **common,
    )
    assert fingerprint(embedding_payload) == embedding_fp
    assert fingerprint(first) != fingerprint(second)


def test_embedding_change_propagates_into_conditioning_fingerprint():
    first_embedding = _embedding_payload()
    second_embedding = dict(first_embedding, eos_policy="drop")
    common = dict(
        role_b_material_sha256="1" * 64,
        representation=slot_channel_representation(),
    )
    first = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(first_embedding), **common
    )
    second = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(second_embedding), **common
    )
    assert fingerprint(first) != fingerprint(second)


def test_representation_layout_is_part_of_conditioning_fingerprint():
    common = dict(
        embedding_fingerprint=fingerprint(_embedding_payload()),
        role_b_material_sha256="1" * 64,
    )
    slots = conditioning_contract_payload(
        representation=slot_channel_representation(), **common
    )
    # The live alternative: adopting it must not be able to look like
    # the same contract to a checkpoint trained on slot channels.
    tokenized = conditioning_contract_payload(
        representation={"kind": "k_token", "max_tokens": 8}, **common
    )
    assert fingerprint(slots) != fingerprint(tokenized)
