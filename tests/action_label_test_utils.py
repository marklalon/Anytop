"""Shared fixtures for the action-label conditioning tests.

Every test that needs a model with ``--action_label_cond`` needs the frozen word
table that goes with it, so the stand-in table is built here once, through the
real bundle constructor -- a fixture that skipped the validation would let a test
pass against a table training would reject.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loaders.tensors import _build_action_slot_batch  # noqa: E402
from data_loaders.truebones.truebones_utils.action_label_conditioning_contract import (  # noqa: E402
    ACTION_LABEL_SLOTS,
    ACTION_WORD_EMBEDDING_DTYPE,
    ACTION_WORD_EMBEDDING_EOS_POLICY,
    ACTION_WORD_EMBEDDING_POOLING,
    ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
    action_label_slots,
    build_action_conditioning_bundle,
    embedding_contract_payload,
    ordered_token_sources,
    scatter_synthetic_code_rows,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    CONTROLLED_VOCAB,
    T5_ENCODED_VOCAB,
    parse_action_label,
)


# The stand-in table's encoded half is dense random, so every slot's sources are
# independent -- the same full-rank property the real T5 table has and the model
# checks for. Its direction and hands blocks are the real orthonormal code, which
# is full rank by construction.
# 768 is t5-base's width, the one the real sidecar is encoded at.
TEST_T5_DIM = 768
# At least the total slot source rank (32 heads + 6 directions + 64 modifiers
# + 2 hands = 104; the slots partition the vocabulary, so the ranks add to its
# size), which model construction refuses to go under: below it the first Linear
# cannot separate every label. 140 is a width at or above the rank that the
# test models' attention can split evenly (num_heads=2).
TEST_LATENT_DIM = 140


def make_test_bundle(dim: int = TEST_T5_DIM, seed: int = 20260906, t5_name: str = "t5-test"):
    """A deterministic stand-in bundle, built and validated like a real one.

    Random rows stand in for the encoded half only. The synthetic half has to be
    the real orthonormal code, because that is what the bundle validator checks
    by value -- a fixture with random direction rows could not stand in for a
    sidecar at all.
    """
    table = scatter_synthetic_code_rows(
        np.random.default_rng(seed).standard_normal(
            (len(T5_ENCODED_VOCAB), dim)
        ).astype(np.float32)
    )
    contract = embedding_contract_payload(
        # The real token -> row source table: a fixture that used bare spellings
        # would not survive the loader's staleness check, so it could not stand
        # in for a sidecar in a round-trip test.
        token_sources=ordered_token_sources(),
        t5_name=t5_name,
        t5_artifact_sha256=f"test-artifact-{seed}",
        tokenizer_class="T5Tokenizer",
        tokenizer_version="0.0.0-test",
        pooling=ACTION_WORD_EMBEDDING_POOLING,
        eos_policy=ACTION_WORD_EMBEDDING_EOS_POLICY,
        vector_postprocess=ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
        embedding_dim=dim,
        dtype=ACTION_WORD_EMBEDDING_DTYPE,
        word_table_sha256=word_table_sha256(table),
    )
    return build_action_conditioning_bundle(
        table, contract, source=f"test-bundle-{seed}",
    )


def sample_action_slots(label: str, group: str):
    """One clip's slot arrays, exactly as the dataset attaches them.

    *group* is accepted so callers can keep spelling (label, group) pairs the
    way the collate sees them; the slot assignment itself does not read it.
    """
    del group
    if not label:
        return None
    slots = action_label_slots(parse_action_label(label))
    return {
        'word_ids': np.asarray(slots['word_ids'], dtype=np.int64),
        'slot_ids': np.asarray(slots['slot_ids'], dtype=np.int64),
        'word_mask': np.asarray(slots['word_mask'], dtype=np.bool_),
    }


def action_cond_fields(labels, groups):
    """The ``y`` fields the collate emits for these labels.

    Goes through the shipped ``_build_action_slot_batch`` rather than hand-rolling
    the padding, so a test model is fed what a training batch actually looks like.
    """
    slots = [sample_action_slots(label, group) for label, group in zip(labels, groups)]
    tensors, valid = _build_action_slot_batch(slots, list(labels))
    fields = {
        'action_label': list(labels),
        'action_group': list(groups),
        'action_label_valid': valid,
    }
    if tensors is not None:
        fields.update(tensors)
    return fields


def reference_channels(bundle, labels, groups, dtype=torch.float64):
    """The numpy contract's channels for the same labels, as one ``[B, S*D]``."""
    rows = []
    for label, group in zip(labels, groups):
        if not label:
            rows.append(np.zeros(len(ACTION_LABEL_SLOTS) * bundle.embedding_dim))
            continue
        channels, _present = bundle.channels_for(parse_action_label(label))
        rows.append(channels.reshape(-1))
    return torch.as_tensor(np.stack(rows), dtype=dtype)
