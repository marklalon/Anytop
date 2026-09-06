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
    ROLE_B_EMBEDDING_DIM,
    ACTION_WORD_EMBEDDING_DTYPE,
    ACTION_WORD_EMBEDDING_EOS_POLICY,
    ACTION_WORD_EMBEDDING_POOLING,
    ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS,
    action_label_slots,
    build_action_conditioning_bundle,
    embedding_contract_payload,
    word_table_sha256,
)
from data_loaders.truebones.truebones_utils.motion_labels import (  # noqa: E402
    CONTROLLED_VOCAB,
    parse_action_label,
    vocab_t5_text,
)


# The stand-in table is dense random, so every slot's sources are independent --
# the same full-rank property the real T5 table has and the model checks for.
# 768 is not a choice: R_B is committed at exactly one dimension, so a word table
# of any other width has no role transform to apply.
TEST_T5_DIM = ROLE_B_EMBEDDING_DIM
# At least the total slot source rank (64 + 6 + 65), which model construction
# refuses to go under: below it the first Linear cannot separate every label.
TEST_LATENT_DIM = 136


def make_test_bundle(dim: int = TEST_T5_DIM, seed: int = 20260906, t5_name: str = "t5-test"):
    """A deterministic stand-in bundle, built and validated like a real one."""
    table = np.random.default_rng(seed).standard_normal(
        (len(CONTROLLED_VOCAB), dim)
    ).astype(np.float32)
    contract = embedding_contract_payload(
        # The real token -> T5 text map: a fixture that used bare spellings would
        # not survive the loader's staleness check, so it could not stand in for
        # a sidecar in a round-trip test.
        token_to_text={token: vocab_t5_text(token) for token in CONTROLLED_VOCAB},
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
    """One clip's slot arrays, exactly as the dataset attaches them."""
    if not label:
        return None
    slots = action_label_slots(group, parse_action_label(label))
    return {
        'word_ids': np.asarray(slots['word_ids'], dtype=np.int64),
        'role_ids': np.asarray(slots['role_ids'], dtype=np.int64),
        'slot_ids': np.asarray(slots['slot_ids'], dtype=np.int64),
        'word_mask': np.asarray(slots['word_mask'], dtype=np.bool_),
        'order_head_mask': np.asarray(slots['order_head_mask'], dtype=np.bool_),
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
        channels, _present = bundle.channels_for(group, parse_action_label(label))
        rows.append(channels.reshape(-1))
    return torch.as_tensor(np.stack(rows), dtype=dtype)
