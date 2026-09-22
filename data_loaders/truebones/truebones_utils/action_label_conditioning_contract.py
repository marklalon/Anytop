"""The word-keyed action-label conditioning contract.

This module deliberately has no torch dependency.  The sidecar builder, the data
loader, model construction and the checkpoint loader all import these helpers, so
slot assignment, slot layout and the two fingerprints cannot acquire competing
definitions across those four call sites; the model mirrors ``assemble_slot_channels``
on tensors against the very ``slot_ids`` produced here.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Iterable, Mapping

import numpy as np

from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    HANDS_VOCAB,
    HEAD_VOCAB,
    SYNTHETIC_CODE_VOCAB,
    T5_ENCODED_VOCAB,
    head_words_in,
)


ACTION_WORD_EMBEDDING_SCHEMA_VERSION = 4
ACTION_CONDITIONING_CONTRACT_SCHEMA_VERSION = 1
# 5: hand0 retired -- an empty hands slot means empty hands (the content
#    default), hand1 / hand2 are the only members; the direction slot gained
#    a training dropout so ITS empty state is the marginal (2026-09-18).
# 6: a label's later head word rejoined the head slot, weighted below the
#    first one (HEAD_SLOT_PRIMARY_WEIGHT), instead of sitting with the
#    modifiers.  "attack, jump" now puts jump into the head channel at every
#    step, which is what the retired --head_aug_words promotion bought a
#    fraction of the time (2026-09-19).
ACTION_LABEL_PARSER_CONTRACT_VERSION = 6

# Slots.  The approved representation gives each slot its own conditioning
# channel, so a word's contribution depends on ITS slot only -- appending a
# NON-HEAD word cannot shrink the head or direction axis, which is the property
# the one-vector weighted mean could not have at any weight setting.  A second
# head word is the one exception, and a deliberate one: see
# HEAD_SLOT_PRIMARY_WEIGHT.
#
# The hands axis has its own channel rather than riding in the modifier slot
# because, once annotated, it sits on nearly every clip of every hand-bearing
# species: pooled into the modifier mean it would halve the signal of every
# real modifier (slash, punch, fast, cast ...) on exactly those species, and
# "attack, slash" would no longer read the same with and without a hand state.
# In its own channel the other three are bit-identical either way, and the
# channel is the token's own vector (the axis admits one member), so the model
# only has to tell two points and the zero row (empty hands) apart.
SLOT_HEAD = 0
SLOT_DIRECTION = 1
SLOT_MODIFIER = 2
SLOT_HANDS = 3
ACTION_LABEL_SLOTS: tuple[str, ...] = ("head", "direction", "modifier", "hands")

# Within the head slot, how much more the label's FIRST head word weighs than a
# later one.  Every other slot pools its members evenly.
#
# A label's later head word ("attack, JUMP, charge", "land, JUMP") names the
# body event the clip is largely about, and for one contract revision it sat
# with the modifiers, which left the head channel -- the channel inference
# queries -- with no share of it at all.  Pooling the two evenly is the other
# extreme and is the one setting that is NOT allowed: at weight 1:1 the head
# channel of "a, b" and "b, a" is literally the same vector, so written head
# order stops being part of the condition.  Any ratio other than 1 keeps the
# two apart (the source rows are independent, and (r, 1) is not a multiple of
# (1, r) unless r == 1), so the choice is only about how much of the axis the
# first word keeps.
#
# 1.5 was measured on the frozen table (dim 768, all rows unit norm) against
# the two costs that matter, with the word-pair null band |cos| p95 = 0.19 for
# scale:
#
#   cos(head channel, first word alone)        "attack, hover"  0.83
#   cos(head channel, later word alone)        "attack, jump"   0.61
#   cos("idle, hover", "attack, hover")                         0.36
#   cos("attack, jump", "land, jump")                           0.24
#   cos("attack, hover", "hover, attack")                       0.92
#
# Raising it sharpens the first three and softens the last; 1.0 collapses the
# last to 1.00 and is rejected above.  It is a constant and not a flag on
# purpose: it is part of the condition's meaning, so inference has to reproduce
# it, and the conditioning fingerprint below is what refuses a checkpoint
# trained under a different value.
HEAD_SLOT_PRIMARY_WEIGHT = 1.5


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole byte representation used by contract fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def word_table_sha256(word_embeddings) -> str:
    """Hash the word vectors themselves, in one fixed byte layout.

    Everything else in the embedding contract records the INPUTS that should
    have produced these vectors -- the text, the encoder, the pooling -- and not
    one of those fields changes when the vectors do.  Without this field a table
    swapped for another of the same shape keeps the same ``embedding_fingerprint``,
    and every guard built on that fingerprint (the sidecar loader, the resume
    check, the checkpoint bind) passes on vectors nobody trained against.

    ``<f4`` rather than the platform float32 so the digest is byte-identical on
    a big-endian host, and ``ascontiguousarray`` so a view or a transposed copy
    of the same table hashes the same.
    """
    table = np.ascontiguousarray(np.asarray(word_embeddings, dtype="<f4"))
    if table.ndim != 2:
        raise ValueError(
            f"word table must be 2-D to be hashed, got shape {tuple(table.shape)}"
        )
    digest = hashlib.sha256()
    digest.update(f"{table.shape[0]}x{table.shape[1]}/<f4\0".encode("utf-8"))
    digest.update(table.tobytes())
    return digest.hexdigest()


def action_label_slots(tokens: Iterable[str]) -> dict[str, tuple]:
    """Map parsed tokens to the ids/masks the loader and the sampler emit.

    This is the canonical slot-assignment implementation, shared by training
    and inference rather than copied into either path.  A word's slot is a
    property of the word and its position only (see :func:`label_slot_ids`):
    the label's group plays no part, so the same label is the same condition
    in every group's model.
    """
    ordered_tokens = tuple(tokens)
    if len(ordered_tokens) > ACTION_LABEL_MAX_WORDS:
        raise ValueError(
            f"action label has {len(ordered_tokens)} tokens; max is "
            f"{ACTION_LABEL_MAX_WORDS}"
        )
    vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
    unknown = tuple(word for word in ordered_tokens if word not in vocab_index)
    if unknown:
        raise ValueError(f"unknown action-label token(s): {unknown}")
    if len(set(ordered_tokens)) != len(ordered_tokens):
        raise ValueError("action label tokens must not repeat")
    heads = tuple(head_words_in(ordered_tokens))
    if not 1 <= len(heads) <= ACTION_LABEL_MAX_HEADS:
        raise ValueError(
            f"action label must have 1..{ACTION_LABEL_MAX_HEADS} head words; "
            f"got {heads}"
        )
    hands = tuple(word for word in ordered_tokens if word in HANDS_VOCAB)
    if len(hands) > 1:
        raise ValueError(
            f"action label names {len(hands)} hand-state words {hands}; the hands "
            "axis admits at most one"
        )

    return {
        "word_ids": tuple(vocab_index[word] for word in ordered_tokens),
        "word_mask": tuple(True for _ in ordered_tokens),
        "slot_ids": label_slot_ids(ordered_tokens),
    }


def word_slots(word: str) -> tuple[int, ...]:
    """Every conditioning channel one vocabulary word can feed.

    Exactly one each: the slots partition a label's words.  A head word feeds
    the head slot whether it leads the label or follows another head word --
    following only costs it weight, not its channel (:func:`label_slot_ids`,
    :data:`HEAD_SLOT_PRIMARY_WEIGHT`).  That is what keeps the rank report a
    sum over disjoint blocks: no vector is a source of two channels.
    """
    if word in HEAD_VOCAB:
        return (SLOT_HEAD,)
    if word in DIRECTION_VOCAB:
        return (SLOT_DIRECTION,)
    if word in HANDS_VOCAB:
        return (SLOT_HANDS,)
    if word not in CONTROLLED_VOCAB:
        raise ValueError(f"unknown action-label token: {word!r}")
    return (SLOT_MODIFIER,)


def label_slot_ids(ordered_tokens: Iterable[str]) -> tuple[int, ...]:
    """The slot each token of one label feeds, in the label's written order.

    Every head word feeds the head slot.  Written order still decides the
    condition, but through weight rather than through the channel: the first
    head word is what the label is about and carries
    :data:`HEAD_SLOT_PRIMARY_WEIGHT`, a later one qualifies it (the posture or
    medium it happens in: "attack, hover", "land, jump") and carries 1.0.  So
    "a, b" and "b, a" stay two conditions -- the corpus spells one word set one
    way per group for exactly that reason
    (motion_labels._validate_head_order_consistency) -- while the head channel
    keeps a real share of the later word, which is the whole point of pooling
    it here instead of with the modifiers.

    Which head word is "first" is a matter of position, so it is read off this
    tuple rather than stored: see :func:`slot_member_weights`.
    """
    return tuple(word_slots(word)[0] for word in ordered_tokens)


def slot_member_weights(slot_ids: Iterable[int]) -> tuple[float, ...]:
    """The pooling weight of each of one label's words, in written order.

    The head slot's FIRST member takes :data:`HEAD_SLOT_PRIMARY_WEIGHT` and
    every other word takes 1.0.  "First" is positional -- the earliest column
    of this row assigned to the head slot -- because a label's words reach the
    model in written order and the slot ids alone do not say which head word
    led.  The model mirrors this on tensors with a cumulative sum over the same
    order, so the two cannot disagree about which word is the primary one.

    Only the RATIO reaches the condition: each channel is L2-normalised after
    pooling, so scaling every weight of a slot is a no-op.
    """
    weights = []
    head_seen = False
    for slot in slot_ids:
        if slot == SLOT_HEAD and not head_seen:
            weights.append(float(HEAD_SLOT_PRIMARY_WEIGHT))
            head_seen = True
        else:
            weights.append(1.0)
    return tuple(weights)


def assemble_slot_channels(
    word_vectors: np.ndarray,
    slots: Mapping[str, tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one label's slot assignment into its (S, D) conditioning channels.

    This is the ONLY implementation of the approved representation: the sidecar
    builder, the geometry preflight, the loader and the model all call it (the
    model mirrors it on tensors, against the same slot ids) so a channel cannot
    acquire two definitions.

    Each slot holds the weighted mean of its member word vectors,
    L2-normalised.  Every weight is 1.0 except the head slot's first member
    (:func:`slot_member_weights`), so three of the four slots are plain means
    of a set and written order reaches the model only as "which head word
    leads".  Normalising per slot is what makes the head axis independent of
    how many MODIFIERS the label spells; an absent slot is a zero row flagged
    in the returned mask, never a renormalisation of the others.
    """
    vectors = np.asarray(word_vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"word_vectors must be (V, D), got {vectors.shape}")

    word_ids = tuple(slots["word_ids"])
    slot_ids = tuple(slots["slot_ids"])
    weights = slot_member_weights(slot_ids)
    channels = np.zeros((len(ACTION_LABEL_SLOTS), vectors.shape[1]), dtype=np.float64)
    present = np.zeros(len(ACTION_LABEL_SLOTS), dtype=bool)
    for slot in range(len(ACTION_LABEL_SLOTS)):
        members = [
            (weight, vectors[word_id])
            for word_id, assigned, weight in zip(word_ids, slot_ids, weights)
            if assigned == slot
        ]
        if not members:
            continue
        member_weights = np.asarray([weight for weight, _ in members], dtype=np.float64)
        mean = (
            np.stack([vector for _, vector in members]) * member_weights[:, None]
        ).sum(axis=0) / member_weights.sum()
        norm = float(np.linalg.norm(mean))
        if norm <= 1e-9:
            raise ValueError(
                f"slot {ACTION_LABEL_SLOTS[slot]!r} of {word_ids} pooled to a zero "
                "vector; the frozen word table cannot express this label"
            )
        channels[slot] = mean / norm
        present[slot] = True
    return channels, present


def slot_channel_representation() -> dict[str, Any]:
    """The ``representation`` block of the approved conditioning contract."""
    return {
        "kind": "slot_channels",
        "slots": list(ACTION_LABEL_SLOTS),
        "slot_assignment": (
            "head = every HEAD_VOCAB word of the label (at most ACTION_LABEL_MAX_HEADS); "
            "direction = DIRECTION_VOCAB member; "
            "hands = HANDS_VOCAB member (at most one); "
            "modifier = every other vocabulary word"
        ),
        "slot_aggregation": "weighted mean of member word vectors, then L2 normalisation",
        "absent_slot": "zero row, reported in slot_mask; never renormalises the other slots",
        "channel_layout": "concatenated in ACTION_LABEL_SLOTS order",
        "projection_input": (
            "each slot's channel cut to slot_channel_widths: the full embedding "
            "for a T5-encoded slot, the SYNTHETIC_CODE_DIM code axes for a code slot"
        ),
        "per_word_weights": {
            "head_first": float(HEAD_SLOT_PRIMARY_WEIGHT),
            "default": 1.0,
            "rule": (
                "the head slot's first member by written order takes head_first; "
                "every other word of every slot takes default"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Synthetic code rows
# ---------------------------------------------------------------------------
# The direction and hands rows are not encoded from anything: they are a
# one-hot written into the same (V, D) table, so downstream -- pooling, the
# rank report, ``word_table_sha256`` -- treats them like any other row.
#
# Axis-aligned, not a seeded random orthonormal set: a one-hot is what the
# axis means, and a deterministic table is what a rebuild must reproduce byte
# for byte.  The axis is the token's position in SYNTHETIC_CODE_VOCAB, so
# direction takes e0..e5 and hands e6..e7.
#
# The two slots keep disjoint axes so no two rows of the table are equal --
# a row swap would otherwise be invisible to anything that reads by value.
#
# The (D - 8) unused columns of those two blocks are inert: zero input for
# every legal label, so they take no gradient.
SYNTHETIC_CODE_SCHEME = "axis_orthonormal"
# Every code row lives on axes 0..len(SYNTHETIC_CODE_VOCAB)-1, so a code
# slot's pooled channel (a mean of code rows, renormalised) is zero outside
# those axes for every legal label.  The model's projection therefore reads
# only that prefix of the two code slots: (D - SYNTHETIC_CODE_DIM) columns per
# code slot would otherwise be weights that never see a nonzero input and
# never take a gradient (2 x 760 x latent_dim of them at D = 768 -- 2% of the
# v24 model, measured dead in its Adam state).
SYNTHETIC_CODE_DIM = len(SYNTHETIC_CODE_VOCAB)
CODE_SLOTS: tuple[int, ...] = (SLOT_DIRECTION, SLOT_HANDS)


def slot_channel_widths(embedding_dim: int) -> tuple[int, ...]:
    """Columns of each slot's channel the projection consumes, in
    ACTION_LABEL_SLOTS order: the full ``embedding_dim`` for a T5-encoded slot,
    :data:`SYNTHETIC_CODE_DIM` for a code slot."""
    return tuple(
        SYNTHETIC_CODE_DIM if slot in CODE_SLOTS else int(embedding_dim)
        for slot in range(len(ACTION_LABEL_SLOTS))
    )


def projection_input_dim(embedding_dim: int) -> int:
    """Width of the compacted channel vector ``action_label_projection`` reads."""
    return sum(slot_channel_widths(embedding_dim))


def compact_slot_channels(channels: np.ndarray) -> np.ndarray:
    """``(S, D)`` channels -> the ``(projection_input_dim(D),)`` vector the model
    projects: each slot's channel cut to :func:`slot_channel_widths`, concatenated
    in slot order.  The numpy definition the model's tensor slicing mirrors."""
    channels = np.asarray(channels)
    if channels.ndim != 2 or channels.shape[0] != len(ACTION_LABEL_SLOTS):
        raise ValueError(f"channels must be ({len(ACTION_LABEL_SLOTS)}, D), got {channels.shape}")
    widths = slot_channel_widths(channels.shape[1])
    for slot in CODE_SLOTS:
        if np.any(channels[slot][SYNTHETIC_CODE_DIM:] != 0):
            raise ValueError(
                f"slot {ACTION_LABEL_SLOTS[slot]!r} has energy outside the "
                f"{SYNTHETIC_CODE_DIM} code axes; its rows are not code rows"
            )
    return np.concatenate([channels[slot][:width] for slot, width in enumerate(widths)])


def synthetic_code_axis(word: str) -> int:
    """The coordinate :data:`SYNTHETIC_CODE_VOCAB` token *word* occupies."""
    try:
        return SYNTHETIC_CODE_VOCAB.index(word)
    except ValueError:
        raise ValueError(f"{word!r} does not carry a synthetic code row") from None


def synthetic_code_rows(embedding_dim: int) -> np.ndarray:
    """The ``(len(SYNTHETIC_CODE_VOCAB), embedding_dim)`` code block.

    Row *k* is the unit vector on axis *k*, so the rows are orthonormal: every
    within-slot cosine is exactly 0 and every block is exactly conditioned.
    """
    dim = int(embedding_dim)
    if dim < len(SYNTHETIC_CODE_VOCAB):
        raise ValueError(
            f"embedding_dim {dim} cannot hold {len(SYNTHETIC_CODE_VOCAB)} orthonormal "
            "code rows"
        )
    rows = np.zeros((len(SYNTHETIC_CODE_VOCAB), dim), dtype=np.float32)
    for index in range(len(SYNTHETIC_CODE_VOCAB)):
        rows[index, index] = 1.0
    return rows


def scatter_synthetic_code_rows(t5_rows: np.ndarray) -> np.ndarray:
    """Interleave encoded rows and code rows into one ``(V, D)`` table.

    *t5_rows* is in :data:`T5_ENCODED_VOCAB` order; the result is in
    ``CONTROLLED_VOCAB`` order, which is what a word id indexes.  One
    implementation so the builder cannot lay the table out one way and a test
    check it another.
    """
    encoded = np.asarray(t5_rows, dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(T5_ENCODED_VOCAB):
        raise ValueError(
            f"t5_rows must be ({len(T5_ENCODED_VOCAB)}, D), got {tuple(encoded.shape)}"
        )
    dim = int(encoded.shape[1])
    table = np.zeros((len(CONTROLLED_VOCAB), dim), dtype=np.float32)
    code = synthetic_code_rows(dim)
    encoded_index = {word: index for index, word in enumerate(T5_ENCODED_VOCAB)}
    for row, word in enumerate(CONTROLLED_VOCAB):
        if word in encoded_index:
            table[row] = encoded[encoded_index[word]]
        else:
            table[row] = code[synthetic_code_axis(word)]
    return table


def ordered_token_sources() -> list[dict[str, Any]]:
    """Where each vocabulary row comes from, in vocabulary order.

    A T5 token records the text it was encoded from; a synthetic token records
    the scheme and the axis instead.  Both are in the fingerprint, so a token
    that changed sides -- or a code row that moved axis -- invalidates every
    checkpoint bound to the old table, which is what it should do.
    """
    from data_loaders.truebones.truebones_utils.motion_labels import vocab_t5_text

    entries: list[dict[str, Any]] = []
    for token in CONTROLLED_VOCAB:
        if token in SYNTHETIC_CODE_VOCAB:
            entries.append({
                "token": token,
                "code": SYNTHETIC_CODE_SCHEME,
                "axis": synthetic_code_axis(token),
            })
        else:
            entries.append({"token": token, "text": vocab_t5_text(token)})
    return entries


def embedding_contract_payload(
    *,
    token_sources: Iterable[Mapping[str, Any]],
    t5_name: str,
    t5_artifact_sha256: str,
    tokenizer_class: str,
    tokenizer_version: str,
    pooling: str,
    eos_policy: str,
    vector_postprocess: str,
    embedding_dim: int,
    dtype: str,
    word_table_sha256: str,
) -> dict[str, Any]:
    """The frozen token vectors' identity: their inputs AND their content.

    ``word_table_sha256`` is what makes the resulting ``embedding_fingerprint``
    a statement about the table rather than about the recipe.  It is a required
    argument, not a derived convenience: the caller has the table in hand, and a
    contract that could be built without it would go back to fingerprinting
    metadata alone.
    """
    ordered_sources = [dict(entry) for entry in token_sources]
    if [entry.get("token") for entry in ordered_sources] != list(CONTROLLED_VOCAB):
        raise ValueError(
            "token_sources must hold one entry per CONTROLLED_VOCAB token, in "
            "vocabulary order"
        )
    return {
        "schema_version": ACTION_WORD_EMBEDDING_SCHEMA_VERSION,
        "ordered_token_sources": ordered_sources,
        "word_table_sha256": str(word_table_sha256),
        "t5_name": t5_name,
        "t5_artifact_sha256": t5_artifact_sha256,
        "tokenizer_class": tokenizer_class,
        "tokenizer_version": tokenizer_version,
        "pooling": pooling,
        "eos_policy": eos_policy,
        "vector_postprocess": vector_postprocess,
        "embedding_dim": int(embedding_dim),
        "dtype": dtype,
    }


def conditioning_contract_payload(
    *,
    embedding_fingerprint: str,
    representation: Mapping[str, Any],
) -> dict[str, Any]:
    """Inputs that determine how token vectors acquire runtime semantics."""
    return {
        "schema_version": ACTION_CONDITIONING_CONTRACT_SCHEMA_VERSION,
        "parser_contract_version": ACTION_LABEL_PARSER_CONTRACT_VERSION,
        "embedding_fingerprint": embedding_fingerprint,
        "ordered_vocab": list(CONTROLLED_VOCAB),
        "head_vocab": list(HEAD_VOCAB),
        "max_words": ACTION_LABEL_MAX_WORDS,
        "max_heads": ACTION_LABEL_MAX_HEADS,
        "canonicalization": "preserve written head order (one head order per word set and group; every head word is a head-slot member but the first one is weighted above the rest, so the order is the condition); bind directions after turn or final head; sort remaining modifiers by ordered_vocab",
        "slot_fields": ["word_ids", "word_mask", "slot_ids"],
        "slot_names": list(ACTION_LABEL_SLOTS),
        "group_is_checkpoint_local": True,
        "empty_label_semantics": "route to learned action_label_null_emb; do not encode empty text",
        "representation": dict(representation),
    }


# ---------------------------------------------------------------------------
# Slot source ranks
# ---------------------------------------------------------------------------
# Certifies every legal slot subset without enumerating the power set, and is
# what model construction checks ``latent_dim`` against.  Lives here rather than
# in the preflight tool because the training entry point has to run it too, and
# a second copy is a second definition of what "injective" means.
def numerical_rank(vectors: np.ndarray) -> tuple[int, float]:
    """Numerical row rank and the smallest/leading singular-value ratio."""
    singular = np.linalg.svd(np.asarray(vectors, dtype=np.float64), compute_uv=False)
    if not len(singular) or singular[0] == 0.0:
        return 0, 0.0
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    ratio = float(singular[rank - 1] / singular[0]) if rank else 0.0
    return rank, ratio


def slot_source_vectors(word_vectors: np.ndarray) -> dict[str, np.ndarray]:
    """The source rows each slot channel can be a normalised sum of.

    The slots partition the vocabulary (:func:`word_slots`), so every word is a
    source of exactly one block and the blocks' ranks simply add.
    """
    vectors = np.asarray(word_vectors, dtype=np.float64)
    vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
    return {
        name: vectors[[
            vocab_index[word]
            for word in CONTROLLED_VOCAB
            if slot in word_slots(word)
        ]]
        for slot, name in enumerate(ACTION_LABEL_SLOTS)
    }


def slot_source_rank_report(word_vectors: np.ndarray, latent_dim: int) -> dict[str, Any]:
    """Whether the slot channels stay separable and fit the first projection.

    If a slot's source rows are independent, two different membership vectors
    cannot produce proportional sums, so L2-normalising those sums creates
    neither a collision nor a loss of linear membership readability -- for every
    non-empty subset, not just the ones the corpus happens to spell.  Slots
    occupy disjoint blocks of the concatenation, so their ranks add, and a first
    Linear at least that wide can be injective on the whole reachable space.

    The coefficients are drawn from {0, 1, HEAD_SLOT_PRIMARY_WEIGHT} rather than
    {0, 1}, which changes nothing here and rules out one more collision: the
    only way two labels over the same words could share a head channel is the
    swap (r, 1) vs (1, r), and those are proportional only at r == 1.
    """
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    slots: dict[str, Any] = {}
    for name, vectors in slot_source_vectors(word_vectors).items():
        rank, relative_min_singular = numerical_rank(vectors)
        slots[name] = {
            "rank": rank,
            "expected_rank": int(len(vectors)),
            "full_rank": rank == int(len(vectors)),
            "relative_min_singular": relative_min_singular,
        }
    total_rank = sum(entry["rank"] for entry in slots.values())
    return {
        "slots": slots,
        "total_rank": total_rank,
        "expected_total_rank": sum(entry["expected_rank"] for entry in slots.values()),
        "full_rank": all(entry["full_rank"] for entry in slots.values()),
        "latent_dim": int(latent_dim),
        "fits_projection": total_rank <= int(latent_dim),
        "proof_scope": f"all non-empty slot subsets under max_total_words={ACTION_LABEL_MAX_WORDS}",
    }


# ---------------------------------------------------------------------------
# The frozen word table and the runtime bundle
# ---------------------------------------------------------------------------
# Selected by the geometry preflight (variant ``slot/eos_keep/center_l2``) and
# frozen here: the sidecar builder encodes with exactly these, and both
# fingerprints record them, so a rebuild that quietly switched pooling or
# postprocess could not be read as the same contract.
ACTION_WORD_EMBEDDING_POOLING = "masked_mean"
ACTION_WORD_EMBEDDING_EOS_POLICY = "keep"
ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS = "center_l2"
ACTION_WORD_EMBEDDING_KEYING = "word"
ACTION_WORD_EMBEDDING_DTYPE = "float32"

# Padding value for ``slot_ids`` past a label's last word.  Membership is
# ``word_mask & (slot_ids == slot)``; a pad that matched a real slot id would
# leave the mask as the only thing keeping padding out of a channel mean.
SLOT_PAD_ID = -1

# The checkpoint payload format that carries the two fingerprints.
# Distinct from utils.parser_util.CKPT_VERSION, which versions args.json and the
# training semantics: this one versions the .pt layout itself.
# 3: removed the persistent action_role_b_perm/action_role_b_sign buffers.
# 4: action_label_projection reads the compacted slot channels (code slots cut
#    to their SYNTHETIC_CODE_DIM axes), so its first Linear is narrower;
#    canonical_frame_projection is one Linear instead of a two-layer MLP.
ACTION_CHECKPOINT_VERSION = 4


class ActionConditioningError(RuntimeError):
    """A word table, contract or checkpoint that cannot be used as it stands."""


class ActionConditioningBundle:
    """The immutable word table and both fingerprints.

    Built once at a training entry point and handed to BOTH the loader and the
    model, so the ordered vocabulary, the slot rule and the fingerprints cannot
    drift apart between the two halves of one run.  Inference builds none: the
    model's buffers carry the same table out of the checkpoint.
    """

    __slots__ = (
        "_word_embeddings", "_embedding_contract",
        "_conditioning_contract", "_embedding_fingerprint",
        "_conditioning_contract_fingerprint", "_source",
    )

    def __init__(
        self,
        *,
        word_embeddings: np.ndarray,
        embedding_contract: Mapping[str, Any],
        conditioning_contract: Mapping[str, Any],
        source: str,
    ) -> None:
        table = np.array(word_embeddings, dtype=np.float32, copy=True)
        table.flags.writeable = False
        self._word_embeddings = table
        self._embedding_contract = dict(embedding_contract)
        self._conditioning_contract = dict(conditioning_contract)
        self._embedding_fingerprint = fingerprint(self._embedding_contract)
        self._conditioning_contract_fingerprint = fingerprint(self._conditioning_contract)
        self._source = str(source)

    @property
    def word_embeddings(self) -> np.ndarray:
        return self._word_embeddings

    @property
    def embedding_dim(self) -> int:
        return int(self._word_embeddings.shape[1])

    @property
    def ordered_vocab(self) -> tuple[str, ...]:
        return CONTROLLED_VOCAB

    @property
    def embedding_contract(self) -> dict[str, Any]:
        return dict(self._embedding_contract)

    @property
    def conditioning_contract(self) -> dict[str, Any]:
        return dict(self._conditioning_contract)

    @property
    def embedding_fingerprint(self) -> str:
        return self._embedding_fingerprint

    @property
    def conditioning_contract_fingerprint(self) -> str:
        return self._conditioning_contract_fingerprint

    @property
    def source(self) -> str:
        return self._source

    def slots_for(self, tokens: Iterable[str]) -> dict[str, tuple]:
        return action_label_slots(tokens)

    def channels_for(self, tokens: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        return assemble_slot_channels(self._word_embeddings, self.slots_for(tokens))

    def slot_source_rank_report(self, latent_dim: int) -> dict[str, Any]:
        return slot_source_rank_report(self._word_embeddings, latent_dim)

    def checkpoint_metadata(self) -> dict[str, Any]:
        """The ``action_conditioning`` block written into every checkpoint."""
        return {
            "embedding_contract": self.embedding_contract,
            "embedding_fingerprint": self.embedding_fingerprint,
            "conditioning_contract": self.conditioning_contract,
            "conditioning_contract_fingerprint": self.conditioning_contract_fingerprint,
        }


def build_action_conditioning_bundle(
    word_embeddings: np.ndarray,
    embedding_contract: Mapping[str, Any],
    *,
    source: str,
    check_token_sources: bool = True,
) -> ActionConditioningBundle:
    """Validate a frozen word table and pair it with the runtime contract.

    ``check_token_sources`` is on for anything built from a data directory: a
    table whose ``ordered_token_sources`` no longer matches this code was built
    from different text (or a different code layout) and is stale, and its
    synthetic rows have to BE the code this version writes.  It is off for a
    table that came out of a checkpoint, where the stored vectors -- not the
    current source table -- are what those weights were fitted against.
    """
    table = np.asarray(word_embeddings)
    if table.ndim != 2 or table.shape[0] != len(CONTROLLED_VOCAB):
        raise ActionConditioningError(
            f"{source}: word table must be ({len(CONTROLLED_VOCAB)}, D), got "
            f"{tuple(table.shape)}"
        )
    if not np.isfinite(np.asarray(table, dtype=np.float64)).all():
        raise ActionConditioningError(f"{source}: word table holds non-finite values")

    contract = dict(embedding_contract)
    declared_dim = int(contract.get("embedding_dim", -1))
    if declared_dim != int(table.shape[1]):
        raise ActionConditioningError(
            f"{source}: embedding_contract declares embedding_dim {declared_dim} but "
            f"the table is {int(table.shape[1])}-dimensional"
        )
    for field, expected in (
        ("schema_version", ACTION_WORD_EMBEDDING_SCHEMA_VERSION),
        ("pooling", ACTION_WORD_EMBEDDING_POOLING),
        ("eos_policy", ACTION_WORD_EMBEDDING_EOS_POLICY),
        ("vector_postprocess", ACTION_WORD_EMBEDDING_VECTOR_POSTPROCESS),
    ):
        if contract.get(field) != expected:
            raise ActionConditioningError(
                f"{source}: embedding_contract {field}={contract.get(field)!r}, but the "
                f"approved representation is {expected!r}. Rebuild the word sidecar."
            )
    # The vectors themselves, against the hash the contract commits to. Every
    # other field here describes how the table SHOULD have been made; this is
    # the only one that fails when the table is not the one that was made.
    declared_table_hash = contract.get("word_table_sha256")
    actual_table_hash = word_table_sha256(table)
    if declared_table_hash != actual_table_hash:
        raise ActionConditioningError(
            f"{source}: its embedding_contract commits to word_table_sha256 "
            f"{declared_table_hash!r}, but the vectors present hash to "
            f"{actual_table_hash}. The table was replaced or edited after the "
            "contract was written; rebuild it with "
            "tools/build_action_label_embeddings.py --force."
        )
    if check_token_sources:
        if contract.get("ordered_token_sources") != ordered_token_sources():
            raise ActionConditioningError(
                f"{source}: the token -> row source table moved since this word sidecar "
                "was built, so its vectors came from different text or a different code "
                "layout. Rebuild it with tools/build_action_label_embeddings.py --force."
            )
        # The contract says which rows are code; this checks that they ARE.
        # word_table_sha256 cannot: it is computed from the same table the
        # contract was written for, so a builder that scattered the code block
        # wrong would agree with itself. Compared by value rather than by hash
        # because the failure worth naming is WHICH token drifted.
        expected_code = synthetic_code_rows(int(table.shape[1]))
        vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
        actual_code = np.asarray(
            table[[vocab_index[word] for word in SYNTHETIC_CODE_VOCAB]],
            dtype=np.float32,
        )
        if not np.array_equal(actual_code, expected_code):
            drifted = [
                word for row, word in enumerate(SYNTHETIC_CODE_VOCAB)
                if not np.array_equal(actual_code[row], expected_code[row])
            ]
            raise ActionConditioningError(
                f"{source}: the rows of {drifted} are not the synthetic "
                f"{SYNTHETIC_CODE_SCHEME} code this version writes. Rebuild the table "
                "with tools/build_action_label_embeddings.py --force."
            )

    conditioning_contract = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(contract),
        representation=slot_channel_representation(),
    )
    return ActionConditioningBundle(
        word_embeddings=table,
        embedding_contract=contract,
        conditioning_contract=conditioning_contract,
        source=source,
    )


def action_word_embedding_payload(
    word_embeddings: np.ndarray, embedding_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """The on-disk form of the word-keyed sidecar."""
    return {
        "schema_version": ACTION_WORD_EMBEDDING_SCHEMA_VERSION,
        "keying": ACTION_WORD_EMBEDDING_KEYING,
        "ordered_vocab": list(CONTROLLED_VOCAB),
        "embeddings": np.asarray(word_embeddings, dtype=np.float32),
        "embedding_contract": dict(embedding_contract),
        "embedding_fingerprint": fingerprint(dict(embedding_contract)),
    }


def load_action_conditioning_bundle(sidecar_path) -> ActionConditioningBundle:
    """Read and fully validate the word-keyed sidecar into a runtime bundle."""
    path = pathlib.Path(sidecar_path)
    if not path.is_file():
        raise ActionConditioningError(
            f"the action word-embedding sidecar is missing at {path}, but action-label "
            "conditioning is enabled. Build it with: "
            "python tools/build_action_label_embeddings.py"
        )
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ActionConditioningError(f"{path}: not a sidecar payload dictionary")
    if int(payload.get("schema_version", -1)) != ACTION_WORD_EMBEDDING_SCHEMA_VERSION:
        raise ActionConditioningError(
            f"{path}: schema_version {payload.get('schema_version')!r}, expected "
            f"{ACTION_WORD_EMBEDDING_SCHEMA_VERSION}. A label-keyed sidecar cannot be "
            "read as a word table -- rebuild it."
        )
    if payload.get("keying") != ACTION_WORD_EMBEDDING_KEYING:
        raise ActionConditioningError(
            f"{path}: keying={payload.get('keying')!r}, expected "
            f"{ACTION_WORD_EMBEDDING_KEYING!r}"
        )
    if list(payload.get("ordered_vocab") or ()) != list(CONTROLLED_VOCAB):
        raise ActionConditioningError(
            f"{path}: its ordered vocabulary is not this code's CONTROLLED_VOCAB. A word "
            "id is a position in that list, so a stale sidecar would silently rename "
            "every token. Rebuild it."
        )
    contract = payload.get("embedding_contract")
    if not isinstance(contract, Mapping):
        raise ActionConditioningError(f"{path}: no embedding_contract block")
    stored_fingerprint = payload.get("embedding_fingerprint")
    if stored_fingerprint != fingerprint(dict(contract)):
        raise ActionConditioningError(
            f"{path}: embedding_fingerprint {stored_fingerprint!r} does not hash its own "
            "embedding_contract; the file was edited after it was written."
        )
    return build_action_conditioning_bundle(
        payload.get("embeddings"), contract, source=str(path)
    )


def validate_action_conditioning_metadata(
    metadata: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """Check a checkpoint's ``action_conditioning`` block against this code.

    Self-consistency first (each fingerprint hashes its own block, and the
    conditioning contract names the embedding contract it was derived from), then
    the part that needs no sidecar: rebuild the conditioning contract from the
    CURRENT vocabulary, parser contract and slot rule, and require the same
    fingerprint.  That is what lets inference stay independent of the data
    directory while still refusing a checkpoint whose runtime semantics this code
    no longer implements.
    """
    if not isinstance(metadata, Mapping):
        raise ActionConditioningError(f"{source}: action_conditioning is not a mapping")
    embedding_contract = metadata.get("embedding_contract")
    conditioning_contract = metadata.get("conditioning_contract")
    if not isinstance(embedding_contract, Mapping) or not isinstance(
        conditioning_contract, Mapping
    ):
        raise ActionConditioningError(
            f"{source}: action_conditioning must carry both contract blocks"
        )
    # Refused here rather than at the buffer hash below: a schema-2 contract has
    # no word_table_sha256 at all, so its embedding_fingerprint says nothing
    # about which vectors those weights were fitted on. That is a retrain, and
    # the reader deserves to be told so by name.
    declared_schema = embedding_contract.get("schema_version")
    if declared_schema != ACTION_WORD_EMBEDDING_SCHEMA_VERSION:
        raise ActionConditioningError(
            f"{source}: its embedding_contract is schema_version {declared_schema!r}, "
            f"this code writes {ACTION_WORD_EMBEDDING_SCHEMA_VERSION}. A pre-{ACTION_WORD_EMBEDDING_SCHEMA_VERSION} "
            "contract does not commit to a word_table_sha256, so its "
            "embedding_fingerprint cannot certify which vectors it was trained "
            "against. Retrain against a rebuilt word sidecar."
        )
    embedding_fp = fingerprint(dict(embedding_contract))
    conditioning_fp = fingerprint(dict(conditioning_contract))
    if metadata.get("embedding_fingerprint") != embedding_fp:
        raise ActionConditioningError(
            f"{source}: embedding_fingerprint does not hash its own embedding_contract"
        )
    if metadata.get("conditioning_contract_fingerprint") != conditioning_fp:
        raise ActionConditioningError(
            f"{source}: conditioning_contract_fingerprint does not hash its own "
            "conditioning_contract"
        )
    if conditioning_contract.get("embedding_fingerprint") != embedding_fp:
        raise ActionConditioningError(
            f"{source}: its conditioning contract was derived from a different word "
            "table than the one it records"
        )
    expected = conditioning_contract_payload(
        embedding_fingerprint=embedding_fp,
        representation=slot_channel_representation(),
    )
    if fingerprint(expected) != conditioning_fp:
        raise ActionConditioningError(
            f"{source}: its conditioning contract ({conditioning_fp}) is not the one "
            f"this code implements ({fingerprint(expected)}). The vocabulary, the parser "
            "contract or the slot layout changed since it was trained, so its weights "
            "would run under semantics they were never fitted for. Retrain, or migrate "
            "it with an explicit tool."
        )
    return {
        "embedding_contract": dict(embedding_contract),
        "embedding_fingerprint": embedding_fp,
        "conditioning_contract": dict(conditioning_contract),
        "conditioning_contract_fingerprint": conditioning_fp,
    }


def assert_bundle_matches_metadata(
    bundle: ActionConditioningBundle, metadata: Mapping[str, Any], *, source: str
) -> None:
    """Refuse a resume whose word table or runtime contract has moved."""
    validated = validate_action_conditioning_metadata(metadata, source=source)
    if validated["embedding_fingerprint"] != bundle.embedding_fingerprint:
        raise ActionConditioningError(
            f"{source} was trained on a different frozen word table "
            f"(embedding_fingerprint {validated['embedding_fingerprint']}) than "
            f"{bundle.source} provides ({bundle.embedding_fingerprint}). Resuming would "
            "re-fit the same weights onto moved word vectors. Start a new run, or "
            "restore the word sidecar that checkpoint was trained with."
        )
    if (
        validated["conditioning_contract_fingerprint"]
        != bundle.conditioning_contract_fingerprint
    ):
        raise ActionConditioningError(
            f"{source} records conditioning contract "
            f"{validated['conditioning_contract_fingerprint']}, this run assembles "
            f"{bundle.conditioning_contract_fingerprint}."
        )
