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
    head_words_in,
)


# 3: the embedding contract carries word_table_sha256, so the embedding
# fingerprint covers the VECTORS and not only the inputs that should have
# produced them.  A schema-2 sidecar or checkpoint has no such field and its
# fingerprint certifies nothing about its table, so it is refused rather than
# read under a guarantee it cannot make.
ACTION_WORD_EMBEDDING_SCHEMA_VERSION = 3
ACTION_CONDITIONING_CONTRACT_SCHEMA_VERSION = 1
# 2: HANDS_VOCAB (hand0/hand1/hand2) -- exclusive, at most one per label, its
#    own slot channel; replaced weapon + 1hand/2hand in the modifier slot.
# 3: head order carries no direction.  The transition group pools its head
#    words as a set exactly like the other two groups; the signed-permutation
#    role transform (R_B) on a transition's second head, its role ids and the
#    order-head mask are gone, and a word set has one head order in every group.
ACTION_LABEL_PARSER_CONTRACT_VERSION = 3

# Slots.  The approved representation gives each slot its own conditioning
# channel, so a word's contribution depends on ITS slot only -- appending
# modifiers cannot shrink the head or direction axis, which is the property the
# one-vector weighted mean could not have at any weight setting.
#
# The hands axis has its own channel rather than riding in the modifier slot
# because, once annotated, it sits on nearly every clip of every hand-bearing
# species: pooled into the modifier mean it would halve the signal of every
# real modifier (slash, punch, fast, cast ...) on exactly those species, and
# "attack, slash" would no longer read the same with and without a hand state.
# In its own channel the other three are bit-identical either way, and the
# channel is the token's own vector (the axis admits one member), so the model
# only has to tell three points apart.
SLOT_HEAD = 0
SLOT_DIRECTION = 1
SLOT_MODIFIER = 2
SLOT_HANDS = 3
ACTION_LABEL_SLOTS: tuple[str, ...] = ("head", "direction", "modifier", "hands")


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
    property of the word alone: the label's group and the head order play no
    part, so the same label is the same condition in every group's model.
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
        "slot_ids": tuple(word_slot(word) for word in ordered_tokens),
    }


def word_slot(word: str) -> int:
    """Which conditioning channel one vocabulary word feeds."""
    if word in HEAD_VOCAB:
        return SLOT_HEAD
    if word in DIRECTION_VOCAB:
        return SLOT_DIRECTION
    if word in HANDS_VOCAB:
        return SLOT_HANDS
    if word not in CONTROLLED_VOCAB:
        raise ValueError(f"unknown action-label token: {word!r}")
    return SLOT_MODIFIER


def assemble_slot_channels(
    word_vectors: np.ndarray,
    slots: Mapping[str, tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one label's slot assignment into its (S, D) conditioning channels.

    This is the ONLY implementation of the approved representation: the sidecar
    builder, the geometry preflight, the loader and the model all call it (the
    model mirrors it on tensors, against the same slot ids) so a channel cannot
    acquire two definitions.

    Each slot holds the mean of its member word vectors, L2-normalised: a set,
    so the order the words were written in does not reach the model.
    Normalising per slot is what makes the head axis independent of how many
    modifiers the label spells; an absent slot is a zero row flagged in the
    returned mask, never a renormalisation of the others.
    """
    vectors = np.asarray(word_vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"word_vectors must be (V, D), got {vectors.shape}")

    word_ids = tuple(slots["word_ids"])
    slot_ids = tuple(slots["slot_ids"])
    channels = np.zeros((len(ACTION_LABEL_SLOTS), vectors.shape[1]), dtype=np.float64)
    present = np.zeros(len(ACTION_LABEL_SLOTS), dtype=bool)
    for slot in range(len(ACTION_LABEL_SLOTS)):
        members = [
            vectors[word_id]
            for word_id, assigned in zip(word_ids, slot_ids)
            if assigned == slot
        ]
        if not members:
            continue
        mean = np.mean(np.stack(members), axis=0)
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
            "head = HEAD_VOCAB member; direction = DIRECTION_VOCAB member; "
            "hands = HANDS_VOCAB member (at most one); "
            "modifier = every other vocabulary word"
        ),
        "slot_aggregation": "mean of member word vectors (a set: word order is not encoded), then L2 normalisation",
        "absent_slot": "zero row, reported in slot_mask; never renormalises the other slots",
        "channel_layout": "concatenated in ACTION_LABEL_SLOTS order",
        "per_word_weights": None,
    }


def embedding_contract_payload(
    *,
    token_to_text: Mapping[str, str],
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
    ordered_mapping = [
        {"token": token, "text": token_to_text[token]} for token in CONTROLLED_VOCAB
    ]
    if set(token_to_text) != set(CONTROLLED_VOCAB):
        raise ValueError("token_to_text must cover CONTROLLED_VOCAB exactly")
    return {
        "schema_version": ACTION_WORD_EMBEDDING_SCHEMA_VERSION,
        "ordered_token_text": ordered_mapping,
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
        "canonicalization": "preserve written head order (one head order per word set and group; the order itself is not encoded); bind directions after turn or final head; sort remaining modifiers by ordered_vocab",
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
    """The source rows each slot channel can be a normalised sum of."""
    vectors = np.asarray(word_vectors, dtype=np.float64)
    vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
    return {
        "head": vectors[[vocab_index[word] for word in HEAD_VOCAB]],
        "direction": vectors[[vocab_index[word] for word in DIRECTION_VOCAB]],
        "modifier": vectors[[
            vocab_index[word]
            for word in CONTROLLED_VOCAB
            if word_slot(word) == SLOT_MODIFIER
        ]],
        "hands": vectors[[vocab_index[word] for word in HANDS_VOCAB]],
    }


def slot_source_rank_report(word_vectors: np.ndarray, latent_dim: int) -> dict[str, Any]:
    """Whether the slot channels stay separable and fit the first projection.

    If a slot's source rows are independent, two different 0/1 membership vectors
    cannot produce proportional sums, so L2-normalising those sums creates
    neither a collision nor a loss of linear membership readability -- for every
    non-empty subset, not just the ones the corpus happens to spell.  Slots
    occupy disjoint blocks of the concatenation, so their ranks add, and a first
    Linear at least that wide can be injective on the whole reachable space.
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
ACTION_CHECKPOINT_VERSION = 3


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
    check_token_text: bool = True,
) -> ActionConditioningBundle:
    """Validate a frozen word table and pair it with the runtime contract.

    ``check_token_text`` is on for anything built from a data directory: a table
    whose ``ordered_token_text`` no longer matches this code's ``_VOCAB_T5_TEXT``
    was encoded from different text and is stale.  It is off for a table that
    came out of a checkpoint, where the stored vectors -- not the current text
    table -- are what those weights were fitted against.
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
    if check_token_text:
        from data_loaders.truebones.truebones_utils.motion_labels import vocab_t5_text

        expected_text = [
            {"token": token, "text": vocab_t5_text(token)} for token in CONTROLLED_VOCAB
        ]
        if contract.get("ordered_token_text") != expected_text:
            raise ActionConditioningError(
                f"{source}: the token -> T5 text table moved since this word sidecar "
                "was built, so its vectors were encoded from different text. Rebuild "
                "it with tools/build_action_label_embeddings.py --force."
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
