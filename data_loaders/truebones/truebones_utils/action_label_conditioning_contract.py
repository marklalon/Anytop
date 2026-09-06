"""The word-keyed action-label conditioning contract.

This module deliberately has no torch dependency.  The sidecar builder, the data
loader, model construction and the checkpoint loader all import these helpers, so
role assignment, slot layout and the two fingerprints cannot acquire competing
definitions across those four call sites; the model mirrors ``assemble_slot_channels``
on tensors against the very ``slot_ids`` produced here.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib
from typing import Any, Iterable, Mapping

import numpy as np

from data_loaders.truebones.truebones_utils.motion_labels import (
    ACTION_LABEL_MAX_HEADS,
    ACTION_LABEL_MAX_WORDS,
    ACTION_GROUPS,
    CONTROLLED_VOCAB,
    DIRECTION_VOCAB,
    STATE_VOCAB,
    head_words_in,
)


# 3: the embedding contract carries word_table_sha256, so the embedding
# fingerprint covers the VECTORS and not only the inputs that should have
# produced them.  A schema-2 sidecar or checkpoint has no such field and its
# fingerprint certifies nothing about its table, so it is refused rather than
# read under a guarantee it cannot make.
ACTION_WORD_EMBEDDING_SCHEMA_VERSION = 3
ACTION_CONDITIONING_CONTRACT_SCHEMA_VERSION = 1
ACTION_LABEL_PARSER_CONTRACT_VERSION = 1
ROLE_B_ARTIFACT_SCHEMA_VERSION = 1
ROLE_NONE = 0
ROLE_HEAD_1 = 1

# The fixed role transform is derived on demand from a committed namespace
# (~1 ms) instead of being carried as a checked-in array file.  What pins the
# material is not a file but ROLE_B_MATERIAL_SHA256 below: any edit to the
# namespace, the dimension or the derivation changes the hash and makes
# role_b_material() fail loudly, and the same hash travels inside
# conditioning_contract_payload, so a checkpoint trained against other material
# is rejected on load exactly as before.
ROLE_B_NAMESPACE = "anytop/action-label/role-b/v1/t5-base/768"
ROLE_B_EMBEDDING_DIM = 768
ROLE_B_MATERIAL_SHA256 = (
    "0204f95ca92d163554ed17bc8ff22ee2858ae128c2691b4893cc0f9c958c4c2b"
)
ROLE_B_CONSTRUCTION = (
    "Derived on demand: indices sorted by SHA-256(namespace/perm/index), signs "
    "from SHA-256(namespace/sign/output_index) byte-0 parity. Pure stdlib and "
    "integer-only, so every host reproduces the same material; the result is "
    "checked against the committed ROLE_B_MATERIAL_SHA256 before use. A "
    "checkpoint keeps its own perm/sign, which stay authoritative for it."
)

# Role slots.  The approved representation gives each slot its own conditioning
# channel, so a word's contribution depends on ITS slot only -- appending
# modifiers cannot shrink the head or direction axis, which is the property the
# one-vector weighted mean could not have at any weight setting.
SLOT_HEAD = 0
SLOT_DIRECTION = 1
SLOT_MODIFIER = 2
ACTION_LABEL_SLOTS: tuple[str, ...] = ("head", "direction", "modifier")


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


def action_order_enabled(action_group: str, tokens: Iterable[str]) -> bool:
    """Whether one label carries an ordered two-state transition."""
    ordered_tokens = tuple(tokens)
    if action_group not in ACTION_GROUPS:
        raise ValueError(f"unknown action_group: {action_group!r}")
    heads = tuple(head_words_in(ordered_tokens))
    return (
        action_group == "transition"
        and len(heads) == ACTION_LABEL_MAX_HEADS
        and heads[0] != "turn"
    )


def action_label_slots(action_group: str, tokens: Iterable[str]) -> dict[str, tuple]:
    """Map parsed tokens to the IDs/masks the future loader will emit.

    This is the canonical role-assignment implementation.  It is usable before
    the model work and later becomes the common implementation for training and
    inference rather than being copied into either path.
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

    enabled = action_order_enabled(action_group, ordered_tokens)
    head_positions = tuple(
        index for index, word in enumerate(ordered_tokens) if word in STATE_VOCAB
    )
    order_positions = frozenset(head_positions if enabled else ())
    second_head = head_positions[1] if enabled else None
    return {
        "word_ids": tuple(vocab_index[word] for word in ordered_tokens),
        "role_ids": tuple(
            ROLE_HEAD_1 if index == second_head else ROLE_NONE
            for index in range(len(ordered_tokens))
        ),
        "word_mask": tuple(True for _ in ordered_tokens),
        "order_head_mask": tuple(
            index in order_positions for index in range(len(ordered_tokens))
        ),
        "slot_ids": tuple(word_slot(word) for word in ordered_tokens),
    }


def word_slot(word: str) -> int:
    """Which conditioning channel one vocabulary word feeds."""
    if word in STATE_VOCAB:
        return SLOT_HEAD
    if word in DIRECTION_VOCAB:
        return SLOT_DIRECTION
    if word not in CONTROLLED_VOCAB:
        raise ValueError(f"unknown action-label token: {word!r}")
    return SLOT_MODIFIER


def assemble_slot_channels(
    word_vectors: np.ndarray,
    slots: Mapping[str, tuple],
    role_b_perm: Iterable[int],
    role_b_sign: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one label's slot assignment into its (S, D) conditioning channels.

    This is the ONLY implementation of the approved representation: the sidecar
    builder, the geometry preflight, the loader and the model all call it (the
    model mirrors it on tensors, against the same slot ids) so a channel cannot
    acquire two definitions.

    Each slot holds the mean of its member word vectors, L2-normalised, with the
    committed ``R_B`` applied to the ``ROLE_HEAD_1`` word before the head mean.
    Normalising per slot is what makes the head axis independent of how many
    modifiers the label spells; an absent slot is a zero row flagged in the
    returned mask, never a renormalisation of the others.
    """
    vectors = np.asarray(word_vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"word_vectors must be (V, D), got {vectors.shape}")
    perm = np.asarray(list(role_b_perm), dtype=np.int64)
    sign = np.asarray(list(role_b_sign), dtype=np.float64)
    if perm.shape != (vectors.shape[1],) or sign.shape != (vectors.shape[1],):
        raise ValueError("R_B perm/sign must match the embedding dimension")

    word_ids = tuple(slots["word_ids"])
    role_ids = tuple(slots["role_ids"])
    slot_ids = tuple(slots["slot_ids"])
    channels = np.zeros((len(ACTION_LABEL_SLOTS), vectors.shape[1]), dtype=np.float64)
    present = np.zeros(len(ACTION_LABEL_SLOTS), dtype=bool)
    for slot in range(len(ACTION_LABEL_SLOTS)):
        members = [
            sign * vectors[word_id][perm] if role == ROLE_HEAD_1 else vectors[word_id]
            for word_id, role, assigned in zip(word_ids, role_ids, slot_ids)
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
        "kind": "role_slot_channels",
        "slots": list(ACTION_LABEL_SLOTS),
        "slot_assignment": (
            "head = STATE_VOCAB member; direction = DIRECTION_VOCAB member; "
            "modifier = every other vocabulary word"
        ),
        "slot_aggregation": "mean of member word vectors, then L2 normalisation",
        "role_transform": "R_B applied to the ROLE_HEAD_1 word before the head-slot mean",
        "absent_slot": "zero row, reported in slot_mask; never renormalises the other slots",
        "channel_layout": "concatenated in ACTION_LABEL_SLOTS order",
        "per_word_weights": None,
    }


def role_b_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash the material role transform, excluding descriptive metadata."""
    material = {
        "schema_version": int(payload["schema_version"]),
        "embedding_dim": int(payload["embedding_dim"]),
        "perm": [int(value) for value in payload["perm"]],
        "sign": [int(value) for value in payload["sign"]],
    }
    return fingerprint(material)


@functools.lru_cache(maxsize=None)
def _derive_role_b(namespace: str, embedding_dim: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Derive the signed permutation from a namespace, deterministically.

    Integer-only and stdlib-only: SHA-256 digests are byte-exact everywhere and
    the sort keys are distinct, so ``sorted`` is order-stable across hosts,
    Python builds and runs.
    """
    def key(kind: str, index: int) -> bytes:
        return hashlib.sha256(f"{namespace}/{kind}/{index}".encode("utf-8")).digest()

    perm = tuple(sorted(range(embedding_dim), key=lambda index: key("perm", index)))
    sign = tuple(1 if key("sign", index)[0] & 1 else -1 for index in range(embedding_dim))
    return perm, sign


def validate_role_b_payload(
    payload: Mapping[str, Any],
    expected_dim: int | None = None,
    *,
    source: str = "R_B payload",
) -> dict[str, Any]:
    """Fully validate one signed-permutation payload, wherever it came from.

    Used for material that is not derived here -- a checkpoint's stored buffers,
    an exported audit copy -- so those paths cannot skip the structural and hash
    checks the derivation itself gets.
    """
    if int(payload.get("schema_version", -1)) != ROLE_B_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported R_B schema in {source}")

    dim = int(payload.get("embedding_dim", -1))
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(
            f"R_B dimension {dim} does not match embedding dimension {expected_dim}"
        )
    perm = np.asarray(payload.get("perm"), dtype=np.int64)
    sign = np.asarray(payload.get("sign"), dtype=np.int8)
    if perm.shape != (dim,) or set(perm.tolist()) != set(range(dim)):
        raise ValueError(f"R_B perm in {source} is not a permutation of 0..{dim - 1}")
    if sign.shape != (dim,) or not np.isin(sign, (-1, 1)).all():
        raise ValueError(f"R_B sign in {source} must contain exactly +/-1 values")

    actual_hash = role_b_payload_hash(payload)
    if payload.get("material_sha256") != actual_hash:
        raise ValueError(
            f"R_B material hash mismatch in {source}: "
            f"stored={payload.get('material_sha256')!r}, actual={actual_hash}"
        )
    return dict(payload)


def role_b_material(expected_dim: int | None = None) -> dict[str, Any]:
    """Return the committed role transform, derived on demand.

    The sidecar builder, the geometry preflight and model construction all call
    this instead of reading an artifact file.  Derivation is ~1 ms and cached,
    and the result is pinned to ``ROLE_B_MATERIAL_SHA256``: changing the
    namespace, the dimension or the derivation fails here rather than silently
    re-defining what ``ROLE_HEAD_1`` means.
    """
    perm, sign = _derive_role_b(ROLE_B_NAMESPACE, ROLE_B_EMBEDDING_DIM)
    payload = {
        "schema_version": ROLE_B_ARTIFACT_SCHEMA_VERSION,
        "name": "R_B",
        "construction": ROLE_B_CONSTRUCTION,
        "namespace": ROLE_B_NAMESPACE,
        "embedding_dim": ROLE_B_EMBEDDING_DIM,
        "perm": list(perm),
        "sign": list(sign),
    }
    payload["material_sha256"] = role_b_payload_hash(payload)
    validate_role_b_payload(payload, expected_dim, source="derived R_B material")
    if payload["material_sha256"] != ROLE_B_MATERIAL_SHA256:
        raise ValueError(
            "derived R_B material does not match the committed hash: "
            f"derived={payload['material_sha256']}, "
            f"committed={ROLE_B_MATERIAL_SHA256}. The role transform is frozen; "
            "a deliberate change needs a new namespace, a new committed hash and "
            "a new conditioning contract fingerprint."
        )
    return payload


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
    role_b_material_sha256: str,
    representation: Mapping[str, Any],
    role_gate: str = "transition && two_heads && first_head != turn",
) -> dict[str, Any]:
    """Inputs that determine how token vectors acquire runtime semantics."""
    return {
        "schema_version": ACTION_CONDITIONING_CONTRACT_SCHEMA_VERSION,
        "parser_contract_version": ACTION_LABEL_PARSER_CONTRACT_VERSION,
        "embedding_fingerprint": embedding_fingerprint,
        "ordered_vocab": list(CONTROLLED_VOCAB),
        "state_vocab": list(STATE_VOCAB),
        "max_words": ACTION_LABEL_MAX_WORDS,
        "max_heads": ACTION_LABEL_MAX_HEADS,
        "canonicalization": "preserve head order; bind directions after turn or final head; sort remaining modifiers by ordered_vocab",
        "slot_fields": ["word_ids", "role_ids", "word_mask", "order_head_mask", "slot_ids"],
        "slot_names": list(ACTION_LABEL_SLOTS),
        "group_is_checkpoint_local": True,
        "empty_label_semantics": "route to learned action_label_null_emb; do not encode empty text",
        "role_gate": role_gate,
        "role_b_material_sha256": role_b_material_sha256,
        "role_ids": {"NONE": ROLE_NONE, "HEAD_1": ROLE_HEAD_1},
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


def slot_source_vectors(
    word_vectors: np.ndarray,
    role_b_perm: Iterable[int],
    role_b_sign: Iterable[int],
) -> dict[str, np.ndarray]:
    """The source rows each slot channel can be a normalised sum of.

    The head sources carry both the plain and the ``R_B``-transformed version of
    every state word, so the independence argument covers ordered transitions as
    well as plain ones.
    """
    vectors = np.asarray(word_vectors, dtype=np.float64)
    perm = np.asarray(list(role_b_perm), dtype=np.int64)
    sign = np.asarray(list(role_b_sign), dtype=np.float64)
    vocab_index = {word: index for index, word in enumerate(CONTROLLED_VOCAB)}
    head = vectors[[vocab_index[word] for word in STATE_VOCAB]]
    return {
        "head": np.concatenate((head, sign * head[:, perm]), axis=0),
        "direction": vectors[[vocab_index[word] for word in DIRECTION_VOCAB]],
        "modifier": vectors[[
            vocab_index[word]
            for word in CONTROLLED_VOCAB
            if word_slot(word) == SLOT_MODIFIER
        ]],
    }


def slot_source_rank_report(
    word_vectors: np.ndarray,
    role_b_perm: Iterable[int],
    role_b_sign: Iterable[int],
    latent_dim: int,
) -> dict[str, Any]:
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
    for name, vectors in slot_source_vectors(word_vectors, role_b_perm, role_b_sign).items():
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
ACTION_CHECKPOINT_VERSION = 2


class ActionConditioningError(RuntimeError):
    """A word table, contract or checkpoint that cannot be used as it stands."""


class ActionConditioningBundle:
    """The immutable word table, role material and both fingerprints.

    Built once at a training entry point and handed to BOTH the loader and the
    model, so the ordered vocabulary, the slot rule and the fingerprints cannot
    drift apart between the two halves of one run.  Inference builds none: the
    model's buffers carry the same table out of the checkpoint.
    """

    __slots__ = (
        "_word_embeddings", "_role_b", "_embedding_contract",
        "_conditioning_contract", "_embedding_fingerprint",
        "_conditioning_contract_fingerprint", "_source",
    )

    def __init__(
        self,
        *,
        word_embeddings: np.ndarray,
        role_b: Mapping[str, Any],
        embedding_contract: Mapping[str, Any],
        conditioning_contract: Mapping[str, Any],
        source: str,
    ) -> None:
        table = np.array(word_embeddings, dtype=np.float32, copy=True)
        table.flags.writeable = False
        self._word_embeddings = table
        self._role_b = dict(role_b)
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
    def role_b_perm(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._role_b["perm"])

    @property
    def role_b_sign(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._role_b["sign"])

    @property
    def role_b_material_sha256(self) -> str:
        return str(self._role_b["material_sha256"])

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

    def slots_for(self, action_group: str, tokens: Iterable[str]) -> dict[str, tuple]:
        return action_label_slots(action_group, tokens)

    def channels_for(
        self, action_group: str, tokens: Iterable[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        return assemble_slot_channels(
            self._word_embeddings,
            self.slots_for(action_group, tokens),
            self.role_b_perm,
            self.role_b_sign,
        )

    def slot_source_rank_report(self, latent_dim: int) -> dict[str, Any]:
        return slot_source_rank_report(
            self._word_embeddings, self.role_b_perm, self.role_b_sign, latent_dim
        )

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

    try:
        role_b = role_b_material(expected_dim=int(table.shape[1]))
    except ValueError as exc:
        # The role transform is committed at ONE dimension: a table of another
        # width has no ROLE_HEAD_1 to apply, so this is a wrong-encoder error
        # rather than something to derive around.
        raise ActionConditioningError(
            f"{source}: {exc}. The committed role transform is "
            f"{ROLE_B_EMBEDDING_DIM}-dimensional ({ROLE_B_NAMESPACE}), so the word "
            "table has to come from that encoder."
        ) from exc
    conditioning_contract = conditioning_contract_payload(
        embedding_fingerprint=fingerprint(contract),
        role_b_material_sha256=role_b["material_sha256"],
        representation=slot_channel_representation(),
    )
    return ActionConditioningBundle(
        word_embeddings=table,
        role_b=role_b,
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
    CURRENT vocabulary, parser contract, slot rule and ``R_B``, and require the
    same fingerprint.  That is what lets inference stay independent of the data
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
        role_b_material_sha256=role_b_material()["material_sha256"],
        representation=slot_channel_representation(),
    )
    if fingerprint(expected) != conditioning_fp:
        raise ActionConditioningError(
            f"{source}: its conditioning contract ({conditioning_fp}) is not the one "
            f"this code implements ({fingerprint(expected)}). The vocabulary, the parser "
            "contract, the slot layout or the role transform changed since it was "
            "trained, so its weights would run under semantics they were never fitted "
            "for. Retrain, or migrate it with an explicit tool."
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
