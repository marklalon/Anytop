from __future__ import annotations

import json
import sys
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
    ACTION_LABELS_FILE,
)

# Key in action_labels.jsonl holding the per-clip loop verdict. An annotation,
# not a measurement: preprocessing proposes it, a person verifies or flips it
# in dataset/review, and nothing downstream re-derives it.
LOOP_FLAG_KEY = "is_loop"

# Flag a prefill tool sets on a row whose label IT wrote (``tools/prefill_*``).
# Just ``true``: what the label said before and what the measurement was live
# in the run's output, not in the sidecar. The row's ``reviewed`` goes to false
# at the same time, so the review UI lists it as unverified although it was
# once signed off.
AUTOFILL_KEY = "autofill"


# Bump when the on-disk shape of motion_metadata.json changes: a field is
# added, dropped, or moves between this file and action_labels.jsonl.
MOTION_METADATA_SCHEMA_VERSION = 8

# Auxiliary action groups (schema 8) are RETIRED: they let a clip join another
# group's training pool back when each group trained its own model, and
# --action_group all makes that a no-op -- every clip is already in the corpus.
# The key is still stripped from any metadata.json an older build wrote, and
# ignored (like any unknown key) in action_labels.jsonl.
_RETIRED_AUX_ACTION_GROUPS_KEY = "aux_action_groups"

# ---------------------------------------------------------------------------
# Action groups + controlled label vocabulary  (action_labels.jsonl)
# ---------------------------------------------------------------------------
# Two fields carry the action signal:
#
#   action_group  -- one of ACTION_GROUPS; partitions the dataset, each group
#                    trains its own model.
#   action_label  -- CONTROLLED KEYWORDS from the vocabulary below, in canonical
#                    order ("run, forward, left, fast"); may be empty (= no
#                    condition, routed to the learned null embedding).
#
# The vocabulary is CONTROLLED, not mutually exclusive classes: a label names
# as many words as apply ("idle, roar"), and naming only part of what a clip
# does is legal ("run" with no direction = the marginal over directions).
#
# Labels are keywords, not prose: mean-pooled T5 dilutes a modifier in
# proportion to the surrounding text; keywords keep each word's signal intact.

ACTION_GROUPS: tuple[str, ...] = ("locomotion", "stationary", "transition")

# The action vocabulary (flat -- no core/detail split). Every word reaches the
# model through the same frozen-T5 embedding, so a rare word is just a point
# next to its pretrained neighbours.
#
# ADMISSION RULE: a word stays in only if the corpus shows variation
# attributable to it after species, group and base action are held fixed; a
# maintained controlled set, not a fixed list.
#
# HEAD_VOCAB is the closed set of words a label can be ABOUT: a state the body
# is in (idle, run, hover, rear ...) or an event that is the whole clip (die,
# getup, land, draw, sheathe, stop ...). Words are spelled in WRITTEN order;
# tuple position decides nothing. Every head word feeds the head slot, the
# FIRST weighted above the later ones (see slot_member_weights), so head order
# IS the condition: "attack, hover" and "hover, attack" are two labels, and one
# word set has only one head order per group
# (_validate_head_order_consistency). Head order carries no DIRECTION. At most
# ACTION_LABEL_MAX_HEADS heads per label.
#
# MODIFIER_VOCAB is every other action word; ITS ORDER IS THE CANONICAL
# SPELLING ORDER: direction words bind after a ``turn`` head (or the last head),
# the modifiers follow in tuple order -- one combination, exactly one spelling
# (see canonical_action_label).
HEAD_VOCAB: tuple[str, ...] = (
    "attack", "burrow", "crawl", "die", "draw", "fall",
    "fly", "getup", "hover", "hurt", "idle", "jump", "kneel", "land", "laydown",
    "lift", "pickup", "putdown", "rear", "rest", "roll", "run", "sheathe",
    "sitdown", "spawn", "stop", "swim", "takeoff", "turn", "walk", "work",
)

# Blocks order the spelling only; nothing weights a word by its block.
MODIFIER_VOCAB: tuple[str, ...] = (
    # -- block A: how the head is executed (gait, speed, wing state) --
    "trot", "fast", "glide", "slow", "retreat", "dive", "flopping",
    # -- block B: secondary action layered on the head --
    "bite", "roar", "eat", "look", "shake", "throw", "taunt",
    "sniff", "yawn", "catch", "sting", "kick", "spit", "wag", "scratch",
    "crouch", "dead", "sit", "sleep",
    # -- block C: how a strike is delivered (manner before strike type) --
    "spin", "flip", "charge",
    "headbutt", "punch", "swat", "slash", "stab", "smash", "whip",
    "block", "cast", "projectile",
    # -- block D: affect / social gesture --
    "happy", "talk", "clap", "wave", "cry", "salute",
    # -- block E: activity and object handling --
    "aim", "carry", "fishing", "cook", "reload",
    "saw", "shovel", "pull", "push",
    # -- block F: the implement an action is performed with (never the asset
    # itself): archery, shooting, tool work, shield bash. The word names the
    # MOTION the implement produces, never the fact of holding one -- see the
    # note under DIRECTION_VOCAB.
    "bow", "gun", "hammer", "shield",
)

ACTION_VOCAB: tuple[str, ...] = HEAD_VOCAB + MODIFIER_VOCAB

# The direction axis -- travel / facing direction. Directions bind after a
# ``turn`` head (or the last head), before the remaining modifiers, spelled
# BARE ("forward", not "leftward": the derived adjectives collapse to nearly
# the same T5 point). up/down are directions, not actions (dive stays an
# action); the vertical word is spelled LAST after the planar ones, at most
# one per label.
DIRECTION_VOCAB: tuple[str, ...] = ("forward", "backward", "left", "right", "up", "down")

# WHAT THE CHARACTER HOLDS IS NOT PART OF THE LABEL. There is no hand-state
# word and no hand slot: an armed and an unarmed clip of the same strike are
# ONE condition, and the annotation says only what the body does. A visible
# prop never changes a label. Where an implement IS the motion it reaches the
# model as a modifier (``bow`` / ``gun`` / ``hammer`` / ``shield``) -- that
# word names the strike, not the grip.
CONTROLLED_VOCAB: tuple[str, ...] = ACTION_VOCAB + DIRECTION_VOCAB

# The closed axis whose table rows are NOT T5 encodings but a synthetic
# orthonormal code (one unit axis per token; see the contract module's
# ``synthetic_code_rows``).
#
# T5 has nothing to offer this slot: the vocabulary is closed (an unknown
# token raises in ``action_label_slots``), so there is no unseen word to place
# and no data-sparse token to borrow from a neighbour. On these axes T5's
# geometry ran BACKWARDS -- measured on the old t5-base table (768d, after
# center_l2, vocabulary-wide |cos| p95 of 0.19),
#
#     left / right        +0.461      forward / backward  +0.384
#     up / down           +0.348
#
# the pairs a prompt most needs kept apart were the closest pairs in the table.
# The replacement is the geometry those axes actually want: mutually orthogonal
# unit rows, i.e. a one-hot in a rotated basis.
#
# This is a table change and not a learned ``nn.Embedding``: the slot channel
# feeds a learned Linear already, and a learned table followed by a learned
# Linear is just a learned Linear on a one-hot -- the embedding would buy no
# reachable geometry, at the cost of splitting ``assemble_slot_channels``
# across numpy and torch.
#
SYNTHETIC_CODE_VOCAB: tuple[str, ...] = DIRECTION_VOCAB
_SYNTHETIC_CODE_SET: frozenset[str] = frozenset(SYNTHETIC_CODE_VOCAB)

# The complement: the tokens the sidecar builder actually hands to T5, in
# vocabulary order. Only these take part in the ``center`` half of ``center_l2``
# -- centring a synthetic row would destroy its orthonormality, and letting the
# code rows drag the T5 mean would make the encoder's output depend on the
# vocabulary's composition.
T5_ENCODED_VOCAB: tuple[str, ...] = tuple(
    word for word in CONTROLLED_VOCAB if word not in _SYNTHETIC_CODE_SET
)

_CONTROLLED_VOCAB_ORDER: dict[str, int] = {
    word: index for index, word in enumerate(CONTROLLED_VOCAB)
}

assert len(_CONTROLLED_VOCAB_ORDER) == len(CONTROLLED_VOCAB), (
    "a word may appear only once across ACTION_VOCAB + DIRECTION_VOCAB: "
    + str(sorted({w for w in CONTROLLED_VOCAB if CONTROLLED_VOCAB.count(w) > 1}))
)
assert not any(char.isspace() for word in CONTROLLED_VOCAB for char in word), (
    "a vocabulary token must not contain whitespace -- multi-word text belongs "
    "on the T5 side only (_VOCAB_T5_TEXT): "
    + str([w for w in CONTROLLED_VOCAB if any(c.isspace() for c in w)])
)


_HEAD_VOCAB_SET: frozenset[str] = frozenset(HEAD_VOCAB)

assert _HEAD_VOCAB_SET.isdisjoint(DIRECTION_VOCAB), (
    "a direction word is not a head: "
    + str(sorted(_HEAD_VOCAB_SET & set(DIRECTION_VOCAB)))
)
assert _HEAD_VOCAB_SET.isdisjoint(MODIFIER_VOCAB), (
    "a word is a head or a modifier, never both: "
    + str(sorted(_HEAD_VOCAB_SET & set(MODIFIER_VOCAB)))
)
assert len(_HEAD_VOCAB_SET) == len(HEAD_VOCAB), "HEAD_VOCAB has a repeat"

# At most this many heads per label: a primary head, optionally qualified by
# one more. Three would have no defined reading.
ACTION_LABEL_MAX_HEADS = 2


# ---------------------------------------------------------------------------
# token -> T5 text
# ---------------------------------------------------------------------------
# A token is the canonical ID (what the annotation writes, what keys the
# embedding sidecar); the T5 TEXT is what is actually encoded. A missing key
# means "encode the token itself". The entries below are the tokens whose bare
# spelling lands in the WRONG T5 neighbourhood (chosen by measurement: only
# tokens where the wrong sense WON as a different referent). An override
# carries only what the token itself contributes, not what a co-occurring
# token already spells.
#
# Only T5-encoded tokens may appear here: a SYNTHETIC_CODE_VOCAB token is
# never encoded, so text for one would be dead weight.
#
# Constraints, all asserted below: one-to-one on the EXPANDED table, no
# whitespace in a token, every key a T5-encoded vocabulary word. No reverse
# lookup -- this is not a synonym table.
_VOCAB_T5_TEXT: dict[str, str] = {
    "aim": "aiming a weapon",            # bare "aim" is a goal or an ambition
    "block": "raising a guard",          # bare "block" is a brick or a city block
    "bow": "archery bow",                # bare "bow" is bending forward -- a POSE
    "burrow": "digging underground",     # bare "burrow" is the hole, not the act
    "cast": "spellcasting",              # bare "cast" is plaster, or a film cast
    "charge": "rushing forward",         # bare "charge" is voltage or a fee
    "draw": "drawing a weapon",          # bare "draw" is pulling a line or a card
    "cry": "weeping",                    # bare "cry" reads as shouting out
    "flip": "somersault",                # bare "flip" is a coin or a switch
    "land": "touching down",             # bare "land" is terrain -- overwhelmingly
    "punch": "punching",                 # bare "punch" is the drink
    "rear": "rearing up",                # bare "rear" is the back side
    "rest": "resting",                   # bare "rest" is the remainder
    "saw": "sawing wood",                # bare "saw" is the past tense of see
    "shake": "shaking",                  # bare "shake" is a milkshake
    "shield": "shield bash",             # bare "shield" is the verb "to protect"
    "stop": "run to stop",               # bare "stop" is ceasing in general, or a bus stop
    "wave": "waving a hand",             # bare "wave" is an ocean wave
}

assert set(_VOCAB_T5_TEXT) <= set(T5_ENCODED_VOCAB), (
    "_VOCAB_T5_TEXT has keys that are not T5-encoded vocabulary tokens: "
    + str(sorted(set(_VOCAB_T5_TEXT) - set(T5_ENCODED_VOCAB)))
)


def vocab_t5_text(word: str) -> str:
    """The text *word* is T5-encoded from. Identity unless overridden above.

    Raises on a :data:`SYNTHETIC_CODE_VOCAB` token: those rows come from
    ``synthetic_code_rows`` and are never encoded, so a caller that reaches one
    here has forgotten to split the vocabulary.
    """
    if word in _SYNTHETIC_CODE_SET:
        raise ValueError(
            f"{word!r} carries a synthetic orthonormal code, not a T5 encoding; it "
            "has no T5 text. Iterate T5_ENCODED_VOCAB instead of CONTROLLED_VOCAB."
        )
    return _VOCAB_T5_TEXT.get(word, word)


# One-to-one on the EXPANDED table, not the override dict: checking the
# overrides against each other would miss a collision with an identity token
# (an override reading "run" would share a vector with the run token).
# Synthetic-code tokens have no text to collide.
_EFFECTIVE_T5_TEXT: dict[str, str] = {w: vocab_t5_text(w) for w in T5_ENCODED_VOCAB}
assert len(set(_EFFECTIVE_T5_TEXT.values())) == len(T5_ENCODED_VOCAB), (
    "two tokens resolve to the same T5 text: "
    + str(sorted(
        text for text in set(_EFFECTIVE_T5_TEXT.values())
        if list(_EFFECTIVE_T5_TEXT.values()).count(text) > 1
    ))
)


class ActionLabelError(ValueError):
    """A label that breaks the canonical spelling contract."""


def vocab_words_in(text: str) -> list[str]:
    """Controlled-vocabulary tokens present in *text*, in canonical vocab order.

    Exact token matching: split on commas and whitespace, each piece must be a
    vocabulary token verbatim, anything else is ignored -- no synonym
    translation.

    Returns a SET in vocab order, not a spelling: use :func:`parse_action_label`
    for the written head order.
    """
    if not text:
        return []
    present = {
        piece
        for chunk in str(text).split(",")
        for piece in chunk.split()
        if piece in _CONTROLLED_VOCAB_ORDER
    }
    return sorted(present, key=_CONTROLLED_VOCAB_ORDER.__getitem__)


def action_words_in(text: str) -> list[str]:
    """The :data:`ACTION_VOCAB` subset of :func:`vocab_words_in`, in vocab order."""
    action = set(ACTION_VOCAB)
    return [word for word in vocab_words_in(text) if word in action]


def direction_words_in(text: str) -> list[str]:
    """The :data:`DIRECTION_VOCAB` subset of :func:`vocab_words_in`, in vocab order."""
    direction = set(DIRECTION_VOCAB)
    return [word for word in vocab_words_in(text) if word in direction]


def head_words_in(words) -> list[str]:
    """The :data:`HEAD_VOCAB` members of *words*, in the order given (the
    written order: primary word first)."""
    return [word for word in words if word in _HEAD_VOCAB_SET]


def parse_action_label(label: str) -> list[str]:
    """Split a label into its tokens IN WRITTEN ORDER, enforcing the contract.

    Every comma-separated piece must be a vocabulary token verbatim: no empty
    segment, no repeat, at most :data:`ACTION_LABEL_MAX_WORDS` tokens, 1..
    :data:`ACTION_LABEL_MAX_HEADS` head words. An empty label parses to ``[]``
    (= no condition).

    Raises :class:`ActionLabelError` rather than dropping anything: a silently
    dropped token is a silently changed condition.
    """
    text = "" if label is None else str(label).strip()
    if not text:
        return []
    tokens = [piece.strip() for piece in text.split(",")]
    if any(not token for token in tokens):
        raise ActionLabelError(f"action_label {label!r} has an empty comma segment")
    unknown = [token for token in tokens if token not in _CONTROLLED_VOCAB_ORDER]
    if unknown:
        raise ActionLabelError(
            f"action_label {label!r} names token(s) {unknown} that are not in the "
            f"controlled vocabulary. Labels are exact tokens now -- there is no "
            f"synonym translation. Valid tokens: {list(CONTROLLED_VOCAB)}"
        )
    seen = [token for token in tokens if tokens.count(token) > 1]
    if seen:
        raise ActionLabelError(
            f"action_label {label!r} repeats token(s) {sorted(set(seen))}"
        )
    if len(tokens) > ACTION_LABEL_MAX_WORDS:
        raise ActionLabelError(
            f"action_label {label!r} has {len(tokens)} tokens (max "
            f"{ACTION_LABEL_MAX_WORDS}). The model truncates past this silently."
        )
    heads = head_words_in(tokens)
    if not heads:
        raise ActionLabelError(
            f"action_label {label!r} names no head word. Every label needs at "
            f"least one HEAD_VOCAB word: {list(HEAD_VOCAB)}"
        )
    if len(heads) > ACTION_LABEL_MAX_HEADS:
        raise ActionLabelError(
            f"action_label {label!r} names {len(heads)} head words {heads} (max "
            f"{ACTION_LABEL_MAX_HEADS}). A label names a state, optionally "
            f"qualified by one more, and nothing longer has a defined reading."
        )
    return tokens


def canonical_action_label(words) -> str:
    """Spell *words* with stable head order and canonical modifier placement.

    HEAD ORDER IS NEVER TOUCHED -- it is the written order (primary word first),
    and the first head word is weighted above the rest in the head slot, so
    re-sorting here would change the condition, not just the spelling.
    Directions bind after a ``turn`` head (or the last head) and precede other
    modifiers; the rest are sorted by :data:`CONTROLLED_VOCAB` index: one
    combination, exactly one spelling.

    Repeats are dropped (first occurrence wins); an out-of-vocabulary word
    raises -- dropping it would quietly delete part of the condition.
    """
    ordered: list[str] = []
    for word in words:
        if word not in ordered:
            ordered.append(word)
    unknown = [word for word in ordered if word not in _CONTROLLED_VOCAB_ORDER]
    if unknown:
        raise ActionLabelError(
            f"{unknown} are not controlled-vocabulary tokens. "
            f"Valid tokens: {list(CONTROLLED_VOCAB)}"
        )
    heads = [word for word in ordered if word in _HEAD_VOCAB_SET]
    directions = sorted(
        (word for word in ordered if word in DIRECTION_VOCAB),
        key=_CONTROLLED_VOCAB_ORDER.__getitem__,
    )
    modifiers = sorted(
        (
            word for word in ordered
            if word not in _HEAD_VOCAB_SET and word not in directions
        ),
        key=_CONTROLLED_VOCAB_ORDER.__getitem__,
    )
    if not heads:
        return ", ".join(directions + modifiers)
    direction_anchor = "turn" if "turn" in heads else heads[-1]
    canonical: list[str] = []
    for head in heads:
        canonical.append(head)
        if head == direction_anchor:
            canonical.extend(directions)
    return ", ".join(canonical + modifiers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_action_group(raw_action_group) -> str:
    """Lower-case / strip an ``action_group`` value. Never validates membership."""
    if raw_action_group is None:
        return ""
    return str(raw_action_group).strip().lower()


def normalize_action_label(raw_action_label) -> str:
    """Canonicalize an ``action_label``: ``word, word, ...``, no repeats.

    Splits on commas, collapses whitespace, dedupes case-insensitively (first
    occurrence wins). Empty stays empty (= no condition).
    """
    if raw_action_label is None:
        return ""
    seen = set()
    tokens = []
    for token in str(raw_action_label).split(","):
        token = " ".join(token.split())
        if not token or token.lower() in seen:
            continue
        seen.add(token.lower())
        tokens.append(token)
    return ", ".join(tokens)


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_motion_labels(
    object_type: str,
    motion_name: str | None = None,
    source_file: str | None = None,
) -> dict[str, object]:
    """Build the (non-action) label fields for a motion clip.

    Action group/label are not produced here — they are maintained by hand in
    ``action_labels.jsonl`` and merged in by :func:`load_motion_metadata`.
    """
    payload: dict[str, object] = {"object_type": object_type}
    if motion_name is not None:
        payload["motion_name"] = motion_name
    return payload


# Hard cap on tokens per label: a label is a compact prompt, not a caption --
# past this the T5 mean-pool dilutes the words that carry the action. Total
# across slots, not a per-slot allowance.
ACTION_LABEL_MAX_WORDS = 8


def _fail_action_labels(line_number: int, message: str) -> None:
    print(
        f"\n❌ {ACTION_LABELS_FILE}:{line_number}: {message}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def reset_action_label_warning_state() -> None:
    """No-op kept so callers do not have to care which regime they are on.

    Old validation silenced a word after printing it once per process, so tools
    auditing a second file had to clear that state. Label validation hard-fails
    on the first bad row now, so there is nothing left to reset.
    """


def _validate_action_label_entry(
    group: str, label: str, clip: str, line_number: int
) -> None:
    """Hard-fail on an ``action_labels.jsonl`` row that breaks the label contract.

    The group must be one of the three closed values (it selects which model the
    clip trains). A non-empty label must parse under :func:`parse_action_label`
    and must already be spelled the way :func:`canonical_action_label` would
    spell it. THESE ARE GATES, NOT HINTS: the vocabulary is closed and the
    corpus is spelled to match, so a warning could only buy a silent regression.

    An *empty* label is legal and means "no condition" -- routed to the learned
    null embedding, never encoded as an empty string (which would poison the CFG
    unconditional branch). Naming only SOME of what a clip does is legal too:
    the model learns the marginal.
    """
    if group not in ACTION_GROUPS:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has invalid action_group {group!r}. "
            f"Valid groups are: {list(ACTION_GROUPS)}",
        )
    if not label:
        return

    try:
        tokens = parse_action_label(label)
    except ActionLabelError as exc:
        _fail_action_labels(line_number, f"clip '{clip}': {exc}")
        return

    canonical = canonical_action_label(tokens)
    if label != canonical:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has action_label {label!r}, which is not the canonical "
            f"spelling. Write it as {canonical!r}: head words "
            f"({', '.join(head_words_in(tokens))}) as written, "
            f"directions next to their head, then the remaining modifiers "
            f"in CONTROLLED_VOCAB order. One word combination must "
            f"have exactly one spelling, or its training mass splits across "
            f"several T5 vectors.",
        )


def _validate_head_order_consistency(rows) -> None:
    """Within a group, one word set has one head order.

    The first head word is weighted above the rest, so two head orders of one
    word set are two DIFFERENT conditions; two spellings of the same word set
    inside one group would be the corpus contradicting itself. Applies to every
    group alike: head order carries no direction anywhere.

    *rows* is an iterable of ``(line_number, group, clip, tokens)``.
    """
    seen: dict = {}
    for line_number, group, clip, tokens in rows:
        if not tokens:
            continue
        key = (group, frozenset(tokens))
        heads = tuple(head_words_in(tokens))
        first = seen.get(key)
        if first is None:
            seen[key] = (heads, line_number, clip)
            continue
        first_heads, first_line, first_clip = first
        if heads != first_heads:
            _fail_action_labels(
                line_number,
                f"clip '{clip}' spells the head words of {sorted(key[1])} as "
                f"{list(heads)}, but {ACTION_LABELS_FILE}:{first_line} "
                f"('{first_clip}') spells the same word set as "
                f"{list(first_heads)}. The first head word outweighs the second "
                f"in the head slot, so these are two different conditions; "
                f"within {group} one word set must have one spelling.",
            )

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def clip_key(name: str) -> str:
    """The sidecar key for a clip: its file name WITHOUT the ``.npy`` extension."""
    name = str(name)
    return name[:-4] if name.endswith(".npy") else name


def set_loop_flag(entry: dict, verdict: bool) -> dict:
    """Return a copy of *entry* carrying ``is_loop`` = *verdict*, in its place.

    A row that already has the key keeps it where it is; a FIRST verdict goes in
    right after ``action_label``, so every row reads clip / group / label /
    is_loop whoever wrote it. Which rows get a verdict, and whether an existing
    one may be replaced, is the annotating tool's policy; this only defines
    where the key sits.
    """
    verdict = bool(verdict)
    if LOOP_FLAG_KEY in entry:
        return {**entry, LOOP_FLAG_KEY: verdict}
    rebuilt: dict[str, object] = {}
    for key, value in entry.items():
        rebuilt[key] = value
        if key == "action_label":
            rebuilt[LOOP_FLAG_KEY] = verdict
    rebuilt.setdefault(LOOP_FLAG_KEY, verdict)
    return rebuilt


def load_action_labels(dataset_dir: str | Path) -> dict[str, dict[str, object]]:
    """Load the hand-maintained ``action_labels.jsonl`` sidecar.

    Each line is a JSON object
    ``{"clip": "<name>", "action_group": "...", "action_label": "...", "is_loop": true}``,
    where ``<name>`` is the file name without its ``.npy`` extension (a row that
    still carries it is normalized to the same key). Returns a mapping
    ``clip -> {"action_group": ..., "action_label": ..., ["is_loop": ...]}``;
    the keys are ALWAYS extension-less. Raises ``FileNotFoundError`` if the file
    is absent so callers fail fast rather than silently training without action
    conditioning.

    ``is_loop`` is OPTIONAL per row: ``tools/prefill_loop_flags.py`` fills it in
    for a clip nobody has annotated yet, and a value already there is never
    overwritten. It is returned only when the row has it, so a caller can tell
    "annotated" from "not yet".
    """
    labels_path = Path(dataset_dir) / ACTION_LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{ACTION_LABELS_FILE} not found at {labels_path}. Action groups and "
            f"labels are maintained by hand in this file (one "
            f'{{"clip": "<name>", "action_group": "...", "action_label": "..."}} '
            f"object per line, <name> without the .npy extension)."
        )

    action_labels: dict[str, dict[str, str]] = {}
    rows: list[tuple[int, str, str, list[str]]] = []
    with open(labels_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{ACTION_LABELS_FILE}:{line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{ACTION_LABELS_FILE}:{line_number} must be a JSON object, "
                    f"got {type(entry).__name__}"
                )
            clip = entry.get("clip")
            if not clip:
                raise ValueError(
                    f"{ACTION_LABELS_FILE}:{line_number} is missing the 'clip' field"
                )
            clip = clip_key(clip)   # rows may still spell the .npy name; the key is the stem
            group = normalize_action_group(entry.get("action_group"))
            raw_label = entry.get("action_label")
            label = normalize_action_label(raw_label)
            # normalize_action_label silently drops repeats and blanks, so this
            # check runs on the RAW text -- past it 'walk, walk' == 'walk'.
            raw_tokens = [
                piece.strip() for piece in str(raw_label or "").split(",")
            ] if raw_label else []
            if raw_tokens and len(raw_tokens) != len(label.split(", ")):
                _fail_action_labels(
                    line_number,
                    f"clip '{clip}' has action_label {raw_label!r} with a repeated "
                    f"token or an empty comma segment",
                )
            _validate_action_label_entry(group, label, str(clip), line_number)
            row: dict[str, object] = {
                "action_group": group,
                "action_label": label,
            }
            if LOOP_FLAG_KEY in entry:
                is_loop = entry[LOOP_FLAG_KEY]
                # JSON true/false only. "true", 1 or null would each read as a
                # verdict somebody never made, and the flag decides the terminal
                # velocity row and the loop-period statistics.
                if not isinstance(is_loop, bool):
                    _fail_action_labels(
                        line_number,
                        f"clip '{clip}' has {LOOP_FLAG_KEY} {is_loop!r}; it must be "
                        f"JSON true or false (or absent, for tools/prefill_loop_flags.py "
                        f"to propose one)",
                    )
                row[LOOP_FLAG_KEY] = is_loop
            action_labels[str(clip)] = row
            rows.append((line_number, group, str(clip), label.split(", ") if label else []))
    # Cross-row rule, so it can only run once the whole file is in.
    _validate_head_order_consistency(rows)
    return action_labels


def load_motion_metadata(
    dataset_dir: str | Path,
    *,
    require_loop_flag: bool = True,
    only: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Load ``motion_metadata.json`` joined with per-clip action group/label/loop.

    A clip present in the metadata but absent from ``action_labels.jsonl`` is a
    fatal error: the group decides which model the clip trains, so there is no
    safe default -- a missing entry is always an incomplete sidecar, never a
    clip that is "labeled later". Bookkeeping-only reads that need no action
    fields use ``_load_motion_metadata_raw`` (dataset_pipeline) instead.

    ``is_loop`` is joined the same way and is just as fatal when a row lacks it:
    the flag is a prerequisite annotation (proposed by
    ``tools/prefill_loop_flags.py``, and preprocessing refuses a clip without
    one), so a clip on disk with no flag means its row was edited after the
    build. ``require_loop_flag=False`` is for a bookkeeping read that must not
    fail on such a row: the joined entry then simply has no ``is_loop`` key.

    ``only`` restricts the join (and its checks) to those motion names, for a
    caller that is about to rebuild the rest and must not trip over their rows.
    """
    metadata_path = Path(dataset_dir) / MOTION_METADATA_FILE
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    motions = payload.get("motions", payload)
    if not isinstance(motions, dict):
        return {}

    action_labels = load_action_labels(dataset_dir)

    normalized: dict[str, dict[str, object]] = {}
    missing_labels: list[str] = []
    missing_loop_flags: list[str] = []
    for motion_name, metadata in motions.items():
        if not isinstance(metadata, dict):
            continue
        if only is not None and motion_name not in only:
            continue
        # metadata keys are the motions/ file names ("<name>.npy"); the sidecar
        # is keyed by the extension-less clip name.
        action = action_labels.get(clip_key(motion_name))
        if action is None:
            missing_labels.append(motion_name)
            continue
        entry = dict(metadata)
        # The sidecar is the only source: a copy an older build baked into the
        # metadata is stale the moment the row is edited, so it never survives
        # the join (write_motion_metadata strips it on the way out too).
        entry.pop(LOOP_FLAG_KEY, None)
        entry.pop(_RETIRED_AUX_ACTION_GROUPS_KEY, None)
        entry["action_group"] = action["action_group"]
        entry["action_label"] = action["action_label"]
        if LOOP_FLAG_KEY in action:
            entry[LOOP_FLAG_KEY] = bool(action[LOOP_FLAG_KEY])
        elif require_loop_flag:
            missing_loop_flags.append(motion_name)
        normalized[motion_name] = entry

    if missing_labels:
        preview = ", ".join(sorted(missing_labels)[:10])
        more = "" if len(missing_labels) <= 10 else f" (+{len(missing_labels) - 10} more)"
        msg = (
            f"\n❌ {ACTION_LABELS_FILE} is missing entries for {len(missing_labels)} "
            f"clip(s): {preview}{more}\n\n"
            f"   Please open {ACTION_LABELS_FILE} and add an entry for each missing clip:\n"
            f'   {{"clip": "clip_name", "action_group": "{ACTION_GROUPS[0]}", '
            f'"action_label": "run, gallops with head lowered"}}\n'
            f"   (clip name without the .npy extension)\n"
        )
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)
    if missing_loop_flags:
        preview = ", ".join(sorted(missing_loop_flags)[:10])
        more = "" if len(missing_loop_flags) <= 10 else f" (+{len(missing_loop_flags) - 10} more)"
        msg = (
            f"\n❌ {ACTION_LABELS_FILE} has no {LOOP_FLAG_KEY} for {len(missing_loop_flags)} "
            f"clip(s): {preview}{more}\n\n"
            f"   The loop flag lives in {ACTION_LABELS_FILE} (proposed by "
            f"tools/prefill_loop_flags.py from the source animation, verified by hand "
            f"in dataset/review). Run\n"
            f"   python tools/prefill_loop_flags.py --dataset-dir <dataset> --raw-data-dir <raw>\n"
            f"   to fill in every row that has none, or set "
            f'"{LOOP_FLAG_KEY}": true/false on the rows yourself.\n'
        )
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)
    return normalized


def write_motion_metadata(
    save_dir: str | Path,
    motion_entries: dict[str, dict[str, object]],
    total_clips: int,
) -> Path:
    """Write ``motion_metadata.json``, stripping the joined action fields.

    ``load_motion_metadata`` joins ``action_group`` / ``action_label`` in from the
    sidecar, and every rebuild path round-trips loaded entries back through here.
    Persisting them would leave a second copy that silently diverges the moment
    ``action_labels.jsonl`` is edited -- the sidecar is the single source of
    truth, so the joined fields (including ``is_loop`` from schema 7) are
    dropped on the way out. Stripping ``action_tags``, ``species_label`` and the
    retired ``aux_action_groups`` clears the stale copies earlier rebuilds baked
    in.
    """
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    dropped_keys = (
        "action_group",
        "action_label",
        LOOP_FLAG_KEY,
        _RETIRED_AUX_ACTION_GROUPS_KEY,
        "action_tags",
        "species_label",
    )
    sanitized_entries = {
        motion_name: {
            key: value for key, value in metadata.items() if key not in dropped_keys
        }
        for motion_name, metadata in motion_entries.items()
        if isinstance(metadata, dict)
    }
    payload = {
        "schema_version": MOTION_METADATA_SCHEMA_VERSION,
        "total_clips": int(total_clips),
        "motions": dict(sorted(sanitized_entries.items())),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path
