from __future__ import annotations

import json
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
    ACTION_LABELS_FILE,
)


MOTION_METADATA_SCHEMA_VERSION = 6

# ---------------------------------------------------------------------------
# Action groups + controlled label vocabulary  (action_labels.jsonl)
# ---------------------------------------------------------------------------
# Two fields carry the action signal:
#
#   action_group  -- one of ACTION_GROUPS. Partitions the dataset; each group
#                    trains its own model.
#   action_label  -- CONTROLLED KEYWORDS from the vocabulary below, in canonical
#                    order, comma-separated ("run, forward, left, fast").
#                    Conditions the model through T5. May be empty (= no
#                    condition, routed to the learned null embedding).
#
# The vocabulary is a CONTROLLED VOCABULARY, *not* a set of mutually exclusive
# classes: a label names as many words as apply ("idle, roar"), and naming only
# part of what a clip does is legal ("run" with no direction = the marginal over
# directions, a defined state and not a defect).
#
# Labels are keywords, not prose: mean-pooled T5 dilutes a modifier in proportion
# to how much other text surrounds it, and under prose a direction word was the
# least separable signal in the vector. Keywords keep each word's signal intact,
# so direction stays as distinct as the action axis.

ACTION_GROUPS: tuple[str, ...] = ("locomotion", "stationary", "transition")

# The action vocabulary (flat -- no core/detail split). Every word reaches the
# model through the same path, the frozen-T5 embedding of the label text, so a
# rare word is just a point next to its pretrained neighbours.
#
# ADMISSION RULE: a word stays in only if the corpus shows variation
# attributable to it after species, group and base action are held fixed; words
# that fail that test are dropped rather than carried as noise. The table is a
# maintained controlled set, not a fixed list -- it changes as the corpus grows.
#
# TUPLE ORDER IS THE CANONICAL SPELLING ORDER FOR MODIFIERS ONLY. Head words
# keep the written (time) order; direction words bind after a ``turn`` head (or
# the last head); the rest follow this tuple's order -- one combination, exactly
# one spelling. See canonical_action_label.
#
# The tuple is ordered in ROLE BLOCKS (A basic mode, B how it is executed,
# C..I secondary actions, states, manners, affect, activities, dance qualifiers
# and equipment). DIRECTION_VOCAB is a separate axis, not a block. The blocks
# order the vocabulary only; nothing weights a word by its block.
ACTION_VOCAB: tuple[str, ...] = (
    # -- block A: basic mode --
    # Travel modes first; "attack" closes the block as a mode of its own, so
    # "run, attack" still leads with the gait and "attack, fast" does not invert.
    "idle", "walk", "run", "fly", "swim", "crawl", "jump", "turn",
    "fall", "roll", "attack",
    # -- block B: how that mode is executed (gait, speed, wing state) --
    "trot", "fast", "strafe", "glide", "slow", "retreat", "dive",
    # -- block C: secondary action layered on the mode (existing order kept) --
    "bite", "roar", "eat", "die", "hurt", "getup", "rest", "look",
    "shake", "throw", "taunt", "land", "takeoff", "sit", "sleep",
    "sniff", "yawn", "catch", "sting", "kick", "spit",
    "dance", "wag", "scratch", "rear", "crouch",
    # -- block D: body states and one-shot posture changes --
    "hover", "burrow", "laydown", "sitdown", "dead", "spawn", "work",
    "lift", "pickup", "putdown",
    # -- block E: how a strike is delivered (manner before strike type) --
    "spin", "flip", "twist", "charge",
    "headbutt", "punch", "swat", "slash", "stab", "smash", "swipe", "whip",
    "block", "cast", "projectile",
    # -- block F: affect / social gesture --
    "happy", "talk", "clap", "wave", "cry", "salute",
    # -- block G: activity and object handling --
    "clean", "aim", "carry", "fishing", "cook", "reload",
    "saw", "shovel", "water", "pull", "push",
    # -- block H: which body part leads a dance --
    "footwork", "fullbody", "armwork", "sway",
    # -- block I: equipment the pose is constrained by (never the asset itself) --
    "weapon", "1hand", "2hand", "bow", "gun", "hammer", "shield",
)

# The direction axis -- travel / facing direction. Separate vocabulary from the
# ACTION_VOCAB role blocks above. Directions bind after a ``turn`` head (or the
# last head), before the remaining modifiers.
# Spelled BARE ("forward", not "leftward"): the derived adjectives collapse to
# nearly the same T5 point, while bare left/right stay distinct.
#
# up/down are DIRECTIONS (where the net travel goes), not actions -- dive stays
# an action (what the body is doing). The vertical word is spelled LAST, after
# the planar ones; at most one vertical word per label. T5 carries a direction
# as a roughly linear offset that composes with unseen actions.
DIRECTION_VOCAB: tuple[str, ...] = ("forward", "backward", "left", "right", "up", "down")

CONTROLLED_VOCAB: tuple[str, ...] = ACTION_VOCAB + DIRECTION_VOCAB

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


# ---------------------------------------------------------------------------
# Head words
# ---------------------------------------------------------------------------
# STATE_VOCAB is the closed set of HEAD words: the ones a label spells in WRITE
# order instead of vocabulary order. The test is "can the body BE IN this" --
# hover / roll / rear qualify, weapon / 1hand / forward / cast / spin do not.
# Event verbs (die, getup, spawn, land, takeoff, laydown, sitdown, lift, pickup,
# putdown) are in as well: each is the load-bearing word of a label that names
# nothing else.
#
# Head order is the clip's TIME order, the only place a transition's direction is
# recorded ("idle, attack" is a draw, "attack, idle" a sheathe). Nothing may
# reorder head words -- see canonical_action_label.
#
# "sit"/"sleep" are in (postures a body is in; a future "sleep, idle" get-up
# should carry its direction). "block" is deliberately OUT: "idle, hover, block"
# would hold three heads and break the at-most-two rule.
STATE_VOCAB: tuple[str, ...] = (
    "attack", "burrow", "crawl", "crouch", "dance", "dead", "die", "fall", "fly",
    "getup", "hover", "hurt", "idle", "jump", "land", "laydown", "lift", "pickup",
    "putdown", "rear", "rest", "roll", "run", "sit", "sitdown", "sleep", "spawn",
    "swim", "takeoff", "turn", "walk", "work",
)

_STATE_VOCAB_SET: frozenset[str] = frozenset(STATE_VOCAB)

assert _STATE_VOCAB_SET <= set(CONTROLLED_VOCAB), (
    "STATE_VOCAB must be a subset of CONTROLLED_VOCAB: "
    + str(sorted(_STATE_VOCAB_SET - set(CONTROLLED_VOCAB)))
)
assert len(_STATE_VOCAB_SET) == len(STATE_VOCAB), "STATE_VOCAB has a repeat"

# At most this many heads per label: a label names a state, or a transition
# between two states. Three would have no defined reading.
ACTION_LABEL_MAX_HEADS = 2


# ---------------------------------------------------------------------------
# token -> T5 text
# ---------------------------------------------------------------------------
# A token is the canonical ID: what the annotation writes, what keys the
# embedding sidecar, what indexes the per-word weight. The T5 TEXT is what is
# actually encoded; a missing key means "encode the token itself". The entries
# below are the tokens whose bare spelling lands in the WRONG T5 neighbourhood.
#
# Chosen by measurement (mean-centred t5-base cosines against single-word
# probes of the intended and the dominant-wrong sense): only tokens where the
# wrong sense WON, as a different referent rather than a near synonym, are
# overridden. Glued compounds are FINE -- cos("1hand", "one handed") = 0.59,
# ("laydown", "lay down") = 0.70, ("takeoff", "take off") = 0.88, vs a p95 of
# 0.12 over unrelated pairs; 1hand/2hand need the override for the numeral, not
# the fragmentation.
#
# SECOND RULE (word-keyed conditioning): an override carries only what the token
# itself contributes, NOT what a co-occurring token already spells. 1hand/2hand
# never appear without a weapon word (47/47 labels), so "weapon in one/both
# hands" (cos 0.784, the table's closest pair) made `weapon, 1hand` vs
# `weapon, 2hand` the corpus's worst near-collision; dropping the shared anchor
# takes the pair to 0.526 and leaves them carrying the COUNT only.
#
# Constraints, all asserted below: one-to-one on the EXPANDED table, no
# whitespace in a token, every key a real vocabulary word. No reverse lookup --
# this is not a synonym table; text never resolves back to a token.
_VOCAB_T5_TEXT: dict[str, str] = {
    "1hand": "one hand",                 # bare form reads as the numeral one
    "2hand": "both hands",               # bare form reads as the numeral two
    "aim": "aiming a weapon",            # bare "aim" is a goal or an ambition
    "block": "raising a guard",          # bare "block" is a brick or a city block
    "bow": "archery bow",                # bare "bow" is bending forward -- a POSE
    "burrow": "digging underground",     # bare "burrow" is the hole, not the act
    "cast": "spellcasting",              # bare "cast" is plaster, or a film cast
    "charge": "rushing forward",         # bare "charge" is voltage or a fee
    "clean": "grooming",                 # bare "clean" is the adjective, not the act
    "cry": "weeping",                    # bare "cry" reads as shouting out
    "flip": "somersault",                # bare "flip" is a coin or a switch
    "land": "touching down",             # bare "land" is terrain -- overwhelmingly
    "punch": "punching",                 # bare "punch" is the drink
    "rear": "rearing up",                # bare "rear" is the back side
    "rest": "resting",                   # bare "rest" is the remainder
    "saw": "sawing wood",                # bare "saw" is the past tense of see
    "shake": "shaking",                  # bare "shake" is a milkshake
    "shield": "shield bash",             # bare "shield" is the verb "to protect"
    "water": "watering",                 # bare "water" is the substance
    "wave": "waving a hand",             # bare "wave" is an ocean wave
    "weapon": "wielding a weapon",       # the pose constraint, not the object
}

assert set(_VOCAB_T5_TEXT) <= set(CONTROLLED_VOCAB), (
    "_VOCAB_T5_TEXT has keys that are not vocabulary tokens: "
    + str(sorted(set(_VOCAB_T5_TEXT) - set(CONTROLLED_VOCAB)))
)


def vocab_t5_text(word: str) -> str:
    """The text *word* is T5-encoded from. Identity unless overridden above."""
    return _VOCAB_T5_TEXT.get(word, word)


# One-to-one on the EXPANDED table, not on the override dict: checking the
# overrides against each other would miss a collision with an identity token
# (an override reading "run" would silently share a vector with the run token).
_EFFECTIVE_T5_TEXT: dict[str, str] = {w: vocab_t5_text(w) for w in CONTROLLED_VOCAB}
assert len(set(_EFFECTIVE_T5_TEXT.values())) == len(CONTROLLED_VOCAB), (
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

    Exact token matching: *text* is split on commas and whitespace, each piece
    must be a vocabulary token verbatim, anything else is ignored -- no synonym
    translation; inference offers an autocomplete over the vocabulary.

    Returns a SET in vocab order, not a spelling: do not feed it to
    :func:`canonical_action_label`, which needs the written head order -- use
    :func:`parse_action_label` for that.
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
    """The :data:`STATE_VOCAB` members of *words*, in the order given -- the
    clip's time order for transitions, the only record of which way they run."""
    return [word for word in words if word in _STATE_VOCAB_SET]


def parse_action_label(label: str) -> list[str]:
    """Split a label into its tokens IN WRITTEN ORDER, enforcing the contract.

    Every comma-separated piece must be a vocabulary token verbatim: no empty
    segment, no repeat, at most :data:`ACTION_LABEL_MAX_WORDS` tokens, between 1
    and :data:`ACTION_LABEL_MAX_HEADS` head words. An empty label is legal and
    parses to ``[]`` (= no condition).

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
            f"least one STATE_VOCAB word: {list(STATE_VOCAB)}"
        )
    if len(heads) > ACTION_LABEL_MAX_HEADS:
        raise ActionLabelError(
            f"action_label {label!r} names {len(heads)} head words {heads} (max "
            f"{ACTION_LABEL_MAX_HEADS}). A label names a state, or a transition "
            f"between two of them, and nothing longer has a defined reading."
        )
    return tokens


def canonical_action_label(words) -> str:
    """Spell *words* with stable head order and canonical modifier placement.

    HEAD ORDER IS NEVER TOUCHED -- it is the clip's time order, the only record
    of which way a transition runs. Directions bind after a ``turn`` head (or the
    last head) and precede other modifiers, so they qualify the motion rather
    than a trailing word (``walk, right, weapon``). Other modifiers are sorted by
    :data:`CONTROLLED_VOCAB` index: one combination, exactly one spelling.

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
    heads = [word for word in ordered if word in _STATE_VOCAB_SET]
    directions = sorted(
        (word for word in ordered if word in DIRECTION_VOCAB),
        key=_CONTROLLED_VOCAB_ORDER.__getitem__,
    )
    modifiers = sorted(
        (
            word for word in ordered
            if word not in _STATE_VOCAB_SET and word not in directions
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
# across head, direction and modifier slots, not a per-slot allowance.
ACTION_LABEL_MAX_WORDS = 8


def _fail_action_labels(line_number: int, message: str) -> None:
    import sys

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
    spell it.

    THESE ARE GATES, NOT HINTS: the vocabulary is closed and the corpus is
    spelled to match, so a warning could only buy a silent regression -- and
    reordering head words flips a transition's direction without changing
    anything a loss can see.

    An *empty* label is legal and means "no condition" -- it is routed to the
    learned null embedding, never encoded as an empty string, which would
    otherwise teach the model that empty text means any motion at all and poison
    the CFG unconditional branch. Naming only SOME of what a clip does is legal
    too ("run" with no direction): the model learns the marginal over the
    directions, which is the right answer to a query that did not ask for one.
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
            f"({', '.join(head_words_in(tokens))}) in the order they happen, "
            f"directions next to their head, then the remaining modifiers "
            f"in CONTROLLED_VOCAB order. One word combination must "
            f"have exactly one spelling, or its training mass splits across "
            f"several T5 vectors.",
        )


def _validate_head_order_consistency(rows) -> None:
    """Outside ``transition``, one word set has one head order.

    Head order is the clip's time order and carries meaning only in transitions;
    in the other groups two spellings of the same word set are an inconsistent
    annotation, and an inconsistent one would train the role transform on noise.

    *rows* is an iterable of ``(line_number, group, clip, tokens)``.
    """
    seen: dict = {}
    for line_number, group, clip, tokens in rows:
        if group == "transition" or not tokens:
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
                f"{list(first_heads)}. Head order may only differ in the "
                f"transition group, where it is the clip's time order; in "
                f"{group} a divergence is an inconsistent annotation.",
            )

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_action_labels(dataset_dir: str | Path) -> dict[str, dict[str, str]]:
    """Load the hand-maintained ``action_labels.jsonl`` sidecar.

    Each line is a JSON object
    ``{"clip": "<name>.npy", "action_group": "...", "action_label": "..."}``.
    Returns a mapping ``clip -> {"action_group": ..., "action_label": ...}``.
    Raises ``FileNotFoundError`` if the file is absent so callers fail fast rather
    than silently training without action conditioning.
    """
    labels_path = Path(dataset_dir) / ACTION_LABELS_FILE
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{ACTION_LABELS_FILE} not found at {labels_path}. Action groups and "
            f"labels are maintained by hand in this file (one "
            f'{{"clip": "<name>.npy", "action_group": "...", "action_label": "..."}} '
            f"object per line)."
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
            action_labels[str(clip)] = {
                "action_group": group,
                "action_label": label,
            }
            rows.append((line_number, group, str(clip), label.split(", ") if label else []))
    # Cross-row rule, so it can only run once the whole file is in.
    _validate_head_order_consistency(rows)
    return action_labels


def load_motion_metadata(
    dataset_dir: str | Path,
    require_action_labels: bool = True,
) -> dict[str, dict[str, object]]:
    """Load ``motion_metadata.json`` joined with per-clip action group/label.

    By default a clip present in the metadata but absent from
    ``action_labels.jsonl`` is a fatal error (the group decides which model the
    clip trains, so there is no safe default). Pass ``require_action_labels=False``
    for bookkeeping reads that only preserve / carry-forward existing metadata
    (e.g. incremental preprocessing): unlabeled clips are then kept as-is without
    the action fields instead of exiting.
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
    for motion_name, metadata in motions.items():
        if not isinstance(metadata, dict):
            continue
        action = action_labels.get(motion_name)
        if action is None:
            missing_labels.append(motion_name)
            if require_action_labels:
                continue
            # Tolerant mode: carry the entry forward untouched (no action fields).
            normalized[motion_name] = dict(metadata)
            continue
        entry = dict(metadata)
        entry["action_group"] = action["action_group"]
        entry["action_label"] = action["action_label"]
        normalized[motion_name] = entry

    if missing_labels and require_action_labels:
        preview = ", ".join(sorted(missing_labels)[:10])
        more = "" if len(missing_labels) <= 10 else f" (+{len(missing_labels) - 10} more)"
        import sys

        msg = (
            f"\n❌ {ACTION_LABELS_FILE} is missing entries for {len(missing_labels)} "
            f"clip(s): {preview}{more}\n\n"
            f"   Please open {ACTION_LABELS_FILE} and add an entry for each missing clip:\n"
            f'   {{"clip": "clip_name.npy", "action_group": "{ACTION_GROUPS[0]}", '
            f'"action_label": "run, gallops with head lowered"}}\n'
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
    ``action_labels.jsonl`` is edited -- the sidecar is the single source of truth,
    so the joined fields are dropped on the way out. (``action_tags`` and
    ``species_label`` are removed predecessors -- stripping them clears the stale
    copies earlier rebuilds baked in.)
    """
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    dropped_keys = ("action_group", "action_label", "action_tags", "species_label")
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
