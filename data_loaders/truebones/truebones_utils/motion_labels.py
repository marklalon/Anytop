from __future__ import annotations

import json
import re
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
#                    order, comma-separated ("run, sprint, forward, left").
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
# TUPLE ORDER IS THE CANONICAL SPELLING ORDER: a label lists its action words
# in this order, then its direction words in DIRECTION_VOCAB order, so one word
# combination has exactly one spelling ("walk, retreat, crouch, backward", never
# "crouch, retreat, walk"). _validate_action_label_entry enforces it.
#
# The modifier words (shuffle, sprint, ...) are surface forms of the base word
# they qualify AND words in their own right: a Sprint clip resolves to
# "run, sprint" while a plain Run clip stays "run". "strafe" is the exception:
# it qualifies any travel mode ("run, strafe", "fly, strafe"), so it names only
# itself and never drags in a base word.
#
# The tuple is ordered in ROLE BLOCKS: A is the basic MODE (the label's first
# word), B how that mode is executed, C a secondary action layered on top.
# DIRECTION_VOCAB is NOT one of these blocks -- it is a separate axis appended
# after every action word. Ordering is NOT a weighting: T5 mean-pools over
# tokens, so every word of a label carries the same weight wherever it sits.
ACTION_VOCAB: tuple[str, ...] = (
    # -- block A: basic mode (the label's first word) --
    # Travel modes first; "attack" closes the block as a mode of its own, so
    # "run, attack" still leads with the gait and "attack, dash" does not invert.
    "idle", "walk", "run", "fly", "swim", "crawl", "climb", "jump", "turn",
    "fall", "roll", "attack",
    # -- block B: how that mode is executed (gait, speed, wing state) --
    "trot", "sprint", "dash", "gallop", "shuffle", "strafe", "glide", "slow",
    "sneak", "retreat", "flap", "dive",
    # -- block C: secondary action layered on the mode (existing order kept) --
    "bite", "roar", "eat", "die", "hurt", "getup", "rest", "look",
    "shake", "throw", "taunt", "land", "takeoff", "sit", "sleep", "stand",
    "sniff", "stretch", "yawn", "dig", "catch", "peck", "sting", "kick", "spit",
    "drag", "dance", "breathe", "drink", "graze", "wag", "scratch",
    "rear", "crouch",
)

# The direction axis -- travel / facing direction. Separate vocabulary from the
# ACTION_VOCAB role blocks above, emitted after every action word in a label.
# Spelled BARE ("forward", not "leftward"): the derived adjectives collapse to
# nearly the same T5 point, while bare left/right stay distinct.
#
# up/down are DIRECTIONS (where the net travel goes), not actions -- climb/dive
# stay actions (what the body is doing). The vertical word is spelled LAST, after
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


# Surface forms recognized for each vocabulary word. Kept explicit rather than
# stemmed: a stemmer would drag in false positives that land silently in the
# conditioning signal. Resolves a free-text --action_label at inference, mapping
# a user's wording to the canonical words the model trained on.
_VOCAB_SURFACE_FORMS: dict[str, tuple[str, ...]] = {
    # "in place" / "stationary" are NOT idle forms: most clips are animated in
    # place, so those phrases appear as filler in run/fly descriptions and would
    # light up the idle slot.
    "idle": ("idle", "idles", "idling", "motionless", "stands still",
             "standing still", "stand still", "stays still"),
    # shuffle is a walk form AND a word of its own: the equal-length rule in
    # vocab_words_in keeps both, so "shuffles" -> "walk, shuffle".
    # "strafe" is NOT a walk form: it qualifies any travel mode, so a bare
    # strafe names only strafe and "run, strafe" stays "run, strafe".
    "walk": ("walk", "walks", "walking", "trot", "trots", "trotting",
             "pace", "paces", "pacing", "march", "marches", "marching",
             "strut", "struts", "strutting", "amble", "ambles", "ambling",
             "shuffle", "shuffles", "shuffling"),
    "run": ("run", "runs", "running", "gallop", "gallops", "galloping",
            "sprint", "sprints", "sprinting", "dash", "dashes", "dashing",
            "jog", "jogs", "jogging", "charge", "charges", "charging"),
    "fly": ("fly", "flies", "flying", "flap", "flaps", "flapping",
            "glide", "glides", "gliding", "soar", "soars", "soaring",
            "hover", "hovers", "hovering"),
    "swim": ("swim", "swims", "swimming", "paddle", "paddles", "paddling"),
    "jump": ("jump", "jumps", "jumping", "leap", "leaps", "leaping",
             "hop", "hops", "hopping", "pounce", "pounces", "pouncing",
             "bound", "bounds", "bounding"),
    # "strafe" is NOT a turn form: pure translation, facing unchanged. "circle"
    # and "bank" stay -- they do change facing.
    "turn": ("turn", "turns", "turning", "spin", "spins", "spinning",
             "rotate", "rotates", "rotating", "pivot", "pivots", "pivoting",
             "circle", "circles", "circling",
             "bank", "banks", "banking"),
    "attack": ("attack", "attacks", "attacking", "strike", "strikes",
               "striking", "lunge", "lunges", "lunging", "swipe", "swipes",
               "swiping", "slash", "slashes", "slashing", "claw", "claws",
               "clawing", "maul", "mauls", "mauling", "slam", "slams",
               "slamming", "swat", "swats", "swatting"),
    "bite": ("bite", "bites", "biting", "bit", "chomp", "chomps", "chomping",
             "snaps its jaws", "snapping its jaws"),
    "roar": ("roar", "roars", "roaring", "growl", "growls", "growling",
             "howl", "howls", "howling", "bark", "barks", "barking",
             "hiss", "hisses", "hissing", "screech", "screeches", "screeching",
             "scream", "screams", "screaming", "bellow", "bellows",
             "bellowing"),
    "eat": ("eat", "eats", "eating", "feed", "feeds", "feeding", "graze",
            "grazes", "grazing", "chew", "chews", "chewing", "devour",
            "devours", "devouring", "drink", "drinks", "drinking"),
    # "collapse" belongs to fall, not die: here it means going down.
    "die": ("die", "dies", "dying", "death", "dead", "perish", "perishes",
            "passes out"),
    "fall": ("fall", "falls", "falling", "fell", "collapse", "collapses",
             "collapsing", "tumble", "tumbles",
             "tumbling", "trip", "trips", "tripping", "stumble", "stumbles",
             "stumbling", "topple", "topples", "toppling",
             "fall down", "falls down", "falling down"),
    # "gethurt" (one token) is a legacy tag spelling; the word-boundary match
    # keeps "hurt" out of it, so it is listed explicitly.
    "hurt": ("hurt", "hurts", "gethurt", "get hurt", "get-hurt", "gets hurt",
             "injured", "wounded", "flinch", "flinches",
             "flinching", "recoil", "recoils", "recoiling", "stagger",
             "staggers", "staggering", "stunned", "limp", "limps", "limping",
             "takes a hit", "knocked back"),
    # Bare "rise"/"rising" are NOT getup forms: they denote any upward motion
    # (swimming, rearing, breathing). Only destination/down-state phrases count.
    "getup": ("getup", "get up", "gets up", "getting up", "get-up",
              "stands up", "standing up",
              "rises to stand", "rises to standing", "rises to its feet",
              "rise back up", "rises back up", "stand up",
              "recover", "recovers", "recovering",
              "wakes up", "waking up", "revive", "revives", "reviving"),
    # Bare "lie"/"lying" are NOT rest forms: in this corpus that posture is
    # nearly always death or the state a get-up departs from.
    # "sit down" is listed on BOTH rest and sit so the two keep firing together
    # (equal-length matches both survive) while the "down" inside is subsumed.
    "rest": ("rest", "rests", "resting", "lie down",
             "lies down", "lying down", "sit", "sits", "sitting", "seated",
             "sit down", "sits down", "sitting down",
             "sleep", "sleeps", "sleeping", "dozing", "napping"),
    "look": ("look", "looks", "looking", "glance", "glances", "glancing",
             "gaze", "gazes", "gazing", "observe", "observes", "observing",
             "scan", "scans", "scanning", "watch", "watches", "watching",
             "look up", "looks up", "looking up",
             "look down", "looks down", "looking down"),
    "shake": ("shake", "shakes", "shaking", "shudder", "shudders",
              "shuddering", "twitch", "twitches", "twitching", "tremble",
              "trembles", "trembling", "wag", "wags", "wagging"),
    "crouch": ("crouch", "crouches", "crouching", "squat", "squats",
               "squatting", "hunker", "hunkers", "hunkering",
               "crouch down", "crouches down", "crouching down"),
    "retreat": ("retreat", "retreats", "retreating", "backs away",
                "backing away", "backs up", "backing up"),
    "rear": ("rear", "rears", "rearing", "on its hind legs",
             "onto its hind legs", "rear up", "rears up", "rearing up"),
    "throw": ("throw", "throws", "throwing", "toss", "tosses", "tossing",
              "fling", "flings", "flinging", "hurl", "hurls", "hurling"),
    "crawl": ("crawl", "crawls", "crawling", "slither", "slithers",
              "slithering", "creep", "creeps", "creeping", "scurry",
              "scurries", "scurrying"),
    "taunt": ("taunt", "taunts", "taunting", "threaten", "threatens",
              "threatening", "intimidate", "intimidates", "intimidating"),
    # -- remaining ACTION_VOCAB words --
    "climb": ("climb", "climbs", "climbing"),
    "sneak": ("sneak", "sneaks", "sneaking", "stalk", "stalks", "stalking",
              "prowl", "prowls", "prowling"),
    "land": ("land", "lands", "landing", "touches down", "touching down"),
    "takeoff": ("take off", "takes off", "taking off", "takeoff", "lift off",
                "lifts off"),
    "dive": ("dive", "dives", "diving", "plunge", "plunges", "plunging"),
    "roll": ("roll", "rolls", "rolling"),
    "sit": ("sit", "sits", "sitting", "seated",
            "sit down", "sits down", "sitting down"),
    "sleep": ("sleep", "sleeps", "sleeping", "dozing", "napping", "asleep"),
    "stand": ("stand", "stands", "standing", "upright"),
    "sniff": ("sniff", "sniffs", "sniffing", "smell", "smells", "smelling"),
    "stretch": ("stretch", "stretches", "stretching"),
    "yawn": ("yawn", "yawns", "yawning"),
    "dig": ("dig", "digs", "digging", "burrow", "burrows", "burrowing",
            "scrape", "scrapes", "scraping"),
    "catch": ("catch", "catches", "catching", "grab", "grabs", "grabbing",
              "seize", "seizes", "seizing", "snatch", "snatches", "snatching",
              "pick up", "picks up", "picking up"),
    "peck": ("peck", "pecks", "pecking"),
    "sting": ("sting", "stings", "stinging"),
    "kick": ("kick", "kicks", "kicking", "stomp", "stomps", "stomping",
             "trample", "tramples", "trampling", "buck", "bucks", "bucking"),
    "spit": ("spit", "spits", "spitting", "spray", "sprays", "spraying",
             "breathes fire"),
    "drag": ("drag", "drags", "dragging"),
    "dance": ("dance", "dances", "dancing", "celebrate", "celebrates",
              "celebrating"),
    "breathe": ("breathe", "breathes", "breathing", "pant", "pants",
                "panting"),
    "drink": ("drink", "drinks", "drinking", "laps at water"),
    "graze": ("graze", "grazes", "grazing"),
    "flap": ("flap", "flaps", "flapping"),
    "wag": ("wag", "wags", "wagging"),
    "scratch": ("scratch", "scratches", "scratching", "groom", "grooms",
                "grooming", "rub", "rubs", "rubbing", "itch", "itches",
                "itching"),
    # -- gait modifiers (also surface forms of walk / run / fly above: "trots" -> "walk, trot") --
    "shuffle": ("shuffle", "shuffles", "shuffling"),
    "strafe": ("strafe", "strafes", "strafing"),
    "sprint": ("sprint", "sprints", "sprinting"),
    "trot": ("trot", "trots", "trotting"),
    "dash": ("dash", "dashes", "dashing"),
    "gallop": ("gallop", "gallops", "galloping"),
    "glide": ("glide", "glides", "gliding"),
    "slow": ("slow", "slowly"),
    # -- direction --
    "forward": ("forward", "forwards"),
    "backward": ("backward", "backwards"),
    "left": ("left", "leftward", "leftwards"),
    "right": ("right", "rightward", "rightwards"),
    "up": ("up", "upward", "upwards"),
    "down": ("down", "downward", "downwards"),
}

# A canonical word must always match itself: labels are written using the
# canonical spelling, so a word missing from its own surface-form list silently
# fails to match every label that uses it ("getup, lifts head" hitting nothing).
assert all(
    word in {form.lower() for form in _VOCAB_SURFACE_FORMS.get(word, ())}
    for word in CONTROLLED_VOCAB
), (
    "every vocabulary word must appear in its own surface forms: "
    + str([w for w in CONTROLLED_VOCAB
           if w not in {f.lower() for f in _VOCAB_SURFACE_FORMS.get(w, ())}])
)
assert set(_VOCAB_SURFACE_FORMS) == set(CONTROLLED_VOCAB), (
    "surface-form table and CONTROLLED_VOCAB disagree: "
    f"{set(_VOCAB_SURFACE_FORMS) ^ set(CONTROLLED_VOCAB)}"
)
# Longest-first so multi-word forms ("stands still") win over their single-word
# prefixes ("stand") when both belong to the SAME word. Precedence across
# different words is resolved by span containment in ``vocab_words_in``.
_VOCAB_MATCHERS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (
        word,
        re.compile(
            r"(?<![A-Za-z])(?:"
            + "|".join(
                re.escape(form).replace(r"\ ", r"\s+")
                for form in sorted(forms, key=len, reverse=True)
            )
            + r")(?![A-Za-z])",
            re.IGNORECASE,
        ),
    )
    for word, forms in (
        (w, _VOCAB_SURFACE_FORMS[w]) for w in CONTROLLED_VOCAB
    )
)


def vocab_words_in(text: str) -> list[str]:
    """Controlled-vocabulary words present in *text*, in canonical vocab order.

    Canonical order means ACTION_VOCAB order first, then DIRECTION_VOCAB -- i.e.
    exactly the spelling a keyword label must use, so ``", ".join(vocab_words_in(t))``
    is the canonical label for whatever *t* names.

    Matching is over the whole string, not just a prefix -- a text may name its
    action anywhere ("stands still and growls" hits idle *and* roar), and may hit
    several words at once. That is the point of a controlled vocabulary: it
    anchors recall without forcing a mutually exclusive choice.

    A word is dropped when every one of its matches sits strictly inside a longer
    match of a *different* word: "stands still" is idle, so the "stands" inside it
    must not also light up ``stand``, and "breathes fire" is spit, not breathe.
    Equal-length matches both survive, which is what keeps a modifier firing
    alongside the base word that shares its spelling ("shuffles" -> walk +
    shuffle, "grazes" -> eat + graze).
    """
    if not text:
        return []

    spans: dict[str, list[tuple[int, int]]] = {}
    for word, pattern in _VOCAB_MATCHERS:
        found = [match.span() for match in pattern.finditer(text)]
        if found:
            spans[word] = found

    def subsumed(word: str, span: tuple[int, int]) -> bool:
        """True when *span* sits strictly inside a longer match of another word."""
        start, end = span
        for other_word, other_spans in spans.items():
            if other_word == word:
                continue
            for other_start, other_end in other_spans:
                if (other_start <= start and end <= other_end
                        and other_end - other_start > end - start):
                    return True
        return False

    # ``spans`` is filled in _VOCAB_MATCHERS order, which is canonical vocab
    # order, and dicts preserve insertion order -- so the result is already sorted.
    return [
        word
        for word in spans
        if any(not subsumed(word, span) for span in spans[word])
    ]


def action_words_in(text: str) -> list[str]:
    """The :data:`ACTION_VOCAB` subset of :func:`vocab_words_in`, in vocab order."""
    action = set(ACTION_VOCAB)
    return [word for word in vocab_words_in(text) if word in action]


def direction_words_in(text: str) -> list[str]:
    """The :data:`DIRECTION_VOCAB` subset of :func:`vocab_words_in`, in vocab order."""
    direction = set(DIRECTION_VOCAB)
    return [word for word in vocab_words_in(text) if word in direction]


def canonical_action_label(words) -> str:
    """Spell a set of controlled words as a label: canonical order, deduplicated.

    One combination of words has exactly ONE spelling in the training
    distribution, which is what makes a user's query land on the vectors the
    model actually trained on. Unknown words are dropped rather than appended,
    since they are out of vocabulary.
    """
    hits = {word for word in words if word in _CONTROLLED_VOCAB_ORDER}
    return ", ".join(sorted(hits, key=_CONTROLLED_VOCAB_ORDER.__getitem__))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_action_group(raw_action_group) -> str:
    """Lower-case / strip an ``action_group`` value. Never validates membership."""
    if raw_action_group is None:
        return ""
    return str(raw_action_group).strip().lower()


def normalize_action_label(raw_action_label) -> str:
    """Collapse whitespace in an ``action_label``. Empty stays empty (= no condition)."""
    if raw_action_label is None:
        return ""
    return " ".join(str(raw_action_label).split())


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


# Hard cap on how many controlled words one label may name. A label is a compact
# prompt, not a caption: past this the T5 mean-pool dilutes the words that carry
# the action. Labels are keywords, so this counts words; the real corpus tops
# out at 4.
ACTION_LABEL_MAX_WORDS = 8


def _fail_action_labels(line_number: int, message: str) -> None:
    import sys

    print(
        f"\n❌ {ACTION_LABELS_FILE}:{line_number}: {message}",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)


def _validate_action_label_entry(
    group: str, label: str, clip: str, line_number: int
) -> None:
    """Enforce the hard constraints on an ``action_labels.jsonl`` row.

    The group must be one of the three closed values (it selects which model the
    clip trains). A non-empty label must be KEYWORDS: every comma-separated token
    a controlled-vocabulary word, no repeats, in canonical order
    (:data:`ACTION_VOCAB` words first, then :data:`DIRECTION_VOCAB`).

    Strict because a keyword label is exactly checkable: a misspelling would
    otherwise become a quietly different T5 vector, and a reordering would split
    one word combination's training mass across several points in embedding
    space.

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

    tokens = [token.strip() for token in label.split(",")]
    unknown = [token for token in tokens if token not in _CONTROLLED_VOCAB_ORDER]
    if unknown:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has action_label {label!r} whose token(s) {unknown} are "
            f"not controlled-vocabulary words. Labels are keywords now, not prose: "
            f"write the words themselves, comma-separated, in canonical order. "
            f"Valid words are {list(CONTROLLED_VOCAB)}, or leave the label empty "
            f"for an unconditioned clip.",
        )
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' repeats {duplicates} in action_label {label!r}. Each "
            f"word may appear at most once.",
        )
    canonical = canonical_action_label(tokens)
    if label != canonical:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has action_label {label!r}, which is not in canonical "
            f"order. Write it as {canonical!r}: action words in ACTION_VOCAB order, "
            f"then direction words. One word combination must have exactly one "
            f"spelling, or its training mass splits across several T5 vectors.",
        )
    if len(tokens) > ACTION_LABEL_MAX_WORDS:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has a {len(tokens)}-word action_label (max "
            f"{ACTION_LABEL_MAX_WORDS}): {label!r}",
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
            label = normalize_action_label(entry.get("action_label"))
            _validate_action_label_entry(group, label, str(clip), line_number)
            action_labels[str(clip)] = {
                "action_group": group,
                "action_label": label,
            }
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
