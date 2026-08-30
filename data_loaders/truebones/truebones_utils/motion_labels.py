from __future__ import annotations

import json
import re
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
    ACTION_LABELS_FILE,
)


MOTION_METADATA_SCHEMA_VERSION = 5

_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")

# ---------------------------------------------------------------------------
# Action groups + controlled label vocabulary  (action_labels.jsonl)
# ---------------------------------------------------------------------------
# See docs/action_group_label_refactor.md. Two fields carry the action signal:
#
#   action_group  -- one of ACTION_GROUPS. Partitions the dataset; each group
#                    trains its own model.
#   action_label  -- a short free-text prompt ("run, gallops with head lowered").
#                    Conditions the model through T5. May be empty (= no
#                    condition, routed to the learned null embedding).
#
# The vocabulary below is a CONTROLLED VOCABULARY / recall anchor, *not* a set of
# mutually exclusive classes. A label may hit several core words at once
# ("idle, growls occasionally" -> idle + roar); that is expected, not an error.

ACTION_GROUPS: tuple[str, ...] = ("locomotion", "stationary", "transition")

# Core vocabulary: the coarse actions with enough support to earn a dedicated
# multi-hot slot, one per entry. Order is significant -- it defines the multi-hot
# index layout (trained checkpoints depend on it) and the word order of
# synthesized coarse strings.
#
# A label is expected to name a core word but is NOT required to: some clips
# (rearing, sniffing, burrowing, crawling) have no coarse counterpart, and forcing
# one on would put a word into the multi-hot that the animal never does. Those
# labels carry detail words only and derive an all-zero multi-hot -- a defined
# state, not a defect.
#
# Membership threshold: a word earns a core slot when it (a) has >= ~20 supporting
# clips AND (b) names a genuinely independent coarse action -- not an aspect of an
# existing core word (stand->idle, flap->fly, kick->attack). Rarer or single-species
# actions live in ACTION_VOCAB_DETAIL, where they still reach the model through the
# T5 text path but get no dedicated multi-hot slot.
ACTION_VOCAB_CORE: tuple[str, ...] = (
    "idle",
    "walk",
    "run",
    "fly",
    "swim",
    "jump",
    "turn",
    "attack",
    "bite",
    "roar",
    "eat",
    "die",
    "fall",
    "hurt",
    "getup",
    "rest",
    "look",
    "shake",
    "crouch",
    "retreat",
    "rear",
    "throw",
    "crawl",
    "taunt",
)

# Detail vocabulary: recognized and allowed in labels, but NOT given a multi-hot
# slot -- too few supporting clips, a single species, or an aspect of a core word
# to learn a reliable response from. These words still reach the model through the
# T5 text path.
ACTION_VOCAB_DETAIL: tuple[str, ...] = (
    "climb", "sneak", "land", "takeoff", "dive", "roll",
    "sit", "sleep", "stand", "sniff", "stretch", "yawn",
    "dig", "catch", "peck", "sting", "kick", "spit", "drag", "dance",
    "breathe", "drink", "graze", "flap", "wag", "scratch",
)

CONTROLLED_VOCAB: tuple[str, ...] = ACTION_VOCAB_CORE + ACTION_VOCAB_DETAIL

# ---------------------------------------------------------------------------
# Per-group multi-hot mask
# ---------------------------------------------------------------------------
# Each group trains its own model, so the ">= ~20 clips" threshold that earned a
# word its core slot is the wrong yardstick: a word that is healthy library-wide
# can be down to a handful of clips *inside* one group. A slot fitted on five
# clips is memorized, not learned -- and on a cross-skeleton model the failure
# mode is that the word binds to whichever two species happened to supply those
# clips, which a pure clip count cannot catch. So the threshold has a species
# axis too:
#
#     keep a slot in a group iff  clips >= 10  AND  species >= 5
#
# A masked-out word is NOT removed from the vocabulary. It keeps its place in the
# label text and therefore in the frozen-T5 path, where 'roar' already sits next
# to 'growl' from pretraining and needs no support from these five clips. Only
# the from-scratch multi-hot column -- an isolated, unambiguous memorization
# handle -- is switched off.
#
# The layout stays the 24-slot global one in every group: three per-group
# vocabularies would mean three index layouts and structurally incompatible
# checkpoints. A permanently-zero column simply never receives gradient.
#
# This mask is a FROZEN CONSTANT, deliberately not recomputed from the dataset at
# import time -- otherwise adding clips would silently redefine what a slot
# means. Recompute it by the rule above when the corpus grows, and commit the new
# values explicitly.
#
GROUP_MULTIHOT_MASK: dict[str, tuple[int, ...]] = {
    #                idle walk run  fly swim jump turn attk bite roar  eat  die fall hurt getup rest look shak crou retr rear thro craw taun
    "locomotion": (    0,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   1,   0,   0,   1,   0),
    "stationary": (    1,   0,   0,   1,   0,   1,   1,   1,   1,   1,   1,   0,   1,   1,   0,   1,   1,   1,   1,   0,   1,   1,   0,   1),
    "transition": (    1,   0,   1,   1,   0,   1,   1,   1,   0,   0,   0,   1,   1,   1,   1,   1,   0,   1,   1,   1,   1,   0,   0,   0),
}

assert set(GROUP_MULTIHOT_MASK) == set(ACTION_GROUPS), (
    "GROUP_MULTIHOT_MASK must cover exactly ACTION_GROUPS: "
    f"{set(GROUP_MULTIHOT_MASK) ^ set(ACTION_GROUPS)}"
)
assert all(
    len(mask) == len(ACTION_VOCAB_CORE) and set(mask) <= {0, 1}
    for mask in GROUP_MULTIHOT_MASK.values()
), "each GROUP_MULTIHOT_MASK row must be len(ACTION_VOCAB_CORE) zeros/ones"


def group_multihot_mask(group: str) -> tuple[int, ...]:
    """The frozen 0/1 mask over :data:`ACTION_VOCAB_CORE` for one action group.

    Training and inference must both multiply the derived multi-hot by this, or
    inference lights up columns that were held at zero throughout training.
    """
    normalized = normalize_action_group(group)
    mask = GROUP_MULTIHOT_MASK.get(normalized)
    if mask is None:
        raise ValueError(
            f"unknown action_group {group!r}; expected one of {list(ACTION_GROUPS)}"
        )
    return mask


# Default group for each core word. Used ONLY to seed the clip-name fallback
# (:func:`infer_action_label_from_clip_name`) for clips not yet hand-labeled --
# never to route a request at inference, which takes the group explicitly.
#
# These are action semantics, not the corpus argmax. 'fly' is the one place the
# two disagree (321 stationary hits vs 218 locomotion, because hovering and
# wing-flapping in place are labeled stationary): a clip *named* "Fly" is a
# flight cycle, so the seed says locomotion and a human moves the exceptions.
CORE_WORD_GROUP: dict[str, str] = {
    "idle": "stationary",
    "walk": "locomotion",
    "run": "locomotion",
    "fly": "locomotion",
    "swim": "locomotion",
    "jump": "transition",
    "turn": "transition",
    "attack": "stationary",
    "bite": "stationary",
    "roar": "stationary",
    "eat": "stationary",
    "die": "transition",
    "fall": "transition",
    "hurt": "stationary",
    "getup": "transition",
    "rest": "stationary",
    "look": "stationary",
    "shake": "stationary",
    "crouch": "stationary",
    "retreat": "locomotion",
    "rear": "stationary",
    "throw": "stationary",
    "crawl": "locomotion",
    "taunt": "stationary",
}

assert set(CORE_WORD_GROUP) == set(ACTION_VOCAB_CORE), (
    "CORE_WORD_GROUP must cover exactly ACTION_VOCAB_CORE: "
    f"{set(CORE_WORD_GROUP) ^ set(ACTION_VOCAB_CORE)}"
)
assert set(CORE_WORD_GROUP.values()) <= set(ACTION_GROUPS), (
    "CORE_WORD_GROUP values must be ACTION_GROUPS members"
)

# Surface forms recognized for each vocabulary word. Labels are written using the
# canonical word, so this mainly serves (a) inflected forms inside labels and
# (b) normalizing free-text user prompts at inference ("sprint" -> "run") so they
# land on the same wording the model was trained on.
#
# Kept explicit rather than stemmed: a stemmer would collapse "stalking" into
# "stalk" but also drag unrelated words in, and the false positives land silently
# in the conditioning signal.
_VOCAB_SURFACE_FORMS: dict[str, tuple[str, ...]] = {
    # NOTE: "in place" / "stationary" are deliberately NOT idle forms. Most of
    # this dataset is animated in place, so those phrases show up as filler in
    # descriptions of running and flying clips too ("opens its jaws while
    # stationary", "runs in place") and would silently light up the idle slot.
    "idle": ("idle", "idles", "idling", "motionless", "stands still",
             "standing still", "stand still", "stays still"),
    "walk": ("walk", "walks", "walking", "trot", "trots", "trotting",
             "pace", "paces", "pacing", "march", "marches", "marching",
             "strut", "struts", "strutting", "amble", "ambles", "ambling"),
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
    "turn": ("turn", "turns", "turning", "spin", "spins", "spinning",
             "rotate", "rotates", "rotating", "pivot", "pivots", "pivoting",
             "strafe", "strafes", "strafing", "circle", "circles", "circling",
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
    # "collapse" belongs to fall, not die: in this dataset it describes going
    # down ("fall, collapses onto its side"), and every clip it matched on its
    # own turned out to be a fall or a lie-down, not a death.
    "die": ("die", "dies", "dying", "death", "dead", "perish", "perishes",
            "passes out"),
    "fall": ("fall", "falls", "falling", "fell", "collapse", "collapses",
             "collapsing", "tumble", "tumbles",
             "tumbling", "trip", "trips", "tripping", "stumble", "stumbles",
             "stumbling", "topple", "topples", "toppling"),
    # "gethurt" (one token) is the legacy tag spelling. The word-boundary match
    # means it does NOT fall out of "hurt" on its own -- the 't' in front blocks
    # it -- so it has to be listed explicitly.
    "hurt": ("hurt", "hurts", "gethurt", "get hurt", "get-hurt", "gets hurt",
             "injured", "wounded", "flinch", "flinches",
             "flinching", "recoil", "recoils", "recoiling", "stagger",
             "staggers", "staggering", "stunned", "limp", "limps", "limping",
             "takes a hit", "knocked back"),
    # Bare "rise"/"rises"/"rising" are deliberately absent: they denote any
    # upward motion ("rises upward undulating fins", "rise onto hind legs",
    # "chest rising and falling") and fired on swimming, rearing and breathing
    # more often than on a recovery. Only the phrases naming the destination or
    # the down-state are kept.
    "getup": ("getup", "get up", "gets up", "getting up", "get-up",
              "stands up", "standing up",
              "rises to stand", "rises to standing", "rises to its feet",
              "rise back up", "rises back up",
              "recover", "recovers", "recovering",
              "wakes up", "waking up", "revive", "revives", "reviving"),
    # Bare "lie"/"lies"/"lying" are deliberately absent: they name a posture,
    # and in this dataset that posture is nearly always death ("die, lies
    # motionless on its side") or the state a get-up departs from ("getup, rises
    # from lying to standing"). The explicit settle-down phrases stay -- those
    # do mean going to rest.
    "rest": ("rest", "rests", "resting", "lie down",
             "lies down", "lying down", "sit", "sits", "sitting", "seated",
             "sleep", "sleeps", "sleeping", "dozing", "napping"),
    # "alert" is deliberately NOT a look form: here it is a posture adjective
    # ("stands alert", "low alert posture", "ears alert"), and all 18 clips it
    # matched on its own were idle stances, not the act of looking.
    "look": ("look", "looks", "looking", "glance", "glances", "glancing",
             "gaze", "gazes", "gazing", "observe", "observes", "observing",
             "scan", "scans", "scanning", "watch", "watches", "watching"),
    "shake": ("shake", "shakes", "shaking", "shudder", "shudders",
              "shuddering", "twitch", "twitches", "twitching", "tremble",
              "trembles", "trembling", "wag", "wags", "wagging"),
    "crouch": ("crouch", "crouches", "crouching", "squat", "squats",
               "squatting", "hunker", "hunkers", "hunkering"),
    "retreat": ("retreat", "retreats", "retreating", "backs away",
                "backing away", "backs up", "backing up", "backward",
                "backwards"),
    "rear": ("rear", "rears", "rearing", "on its hind legs",
             "onto its hind legs"),
    "throw": ("throw", "throws", "throwing", "toss", "tosses", "tossing",
              "fling", "flings", "flinging", "hurl", "hurls", "hurling"),
    "crawl": ("crawl", "crawls", "crawling", "slither", "slithers",
              "slithering", "creep", "creeps", "creeping", "scurry",
              "scurries", "scurrying"),
    "taunt": ("taunt", "taunts", "taunting", "threaten", "threatens",
              "threatening", "intimidate", "intimidates", "intimidating"),
    # -- detail tier --
    "climb": ("climb", "climbs", "climbing"),
    "sneak": ("sneak", "sneaks", "sneaking", "stalk", "stalks", "stalking",
              "prowl", "prowls", "prowling"),
    "land": ("land", "lands", "landing", "touches down", "touching down"),
    "takeoff": ("take off", "takes off", "taking off", "takeoff", "lift off",
                "lifts off"),
    "dive": ("dive", "dives", "diving", "plunge", "plunges", "plunging"),
    "roll": ("roll", "rolls", "rolling"),
    "sit": ("sit", "sits", "sitting", "seated"),
    "sleep": ("sleep", "sleeps", "sleeping", "dozing", "napping", "asleep"),
    "stand": ("stand", "stands", "standing", "upright"),
    "sniff": ("sniff", "sniffs", "sniffing", "smell", "smells", "smelling"),
    "stretch": ("stretch", "stretches", "stretching"),
    "yawn": ("yawn", "yawns", "yawning"),
    "dig": ("dig", "digs", "digging", "burrow", "burrows", "burrowing",
            "scrape", "scrapes", "scraping"),
    "catch": ("catch", "catches", "catching", "grab", "grabs", "grabbing",
              "seize", "seizes", "seizing", "snatch", "snatches", "snatching"),
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
assert not (set(ACTION_VOCAB_CORE) & set(ACTION_VOCAB_DETAIL)), (
    "a word cannot be in both the core and detail vocabulary"
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


def vocab_words_in(text: str, core_only: bool = False) -> list[str]:
    """Controlled-vocabulary words present in *text*, in canonical vocab order.

    Matching is over the whole string, not just a prefix -- a label may name its
    coarse action anywhere ("stands still and growls" hits idle *and* roar), and
    may hit several words at once. That is the point of a controlled vocabulary:
    it anchors recall without forcing a mutually exclusive choice.

    A word is dropped when every one of its matches sits strictly inside a longer
    match of a *different* word: "stands still" is idle, so the "stands" inside it
    must not also light up ``stand``, and "breathes fire" is spit, not breathe.
    Equal-length matches both survive, which is what keeps a detail word firing
    alongside the core word that shares its spelling ("grazes" -> eat + graze).
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

    allowed = set(ACTION_VOCAB_CORE) if core_only else None
    # ``spans`` is filled in _VOCAB_MATCHERS order, which is canonical vocab
    # order, and dicts preserve insertion order -- so the result is already sorted.
    return [
        word
        for word in spans
        if (allowed is None or word in allowed)
        and any(not subsumed(word, span) for span in spans[word])
    ]


def action_multihot_words(label: str, group: str | None = None) -> list[str]:
    """Core words a label activates -- the derived multi-hot, as words.

    Derived automatically from the label text, so it costs the annotator nothing
    and naturally supports multiple hits.

    Pass *group* to apply that group's frozen :data:`GROUP_MULTIHOT_MASK`, which
    is what any consumer building an actual multi-hot vector must do. Leave it
    ``None`` to get the unmasked hits -- that is the right input for the
    coarse-string synthesis of :func:`coarse_label_from_words`, which feeds T5
    text and must keep the down-weighted words (see the note there).
    """
    words = vocab_words_in(label, core_only=True)
    if group is None:
        return words
    mask = group_multihot_mask(group)
    allowed = {word for word, bit in zip(ACTION_VOCAB_CORE, mask) if bit}
    return [word for word in words if word in allowed]


def action_multihot_vector(label: str, group: str | None = None) -> list[float]:
    """The derived multi-hot over :data:`ACTION_VOCAB_CORE` as a 0/1 list.

    With *group* given, columns masked out for that group are held at zero (see
    :data:`GROUP_MULTIHOT_MASK`). An all-zero result is a defined state -- it maps
    to the projection's bias -- and is distinct from the hard-dropped null the
    model uses for an absent condition.
    """
    hits = set(action_multihot_words(label, group))
    return [1.0 if word in hits else 0.0 for word in ACTION_VOCAB_CORE]


def coarse_label_from_words(words) -> str:
    """Synthesize the coarse training string from core words ('idle, roar').

    Used by the training-time coarse augmentation: with some probability the
    model sees this instead of the full label, so it learns to answer the short
    queries users actually type. Word order follows ACTION_VOCAB_CORE, so
    'idle, roar' is the only spelling of that combination.

    Detail-only labels fall back to their detail words ('rear', 'sniff'). Users
    type those bare too, and returning '' for them would hand the augmentation
    the null condition instead -- the model would then train on "no condition"
    for exactly the clips whose action is only reachable through a detail word,
    and could never learn to answer that query. The fallback is a T5 string, not
    a multi-hot, so it costs no index slot.
    """
    order = {word: i for i, word in enumerate(ACTION_VOCAB_CORE)}
    hits = sorted({w for w in words if w in order}, key=order.__getitem__)
    if not hits:
        detail_order = {word: i for i, word in enumerate(ACTION_VOCAB_DETAIL)}
        hits = sorted({w for w in words if w in detail_order},
                      key=detail_order.__getitem__)
    return ", ".join(hits)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _split_identifier_tokens(value: str) -> list[str]:
    raw_parts = re.split(r"[^A-Za-z0-9]+", value)
    tokens: list[str] = []
    for part in raw_parts:
        if not part:
            continue
        matches = _TOKEN_PATTERN.findall(part)
        if matches:
            tokens.extend(matches)
        else:
            tokens.append(part)
    return [token.lower() for token in tokens if token]


def _strip_species_variant(object_type: str) -> str:
    base = re.sub(r"[-_\s]*\d+$", "", object_type).strip("-_")
    if len(base) > 1 and base[-1].isupper() and base[-2].islower():
        return base[:-1]
    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_species_label(object_type: str) -> str:
    base = _strip_species_variant(object_type)
    tokens = _split_identifier_tokens(base)
    return " ".join(tokens) if tokens else object_type.lower()


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
# Action-label fallback inference
# ---------------------------------------------------------------------------
# Action labels are normally hand-maintained in ``action_labels.jsonl``, and
# ``load_action_labels`` hard-exits when a clip on disk has no entry. When clips
# are added incrementally, hand-labeling lags behind, so callers can backfill
# missing entries with a best-effort (group, label) pair inferred from the
# Truebones ``Species_Action_id`` clip name. These guesses are HEURISTIC and meant
# to be reviewed by hand -- the clip name knows the verb, never the detail that
# makes a label worth conditioning on.

# Tokens that, paired with a trailing "Up", denote a get-up recovery ("DeadUp",
# "DieUp", "SleepUp", "StandUp"), which would otherwise mis-resolve to
# death/rest/idle on the bare verb.
_GETUP_UP_CONTEXT: frozenset[str] = frozenset(
    {"dead", "die", "stand", "sleep", "get", "wake", "knock", "rise", "sit", "lay", "lie", "fall"}
)

# Ordered fallback rules: the first word whose keyword set intersects the
# clip-name tokens wins, so specific/event-like actions precede generic
# locomotion/idle. Keyword sets carry over from the previous action-tag fallback
# (~84% agreement with the hand labels there); the right-hand side is now a
# CONTROLLED_VOCAB word, so the synthesized label matches on its own vocabulary.
_FALLBACK_LABEL_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("getup", frozenset({"getup", "get", "rise", "rising", "wake", "waking", "revive", "recover", "standup", "spawn"})),
    ("die", frozenset({"die", "death", "dead", "dying", "dies", "deceased", "despawn"})),
    ("hurt", frozenset({"hit", "hurt", "gethurt", "damage", "knock", "knocked", "knockback", "stunned",
                        "stun", "shot", "recoil", "flinch", "pain", "wound", "injured", "limp", "sorr"})),
    ("bite", frozenset({"bite", "chomp", "fang", "snap"})),
    ("attack", frozenset({"attack", "atk", "sting", "strike", "kill", "swat", "swipe", "whip",
                          "throw", "rip", "claw", "punch", "slash", "peck", "stab", "spit", "spear", "maul",
                          "tail", "butt", "gore", "charge", "bash", "headbutt", "tackle",
                          "spray", "kick", "pounce", "catch", "smash", "slam", "grab"})),
    ("fall", frozenset({"fall", "falling", "trip", "stumble", "fallen"})),
    ("jump", frozenset({"jump", "leap", "hop"})),
    ("swim", frozenset({"swim", "dive", "paddle"})),
    ("fly", frozenset({"fly", "flying", "glide", "flap", "hover", "soar", "soaring", "takeoff", "take",
                       "land", "lander", "wing", "float", "floater"})),
    ("turn", frozenset({"turn", "spin", "rotate", "strafe", "circle", "pivot", "arc"})),
    ("eat", frozenset({"eat", "drink", "fish", "graze", "graz", "grazing", "feed", "feast"})),
    ("dig", frozenset({"dig", "burrow", "burrough", "egg"})),
    ("rest", frozenset({"rest", "sleep", "sit", "lay", "laydown", "lie", "lying", "relax", "down", "sleepy"})),
    ("roar", frozenset({"roar", "growl", "yell", "scream", "hiss", "howl", "bark", "cry", "call"})),
    ("shake", frozenset({"shake", "twitch", "twitching", "wag", "tremble", "shudder"})),
    ("scratch", frozenset({"scratch", "groom", "rub", "itch", "lick", "scrape"})),
    ("look", frozenset({"look", "observ", "observing", "listen", "watch", "gaze", "scan"})),
    ("rear", frozenset({"rear", "buck", "hoof", "hind", "hind2"})),
    ("taunt", frozenset({"emote", "yawn", "sniff", "stretch", "purr", "taunt",
                         "celebrate", "dance", "nod", "alert", "angry", "pissed", "scared",
                         "curious", "sneeze", "ear", "restless", "mean", "scary",
                         "pant", "mope", "wild", "special", "threaten", "threat", "clear", "clearing"})),
    ("sneak", frozenset({"sneak", "prowl", "stalk", "stalking", "crawl", "slither", "scurry", "creep"})),
    ("climb", frozenset({"climb"})),
    ("retreat", frozenset({"retreat", "back", "backing", "backward", "chase"})),
    ("run", frozenset({"run", "jog", "gallop", "sprint", "dash", "fast"})),
    ("walk", frozenset({"walk", "trot", "move", "locomotion", "step", "pace", "slide", "march",
                        "strut", "wander", "roam", "forward", "slow", "slowwalk"})),
    ("idle", frozenset({"idle", "stand", "ready", "breath", "breathe", "wait", "stance", "energetic",
                        "tired", "steady", "clean", "cud", "start"})),
)

# Sanity guard: every fallback word must be a member of the controlled vocabulary,
# so a synthesized label always passes ``load_action_labels`` validation.
assert all(word in CONTROLLED_VOCAB for word, _ in _FALLBACK_LABEL_RULES), (
    "fallback rules reference a word outside CONTROLLED_VOCAB: "
    + str([w for w, _ in _FALLBACK_LABEL_RULES if w not in CONTROLLED_VOCAB])
)

# Group for the detail words a fallback rule can produce. Core words take their
# group from CORE_WORD_GROUP; only the detail-tier right-hand sides above need an
# entry here.
_FALLBACK_DETAIL_GROUP: dict[str, str] = {
    "dig": "stationary",
    "sneak": "locomotion",
    "climb": "locomotion",
    "scratch": "stationary",
}

assert all(
    word in CORE_WORD_GROUP or word in _FALLBACK_DETAIL_GROUP
    for word, _ in _FALLBACK_LABEL_RULES
), "every fallback word needs a group (CORE_WORD_GROUP or _FALLBACK_DETAIL_GROUP)"


def _strip_species_prefix(parts: list[str], object_type: str | None) -> list[str]:
    """Drop the species tokens from a split clip name.

    With *object_type* known the exact prefix is removed, which is the only way
    to handle a multi-token species: ``MU06_DeathMage_Idle_1`` must not leave
    'death' in the action tokens. Without it only one token can be dropped, so
    pass *object_type* whenever the caller has it.

    The comparison is case-insensitive, like the filename->species inference that
    produced *object_type*: ``deer_buck_Idle_1.npy`` resolves to ``Deer_Buck``,
    and a case-sensitive prefix test would drop only 'deer' and let 'buck' reach
    the keyword rules.
    """
    if object_type:
        species_parts = str(object_type).split("_")
        head = [part.lower() for part in parts[: len(species_parts)]]
        if head == [part.lower() for part in species_parts] and len(parts) > len(species_parts):
            return parts[len(species_parts):]
    if len(parts) > 1:
        return parts[1:]
    return parts


def _tokenize_action_name(clip_name: str, object_type: str | None = None) -> set[str]:
    """Tokenize the action portion of a "Species_Action_id" clip name.

    Drops the species tokens (so species names like 'Ant' don't pollute keyword
    matching) and trailing numeric ids, lower-cases via the shared
    :data:`_TOKEN_PATTERN`, and adds lightly stemmed variants (-ing/-ed/-er/-s) so
    'Trotting' matches 'trot', etc.
    """
    stem = Path(clip_name).stem
    parts = _strip_species_prefix(stem.split("_"), object_type)
    tokens: set[str] = set()
    for part in parts:
        for match in _TOKEN_PATTERN.findall(part):
            token = match.lower()
            if token.isdigit():
                continue
            tokens.add(token)
            for suffix in ("ing", "ed", "er", "s"):
                if len(token) > len(suffix) + 2 and token.endswith(suffix):
                    tokens.add(token[: -len(suffix)])
    return tokens


def infer_action_label_from_clip_name(
    clip_name: str,
    object_type: str | None = None,
) -> tuple[str, str]:
    """Best-effort ``(action_group, action_label)`` inferred from a clip name.

    Heuristic fallback for clips not yet hand-labeled in ``action_labels.jsonl``;
    the label is a single :data:`CONTROLLED_VOCAB` word and the group its default
    from :data:`CORE_WORD_GROUP`, so the entry passes validation and is a review
    seed rather than a finished label. Pass *object_type* so that a multi-token
    species name is stripped whole and cannot leak into the keyword match.

    Falls back to ``("stationary", "idle")`` when no rule fires: the schema has no
    "unknown" value, and a wrong-but-legal seed that a human can spot beats an
    entry the loader refuses.
    """
    tokens = _tokenize_action_name(clip_name, object_type)
    word = None
    if "up" in tokens and (tokens & _GETUP_UP_CONTEXT):
        word = "getup"
    else:
        for candidate, keywords in _FALLBACK_LABEL_RULES:
            if tokens & keywords:
                word = candidate
                break
    if word is None:
        return "stationary", "idle"
    group = CORE_WORD_GROUP.get(word) or _FALLBACK_DETAIL_GROUP[word]
    return group, word


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_object_labels(object_type: str) -> dict[str, str]:
    return {"species_label": infer_species_label(object_type)}


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
    payload.update(build_object_labels(object_type))
    if motion_name is not None:
        payload["motion_name"] = motion_name
    return payload


# Soft cap on label length. A label is a compact prompt, not a caption: past this
# the T5 mean-pool dilutes the words that carry the action.
ACTION_LABEL_MAX_WORDS = 15


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
    """Enforce the two hard constraints on an ``action_labels.jsonl`` row.

    The group must be one of the three closed values (it selects which model the
    clip trains), and a non-empty label must hit at least one controlled word so
    the recall anchor actually holds. An *empty* label is legal and means "no
    condition" — it is routed to the learned null embedding, never encoded as an
    empty string, which would otherwise teach the model that empty text means any
    motion at all and poison the CFG unconditional branch.
    """
    if group not in ACTION_GROUPS:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has invalid action_group {group!r}. "
            f"Valid groups are: {list(ACTION_GROUPS)}",
        )
    if not label:
        return
    if not vocab_words_in(label):
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has action_label {label!r}, which hits no controlled "
            f"vocabulary word. Every non-empty label must name at least one of "
            f"{list(CONTROLLED_VOCAB)} (or one of their surface forms), or be left "
            f"empty for an unconditioned clip.",
        )
    word_count = len(label.split())
    if word_count > ACTION_LABEL_MAX_WORDS:
        _fail_action_labels(
            line_number,
            f"clip '{clip}' has a {word_count}-word action_label (max "
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
    so the joined fields are dropped on the way out. (``action_tags`` is the
    removed predecessor; stripping it clears the stale copies earlier rebuilds
    baked in.)
    """
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    joined_keys = ("action_group", "action_label", "action_tags")
    sanitized_entries = {
        motion_name: {
            key: value for key, value in metadata.items() if key not in joined_keys
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
