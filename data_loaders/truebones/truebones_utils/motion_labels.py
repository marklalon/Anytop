from __future__ import annotations

import json
import re
from pathlib import Path

from data_loaders.truebones.truebones_utils.param_utils import (
    MOTION_METADATA_FILE,
    ACTION_TAGS_FILE,
)


MOTION_METADATA_SCHEMA_VERSION = 5

_TOKEN_PATTERN = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")

# ---------------------------------------------------------------------------
# Canonical action-tag vocabulary
# ---------------------------------------------------------------------------
# The full set of valid action tags. Order is significant: it defines the
# multi-hot index layout the model conditions on, so trained checkpoints depend
# on it staying stable. Tags themselves are maintained by hand in
# ``action_tags.jsonl`` (see ``load_action_tags``); this module never generates
# them automatically.

ACTION_TAGS: tuple[str, ...] = (
    "idle",
    "locomotion",
    "getup",
    "swim",
    "fly",
    "jump",
    "turn",
    "attack",
    "gethurt",
    "rest",
    "emote",
    "interact",
    "death",
    "fall",
    "unknown",
)

# ---------------------------------------------------------------------------
# Action groups + controlled label vocabulary  (action_labels.jsonl)
# ---------------------------------------------------------------------------
# See docs/action_group_label_refactor.md. Two fields replace ACTION_TAGS:
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
# Membership threshold: >= ~20 supporting clips out of 1445. Rarer actions live in
# ACTION_VOCAB_DETAIL, where they still reach the model through the T5 text path
# but get no dedicated multi-hot slot (too few samples to learn a reliable
# response from).
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
    "scratch",
)

# Detail vocabulary: recognized and allowed in labels, but NOT given a multi-hot
# slot -- too few supporting clips to learn a reliable response from. These words
# still reach the model through the T5 text path.
ACTION_VOCAB_DETAIL: tuple[str, ...] = (
    "crawl", "climb", "sneak", "retreat", "land", "takeoff", "dive", "roll",
    "rear", "sit", "sleep", "stand", "sniff", "stretch", "yawn", "taunt",
    "dig", "throw", "catch", "peck", "sting", "kick", "spit", "drag", "dance",
    "breathe", "drink", "graze", "flap", "wag",
)

CONTROLLED_VOCAB: tuple[str, ...] = ACTION_VOCAB_CORE + ACTION_VOCAB_DETAIL

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
    "scratch": ("scratch", "scratches", "scratching", "groom", "grooms",
                "grooming", "rub", "rubs", "rubbing", "itch", "itches",
                "itching"),
    # -- detail tier --
    "crawl": ("crawl", "crawls", "crawling", "slither", "slithers",
              "slithering", "creep", "creeps", "creeping", "scurry",
              "scurries", "scurrying"),
    "climb": ("climb", "climbs", "climbing"),
    "sneak": ("sneak", "sneaks", "sneaking", "stalk", "stalks", "stalking",
              "prowl", "prowls", "prowling"),
    "retreat": ("retreat", "retreats", "retreating", "backs away",
                "backing away", "backs up", "backing up", "backward",
                "backwards"),
    "land": ("land", "lands", "landing", "touches down", "touching down"),
    "takeoff": ("take off", "takes off", "taking off", "takeoff", "lift off",
                "lifts off"),
    "dive": ("dive", "dives", "diving", "plunge", "plunges", "plunging"),
    "roll": ("roll", "rolls", "rolling"),
    "rear": ("rear", "rears", "rearing", "on its hind legs",
             "onto its hind legs"),
    "sit": ("sit", "sits", "sitting", "seated"),
    "sleep": ("sleep", "sleeps", "sleeping", "dozing", "napping", "asleep"),
    "stand": ("stand", "stands", "standing", "upright"),
    "sniff": ("sniff", "sniffs", "sniffing", "smell", "smells", "smelling"),
    "stretch": ("stretch", "stretches", "stretching"),
    "yawn": ("yawn", "yawns", "yawning"),
    "taunt": ("taunt", "taunts", "taunting", "threaten", "threatens",
              "threatening", "intimidate", "intimidates", "intimidating"),
    "dig": ("dig", "digs", "digging", "burrow", "burrows", "burrowing",
            "scrape", "scrapes", "scraping"),
    "throw": ("throw", "throws", "throwing", "toss", "tosses", "tossing",
              "fling", "flings", "flinging", "hurl", "hurls", "hurling"),
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


def action_multihot_words(label: str) -> list[str]:
    """Core words a label activates -- the derived multi-hot, as words.

    Derived automatically from the label text, so it costs the annotator nothing
    and naturally supports multiple hits.
    """
    return vocab_words_in(label, core_only=True)


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


def normalize_action_tags(raw_action_tags) -> list[str]:
    if raw_action_tags is None:
        return []
    if isinstance(raw_action_tags, str):
        values = [raw_action_tags]
    elif isinstance(raw_action_tags, (list, tuple, set)):
        values = raw_action_tags
    else:
        values = [raw_action_tags]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = str(value).strip().lower()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


# ---------------------------------------------------------------------------
# Action-tag fallback inference
# ---------------------------------------------------------------------------
# Action tags are normally hand-maintained in ``action_tags.jsonl``, and
# ``load_action_tags`` hard-exits when a clip on disk has no entry. When clips are
# added incrementally, hand-labeling lags behind, so callers can backfill missing
# entries with a best-effort tag inferred from the Truebones ``Species_Action_id``
# clip name. These guesses are HEURISTIC and meant to be reviewed by hand.

# Tokens that, paired with a trailing "Up", denote a get-up recovery ("DeadUp",
# "DieUp", "SleepUp", "StandUp"), which would otherwise mis-resolve to
# death/rest/idle on the bare verb.
_GETUP_UP_CONTEXT: frozenset[str] = frozenset(
    {"dead", "die", "stand", "sleep", "get", "wake", "knock", "rise", "sit", "lay", "lie", "fall"}
)

# Ordered fallback rules: the first tag whose keyword set intersects the clip-name
# tokens wins, so specific/event-like actions precede generic locomotion/idle.
# Keyword sets were derived empirically from the hand-labeled ``action_tags.jsonl``
# vocabulary (~84% agreement with the existing labels; the rest surface as
# ``unknown`` / review items).
_FALLBACK_ACTION_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("getup", frozenset({"getup", "get", "rise", "rising", "wake", "waking", "revive", "recover", "standup", "spawn"})),
    ("death", frozenset({"die", "death", "dead", "dying", "dies", "deceased", "despawn"})),
    ("gethurt", frozenset({"hit", "hurt", "gethurt", "damage", "knock", "knocked", "knockback", "stunned",
                           "stun", "shot", "recoil", "flinch", "pain", "wound", "injured", "limp", "sorr"})),
    ("attack", frozenset({"attack", "atk", "bite", "sting", "strike", "kill", "swat", "swipe", "snap", "whip",
                          "throw", "rip", "claw", "punch", "slash", "peck", "stab", "spit", "spear", "maul",
                          "tail", "butt", "gore", "charge", "bash", "headbutt", "chomp", "tackle", "fang",
                          "spray", "kick", "pounce", "catch", "smash", "slam", "grab"})),
    ("fall", frozenset({"fall", "falling", "trip", "stumble", "fallen"})),
    ("jump", frozenset({"jump", "leap", "hop"})),
    ("swim", frozenset({"swim", "dive", "paddle"})),
    ("fly", frozenset({"fly", "flying", "glide", "flap", "hover", "soar", "soaring", "takeoff", "take",
                       "land", "lander", "wing", "float", "floater"})),
    ("turn", frozenset({"turn", "spin", "rotate", "strafe", "circle", "pivot", "arc"})),
    ("interact", frozenset({"eat", "drink", "fish", "egg", "dig", "burrow", "burrough", "graze", "graz", "grazing", "feed", "feast"})),
    ("rest", frozenset({"rest", "sleep", "sit", "lay", "laydown", "lie", "lying", "relax", "down", "sleepy"})),
    ("emote", frozenset({"emote", "roar", "yawn", "look", "shake", "growl", "yell", "scream", "hiss", "howl",
                         "bark", "cry", "scratch", "sniff", "lick", "stretch", "scrape", "purr", "taunt",
                         "celebrate", "dance", "nod", "listen", "alert", "angry", "pissed", "scared",
                         "curious", "sneeze", "wag", "ear", "restless", "hoof", "rear", "mean", "scary",
                         "pant", "twitch", "twitching", "buck", "mope", "wild", "special", "call", "threaten",
                         "threat", "clear", "clearing"})),
    ("locomotion", frozenset({"walk", "run", "jog", "trot", "gallop", "sprint", "dash", "crawl", "move", "locomotion",
                              "step", "chase", "retreat", "climb", "sneak", "prowl", "pace", "slide", "march",
                              "strut", "wander", "roam", "scurry", "slither", "back", "backing", "forward",
                              "backward", "slow", "fast", "slowwalk", "stalk", "stalking"})),
    ("idle", frozenset({"idle", "stand", "ready", "breath", "breathe", "wait", "stance", "energetic",
                        "tired", "steady", "clean", "cud", "hind", "hind2", "observ", "observing", "start"})),
)

# Sanity guard: every fallback tag must be a member of the canonical vocabulary so
# inferred entries pass ``load_action_tags`` validation.
assert all(tag in ACTION_TAGS for tag, _ in _FALLBACK_ACTION_RULES), (
    "fallback rules reference a tag outside ACTION_TAGS"
)


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


def infer_action_tags_from_clip_name(
    clip_name: str,
    object_type: str | None = None,
) -> list[str]:
    """Best-effort single action tag inferred from a clip name; ``['unknown']`` if no rule fires.

    Heuristic fallback for clips not yet hand-labeled in ``action_tags.jsonl``; the
    result is always a list of canonical :data:`ACTION_TAGS` members and is meant
    to be reviewed by a human before use. Pass *object_type* so that a multi-token
    species name is stripped whole and cannot leak into the keyword match.
    """
    tokens = _tokenize_action_name(clip_name, object_type)
    if "up" in tokens and (tokens & _GETUP_UP_CONTEXT):
        return ["getup"]
    for tag, keywords in _FALLBACK_ACTION_RULES:
        if tokens & keywords:
            return [tag]
    return ["unknown"]


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

    Action tags are no longer produced here — they are maintained by hand in
    ``action_tags.jsonl`` and merged in by :func:`load_motion_metadata`.
    """
    payload: dict[str, object] = {"object_type": object_type}
    payload.update(build_object_labels(object_type))
    if motion_name is not None:
        payload["motion_name"] = motion_name
    return payload


def _validate_action_tags(tags: list[str], clip: str, line_number: int) -> None:
    """Validate that all tags are members of the canonical ``ACTION_TAGS`` vocabulary."""
    import sys

    invalid = [t for t in tags if t not in ACTION_TAGS]
    if invalid:
        print(
            f"\n❌ {ACTION_TAGS_FILE}:{line_number}: clip '{clip}' contains invalid "
            f"action tag(s): {invalid}. Valid tags are: {list(ACTION_TAGS)}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_action_tags(dataset_dir: str | Path) -> dict[str, list[str]]:
    """Load the hand-maintained ``action_tags.jsonl`` sidecar.

    Each line is a JSON object ``{"clip": "<name>.npy", "action_tags": [...]}``.
    Returns a mapping ``clip -> [tag, ...]``. Raises ``FileNotFoundError`` if the
    file is absent so callers fail fast rather than silently training without
    action conditioning.
    """
    tags_path = Path(dataset_dir) / ACTION_TAGS_FILE
    if not tags_path.exists():
        raise FileNotFoundError(
            f"{ACTION_TAGS_FILE} not found at {tags_path}. Action tags are now "
            f"maintained by hand in this file (one "
            f'{{"clip": "<name>.npy", "action_tags": [...]}} object per line).'
        )

    action_tags: dict[str, list[str]] = {}
    with open(tags_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{ACTION_TAGS_FILE}:{line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{ACTION_TAGS_FILE}:{line_number} must be a JSON object, "
                    f"got {type(entry).__name__}"
                )
            clip = entry.get("clip")
            if not clip:
                raise ValueError(
                    f"{ACTION_TAGS_FILE}:{line_number} is missing the 'clip' field"
                )
            normalized = normalize_action_tags(entry.get("action_tags"))
            _validate_action_tags(normalized, clip, line_number)
            action_tags[str(clip)] = normalized
    return action_tags


def load_motion_metadata(
    dataset_dir: str | Path,
    require_action_tags: bool = True,
) -> dict[str, dict[str, object]]:
    """Load ``motion_metadata.json`` joined with per-clip ``action_tags``.

    By default a clip present in the metadata but absent from ``action_tags.jsonl``
    is a fatal error (action tags are a required training-conditioning signal).
    Pass ``require_action_tags=False`` for bookkeeping reads that only preserve /
    carry-forward existing metadata (e.g. incremental preprocessing): missing-tag
    clips are then kept as-is without an ``action_tags`` field instead of exiting.
    """
    metadata_path = Path(dataset_dir) / MOTION_METADATA_FILE
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    motions = payload.get("motions", payload)
    if not isinstance(motions, dict):
        return {}

    action_tags = load_action_tags(dataset_dir)

    normalized: dict[str, dict[str, object]] = {}
    missing_tags: list[str] = []
    for motion_name, metadata in motions.items():
        if not isinstance(metadata, dict):
            continue
        tags = action_tags.get(motion_name)
        if tags is None:
            missing_tags.append(motion_name)
            if require_action_tags:
                continue
            # Tolerant mode: carry the entry forward untouched (no action_tags).
            normalized[motion_name] = dict(metadata)
            continue
        entry = dict(metadata)
        entry["action_tags"] = list(tags)
        normalized[motion_name] = entry

    if missing_tags and require_action_tags:
        preview = ", ".join(sorted(missing_tags)[:10])
        more = "" if len(missing_tags) <= 10 else f" (+{len(missing_tags) - 10} more)"
        import sys

        msg = (
            f"\n❌ {ACTION_TAGS_FILE} is missing action_tags for {len(missing_tags)} "
            f"clip(s): {preview}{more}\n\n"
            f"   Please open {ACTION_TAGS_FILE} and add an entry for each missing clip:\n"
            f"   {{ \"clip_name.npy\": [\"action_tag1\", \"action_tag2\", ...] }}\n"
        )
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)
    return normalized


def write_motion_metadata(
    save_dir: str | Path,
    motion_entries: dict[str, dict[str, object]],
    total_clips: int,
) -> Path:
    output_path = Path(save_dir) / MOTION_METADATA_FILE
    sanitized_entries = {
        motion_name: dict(metadata)
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
