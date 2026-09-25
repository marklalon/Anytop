"""End effector detection and symmetry analysis utilities."""

from collections import Counter, defaultdict

import numpy as np
import re

from data_loaders.truebones.truebones_utils.dataset_tags import dataset_tags


# End effector joint detection tokens
_END_EFFECTOR_DISTAL_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'leg',
)
_END_EFFECTOR_TAIL_TOKENS = (
    'tail',
    'sippo',
    'tai',
)
_END_EFFECTOR_HEAD_TOKENS = (
    'head',
    'jaw',
    'mouth',
    'nose',
    'snout',
    'muzzle',
    'beak',
    'tongue',
    'mandible',
    'fang',
    'chin',
)
_END_EFFECTOR_APPENDAGE_TOKENS = (
    'wing',
    'forearm',
    'clip',
    'pincer',
    'plier',
    'feeler',
    'antenna',
    'horn',
    'spike',
)
_END_EFFECTOR_EXCLUDE_TOKENS = (
    'jiggle',
    'twist',
    'hair',
    'fur',
    'beard',
    'eyebrow',
    'eyelid',
    'eyeball',
    'eye',
    'ear',
    'lip',
    'saddle',
    'halter',
    'reins',
    'handle',
    'trajectory',
    'projectile',
    'magic',
    'mesh',
    'ik',
    'chain',
    'xtra',
    'extra',
    'ponytail',
    'body',
    'spine',
    'shell',
    'center',
    'mascara',
)

# Contact joint detection tokens
_CONTACT_JOINT_KEYWORDS = (
    'toe',
    'foot',
    'hoof',
    'phalanx',
    'ashi',
    'ankle',
    'heel',
    'paw',
)
_CONTACT_JOINT_CONTEXT_KEYWORDS = _CONTACT_JOINT_KEYWORDS + (
    'leg',
)
_CONTACT_JOINT_UPPER_LIMB_TOKENS = (
    'hand',
    'finger',
    'thumb',
    'arm',
    'wrist',
    'elbow',
    'forearm',
    'shoulder',
    'wing',
)
_CONTACT_JOINT_WEAK_KEYWORDS = (
    'leg',
)
_CONTACT_GEOMETRY_DISTAL_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'leg',
)
_CONTACT_CHAIN_STOP_TOKENS = (
    'hip',
    'hips',
    'pelvis',
    'root',
    'cog',
    'spine',
    'chest',
    'thigh',
    'knee',
    'upperleg',
    'upleg',
    'neck',
    'head',
    'tail',
    'jaw',
    'body',
)
_CONTACT_CHAIN_INCLUDE_TOKENS = (
    'toe',
    'foot',
    'hoof',
    'paw',
    'phalanx',
    'claw',
    'finger',
    'thumb',
    'hand',
    'palm',
    'ball',
    'ankle',
    'wrist',
)
_CONTACT_PARENT_OFFSET_RATIO = 0.22
_CONTACT_PARENT_OFFSET_MIN = 0.10
_CONTACT_PARENT_OFFSET_CAP = 0.20
_CONTACT_CUMULATIVE_OFFSET_RATIO = 0.44
_CONTACT_CUMULATIVE_OFFSET_MIN = 0.15
_CONTACT_CUMULATIVE_OFFSET_CAP = 0.34

# Joint name canonicalization
_CANONICAL_NAME_PREFIXES = (
    'BN_Bip01',
    'Bip001',
    'Bip01',
    'Sabrecat',
    'NPC',
    'Rig',
    'BN',
    'jt',
    'Elk',
)
# Trailing rig suffixes stripped from joint names during canonicalization.
# Matched case-insensitively against the raw (pre-lowercased) name.
_CANONICAL_NAME_SUFFIXES = (
    'SHJnt',
)
_JAPANESE_NAME_REPLACEMENTS = {
    'momo': 'Thigh',
    'sippo': 'Tail',
    'shippo': 'Tail',
    'mune': 'Chest',
    'hiza': 'Knee',
    'hara': 'Stomach',
    'ashi': 'Leg',
    'hiji': 'Elbow',
    'koshi': 'Hips',
    'kubi': 'Neck',
    'atama': 'Head',
    'ago': 'Jaw',
    'kata': 'Shoulder',
    'munabire': 'Pectoral Fin',
    'sebire': 'Dorsal Fin',
    'harabire': 'Pelvic Fin',
    'shiribire': 'Anal Fin',
    'shirihire': 'Anal Fin',
    'obire': 'Caudal Fin',
    'tai': 'Tail',
}

# Romaji tokens that mark a rig as Japanese-named. Short or ambiguous tokens
# ("te", "o") are left out so one coincidental match cannot enable the gated
# replacements below.
_JAPANESE_EVIDENCE_TOKENS = frozenset({
    'momo', 'sippo', 'shippo', 'mune', 'hiza', 'hara', 'ashi', 'hiji',
    'koshi', 'kubi', 'atama', 'ago', 'kata',
    'munabire', 'sebire', 'harabire', 'shiribire', 'shirihire', 'obire',
})
_JAPANESE_EVIDENCE_MIN_DISTINCT = 3

# Replacements applied only to a rig confirmed Japanese-named: some are too short
# to map globally ("o" = tail).
_JAPANESE_GATED_REPLACEMENTS = {
    'kao': 'Head',
    'kosi': 'Hips',
    'o': 'Tail',
    'te': 'Hand',
    'era': 'Gill',
}
# Filler dropped from the embedding text: chain position ("tip", "end"), phalanx
# position ("proximal") and the canonical "Variant" uniqueness suffix. Limb
# position words (front/back/mid) are not filler -- they separate a fore limb
# from a hind one.
_EMBED_TEXT_SKIP_TOKENS = {
    'base',
    'tip',
    'nub',
    'end',
    'site',
    'proximal',
    'intermediate',
    'distal',
    'variant',
}
# Words that name no body part: rig markers ("Bip", "Xtra"), controls and tack
# ("Ctrl", "Saddle") and props ("Sword", "Quiver"). Dropped token by token, so a
# name that also carries anatomy keeps it ("XtraSpine" -> "Spine"); a name made
# only of these is blanked (zero embedding).
#
# Leave out words some rig uses for a body part or for motion that follows the
# body ("Fur", "Spline"), and cloth nouns rigs qualify with a direction
# ("CapeBack"): dropping them leaves a bare "Back" that reads as anatomy.
_EMBED_TEXT_NON_ANATOMICAL_TOKENS = {
    'all',
    'armor',
    'arrow',
    'aux',
    'backpack',
    'bag',
    'barrel',
    'base',
    'bip',
    'blade',
    'bnp',
    'bolt',
    'bone',
    'bow',
    'brain',
    'center',
    'cg',
    'chain',
    'container',
    'control',
    'controler',
    'copy',
    'cog',
    'ctrl',
    'dummy',
    'effects',
    'fan',
    'fire',
    'gold',
    'gun',
    'halo',
    'halter',
    'handle',
    'headband',
    'helb',
    'helper',
    'helt',
    'hub',
    'ik',
    'joint',
    'lftb',
    'locator',
    'mag',
    'magic',
    'main',
    'mesh',
    'mount',
    'node',
    'null',
    'opp',
    'passenger',
    'pick',
    'point',
    'pole',
    'ponitail',
    'ponytail',
    'position',
    'projectile',
    'prop',
    'quiver',
    'reins',
    'rftb',
    'robe',
    'saddle',
    'shield',
    'spear',
    'staff',
    'stick',
    'sword',
    'target',
    'trajectory',
    'weapon',
    'wood',
    'xtra',
}
# Side words dropped from the name. build_joint_embedding_texts re-attaches the
# side from the geometry-derived joint_side_labels, so every rig spells it the
# same way ("R_thigh", "RightThigh" -> "Right Thigh").
_EMBED_TEXT_SIDE_TOKENS = {
    'left',
    'right',
}
_EMBED_TEXT_HEAD_FEATURE_TOKENS = {
    'beard',
    'ear',
    'eye',
    'tongue',
}
# Species words glued onto joint names ("GorillaJaw", "MooseNeck"), stripped from
# canonical names and embedding text. Species is conditioned through
# ``species_emb``; left in, the word makes T5 cluster joints by species instead
# of body part. Joints that collide after stripping get a Variant suffix. Words
# that are also anatomy or rig vocabulary stay out ("ant" = antenna, "horse" in
# "HorseLink").
_EMBED_TEXT_CREATURE_TOKENS = {
    'antilope', 'bat', 'bear', 'bee', 'boar', 'buffalo', 'buzzard', 'camel',
    'cat', 'centipede', 'chicken', 'cobra', 'coyote', 'crab', 'cricket', 'crocodile',
    'crow', 'deer', 'dinosaur', 'dog', 'donkey', 'dragon', 'eagle', 'elephant',
    'elk', 'flamingo', 'fox', 'gazelle', 'goat', 'gorilla', 'hamster', 'hen',
    'hippopotamus', 'hound', 'hyena', 'jaguar', 'kappa', 'khitan', 'leapord',
    'leopard', 'lion', 'lynx', 'mammoth', 'monkey', 'moose', 'mouse', 'ostrich',
    'parrot', 'pigeon', 'puppy', 'quilin', 'rabbit', 'raptor', 'rat', 'rhino',
    'roach', 'sabrecat', 'scorpion', 'seagull', 'serpent', 'skunk', 'spider',
    'stego', 'tarantula', 'tiger', 'trex', 'tricera', 'tukan', 'turtle', 'tyranno',
    'wyvern',
}


def joint_name_token_is_species(token):
    """Whether one normalized joint-name token is a known species label."""
    clean_token = re.sub(r'[^a-z0-9]+', '', str(token or '').casefold())
    return clean_token in _EMBED_TEXT_CREATURE_TOKENS


# Quadruped limb codes (Lf/Rf/Lb/Rb = left/right fore/hind), decoded to their
# fore/hind half; the side comes from the geometry label.
_EMBED_TEXT_LIMB_CODE_TOKENS = {
    'lf': 'Front',
    'rf': 'Front',
    'lb': 'Back',
    'rb': 'Back',
}
# The same codes spelled fore/hind first (Fl/Br) plus a hexapod's middle pair
# (Lm/Rm, Ml/Mr). Ambiguous alone ("MouthBL" is bottom-left), so read only when
# the name also carries a limb word. A wing is a limb here: a cicada's hind wing
# is "wingBL".
_EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS = {
    'fl': 'Front',
    'fr': 'Front',
    'bl': 'Back',
    'br': 'Back',
    'lm': 'Mid',
    'rm': 'Mid',
    'ml': 'Mid',
    'mr': 'Mid',
}
_EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS = frozenset({'arm', 'leg', 'wing'})
# Top/bottom x left/right corners of a face part ("RigMouthTL" .. "RigMouthBR").
# Read only next to a face word, which is what tells "Bl" apart from a hind limb.
_EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS = {
    'tl': 'Upper',
    'tr': 'Upper',
    'bl': 'Lower',
    'br': 'Lower',
}
_EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS = frozenset({
    'mouth', 'lip', 'jaw', 'beak', 'eye', 'eyelid', 'brow', 'cheek',
})
# Lm/Rm on a head joint ("Head_LM01") is a mouth corner, not a middle limb. A limb
# word wins when a name carries both.
_EMBED_TEXT_HEAD_SIDE_CODE_TOKENS = {
    'lm': 'Mouth',
    'rm': 'Mouth',
}
_EMBED_TEXT_HEAD_SIDE_CODE_CONTEXT_TOKENS = frozenset({'head'})
# "Digit" is a finger on a hand and a toe on a foot; the limb word in the same
# name decides which.
_EMBED_TEXT_DIGIT_HAND_CONTEXT_TOKENS = frozenset({'arm', 'hand', 'palm'})
_EMBED_TEXT_DIGIT_FOOT_CONTEXT_TOKENS = frozenset({'leg', 'foot', 'ankle', 'toe', 'paw', 'hoof'})
# The limb word a rig repeats inside a distal joint's name ("LeftHandThumb1",
# "RigLLegAnkle"). The part already implies its limb, so the carrier is dropped
# and each finger, ankle or toe lands on one token across rig conventions.
# "Wing" is not a carrier: a wing digit belongs to a different limb.
_EMBED_TEXT_LIMB_CARRIER_TOKENS = frozenset({'Arm', 'Hand', 'Leg'})
# Distal parts that already name the limb they sit on, so a carrier in front of
# one is pure repetition.
_EMBED_TEXT_CARRIED_PART_TOKENS = frozenset({
    'Ankle', 'Finger', 'Foot', 'Hand', 'Heel', 'Hoof', 'Index', 'Little',
    'Middle', 'Paw', 'Pinky', 'Ring', 'Thumb', 'Toe', 'Wrist',
})
# Words ahead of a dropped carrier that say which limb ("LFLegAnkle" = fore
# ankle); they are kept. Behind the carrier they index one limb of many
# ("LegFront1") and the carrier stays.
_EMBED_TEXT_LIMB_QUALIFIER_TOKENS = frozenset({'Back', 'Front', 'Mid', 'Rear'})
# Synonyms, abbreviations and misspellings folded onto the corpus vocabulary, so
# one body part is one T5 point ("Femur" -> "Thigh", "Clav" -> "Clavicle"). Map
# a word by where the joint sits in the tree; ambiguous words ("Ball", "Belly")
# stay unmapped.
_EMBED_TEXT_SYNONYM_TOKENS = {
    # long bones -> the segment word the corpus uses
    'scapula': 'Clavicle',
    'humerus': 'UpperArm',
    'humer': 'UpperArm',
    'radius': 'Forearm',
    'ulna': 'Forearm',
    'carpal': 'Hand',
    'carpus': 'Hand',
    'metacarpal': 'Hand',
    'metacarpus': 'Hand',
    'femur': 'Thigh',
    'tibia': 'Calf',
    'fibula': 'Calf',
    'cannon': 'Foot',
    'tarsal': 'Foot',
    'metatarsal': 'Foot',
    'metatarsus': 'Foot',
    'phalanx': 'Toe',
    'phalanges': 'Toe',
    'palm': 'Hand',
    'collarbone': 'Clavicle',
    'feet': 'Foot',
    'lwing': 'Wing',
    'rwing': 'Wing',
    # head
    'mandible': 'Jaw',
    'chin': 'Jaw',
    'muzzle': 'Nose',
    'snout': 'Nose',
    'brow': 'Eyebrow',
    # trunk
    'ribcage': 'Chest',
    'thorax': 'Chest',
    'spline': 'Spine',
    # arthropod appendages, one family instead of three spellings
    'antenna': 'Feeler',
    'antennae': 'Feeler',
    'piers': 'Pincers',
    'pliers': 'Pincers',
    # misspellings
    'tounge': 'Tongue',
    'thouge': 'Tongue',
    'tunge': 'Tongue',
    'eyeleds': 'Eyelid',
    'scull': 'Head',
    'pevis': 'Pelvis',
    'shouder': 'Shoulder',
    'uppder': 'Upper',
    # an L -> R replace over mirrored bone names ("Rower_Arm_R", "Upper_Reg_R")
    'rower': 'Lower',
    'reg': 'Leg',
    # abbreviations echoing the full word in the same name ("SpineSpn0");
    # expanded so the adjacent-duplicate collapse removes the echo
    'thi': 'Thigh',
    'clf': 'Calf',
    'fot': 'Foot',
    'hnd': 'Hand',
    'uar': 'UpperArm',
    'far': 'Forearm',
    'clv': 'Clavicle',
    'nek': 'Neck',
    'spn': 'Spine',
    'tal': 'Tail',
    # standalone abbreviations
    'clav': 'Clavicle',
    'scap': 'Clavicle',
    'shin': 'Calf',
    'pelv': 'Pelvis',
    'chk': 'Cheek',
    'lips': 'Lip',
    'btm': 'Bottom',
}

# Vocabulary for splitting an all-lowercase glued name ("smallfrontarm") into
# words during canonicalization.
_COMPOUND_MODIFIER_TOKENS = frozenset({
    'back', 'big', 'bottom', 'down', 'first', 'fore', 'front', 'hind', 'inner',
    'large', 'left', 'long', 'low', 'lower', 'mid', 'middle', 'outer', 'outter',
    'rear', 'right', 'second', 'short', 'small', 'third', 'top', 'upper',
})
_COMPOUND_ANATOMY_TOKENS = frozenset({
    'ankle', 'arm', 'belly', 'body', 'calf', 'chest', 'claw', 'elbow', 'fat',
    'fin', 'finger', 'foot', 'forearm', 'hand', 'head', 'hoof', 'horn', 'jaw',
    'knee', 'leg', 'lip', 'neck', 'nose', 'palm', 'paw', 'spine', 'tail', 'thigh',
    'thumb', 'toe', 'tongue', 'tooth', 'wing', 'wrist',
})
_COMPOUND_SPLIT_VOCABULARY = _COMPOUND_MODIFIER_TOKENS | _COMPOUND_ANATOMY_TOKENS
# Real words that decompose into vocabulary entries and must stay whole
# ("ponytail", "eyebrow").
_COMPOUND_SPLIT_PROTECTED_TOKENS = frozenset({
    'backbone', 'collarbone', 'eyeball', 'eyebrow', 'eyelid', 'fingertip',
    'foreleg', 'headtop', 'ponytail', 'ribcage', 'toenail', 'topknot',
})
_COMPOUND_SPLIT_MIN_LENGTH = 6
_COMPOUND_SPLIT_MIN_PART_LENGTH = 3


def _split_glued_compound_token(token):
    """Segment an all-lowercase glued joint token into vocabulary words.

    Returns the parts (>= 2) when the *whole* token is covered by
    ``_COMPOUND_SPLIT_VOCABULARY``, otherwise None -- an all-or-nothing rule so
    an unknown word is never half-split into noise. Prefers the fewest parts,
    breaking ties toward the longest leading word.
    """
    if len(token) < _COMPOUND_SPLIT_MIN_LENGTH:
        return None
    if token in _COMPOUND_SPLIT_PROTECTED_TOKENS or token in _COMPOUND_SPLIT_VOCABULARY:
        return None

    best_by_start = [None] * (len(token) + 1)
    best_by_start[len(token)] = []
    for start in range(len(token) - _COMPOUND_SPLIT_MIN_PART_LENGTH, -1, -1):
        for end in range(len(token), start + _COMPOUND_SPLIT_MIN_PART_LENGTH - 1, -1):
            word = token[start:end]
            if word not in _COMPOUND_SPLIT_VOCABULARY:
                continue
            tail = best_by_start[end]
            if tail is None:
                continue
            candidate = [word] + tail
            if best_by_start[start] is None or len(candidate) < len(best_by_start[start]):
                best_by_start[start] = candidate

    parts = best_by_start[0]
    return parts if parts is not None and len(parts) >= 2 else None


# Bump whenever the text build_joint_embedding_texts produces for a joint can
# change: the token tables above, name canonicalization or refinement, or what
# goes into the sentence. Stored name embeddings are keyed by this version, so
# a bump makes the loader reject stale cond files until preprocessing re-runs.
JOINT_NAME_EMBEDDING_SCHEMA_VERSION = 16

# The text the pipeline encodes: body part and side only; structure-derived
# tokens belong to the structural channel. ``slim=False`` adds them back for
# offline comparison scripts, never for a training corpus.
JOINT_NAME_EMBEDDING_SLIM = True

_CHAIN_INDEX_ORDINAL_TOKENS = {
    1: 'First',
    2: 'Second',
    3: 'Third',
    4: 'Fourth',
    5: 'Fifth',
    6: 'Sixth',
    7: 'Seventh',
    8: 'Eighth',
    9: 'Ninth',
    10: 'Tenth',
}
def normalize_joint_name(name):
    # Split on lowercase→UPPER (e.g. "ElkRFemur" → "Elk RFemur")
    split_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    # Also split on UPPER→UPPER+lower (e.g. "RFemur" → "R Femur")
    split_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', split_name)
    split_name = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', split_name)
    split_name = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', split_name)
    return re.sub(r'[^a-z0-9]+', ' ', split_name.lower()).strip()


def _has_joint_name_prefix(name, prefix, *, case_sensitive=True):
    """Return whether *prefix* is one complete leading identifier token."""
    name = str(name or '')
    prefix = str(prefix or '')
    leading = name[:len(prefix)]
    prefix_matches = leading == prefix if case_sensitive else leading.casefold() == prefix.casefold()
    if not prefix or not prefix_matches:
        return False

    prefix_end = len(prefix)
    return (
        prefix_end == len(name)
        or not name[prefix_end].isalnum()
        or name[prefix_end].isupper()
        or name[prefix_end].isdigit()
    )


def infer_species_joint_name_prefixes(joint_names, species_name=None):
    """Infer a character/species prefix shared by the whole skeleton.

    Dataset identifiers commonly include a pack code (``IAC_Caveman``), while
    their bones use only the species suffix (``Caveman Pelvis``).  Generate all
    separator-preserving suffix forms and accept only the longest form that is
    a complete leading token on *every* joint.  The all-joints gate is what keeps
    an anatomical name such as ``HorseLink`` intact on an ordinary Horse rig.
    """
    names = [] if joint_names is None else [str(name or '') for name in joint_names]
    if not names or not species_name:
        return ()

    bare_species = str(species_name).replace('\\', '/').rsplit('/', 1)[-1]
    parts = [part for part in re.split(r'[^0-9A-Za-z]+', bare_species) if part]
    candidates = set()
    for start in range(len(parts)):
        suffix = parts[start:]
        if not any(any(character.isalpha() for character in part) for part in suffix):
            continue
        candidates.update({
            ''.join(suffix),
            ' '.join(suffix),
            '_'.join(suffix),
            '-'.join(suffix),
        })

    for candidate in sorted(candidates, key=lambda value: (len(value), value), reverse=True):
        if all(
            len(name) > len(candidate)
            and _has_joint_name_prefix(name, candidate, case_sensitive=False)
            for name in names
        ):
            return (candidate,)
    return ()


def strip_joint_name_prefix(name, additional_prefixes=()):
    stripped = name
    prefixes = (
        *((prefix, False) for prefix in tuple(additional_prefixes or ())),
        *((prefix, True) for prefix in _CANONICAL_NAME_PREFIXES),
    )
    for prefix, case_sensitive in sorted(prefixes, key=lambda item: len(item[0]), reverse=True):
        # Prefixes are complete rig/character tokens, not arbitrary character
        # sequences.  A following separator, digit, or CamelCase boundary is
        # valid ("Rig_Head", "Rig01", "RigHead"); a lowercase continuation is
        # not ("RightArm", "RigidBody", "Belly").  This boundary check is
        # especially important for the short Unity prefix "Rig".
        if _has_joint_name_prefix(stripped, prefix, case_sensitive=case_sensitive):
            stripped = stripped[len(prefix):]
            break
    # Strip known rig suffixes (case-insensitive), but never reduce the name
    # to an empty string (e.g. a joint literally named "SHJnt").
    for suffix in sorted(_CANONICAL_NAME_SUFFIXES, key=len, reverse=True):
        suffix_len = len(suffix)
        if len(stripped) > suffix_len and stripped[-suffix_len:].lower() == suffix.lower():
            stripped = stripped[:-suffix_len]
            break
    return stripped


def is_japanese_style_naming(joint_names):
    """True when the joint name set shows clear Japanese romaji rig naming.

    Requires at least ``_JAPANESE_EVIDENCE_MIN_DISTINCT`` distinct unambiguous
    romaji tokens so that a single coincidental match cannot trigger the gated
    Japanese-only replacements.
    """
    if not joint_names:
        return False
    seen = set()
    for name in joint_names:
        for token in normalize_joint_name(name).split():
            if token in _JAPANESE_EVIDENCE_TOKENS:
                seen.add(token)
                if len(seen) >= _JAPANESE_EVIDENCE_MIN_DISTINCT:
                    return True
    return False


def effective_canonical_replacements(joint_names):
    """Base canonical replacements, plus Japanese-only entries when warranted.

    Falls back to the shared ``_JAPANESE_NAME_REPLACEMENTS`` object (no copy)
    unless the skeleton is confirmed Japanese-style, in which case the gated
    ``kao``/``kosi``/``o`` mappings are merged in.
    """
    if is_japanese_style_naming(joint_names):
        return {**_JAPANESE_NAME_REPLACEMENTS, **_JAPANESE_GATED_REPLACEMENTS}
    return _JAPANESE_NAME_REPLACEMENTS


def _collapse_repeated_name_parts(canonical_parts):
    """Drop words a rig name repeats verbatim.

    Rigs that encode the parent path *and* the joint's own name emit the same
    words twice ("Sabrecat_HeadLeftEar_LEar_" -> "Head Left Ear Left Ear"). Two
    exact-match rules, so they can only ever remove a verbatim echo: collapse an
    adjacent duplicate, then drop a trailing block that repeats the block right
    before it.

    Numeric parts are exempt: repeated digits are two index fields that happen to
    hold the same value, not an echo. Boar's "LEFT_Ear_01_01SHJnt" is chain 01
    segment 01 -- its siblings "..._01_02" and "..._01_03" prove it -- so
    collapsing it would desync one member of a chain from the rest.

    A trailing *abbreviation* of an earlier word ("LeftThigh_LThi_") is
    deliberately left alone too: that would take a prefix heuristic, and the
    three joints it covers do not justify the risk of eating a real short word.
    """
    collapsed = []
    for part in canonical_parts:
        if collapsed and collapsed[-1] == part and not part.isdigit():
            continue
        collapsed.append(part)

    for block_length in range(2, len(collapsed) // 2 + 1):
        block = collapsed[-block_length:]
        if any(part.isdigit() for part in block):
            continue
        if block == collapsed[-2 * block_length:-block_length]:
            return collapsed[:-block_length]
    return collapsed


def _drop_species_name_parts(canonical_parts):
    return [part for part in canonical_parts if not joint_name_token_is_species(part)]


def _canonicalize_joint_name(name, replacements=None, additional_prefixes=()):
    replacements = _JAPANESE_NAME_REPLACEMENTS if replacements is None else replacements
    split_name = normalize_joint_name(strip_joint_name_prefix(name, additional_prefixes))
    canonical_parts = []
    for part in split_name.split():
        clean_part = re.sub(r'[^a-z0-9]+', '', part)
        if not clean_part:
            continue
        if clean_part in ('l', 'left'):
            canonical_parts.append('Left')
        elif clean_part in ('r', 'right'):
            canonical_parts.append('Right')
        elif clean_part in replacements:
            canonical_parts.append(replacements[clean_part])
        elif len(clean_part) == 1:
            # Skip single letters (except digits which are preserved for disambiguation)
            if not clean_part.isdigit():
                continue
            canonical_parts.append(clean_part)
        else:
            compound_parts = _split_glued_compound_token(clean_part)
            if compound_parts is None:
                canonical_parts.append(clean_part.capitalize())
            else:
                canonical_parts.extend(part.capitalize() for part in compound_parts)

    canonical_parts = _collapse_repeated_name_parts(canonical_parts)
    canonical_parts = _drop_species_name_parts(canonical_parts)
    return ' '.join(canonical_parts) if canonical_parts else name.strip()


def _titlecase_identifier_tokens(value):
    normalized = normalize_joint_name(str(value))
    if not normalized:
        return []
    return [token.capitalize() for token in normalized.split() if token]


def _collapse_solitary_head_feature_indices(canonical_joint_names):
    normalized_tokens = [normalize_joint_name(name).split() for name in canonical_joint_names]
    base_counts = Counter(
        tuple(tokens[:-1])
        for tokens in normalized_tokens
        if len(tokens) >= 2
        and tokens[-1].isdigit()
        and any(token in _EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
    )

    collapsed_names = []
    for name, tokens in zip(canonical_joint_names, normalized_tokens):
        if (
            len(tokens) >= 2
            and tokens[-1].isdigit()
            and any(token in _EMBED_TEXT_HEAD_FEATURE_TOKENS for token in tokens[:-1])
            and base_counts[tuple(tokens[:-1])] == 1
        ):
            collapsed_names.append(' '.join(token.capitalize() for token in tokens[:-1]))
            continue
        collapsed_names.append(name)
    return collapsed_names


def _species_motion_tokens(object_cond):
    object_type = str(object_cond.get('object_type') or '').strip()
    if not object_type:
        return []
    return list(dataset_tags().tags_for(object_type))


def build_species_embedding_text(object_cond):
    """Return the text describing a species as a whole, encoded once per object
    type into a single ``species_emb`` (T5) vector that conditions the whole
    network -- as opposed to ``build_joint_embedding_texts``, which describes
    each joint. This is the one place to refine the species descriptor; keep it
    open-vocabulary text so novel species still map into the same T5 space.

    Returns the motion-relevant body-plan/dynamics tags from ``species_tags.jsonl``,
    which describe how the animal moves -- the axis that matters for motion and
    that topology alone can't supply. There is no fallback: every species MUST
    be registered in ``species_tags.jsonl`` (enforced by
    assert_species_tags_cover at preprocessing/training time).
    """
    motion_tokens = _species_motion_tokens(object_cond)
    if not motion_tokens:
        object_type = str(object_cond.get('object_type') or '').strip() or '<empty>'
        raise SystemExit(
            f"\033[91mNo species_tags.jsonl entry for object_type '{object_type}'. "
            "Register it in the species_tags.jsonl sidecar.\033[0m"
        )
    return ' '.join(motion_tokens)


# Adjacent tokens that name one part together ("upper leg" -> Thigh). Applied
# before the per-token substitutions, which would otherwise rewrite "arm".
_EMBED_TEXT_TOKEN_PAIR_MERGES = {
    ('upper', 'leg'): 'Thigh',
    ('up', 'leg'): 'Thigh',
    ('fore', 'arm'): 'Forearm',
    ('fore', 'leg'): 'Foreleg',
    ('upper', 'arm'): 'UpperArm',
    ('lower', 'arm'): 'Forearm',
    # below an UpperLeg; Calf even on a fore limb, matching ('upper', 'leg')
    ('lower', 'leg'): 'Calf',
    # Same three segments as spelled by the L -> R mirrored names above.
    ('rower', 'arm'): 'Forearm',
    ('upper', 'reg'): 'Thigh',
    ('rower', 'reg'): 'Calf',
    ('lower', 'reg'): 'Calf',
    # 3ds Max Biped's digitigrade link, Thigh -> Calf -> HorseLink -> Foot
    ('horse', 'link'): 'Ankle',
    # keeps an eyelid from reading as "Eye" (HeadFeature) + "Lid"
    ('eye', 'lid'): 'Eyelid',
    ('eye', 'lids'): 'Eyelid',
}


def _bare_arm_means_upper_arm(joint_names, parents):
    """Per-joint flag: is a "ForeArm" named further down this joint's limb?

    Mixamo-style rigs call the upper arm "Arm" and the next segment "ForeArm",
    so a bare "Arm" there really is the upper arm. Arthropod rigs use "Arm" for
    a whole multi-segment limb and never name a forearm (Crab "BN_Arm_L_01..04",
    Spider "ArmR_01_" -> "ArmRClaw"), and FireAnt hangs an "Arm_Nub" off the
    hand -- mapping those to UpperArm mislabels 48 joints across 6 species, so
    the rewrite is gated on this signal instead of firing unconditionally.
    """
    joint_count = len(joint_names)
    if parents is None or len(parents) != joint_count:
        return [False] * joint_count

    parents = np.asarray(parents, dtype=np.int64)
    is_forearm = [
        'forearm' in normalize_joint_name(str(name)).replace(' ', '')
        for name in joint_names
    ]
    # Same reverse-index sweep as _build_chain_relative_joint_tokens: a child
    # always has a higher index than its parent in these rigs.
    has_forearm_below = [False] * joint_count
    for joint_index in range(joint_count - 1, 0, -1):
        parent_index = int(parents[joint_index])
        if parent_index >= 0 and (is_forearm[joint_index] or has_forearm_below[joint_index]):
            has_forearm_below[parent_index] = True
    return has_forearm_below


# A bare "Leg" between a thigh and a foot is the shank; these sets spell the
# thigh, the foot and the bare leg for that test.
_EMBED_TEXT_THIGH_NAME_TOKENS = frozenset({'thigh', 'upleg', 'upperleg'})
_EMBED_TEXT_DISTAL_LEG_TOKENS = frozenset({
    'foot', 'ankle', 'toe', 'toes', 'paw', 'hoof', 'heel', 'ball', 'tarsus', 'hock',
})
_EMBED_TEXT_BARE_LEG_TOKENS = frozenset({'leg', 'reg'})
# How many links along the limb the thigh and foot may sit: spans a full leg
# without letting an arthropod's long "Leg" chain reach another limb's thigh.
_BARE_LEG_CONTEXT_MAX_LINKS = 4


def _bare_leg_means_calf(joint_names, parents, end_effector_joints=(), additional_prefixes=()):
    """Per-joint flag: is this bare "Leg" the segment *below* a named thigh?

    The leg-side counterpart of ``_bare_arm_means_upper_arm``. In Mixamo-style
    rigs the thigh is "UpLeg" and the shank is "Leg"; the pair merge reads the
    first correctly but leaves the shank as a naked "Leg" (nearest T5 neighbour
    is "Arm", which hinges the knee the wrong way).

    Gated hard because a bare "Leg" means something else in most other rigs
    (Crab/tarantula leg chains with no thigh, Bear "LLeg1/LLeg2", Alligator
    "L_ashi" as ground contact). In the current corpus the gates fire on only
    the Rat's and the Hen's shanks.
    """
    joint_count = len(joint_names)
    if parents is None or len(parents) != joint_count:
        return [False] * joint_count

    parents = np.asarray(parents, dtype=np.int64)
    tokens_per_joint = [
        _body_clean_tokens(str(name), additional_prefixes=additional_prefixes)[1]
        for name in joint_names
    ]

    def names_a_thigh(tokens):
        if _EMBED_TEXT_THIGH_NAME_TOKENS & set(tokens):
            return True
        # "UpLeg" arrives here as the two tokens the pair-merge table folds into
        # a Thigh, so ask that table rather than restating its keys.
        return any(
            _EMBED_TEXT_TOKEN_PAIR_MERGES.get((first, second)) == 'Thigh'
            for first, second in zip(tokens, tokens[1:])
        )

    children = _child_map(parents)
    end_effectors = {int(joint_index) for joint_index in (end_effector_joints or ())}
    flags = [False] * joint_count
    for joint_index in range(joint_count):
        own_tokens = tokens_per_joint[joint_index]
        if len(own_tokens) != 1 or own_tokens[0] not in _EMBED_TEXT_BARE_LEG_TOKENS:
            continue
        # A bare "Leg" with nothing below it is the foot, not the shank.
        if joint_index in end_effectors or not children[joint_index]:
            continue

        ancestor = int(parents[joint_index])
        has_thigh_above = False
        for _ in range(_BARE_LEG_CONTEXT_MAX_LINKS):
            if ancestor < 0:
                break
            if names_a_thigh(tokens_per_joint[ancestor]):
                has_thigh_above = True
                break
            ancestor = int(parents[ancestor])
        if not has_thigh_above:
            continue

        has_foot_below = False
        frontier = list(children[joint_index])
        for _ in range(_BARE_LEG_CONTEXT_MAX_LINKS):
            if not frontier or has_foot_below:
                break
            next_frontier = []
            for descendant in frontier:
                if _EMBED_TEXT_DISTAL_LEG_TOKENS & set(tokens_per_joint[descendant]):
                    has_foot_below = True
                    break
                next_frontier.extend(children[descendant])
            frontier = next_frontier
        flags[joint_index] = has_foot_below
    return flags


def _refine_joint_embedding_tokens(clean_token, bare_arm_is_upper_arm=False,
                                   quadrant_codes_name_a_limb=False,
                                   digit_limb=None, bare_leg_is_calf=False,
                                   side_codes_name_a_head=False,
                                   quadrant_codes_name_a_face=False):
    """Map one canonical token to the embedding token(s) it contributes."""
    if clean_token == 'digit' and digit_limb is not None:
        return ['Finger'] if digit_limb == 'hand' else ['Toe']
    limb_code_token = _EMBED_TEXT_LIMB_CODE_TOKENS.get(clean_token)
    if limb_code_token is not None:
        return [limb_code_token]
    if quadrant_codes_name_a_limb:
        quadrant_token = _EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS.get(clean_token)
        if quadrant_token is not None:
            return [quadrant_token]
    else:
        if side_codes_name_a_head:
            head_code_token = _EMBED_TEXT_HEAD_SIDE_CODE_TOKENS.get(clean_token)
            if head_code_token is not None:
                return [head_code_token]
        if quadrant_codes_name_a_face:
            face_code_token = _EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS.get(clean_token)
            if face_code_token is not None:
                return [face_code_token]
    # Ahead of the synonym lookup: 'reg' is a synonym-mapped spelling of 'leg',
    # and a bare 'reg' below a thigh is a Calf just like a bare 'leg'.
    if bare_leg_is_calf and clean_token in _EMBED_TEXT_BARE_LEG_TOKENS:
        return ['Calf']
    synonym_token = _EMBED_TEXT_SYNONYM_TOKENS.get(clean_token)
    if synonym_token is not None:
        return [synonym_token]
    if clean_token in ('sippo', 'tai') or clean_token.startswith('tail'):
        return ['Tail']
    if clean_token.startswith('toe'):
        return ['Toe']
    if clean_token.startswith('finger'):
        return ['Finger']
    if clean_token == 'arm':
        return ['UpperArm'] if bare_arm_is_upper_arm else ['Arm']
    if clean_token in ('fore', 'forearm'):
        return ['Forearm']
    if clean_token == 'upleg':
        return ['UpperLeg']
    if clean_token == 'clip':
        return ['Appendage']
    if clean_token in _EMBED_TEXT_HEAD_FEATURE_TOKENS:
        # Emit the specific word *and* the shared category. The category token
        # keeps every head appendage close together in T5 space (the point of
        # the grouping), while the specific word stops Jaguar's Eye, Ear and
        # Beard from collapsing onto one identical "HeadFeature Right".
        return [clean_token.capitalize(), 'HeadFeature']
    return [clean_token.capitalize()]


def _clean_embedding_token(token):
    """Lower-case, strip punctuation, drop a trailing index run.

    The token tables are all keyed on this form, so every lookup against them
    has to go through here -- comparing a raw token instead silently misses
    anything spelled with punctuation or an index ("BN_P", "Bip01").
    """
    cleaned = re.sub(r'[^a-z0-9]+', '', token.lower())
    return re.sub(r'\d+$', '', cleaned)


def joint_name_is_non_anatomical(name, additional_prefixes=()):
    """True when a joint's name carries no body-part token.

    Reuses the embedding vocabulary: a joint named ``Sword``, ``Quiver`` or
    ``Backpack`` (or a pure marker like ``joint1``/``Bone02`` that reduces to
    nothing) is taken at its word, with the same test
    ``build_joint_embedding_texts`` uses to blank body tokens.

    A *name* signal only -- armor and saddles are non-anatomical but still
    part of the character's size; pair with geometry (``find_prop_socket_joints``).
    """
    tokens = {
        _clean_embedding_token(token)
        for token in _refine_joint_embedding_name(name, additional_prefixes=additional_prefixes)
    }
    tokens.discard('')
    return not tokens or bool(tokens & _EMBED_TEXT_NON_ANATOMICAL_TOKENS)


def _body_clean_tokens(name, additional_prefixes=()):
    """The canonical body tokens of one joint name, before any refinement.

    Shared by ``_refine_joint_embedding_name`` and the per-skeleton context
    flags it takes (``_bare_leg_means_calf``), so a flag is decided on exactly
    the token list the refinement will see rather than on a substring of the raw
    name.
    """
    canonical_name = _canonicalize_joint_name(name, additional_prefixes=additional_prefixes)
    clean_tokens = []
    for token in canonical_name.split():
        clean_token = _clean_embedding_token(token)
        if not clean_token or clean_token.isdigit() or clean_token in _EMBED_TEXT_SKIP_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_SIDE_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_CREATURE_TOKENS:
            continue
        clean_tokens.append(clean_token)
    return canonical_name, clean_tokens


def _drop_redundant_limb_carrier(refined_tokens):
    """Strip the carrier limb from a "[qualifier] <carrier> <part>" name.

    The carrier has to sit *immediately* in front of the part, with nothing after
    it, because that shape is what makes it a namespace prefix rather than
    anatomy. Anything else in the name means the words are doing work: Raptor2's
    "jt_HandClawMiddle_L" is [Hand, Claw, Middle], the middle claw of the *hand*,
    and the same rig carries a "Claw Middle" on the foot for it to collide with;
    RU01_BotRobot's "RigLArmFingerIn1" is [Arm, Finger, In]; and a spiderling's
    "RigLLegFront1" is [Leg, Front], where the leg word is the only anatomy
    there is.
    """
    if len(refined_tokens) < 2:
        return refined_tokens
    qualifier_tokens, (carrier_token, part_token) = refined_tokens[:-2], refined_tokens[-2:]
    if carrier_token not in _EMBED_TEXT_LIMB_CARRIER_TOKENS:
        return refined_tokens
    if part_token not in _EMBED_TEXT_CARRIED_PART_TOKENS:
        return refined_tokens
    if any(token not in _EMBED_TEXT_LIMB_QUALIFIER_TOKENS for token in qualifier_tokens):
        return refined_tokens
    return [*qualifier_tokens, part_token]


def _refine_joint_embedding_name(name, bare_arm_is_upper_arm=False, additional_prefixes=(),
                                 bare_leg_is_calf=False):
    canonical_name, clean_tokens = _body_clean_tokens(name, additional_prefixes=additional_prefixes)

    # Gate for the ambiguous fore/hind codes: only a name that also spells a limb
    # gets them decoded, so a mouth corner keeps its "Bl" and a leg does not.
    quadrant_codes_name_a_limb = bool(
        set(clean_tokens) & _EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS
    )
    side_codes_name_a_head = bool(set(clean_tokens) & _EMBED_TEXT_HEAD_SIDE_CODE_CONTEXT_TOKENS)
    quadrant_codes_name_a_face = bool(set(clean_tokens) & _EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS)
    # Which limb a "Digit" belongs to, from the same name. Only an unambiguous
    # single side of the fork is read; a name that says both (or neither) keeps
    # the bare word.
    names_a_hand = bool(set(clean_tokens) & _EMBED_TEXT_DIGIT_HAND_CONTEXT_TOKENS)
    names_a_foot = bool(set(clean_tokens) & _EMBED_TEXT_DIGIT_FOOT_CONTEXT_TOKENS)
    digit_limb = (
        'hand' if names_a_hand and not names_a_foot
        else 'foot' if names_a_foot and not names_a_hand
        else None
    )

    merged_tokens = []
    index = 0
    while index < len(clean_tokens):
        merged_token = _EMBED_TEXT_TOKEN_PAIR_MERGES.get(tuple(clean_tokens[index:index + 2]))
        if merged_token is not None:
            merged_tokens.append(merged_token)
            index += 2
            continue
        merged_tokens.extend(
            _refine_joint_embedding_tokens(
                clean_tokens[index],
                bare_arm_is_upper_arm,
                quadrant_codes_name_a_limb=quadrant_codes_name_a_limb,
                digit_limb=digit_limb,
                bare_leg_is_calf=bare_leg_is_calf,
                side_codes_name_a_head=side_codes_name_a_head,
                quadrant_codes_name_a_face=quadrant_codes_name_a_face,
            )
        )
        index += 1

    # Collapse an adjacent repeat left behind by the mappings above. Rigs that
    # spell a part twice in one name -- "LeftThighLeftThi", "SpineSpn0",
    # "NeckNek0" -- only look like two tokens until the abbreviation is expanded;
    # a body part never legitimately repeats back-to-back.
    deduped_tokens = [
        token for position, token in enumerate(merged_tokens)
        if position == 0 or token != merged_tokens[position - 1]
    ]
    if deduped_tokens:
        return _drop_redundant_limb_carrier(deduped_tokens)

    # Nothing survived, so hand back the raw canonical tokens -- minus the side
    # word, which build_joint_embedding_texts is about to re-attach from the
    # geometry label. spider_tarantula names its leg segments "R4_00".."L2_03",
    # which reduce to a side word plus two index runs; keeping the side word here
    # spelled it twice ("Right Right 4 00"). The indices stay: for these 24
    # joints they are the only thing telling leg 4 from leg 2.
    fallback_tokens = [
        token for token in canonical_name.split()
        if _clean_embedding_token(token) not in _EMBED_TEXT_SIDE_TOKENS
    ]
    return fallback_tokens or canonical_name.split()


def _chain_index_token(index):
    index = int(index)
    return _CHAIN_INDEX_ORDINAL_TOKENS.get(index, f'Index{index}')


def _chain_role_token(chain_index, chain_length):
    chain_index = int(chain_index)
    chain_length = int(chain_length)
    if chain_length <= 1:
        return None
    if chain_index <= 1:
        return 'ChainStart'
    if chain_index >= chain_length:
        return 'ChainEnd'
    relative_position = float(chain_index - 1) / float(max(chain_length - 1, 1))
    if relative_position <= 0.34:
        return 'ChainEarly'
    if relative_position >= 0.67:
        return 'ChainLate'
    return 'ChainMiddle'


def _build_chain_relative_joint_tokens(refined_tokens_per_joint, parents):
    joint_count = len(refined_tokens_per_joint)
    if parents is None or len(parents) != joint_count:
        return [[] for _ in range(joint_count)]

    parents = np.asarray(parents, dtype=np.int64)
    children = _child_map(parents)
    signatures = [tuple(tokens) for tokens in refined_tokens_per_joint]
    upward_steps = np.zeros(joint_count, dtype=np.int32)
    downward_steps = np.zeros(joint_count, dtype=np.int32)

    for joint_index in range(joint_count):
        parent_index = int(parents[joint_index])
        if parent_index >= 0 and signatures[parent_index] and signatures[parent_index] == signatures[joint_index]:
            upward_steps[joint_index] = upward_steps[parent_index] + 1

    for joint_index in range(joint_count - 1, -1, -1):
        matching_children = [
            child_index
            for child_index in children[joint_index]
            if signatures[joint_index] and signatures[child_index] == signatures[joint_index]
        ]
        if matching_children:
            downward_steps[joint_index] = 1 + max(downward_steps[child_index] for child_index in matching_children)

    chain_lengths = upward_steps + downward_steps + 1
    chain_tokens = []
    for joint_index in range(joint_count):
        signature = signatures[joint_index]
        chain_length = int(chain_lengths[joint_index])
        if not signature or chain_length <= 1:
            chain_tokens.append([])
            continue

        chain_index = int(upward_steps[joint_index]) + 1
        joint_tokens = ['Segment', _chain_index_token(chain_index), 'Of', str(chain_length)]
        role_token = _chain_role_token(chain_index, chain_length)
        if role_token is not None:
            joint_tokens.append(role_token)
        chain_tokens.append(joint_tokens)

    return chain_tokens


def _sibling_instance_tokens(body_tokens_per_joint, flag_tokens_per_joint, symmetry_partner_indices):
    """Number the joints that would otherwise share one identical text.

    Whatever still collides here is a *sibling* repeat -- a centipede's leg
    pairs, a bat's wing fingers -- which the chain tokens cannot separate
    because the joints do not sit on one parent-child run.

    The ordinal is a plain within-group index, deliberately not a geometric one.
    Ordering siblings along a body axis would need that axis signed (raw PCA
    would mirror-flip left against right), and the only thing it buys over array
    order is mirror consistency -- which ``symmetry_partner_indices`` already
    delivers exactly: propagating ranks across the symmetry links matches
    742/742 paired joints, against 566/742 for bare array order. What is given
    up is cross-species comparability: "Instance First" is a within-skeleton id,
    not "the front-most pair".
    """
    groups = defaultdict(list)
    for joint_index, (body_tokens, flag_tokens) in enumerate(zip(body_tokens_per_joint, flag_tokens_per_joint)):
        if body_tokens:
            groups[' '.join([*body_tokens, *flag_tokens])].append(joint_index)
    groups = {text: indices for text, indices in groups.items() if len(indices) > 1}
    if not groups:
        return [[] for _ in body_tokens_per_joint]

    partners = list(symmetry_partner_indices or [])
    ranks = {}
    # Rank the earliest group by array order, then let each later group inherit
    # its ranks through the symmetry links whenever that yields a clean
    # one-to-one match; otherwise fall back to array order for that group too.
    for text in sorted(groups, key=lambda text: min(groups[text])):
        indices = sorted(groups[text])
        partner_indices = [
            int(partners[joint_index]) if joint_index < len(partners) else -1
            for joint_index in indices
        ]
        partner_ranks = [ranks[partner_index] for partner_index in partner_indices if partner_index in ranks]
        if len(partner_ranks) == len(indices) and len(set(partner_ranks)) == len(indices):
            ranks.update(zip(indices, partner_ranks))
            continue
        ranks.update((joint_index, rank) for rank, joint_index in enumerate(indices))

    instance_tokens = [[] for _ in body_tokens_per_joint]
    for indices in groups.values():
        for joint_index in indices:
            instance_tokens[joint_index] = [
                'Instance', _chain_index_token(ranks[joint_index] + 1), 'Of', str(len(indices)),
            ]
    return instance_tokens


def build_joint_embedding_texts(object_cond, slim=JOINT_NAME_EMBEDDING_SLIM):
    """Per-joint T5 sentences: side + body part, and (only when ``slim=False``)
    the structure-derived tokens the pre-v14 schema also spelled.

    Slimming is done here, at token construction, rather than by deleting words
    from a finished string: the chain grouping and the instance numbering read
    each other, so a post-hoc blacklist would leave whichever of them happened to
    survive keyed on a signature that no longer exists.
    """
    base_joint_names = object_cond.get('canonical_joint_names') or object_cond.get('joints_names') or []
    if not base_joint_names:
        return []

    raw_joint_names = list(object_cond.get('joints_names') or base_joint_names)
    species_prefixes = infer_species_joint_name_prefixes(
        raw_joint_names,
        object_cond.get('species_name') or object_cond.get('object_type'),
    )
    joint_side_labels = list(object_cond.get('joint_side_labels') or ['center'] * len(base_joint_names))
    contact_joints = {int(joint_index) for joint_index in list(object_cond.get('contact_joints') or [])}
    end_effector_joints = {int(joint_index) for joint_index in list(object_cond.get('end_effector_joints') or [])}
    bare_arm_flags = _bare_arm_means_upper_arm(
        raw_joint_names,
        object_cond.get('parents'),
    )
    # Read off the canonical names, not the raw ones: the thigh above a Mixamo
    # shank is spelled "UpLeg", which only becomes a Thigh after canonicalization
    # splits it and the pair-merge table folds it.
    bare_leg_flags = _bare_leg_means_calf(
        base_joint_names,
        object_cond.get('parents'),
        end_effector_joints=object_cond.get('end_effector_joints') or (),
        additional_prefixes=species_prefixes,
    )
    refined_tokens_per_joint = [
        _refine_joint_embedding_name(
            joint_name,
            bare_arm_flags[joint_index],
            additional_prefixes=species_prefixes,
            bare_leg_is_calf=bare_leg_flags[joint_index],
        )
        for joint_index, joint_name in enumerate(base_joint_names)
    ]
    if slim:
        chain_relative_tokens = [[] for _ in refined_tokens_per_joint]
    else:
        # Chain grouping stays side-aware even though the side word is emitted only
        # once, at the end: without it a midline trunk (Buzzard "Tail 01") shares a
        # signature with its left and right forks and swallows both into one chain.
        chain_signature_tokens = [
            [*tokens, joint_side_labels[joint_index] if joint_index < len(joint_side_labels) else 'center']
            if tokens else []
            for joint_index, tokens in enumerate(refined_tokens_per_joint)
        ]
        chain_relative_tokens = _build_chain_relative_joint_tokens(chain_signature_tokens, object_cond.get('parents'))

    body_tokens_per_joint = []
    flag_tokens_per_joint = []
    for joint_index, joint_name in enumerate(base_joint_names):
        refined_tokens = refined_tokens_per_joint[joint_index]
        # Same cleaning the table lookups use. It matters on the fallback path:
        # a name made *only* of markers comes back as the raw canonical tokens
        # ("BN_P", "Bip01"), which a bare .lower() cannot match against the set.
        lowered_tokens = {_clean_embedding_token(token) for token in refined_tokens}
        if lowered_tokens & _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            body_tokens_per_joint.append([])
            flag_tokens_per_joint.append([])
            continue

        # Side leads, so the text opens with the identity attributes as a plain
        # English noun phrase ("Right Finger ...") -- a construction T5 saw in
        # pretraining, unlike a trailing "Right" stranded after chain jargon.
        # Everything derived (chain position, contact, end effector) follows.
        side = joint_side_labels[joint_index] if joint_index < len(joint_side_labels) else 'center'
        body_tokens = [side.capitalize()] if side in ('left', 'right') else []
        body_tokens.extend(refined_tokens)
        body_tokens.extend(chain_relative_tokens[joint_index])
        body_tokens_per_joint.append(body_tokens)

        flag_tokens = []
        if not slim:
            if joint_index in contact_joints:
                flag_tokens.append('Contact')
            if joint_index in end_effector_joints:
                flag_tokens.append('EndEffector')
        flag_tokens_per_joint.append(flag_tokens)

    if slim:
        # Repeated siblings (a centipede's leg pairs, a bat's wing fingers) are
        # left sharing one text on purpose: numbering them is a within-skeleton
        # id, which is exactly what sib_rank / fore_aft_n / lateral_signed encode
        # in the structural channel -- and there they are comparable across
        # species, which "Instance First Of 22" never was.
        instance_tokens_per_joint = [[] for _ in body_tokens_per_joint]
    else:
        # Instance ordinals sit with the other positional tokens, ahead of the
        # derived Contact/EndEffector flags.
        instance_tokens_per_joint = _sibling_instance_tokens(
            body_tokens_per_joint, flag_tokens_per_joint, object_cond.get('symmetry_partner_indices')
        )
    return [
        ' '.join([*body_tokens, *instance_tokens, *flag_tokens]) if body_tokens else ''
        for body_tokens, instance_tokens, flag_tokens
        in zip(body_tokens_per_joint, instance_tokens_per_joint, flag_tokens_per_joint)
    ]


# Limb and face corner codes in the symmetry signature: the side half is dropped
# and the fore/hind/middle (or top/bottom) half kept, so LfLeg01 pairs only with
# RfLeg01 and MouthTL only with MouthTR.
_LIMB_CODE_SIGNATURE_TOKENS = {
    'lf': 'f', 'rf': 'f', 'lb': 'b', 'rb': 'b',
    'fl': 'f', 'fr': 'f', 'bl': 'b', 'br': 'b',
    'lm': 'm', 'rm': 'm', 'ml': 'm', 'mr': 'm',
    'tl': 't', 'tr': 't',
}


# Spelling repairs for the symmetry signature, so both halves of a pair share one
# key ("UpperReg" -> "UpperLeg", "Lwing" -> "wing"). Spelling only: folding the
# synonym table in would regroup existing rigs.
_SIGNATURE_SPELLING_TOKENS = {
    'rower': 'lower',
    'reg': 'leg',
    'piers': 'pliers',
    'lwing': 'wing',
    'rwing': 'wing',
}


def _signature_tokens(tokens, side_tokens):
    signature_tokens = []
    for token in tokens:
        if token in side_tokens:
            continue
        token = _SIGNATURE_SPELLING_TOKENS.get(token, token)
        signature_tokens.append(_LIMB_CODE_SIGNATURE_TOKENS.get(token, token))
    return signature_tokens


def _joint_signature(name):
    signature_tokens = _signature_tokens(
        _canonicalize_joint_name(name).lower().split(), ('left', 'right'),
    )
    if signature_tokens:
        return ' '.join(signature_tokens)

    fallback_tokens = _signature_tokens(
        normalize_joint_name(name).split(), ('left', 'right', 'l', 'r'),
    )
    return ' '.join(fallback_tokens)


def _fallback_child_signature(name):
    return ' '.join(
        token for token in _joint_signature(name).split()
        if not token.isdigit()
    )


def _joint_semantic_text(name):
    normalized = normalize_joint_name(name)
    canonical = _canonicalize_joint_name(name).lower()
    return f'{normalized} {canonical}'.strip()


def _text_matches_keywords(text, keywords):
    return any(keyword in text for keyword in keywords)


def _joint_family_semantic_text(joint_index, joint_names, parents, max_depth=3):
    semantic_chunks = []
    current_index = int(joint_index)
    depth = 0
    while current_index >= 0 and depth <= max_depth:
        semantic_chunks.append(_joint_semantic_text(joint_names[current_index]))
        current_index = int(parents[current_index])
        depth += 1
    return ' '.join(chunk for chunk in semantic_chunks if chunk)


def _is_informative_joint_name(name):
    normalized = normalize_joint_name(name)
    if not normalized:
        return False
    tokens = [token for token in normalized.split() if token]
    return any(len(token) > 1 for token in tokens)


def _child_map(parents):
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)
    return children


def _select_representative_joint(indices, rest_positions, axis, prefer_max=True):
    if not indices:
        return None
    if rest_positions is None or len(rest_positions) <= max(indices):
        return indices[0]

    direction = 1.0 if prefer_max else -1.0
    return max(
        indices,
        key=lambda joint_index: (
            direction * float(rest_positions[joint_index, axis]),
            float(np.linalg.norm(rest_positions[joint_index])),
            -joint_index,
        ),
    )


def _filter_grounded_joint_indices(candidate_indices, rest_positions, margin_ratio=0.18):
    if len(candidate_indices) == 0 or len(rest_positions) == 0:
        return []

    unique_candidates = sorted({int(joint_index) for joint_index in candidate_indices})
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    ground_margin = max(body_height * margin_ratio, 1e-3)
    ground_level = float(np.min(rest_positions[unique_candidates, 1]))
    return [
        joint_index
        for joint_index in unique_candidates
        if rest_positions[joint_index, 1] <= ground_level + ground_margin
    ]


def _expand_grounded_contact_chain(candidate_indices, grounded_indices, parents, rest_positions, margin_ratio=0.2):
    if not grounded_indices:
        return []

    candidate_set = {int(joint_index) for joint_index in candidate_indices}
    expanded = set(int(joint_index) for joint_index in grounded_indices)
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    parent_margin = max(body_height * margin_ratio, 1e-3)
    frontier = list(expanded)

    while frontier:
        joint_index = frontier.pop()
        parent_index = int(parents[joint_index])
        if parent_index < 0 or parent_index not in candidate_set or parent_index in expanded:
            continue
        if abs(float(rest_positions[parent_index, 1] - rest_positions[joint_index, 1])) > parent_margin:
            continue
        expanded.add(parent_index)
        frontier.append(parent_index)

    return sorted(expanded)


def _select_grounded_contact_end_effectors(candidate_indices, joint_names, parents, rest_positions):
    if len(candidate_indices) == 0:
        return []

    candidate_indices = sorted({int(joint_index) for joint_index in candidate_indices})
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    pair_height_margin = max(body_height * 0.24, 1e-3)
    single_height_margin = max(body_height * 0.18, 1e-3)

    _, symmetry_partner_indices, _ = _infer_symmetry_metadata(joint_names, parents, rest_positions)
    paired_groups = []
    paired_joint_indices = set()

    for joint_index in candidate_indices:
        partner_index = int(symmetry_partner_indices[joint_index])
        if partner_index < 0 or partner_index not in candidate_indices or joint_index >= partner_index:
            continue
        paired_groups.append((
            float((rest_positions[joint_index, 1] + rest_positions[partner_index, 1]) / 2.0),
            joint_index,
            partner_index,
        ))
        paired_joint_indices.add(joint_index)
        paired_joint_indices.add(partner_index)

    selected = set()
    if paired_groups:
        min_pair_height = min(group[0] for group in paired_groups)
        for pair_height, left_index, right_index in paired_groups:
            if pair_height <= min_pair_height + pair_height_margin:
                selected.add(left_index)
                selected.add(right_index)

    if not selected:
        min_height = float(np.min(rest_positions[candidate_indices, 1]))
        for joint_index in candidate_indices:
            if rest_positions[joint_index, 1] <= min_height + single_height_margin:
                selected.add(joint_index)

    for joint_index in candidate_indices:
        if joint_index in paired_joint_indices:
            continue
        if rest_positions[joint_index, 1] <= min(float(rest_positions[index, 1]) for index in selected) + single_height_margin:
            selected.add(joint_index)

    return sorted(selected)


def _expand_contact_chain_from_leaves(leaf_indices, joint_names, parents, rest_positions, max_depth=4):
    if not leaf_indices:
        return []

    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    chain_margin = max(body_height * 0.2, 1e-3)
    # Cap support-joint backfilling when the parent-child bone itself is too long.
    # This keeps obvious mid-limb transport bones such as Calf/HorseLink from being
    # mislabeled as direct contact points, while still allowing short foot/hand/palm
    # support bones to remain in the contact chain.
    max_parent_contact_offset = min(
        max(body_height * _CONTACT_PARENT_OFFSET_RATIO, _CONTACT_PARENT_OFFSET_MIN),
        _CONTACT_PARENT_OFFSET_CAP,
    )
    # Also cap the cumulative distance from the terminal contact leaf. Even when
    # every individual bone is short, a long multi-bone chain should not turn a
    # clearly upstream support joint into a direct contact point.
    max_cumulative_contact_offset = min(
        max(body_height * _CONTACT_CUMULATIVE_OFFSET_RATIO, _CONTACT_CUMULATIVE_OFFSET_MIN),
        _CONTACT_CUMULATIVE_OFFSET_CAP,
    )
    expanded = set(int(joint_index) for joint_index in leaf_indices)

    for joint_index in leaf_indices:
        current_index = int(joint_index)
        cumulative_contact_offset = 0.0
        for _ in range(max_depth):
            parent_index = int(parents[current_index])
            if parent_index < 0:
                break
            parent_text = _joint_semantic_text(joint_names[parent_index])
            if _text_matches_keywords(parent_text, _CONTACT_CHAIN_STOP_TOKENS):
                break
            if not _text_matches_keywords(parent_text, _CONTACT_CHAIN_INCLUDE_TOKENS):
                break
            parent_contact_offset = float(np.linalg.norm(rest_positions[parent_index] - rest_positions[current_index]))
            if parent_contact_offset > max_parent_contact_offset:
                break
            cumulative_contact_offset += parent_contact_offset
            if cumulative_contact_offset > max_cumulative_contact_offset:
                break
            if abs(float(rest_positions[parent_index, 1] - rest_positions[current_index, 1])) > chain_margin:
                break
            expanded.add(parent_index)
            current_index = parent_index

    return sorted(expanded)


def _infer_contact_leaf_candidates(parents, joint_names, rest_positions):
    end_effectors = _infer_end_effector_joints(parents, joint_names=joint_names, rest_positions=rest_positions)
    return [
        joint_index
        for joint_index in end_effectors
        if _text_matches_keywords(_joint_semantic_text(joint_names[joint_index]), _CONTACT_GEOMETRY_DISTAL_TOKENS)
    ]


def rest_positions_from_offsets(offsets, parents):
    offsets = np.asarray(offsets, dtype=np.float64)
    rest_positions = np.zeros_like(offsets, dtype=np.float64)
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
    return rest_positions


def _infer_end_effector_joints(parents, joint_names=None, rest_positions=None):
    children = _child_map(parents)
    leaf_joints = [joint_index for joint_index, child_indices in enumerate(children) if not child_indices]
    if joint_names is None:
        return leaf_joints

    distal_joints = []
    tail_joints = []
    head_joints = []
    appendage_joints = []
    filtered_leaf_joints = []

    for joint_index in leaf_joints:
        semantic_text = _joint_semantic_text(joint_names[joint_index])
        if not _is_informative_joint_name(joint_names[joint_index]):
            continue
        if _text_matches_keywords(semantic_text, _END_EFFECTOR_EXCLUDE_TOKENS):
            continue

        filtered_leaf_joints.append(joint_index)
        if _text_matches_keywords(semantic_text, _END_EFFECTOR_DISTAL_TOKENS):
            distal_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_TAIL_TOKENS):
            tail_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_HEAD_TOKENS):
            head_joints.append(joint_index)
        elif _text_matches_keywords(semantic_text, _END_EFFECTOR_APPENDAGE_TOKENS):
            appendage_joints.append(joint_index)

    semantic_end_effectors = set(distal_joints)
    semantic_end_effectors.update(appendage_joints)

    tail_joint = _select_representative_joint(tail_joints, rest_positions, axis=2, prefer_max=False)
    if tail_joint is not None:
        semantic_end_effectors.add(tail_joint)

    head_joint = _select_representative_joint(head_joints, rest_positions, axis=2, prefer_max=True)
    if head_joint is not None:
        semantic_end_effectors.add(head_joint)

    if semantic_end_effectors:
        return sorted(semantic_end_effectors)
    if filtered_leaf_joints:
        return sorted(filtered_leaf_joints)
    return leaf_joints


def _infer_contact_joints_from_names(joint_names, parents, rest_positions):
    strong_candidates = []
    weak_candidates = []
    children = _child_map(parents)

    for joint_index, joint_name in enumerate(joint_names):
        semantic_text = _joint_semantic_text(joint_name)
        family_text = _joint_family_semantic_text(joint_index, joint_names, parents, max_depth=3)
        has_upper_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_UPPER_LIMB_TOKENS)
        has_lower_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_CONTEXT_KEYWORDS)

        is_strong_contact = _text_matches_keywords(semantic_text, _CONTACT_JOINT_KEYWORDS)
        is_ball_contact = 'ball' in semantic_text and has_lower_limb_context and not has_upper_limb_context
        is_claw_contact = 'claw' in semantic_text and has_lower_limb_context and not has_upper_limb_context
        is_end_site_contact = (
            ('nub' in semantic_text or 'end site' in semantic_text)
            and has_lower_limb_context
            and not has_upper_limb_context
        )

        if is_strong_contact or is_ball_contact or is_claw_contact or is_end_site_contact:
            strong_candidates.append(joint_index)
            continue

        if not children[joint_index] and not has_upper_limb_context and _text_matches_keywords(semantic_text, _CONTACT_JOINT_WEAK_KEYWORDS):
            weak_candidates.append(joint_index)

    grounded_candidates = _filter_grounded_joint_indices(strong_candidates, rest_positions, margin_ratio=0.24)
    if grounded_candidates:
        return _expand_grounded_contact_chain(strong_candidates, grounded_candidates, parents, rest_positions)

    grounded_weak_candidates = _filter_grounded_joint_indices(weak_candidates, rest_positions, margin_ratio=0.24)
    if grounded_weak_candidates:
        return grounded_weak_candidates

    return []


def _infer_contact_joints_from_geometry(joint_names, rest_positions, parents):
    if len(rest_positions) == 0:
        return []

    candidates = _infer_contact_leaf_candidates(parents, joint_names, rest_positions)
    if not candidates:
        return []

    grounded_leaves = _select_grounded_contact_end_effectors(candidates, joint_names, parents, rest_positions)
    if not grounded_leaves:
        return []

    return _expand_contact_chain_from_leaves(grounded_leaves, joint_names, parents, rest_positions)


def infer_contact_joints(joint_names, parents, rest_positions):
    contact_joints = _infer_contact_joints_from_geometry(joint_names, rest_positions, parents)
    if contact_joints:
        return contact_joints, 'geometry'

    contact_joints = _infer_contact_joints_from_names(joint_names, parents, rest_positions)
    if contact_joints:
        return contact_joints, 'names'

    return [], 'none'


def _joint_depths(parents):
    depths = [0] * len(parents)
    for joint_index in range(1, len(parents)):
        parent_index = parents[joint_index]
        if parent_index >= 0:
            depths[joint_index] = depths[parent_index] + 1
    return depths


def detect_joint_side(name):
    normalized = normalize_joint_name(name)
    compact = normalized.replace(' ', '')
    tokens = set(normalized.split())
    right_markers = (
        ' right ',
        ' r ',
        ' r_',
        ' rleg',
        ' rarm',
        ' rwing',
        ' rthigh',
        ' rclavicle',
        ' rupperarm',
        ' r momo',
        ' r kata',
        ' r hiji',
    )
    left_markers = (
        ' left ',
        ' l ',
        ' l_',
        ' lleg',
        ' larm',
        ' lwing',
        ' lthigh',
        ' lclavicle',
        ' lupperarm',
        ' l momo',
        ' l kata',
        ' l hiji',
    )
    # Markers are matched on whole words at their start. A rig prefix followed by
    # a bare side letter ("NPC_R_Thigh", "BN_L_Arm") is already ' r ' / ' l ';
    # spelled as a prefix marker (' npc r') it also caught the first letter of the
    # next word, siding Bear's "NPC_Ribcage" right and "NPC_LowerFrontLip" left.
    padded = f' {normalized} '
    if any(marker in padded for marker in right_markers) or compact.startswith(('r_', 'rleg', 'rarm', 'rwing', 'rthigh', 'rmomo', 'rkata', 'rhiji')):
        return 'right'
    if any(marker in padded for marker in left_markers) or compact.startswith(('l_', 'lleg', 'larm', 'lwing', 'lthigh', 'lmomo', 'lkata', 'lhiji')):
        return 'left'

    # Quadruped limb codes. Both halves of the rig use them -- Lf/Rf for the fore
    # limbs, Lb/Rb for the hind -- but only the fore pair was ever read, so every
    # Lb*/Rb* joint in Bear, Dinosaur, Tiger, antilope and rhino (52 joints) came
    # back 'center' and lost its side, taking its symmetry pairing with it.
    # Fires only on an unambiguous single side, same as the explicit markers above.
    right_codes = tokens & {'rf', 'rb'}
    left_codes = tokens & {'lf', 'lb'}
    if right_codes and not left_codes:
        return 'right'
    if left_codes and not right_codes:
        return 'left'

    # The same code with the halves swapped (Fl/Fr, Bl/Br) plus the hexapod's
    # middle pair (Lm/Rm, Ml/Mr). Read only next to a limb word, for the same
    # reason _EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS is gated: a mouth corner named
    # "MouthBL" is a bottom-left corner, not a back-left leg. Without this a
    # whole quadruped ("FlLeg1".."BrLegFoot2") came back 'center' and formed no
    # symmetry pairs at all.
    if tokens & _EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS:
        return _single_side(tokens & {'fr', 'br', 'rm', 'mr'}, tokens & {'fl', 'bl', 'lm', 'ml'})
    side = None
    # Lm/Rm on a head: the mouth corners (_EMBED_TEXT_HEAD_SIDE_CODE_TOKENS).
    if tokens & _EMBED_TEXT_HEAD_SIDE_CODE_CONTEXT_TOKENS:
        side = _single_side(tokens & {'rm'}, tokens & {'lm'})
    # Tl/Tr/Bl/Br on a face part: its top/bottom corners
    # (_EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS).
    if side is None and tokens & _EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS:
        side = _single_side(tokens & {'tr', 'br'}, tokens & {'tl', 'bl'})
    return side


def _single_side(right_codes, left_codes):
    if right_codes and not left_codes:
        return 'right'
    if left_codes and not right_codes:
        return 'left'
    return None


def _symmetry_pair_score(left_index, right_index, rest_positions, depths, parents, joint_names):
    mirror_error = abs(float(rest_positions[left_index, 0] + rest_positions[right_index, 0]))
    yz_error = float(np.linalg.norm(rest_positions[left_index, 1:] - rest_positions[right_index, 1:]))
    depth_error = abs(depths[left_index] - depths[right_index])

    left_parent = parents[left_index]
    right_parent = parents[right_index]
    left_parent_sig = _joint_signature(joint_names[left_parent]) if left_parent >= 0 else ''
    right_parent_sig = _joint_signature(joint_names[right_parent]) if right_parent >= 0 else ''
    parent_penalty = 0 if left_parent_sig == right_parent_sig else 1
    return parent_penalty, depth_error, mirror_error + yz_error, left_index, right_index


def _local_mirror_error(left_index, right_index, left_parent, right_parent, rest_positions):
    left_anchor = rest_positions[left_parent] if left_parent >= 0 else np.zeros(3, dtype=np.float64)
    right_anchor = rest_positions[right_parent] if right_parent >= 0 else np.zeros(3, dtype=np.float64)
    left_delta = rest_positions[left_index] - left_anchor
    right_delta = rest_positions[right_index] - right_anchor
    mirror_error = abs(float(left_delta[0] + right_delta[0]))
    yz_error = float(np.linalg.norm(left_delta[1:] - right_delta[1:]))
    local_scale = max(float(np.linalg.norm(left_delta)), float(np.linalg.norm(right_delta)), 1e-6)
    return mirror_error, yz_error, local_scale


def _passes_conservative_child_mirror_check(left_index, right_index, left_parent, right_parent, rest_positions):
    mirror_error, yz_error, local_scale = _local_mirror_error(
        left_index,
        right_index,
        left_parent,
        right_parent,
        rest_positions,
    )
    tolerance = max(1e-3, local_scale * 0.6)
    return mirror_error <= tolerance and yz_error <= tolerance


# How unanimous the rig's own named joints must be about which X half is its
# left before a name that contradicts it is overruled, and how many off-midline
# named joints that vote needs.
_WRONG_SIDE_NAME_MIN_AGREEMENT = 0.9
_WRONG_SIDE_NAME_MIN_VOTES = 4
# Fraction of a joint's offset from its parent below which it counts as on the
# midline and casts no vote / cannot be on the wrong side.
_WRONG_SIDE_MIDLINE_RATIO = 0.1


def _subtree_mean_x(parents, rest_positions):
    """Mean X over each joint's subtree, the joint included."""
    subtree_x_sum = rest_positions[:, 0].copy()
    subtree_size = np.ones(len(parents), dtype=np.float64)
    depths = _joint_depths(parents)
    for index in sorted(range(len(parents)), key=lambda i: -depths[i]):
        if parents[index] >= 0:
            subtree_x_sum[parents[index]] += subtree_x_sum[index]
            subtree_size[parents[index]] += subtree_size[index]
    return subtree_x_sum / subtree_size


def _midline_tolerance(index, parents, rest_positions):
    parent = parents[index]
    anchor = rest_positions[parent] if parent >= 0 else np.zeros(3, dtype=np.float64)
    return _WRONG_SIDE_MIDLINE_RATIO * max(float(np.linalg.norm(rest_positions[index] - anchor)), 1e-6)


def _correct_wrong_side_mirror_siblings(joint_names, joint_side_labels, parents, rest_positions):
    """Relabel named-side joints that sit on the other half of the rig from
    their mirror twin. Mutates *joint_side_labels*; returns the indices.

    Two slips in the source rigs, one rule. Leopard names both mane joints
    right ("BN_Mane_R_01" at x=-0.08, "BN_Mane_R_02" at x=+0.08), so with no
    left joint the pair can never form. Spider swaps its jaws as a pair
    ("R_Jaw_" at +X, "L_Jaw_" at -X) while every other joint of the rig keeps
    its left at -X. Which X half is left is read off the rig's own names, never
    assumed, and only a near-unanimous vote is trusted. A joint on the wrong
    half is overruled only when a sibling with the same name apart from its
    side and indices ("Mane 01" / "Mane 02") is its mirror image and sits on the
    half this joint's label claims -- whatever that sibling is labelled. The
    same-part test matters: the mirror check alone is loose enough to match a
    neighbouring part (Spider's "R_Jaw_" against "FangR_00_").

    A joint's half is where its subtree lies, not its own point: Biped and jt_
    rigs seat the clavicle and hip joints across the midline ("Bip01_R_Clavicle"
    at x=+0.08 in Lion) while the limb below them sits on the named side.
    """
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    side_x = _subtree_mean_x(parents, rest_positions)

    def midline_tolerance(index):
        return _midline_tolerance(index, parents, rest_positions)

    left_positive = 0
    left_negative = 0
    for index, side in enumerate(joint_side_labels):
        x = float(side_x[index])
        if side not in ('left', 'right') or abs(x) <= midline_tolerance(index):
            continue
        if (side == 'left') == (x > 0):
            left_positive += 1
        else:
            left_negative += 1
    votes = left_positive + left_negative
    if votes < _WRONG_SIDE_NAME_MIN_VOTES:
        return []
    if left_positive >= _WRONG_SIDE_NAME_MIN_AGREEMENT * votes:
        left_sign = 1.0
    elif left_negative >= _WRONG_SIDE_NAME_MIN_AGREEMENT * votes:
        left_sign = -1.0
    else:
        return []

    def claimed_sign(index):
        return left_sign if joint_side_labels[index] == 'left' else -left_sign

    def on_claimed_half(index):
        x = float(side_x[index])
        if abs(x) <= midline_tolerance(index):
            return None
        return x * claimed_sign(index) > 0

    corrected = []
    for index, side in enumerate(joint_side_labels):
        if side not in ('left', 'right') or on_claimed_half(index) is not False:
            continue
        parent = parents[index]
        has_mirror_twin = any(
            sibling != index
            and parents[sibling] == parent
            and joint_side_labels[sibling] in ('left', 'right')
            and _fallback_child_signature(joint_names[sibling]) == _fallback_child_signature(joint_names[index])
            and abs(float(side_x[sibling])) > midline_tolerance(sibling)
            and float(side_x[sibling]) * claimed_sign(index) > 0
            and _passes_conservative_child_mirror_check(sibling, index, parent, parent, rest_positions)
            for sibling in range(len(joint_side_labels))
        )
        if has_mirror_twin:
            corrected.append(index)
    for index in corrected:
        joint_side_labels[index] = 'right' if joint_side_labels[index] == 'left' else 'left'
    return corrected


def _pair_sided_joints_by_geometry(joint_side_labels, symmetry_partner_indices, parents,
                                   rest_positions, depths):
    """Pair named-side joints whose signatures never met, by mirror geometry.

    The name decides the side but the signature can still differ between the
    halves: FireAnt's 3ds Max Biped extras keep their Ponytail index in the name
    ("Bip01_Ponytail2_R_Antenna1" against "Bip01_Ponytail1_L_Antenna"), so the
    two antennae and mandibles land in different signature groups. A left and a
    right joint pair here only when they hang off the same parent or off an
    already-mirrored parent pair, sit at the same depth, pass the child mirror
    check, and are each other's closest match. Side labels are never created
    here -- a 'center' joint stays out.
    """
    left_indices = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'left' and symmetry_partner_indices[index] < 0
    ]
    right_indices = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'right' and symmetry_partner_indices[index] < 0
    ]
    errors = {}
    for left_index in left_indices:
        left_parent = parents[left_index]
        for right_index in right_indices:
            right_parent = parents[right_index]
            if depths[left_index] != depths[right_index]:
                continue
            if left_parent != right_parent and (
                left_parent < 0 or symmetry_partner_indices[left_parent] != right_parent
            ):
                continue
            if not _passes_conservative_child_mirror_check(
                left_index, right_index, left_parent, right_parent, rest_positions,
            ):
                continue
            mirror_error, yz_error, local_scale = _local_mirror_error(
                left_index, right_index, left_parent, right_parent, rest_positions,
            )
            errors[left_index, right_index] = (mirror_error + yz_error) / local_scale

    best_right = {}
    best_left = {}
    for (left_index, right_index), error in errors.items():
        if left_index not in best_right or error < errors[left_index, best_right[left_index]]:
            best_right[left_index] = right_index
        if right_index not in best_left or error < errors[best_left[right_index], right_index]:
            best_left[right_index] = left_index
    return sorted(
        (left_index, right_index) for left_index, right_index in best_right.items()
        if best_left.get(right_index) == left_index
    )


# Smallest ratio of the two subtree sizes an unnamed twin may have: one half
# can carry an extra prop joint (a blade in one hand).
_UNSIDED_TWIN_MIN_SUBTREE_RATIO = 0.8


def _subtree_sizes(parents):
    """Per joint, its subtree size and its number of direct children."""
    sizes = np.ones(len(parents), dtype=np.int64)
    child_counts = np.zeros(len(parents), dtype=np.int64)
    depths = _joint_depths(parents)
    for index in sorted(range(len(parents)), key=lambda i: -depths[i]):
        if parents[index] >= 0:
            sizes[parents[index]] += sizes[index]
            child_counts[parents[index]] += 1
    return sizes, child_counts


def _pair_unsided_twins_by_name(joint_side_labels, symmetry_partner_indices, parents,
                                rest_positions, depths, signatures, subtree_x, subtree_sizes):
    """Pair a named-side joint with a 'center' joint that is its unnamed twin.

    Some rigs spell the side on one half only: serpent_man has "R_Arm_Shoulder"
    and a plain "Arm_Shoulder" on the other half. With the side word stripped
    the two names are the same, so the 'center' joint is a candidate when it
    hangs off the same parent (or the mirror of the sided joint's parent), has
    as many children and a subtree of about the same size, lies off the
    midline on the other half -- judged by its subtree, as Biped clavicles
    cross the midline -- and passes the child mirror check. Each candidate pair
    must be the other's closest match. The 'center' joint takes the opposite
    side; the pairs are returned as (left, right).
    """
    sizes, child_counts = subtree_sizes
    def off_midline(index):
        return abs(float(subtree_x[index])) > _midline_tolerance(index, parents, rest_positions)

    sided = [
        index for index, side in enumerate(joint_side_labels)
        if side in ('left', 'right') and symmetry_partner_indices[index] < 0
        and signatures[index] and off_midline(index)
    ]
    unsided = [
        index for index, side in enumerate(joint_side_labels)
        if side == 'center' and symmetry_partner_indices[index] < 0
        and signatures[index] and off_midline(index)
    ]
    errors = {}
    for sided_index in sided:
        sided_parent = parents[sided_index]
        for unsided_index in unsided:
            unsided_parent = parents[unsided_index]
            if signatures[sided_index] != signatures[unsided_index]:
                continue
            if depths[sided_index] != depths[unsided_index]:
                continue
            if child_counts[sided_index] != child_counts[unsided_index]:
                continue
            if (min(sizes[sided_index], sizes[unsided_index])
                    < _UNSIDED_TWIN_MIN_SUBTREE_RATIO * max(sizes[sided_index], sizes[unsided_index])):
                continue
            if float(subtree_x[sided_index]) * float(subtree_x[unsided_index]) >= 0:
                continue
            if sided_parent != unsided_parent and (
                sided_parent < 0 or symmetry_partner_indices[sided_parent] != unsided_parent
            ):
                continue
            if not _passes_conservative_child_mirror_check(
                sided_index, unsided_index, sided_parent, unsided_parent, rest_positions,
            ):
                continue
            mirror_error, yz_error, local_scale = _local_mirror_error(
                sided_index, unsided_index, sided_parent, unsided_parent, rest_positions,
            )
            errors[sided_index, unsided_index] = (mirror_error + yz_error) / local_scale

    best_unsided = {}
    best_sided = {}
    for (sided_index, unsided_index), error in errors.items():
        if sided_index not in best_unsided or error < errors[sided_index, best_unsided[sided_index]]:
            best_unsided[sided_index] = unsided_index
        if unsided_index not in best_sided or error < errors[best_sided[unsided_index], unsided_index]:
            best_sided[unsided_index] = sided_index

    pairs = []
    for sided_index, unsided_index in sorted(best_unsided.items()):
        if best_sided.get(unsided_index) != sided_index:
            continue
        if joint_side_labels[sided_index] == 'left':
            joint_side_labels[unsided_index] = 'right'
            pairs.append((sided_index, unsided_index))
        else:
            joint_side_labels[unsided_index] = 'left'
            pairs.append((unsided_index, sided_index))
    return pairs


def _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=False):
    depths = _joint_depths(parents)
    joint_side_labels = []
    grouped_indices = {}

    for joint_name in joint_names:
        side = detect_joint_side(joint_name)
        if side is None:
            side = detect_joint_side(_canonicalize_joint_name(joint_name))
        joint_side_labels.append(side if side in ('left', 'right') else 'center')
    _correct_wrong_side_mirror_siblings(joint_names, joint_side_labels, parents, rest_positions)

    for joint_index, joint_name in enumerate(joint_names):
        side = joint_side_labels[joint_index]
        if side == 'center':
            continue

        signature = _joint_signature(joint_name)
        if not signature:
            continue
        if signature not in grouped_indices:
            grouped_indices[signature] = {'left': [], 'right': []}
        grouped_indices[signature][side].append(joint_index)

    symmetry_partner_indices = [-1] * len(joint_names)
    symmetric_joint_pairs = []

    for signature in sorted(grouped_indices):
        left_indices = sorted(grouped_indices[signature]['left'], key=lambda index: (depths[index], index))
        remaining_right_indices = set(grouped_indices[signature]['right'])
        for left_index in left_indices:
            if not remaining_right_indices:
                break
            best_right = min(
                remaining_right_indices,
                key=lambda right_index: _symmetry_pair_score(
                    left_index,
                    right_index,
                    rest_positions,
                    depths,
                    parents,
                    joint_names,
                ),
            )
            remaining_right_indices.remove(best_right)
            symmetry_partner_indices[left_index] = best_right
            symmetry_partner_indices[best_right] = left_index
            symmetric_joint_pairs.append([left_index, best_right])

    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[parent_index].append(joint_index)

    signatures = [_joint_signature(joint_name) for joint_name in joint_names]
    subtree_x = _subtree_mean_x(parents, np.asarray(rest_positions, dtype=np.float64))
    subtree_sizes = _subtree_sizes(parents)

    changed = True
    while changed:
        changed = False
        for left_parent, right_parent in list(symmetric_joint_pairs):
            left_unpaired = [joint_index for joint_index in children[left_parent] if symmetry_partner_indices[joint_index] < 0]
            right_unpaired = [joint_index for joint_index in children[right_parent] if symmetry_partner_indices[joint_index] < 0]
            if len(left_unpaired) != 1 or len(right_unpaired) != 1:
                continue

            left_index = left_unpaired[0]
            right_index = right_unpaired[0]
            if not _passes_conservative_child_mirror_check(
                left_index,
                right_index,
                left_parent,
                right_parent,
                rest_positions,
            ):
                continue

            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            joint_side_labels[left_index] = 'left'
            joint_side_labels[right_index] = 'right'
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

        for left_index, right_index in _pair_sided_joints_by_geometry(
            joint_side_labels, symmetry_partner_indices, parents, rest_positions, depths,
        ):
            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

        for left_index, right_index in _pair_unsided_twins_by_name(
            joint_side_labels, symmetry_partner_indices, parents, rest_positions, depths,
            signatures, subtree_x, subtree_sizes,
        ):
            symmetry_partner_indices[left_index] = right_index
            symmetry_partner_indices[right_index] = left_index
            symmetric_joint_pairs.append([left_index, right_index])
            changed = True

    if return_details:
        return {
            'joint_side_labels': joint_side_labels,
            'symmetry_partner_indices': symmetry_partner_indices,
            'symmetric_joint_pairs': symmetric_joint_pairs,
        }

    return joint_side_labels, symmetry_partner_indices, symmetric_joint_pairs


def _infer_is_symmetric(symmetric_joint_pairs, joint_side_labels):
    """Determine if skeleton has bilateral symmetry based on paired joints and side labels.
    
    Returns True if:
    - At least 2 symmetric pairs were found, OR
    - At least 30% of joints are labeled as left or right (not center)
    """
    num_pairs = len(symmetric_joint_pairs)
    if num_pairs >= 2:
        return True
    
    if joint_side_labels:
        sided_count = sum(1 for label in joint_side_labels if label in ('left', 'right'))
        sided_ratio = sided_count / len(joint_side_labels)
        if sided_ratio >= 0.3:
            return True
    
    return False


def build_semantic_metadata(joint_names, parents, offsets, rest_positions=None, species_name=None):
    parents = np.asarray(parents, dtype=np.int64)
    rest_positions = rest_positions_from_offsets(offsets, parents) if rest_positions is None else np.asarray(rest_positions, dtype=np.float64)
    replacements = effective_canonical_replacements(joint_names)
    species_prefixes = infer_species_joint_name_prefixes(joint_names, species_name)
    canonical_joint_names = [
        _canonicalize_joint_name(name, replacements, species_prefixes)
        for name in joint_names
    ]
    canonical_joint_names = _collapse_solitary_head_feature_indices(canonical_joint_names)
    contact_joints, contact_joint_source = infer_contact_joints(
        joint_names,
        parents,
        rest_positions,
    )
    leaf_contact_joints = {
        int(joint_index)
        for joint_index in contact_joints
        if not np.any(np.asarray(parents) == int(joint_index))
    }
    end_effector_joints = sorted(
        set(_infer_end_effector_joints(parents, joint_names=joint_names, rest_positions=rest_positions))
        | leaf_contact_joints
    )
    symmetry_metadata = _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=True)
    joint_side_labels = symmetry_metadata['joint_side_labels']
    symmetry_partner_indices = symmetry_metadata['symmetry_partner_indices']
    symmetric_joint_pairs = symmetry_metadata['symmetric_joint_pairs']
    is_symmetric = _infer_is_symmetric(symmetric_joint_pairs, joint_side_labels)
    return {
        'canonical_joint_names': canonical_joint_names,
        'end_effector_joints': end_effector_joints,
        'end_effector_names': [joint_names[index] for index in end_effector_joints],
        'contact_joints': list(contact_joints),
        'contact_joint_names': [joint_names[index] for index in contact_joints],
        'contact_joint_source': contact_joint_source,
        'joint_side_labels': joint_side_labels,
        'symmetry_partner_indices': symmetry_partner_indices,
        'symmetric_joint_pairs': symmetric_joint_pairs,
        'symmetric_joint_pair_names': [[joint_names[left], joint_names[right]] for left, right in symmetric_joint_pairs],
        'is_symmetric': bool(is_symmetric),
    }
