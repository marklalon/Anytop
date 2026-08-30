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

# Joint-name tokens that unambiguously signal Japanese romaji rig naming. Used
# only as *evidence* that a skeleton is Japanese-style; deliberately excludes
# short/ambiguous tokens (e.g. "te", "o") so a single coincidental match in an
# otherwise non-Japanese rig cannot flip on the gated replacements below.
_JAPANESE_EVIDENCE_TOKENS = frozenset({
    'momo', 'sippo', 'shippo', 'mune', 'hiza', 'hara', 'ashi', 'hiji',
    'koshi', 'kubi', 'atama', 'ago', 'kata',
    'munabire', 'sebire', 'harabire', 'shiribire', 'shirihire', 'obire',
})
_JAPANESE_EVIDENCE_MIN_DISTINCT = 3

# Replacements that are only safe to apply once a skeleton is confirmed to use
# Japanese romaji naming. "kao" (face → head) and "kosi" (hips, the unaspirated
# spelling of "koshi") would rarely collide elsewhere, but "o" (tail, 尾) is a
# single character that must never be mapped globally.
_JAPANESE_GATED_REPLACEMENTS = {
    'kao': 'Head',
    'kosi': 'Hips',
    'o': 'Tail',
    'te': 'Hand',
    'era': 'Gill',
}
# Chain-position filler that the Segment/ChainStart/ChainEnd tokens already say
# better. Limb-position words (front/back/rear/mid) are deliberately NOT here:
# they are the only thing separating a fore limb from a hind limb once the side
# label is factored out, so dropping them collapsed e.g. Crocodile's
# "Right Front Leg 1" and "Right Back Leg 1" onto one identical embedding text.
_EMBED_TEXT_SKIP_TOKENS = {
    'base',
    'tip',
    'nub',
    'end',
    'site',
}
# Rig scaffolding, props and tack: bones that carry no anatomy at all. A joint
# whose name reduces to nothing but these is blanked by build_joint_embedding_texts
# (zero embedding), which is the honest encoding -- it has no cross-species slot.
# A name that also carries anatomy keeps it ("XtraSpine" -> "Spine"), because the
# per-token filter runs first and only the all-marker case reaches the blanking
# fallback.
#
# Everything here was checked against its actual place in the tree before being
# added: Horse/Camel "Saddle"/"Reins"/"Halter"/"Handle"/"Ctrl"/"IkChain" are tack
# and controls hung off the spine, rhino "Passenger" is a rider node, serpent_man
# "Blade" is a weapon parented to the hand, SabreToothTiger "MagicEffectsNode" and
# FireAnt "ProjectileNodeFire" are VFX emitters, and Flamingo/Roach "Bone02" is a
# bare generic marker. Deliberately NOT here: "spline", which in this corpus is
# Anaconda's Hips->Spline01..06->Neck trunk (a real spine, mapped as a synonym),
# and "down", which may be a lower-body qualifier rather than a control.
#
# The single-rig codes below were each read off their own tree before being
# added: Rabbit's "Bip01" is a bare node between Root and Pelvis; Crow/Pirrana/
# Tukan's "All" is the rig root, Hips -> All -> Locator; Boar's "Aux" hangs off
# an otherwise ordinary "LeftClavicleAux"/"LeftCheekAux" (no plain Clavicle to
# collide with, so the joint simply gains the corpus name); "Helt"/"Helb",
# "Lftb"/"Rftb", "Bnp" and "Mag" are opaque suffix codes whose name already
# carries the anatomy ("HeadEyeLidHelt", "LeftTwistBoneLftb").
#
# The game-character rigs carry far more equipment than the animal ones: held
# weapons ("Sword", "Bow", "Arrow", "Spear", "Staff"), worn gear ("Shield",
# "Armor", "Backpack", "Cape", "Skirt", "Robe", "Headband", "Quiver", "Bag"),
# carried tools ("GoldPick", "BoneWood"), robot hardware ("Gun", "Barrel",
# "Bolt") and bare attachment sockets ("L_hand_container", "Quiver_container",
# "WeaponPosition", "Bone_Mount"). Each was read off its tree first: the
# containers hang as childless leaves off a hand or the spine, the mount node
# sits between a rider's spine and the mount's own root. Rig roots that spell a
# code instead of a body part go here too -- "CG" (centre of gravity), "Hub",
# "Main" and a bare "Base" are all index-0 nodes with no anatomy in the name.
#
# Deliberately NOT here, after checking the tree: "Crown" (RMW_Bat hangs it off
# UpperBody, RMW_Slime off Head -- two rigs, two meanings), "Ice" (an ice
# elemental's shards are its own body, like the plant monster's "Leaf"),
# "Notch" (single joint on a rock golem's spine, could be rock anatomy) and
# "Pad" (keeping it leaves "Shoulder Pad", which reads as the armour it is
# instead of inventing a shoulder the Orc rig does not otherwise have).
#
# "Cape" and "Skirt" are left out for a different reason: those rigs qualify the
# cloth with a direction ("CapeBack01", "FrontSkirt", "RearSkirt"), and blanking
# fires only when *every* token is a marker -- so dropping the noun would leave a
# bare "Back"/"Front" claiming to be the creature's back, which is worse than the
# uninformative-but-true "Cape Back". Same treatment hair already gets.
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
    'fire',
    'fur',
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
    'passenger',
    'pick',
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
    'sword',
    'target',
    'trajectory',
    'weapon',
    'wood',
    'xtra',
}
# Side is re-attached from the geometry-derived joint_side_labels in
# build_joint_embedding_texts, so the name's own side word is dropped here.
# Across all 104 species the geometry label is a strict superset of the name:
# it agrees on every joint that names a side (never conflicts, never falls back
# to "center") and additionally sides 100 joints whose name is silent. Dropping
# it also puts the side at one fixed position for every rig -- "R_thigh",
# "thigh_R" and "RightThigh" all reduce to "Thigh ... Right".
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
# Creature words that some rigs glue onto an otherwise ordinary joint name. Two
# sources: a rig that stamps its own species on every bone (Kappa_gorilla's
# "GorillaJaw", "KappaNeck", "RightGorillaFinger101"), and a *variant* rig that
# carries several interchangeable heads on one skeleton (antilope hangs a Deer,
# Moose, Quilin and Donkey neck+head off Spine02; Tiger hangs a KhitanTiger neck
# beside its own).
#
# The species is already conditioned globally through ``species_emb``, so
# repeating it per joint only dilutes the anatomy. Measured on the current bank,
# the dilution is total, not cosmetic: the nearest neighbour of "Kappa Head" is
# "Neck Nek ..." (0.49) and of "Moose Neck" is "Right Quilin Moustache" (0.57) --
# T5 clusters these joints by *species* instead of by body part, which is the
# exact opposite of what a cross-species model needs. Dropping the word leaves
# four plain "Neck" joints that _sibling_instance_tokens then numbers apart.
#
# Stripped only in the embedding text, never in canonical_joint_names: the
# canonical layer needs them to keep names unique (and would have
# _disambiguate_duplicate_canonical_names re-append them anyway), so stripping
# there would cost BVH-name churn and a renamer bank rebuild for no gain.
#
# Deliberately excluded: 'ant' (spider_tarantula's "RightAnt00" is an antenna
# under Head01, not the insect), 'horse' ("HorseLink" is a 3ds Max Biped leg
# bone used by 33 species -- handled as a pair merge below), and 'jaws' (a
# species here, but one keystroke from the anatomical 'jaw').
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
# Quadruped limb codes: Lf/Rf/Lb/Rb = left/right fore/hind. The side half is
# already recovered by detect_joint_side and re-attached from the geometry label,
# so only the fore/hind half is emitted here. Dropping the code outright would
# collapse a front leg onto a hind leg -- the same failure the front/back words
# are kept out of _EMBED_TEXT_SKIP_TOKENS to avoid.
_EMBED_TEXT_LIMB_CODE_TOKENS = {
    'lf': 'Front',
    'rf': 'Front',
    'lb': 'Back',
    'rb': 'Back',
}
# The same code with the two halves swapped -- Fl/Fr = fore-left/fore-right,
# Bl/Br = back-left/back-right, Lm/Rm = the middle pair of a hexapod. Kept apart
# from the table above because these spellings are ambiguous on their own:
# MU04_Earthworm names the corners of its mouth "MouthTL"/"MouthBL", where "Bl"
# is bottom-left and decoding it as a hind limb would invent anatomy. They are
# only read when the same name also carries a limb word, which is how every rig
# that uses them spells it ("FlLeg1", "BrLegAnkle", "LmLegAnkle").
_EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS = {
    'fl': 'Front',
    'fr': 'Front',
    'bl': 'Back',
    'br': 'Back',
    'lm': 'Mid',
    'rm': 'Mid',
}
_EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS = frozenset({'arm', 'leg'})
# Anatomical synonyms and rig abbreviations folded onto the vocabulary the rest
# of the corpus already uses, so one body part is one point in T5 space instead
# of a dozen singleton families. Left as-is when T5 gets there on its own; these
# are the ones it does not -- it neighbours by spelling, so "Left Carpal" lands
# on "Left Calf" (0.70) and "Left Ulna" on "Left Clavicle" (0.65).
#
# Every mapping was read off the joint's actual position in the tree, not a
# dictionary: Deer runs Pelvis->Femur->Tibia->LargeCannon->PhalanxPrima->Hoof
# (= Thigh/Calf/Foot/Toe) and Ribcage->Scapula->Humerus->Radius->Metacarpus
# (= Clavicle/UpperArm/Forearm/Hand); Hyena runs Scapula->Humerus->Ulna->Carpal;
# Anaconda runs Hips->Spline01..06->Neck. Not mapped, on inspection: 'ball'
# (Bear uses it for both the ball of the foot and the palm), 'belly'/'stomach'
# (abdomen, spine segment and fat jiggle across three rigs) and 'crest' (Boar's
# is a back crest, not a neck one).
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
    # misspellings, which T5 has no reason to place anywhere near the word they
    # meant
    'tounge': 'Tongue',
    'thouge': 'Tongue',
    'tunge': 'Tongue',
    'eyeleds': 'Eyelid',
    'scull': 'Head',
    'pevis': 'Pelvis',
    'shouder': 'Shoulder',
    'uppder': 'Upper',
    # One asset pack mirrored its left-side bones and ran a global L -> R
    # replace over the copied names, which corrupted the words themselves: the
    # left arm is "Lower_Arm_L" but the right one is "Rower_Arm_R", and the left
    # leg is "Upper_Leg_L"/"Lower_Leg_L" against "Upper_Reg_R"/"Rower_Reg_R" on
    # the right. Both spellings sit in the same skeleton, and the tree confirms
    # them (UpperArm -> RowerArm -> Hand, Hips -> UpperReg -> RowerReg -> Foot),
    # so the two sides landed in unrelated corners of T5 space and their
    # symmetry pairs failed to form. The pair merges below turn the decoded
    # words into the corpus segment names; these entries cover a stray single.
    'rower': 'Lower',
    'reg': 'Leg',
    # rig abbreviations that echo the full word already in the same name
    # ("LeftThighLeftThi", "SpineSpn0", "Tail0Tal0"); expanding them lets the
    # adjacent-duplicate collapse in _refine_joint_embedding_name eat the echo.
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
    # standalone abbreviations, read off the tree: Deer_Buck hangs "LeftClav"
    # and "LeftScap" off Spine4 (both girdle helpers, both -> Clavicle, as
    # "scapula" already maps there), Hyena runs Femur -> Shin -> Ankle, and
    # SabreToothTiger's "Pelv" sits Hips -> Pelv -> Thigh/Tail.
    'clav': 'Clavicle',
    'scap': 'Clavicle',
    'shin': 'Calf',
    'pelv': 'Pelvis',
    'chk': 'Cheek',
    'lips': 'Lip',
    'btm': 'Bottom',
}

# Some rigs glue a multi-word joint name together in all lowercase
# ("R_smallfrontarm_J01"), which leaves normalize_joint_name no case or digit
# boundary to split on, so the whole blob survives as one OOV token. Segment
# those against an explicit vocabulary during canonicalization so both the
# canonical name and the T5 embedding text see real words.
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
# Real single words that happen to decompose into vocabulary entries. Splitting
# them would be wrong ("ponytail" is a deliberate non-anatomical marker, and
# "eyebrow" must not decay into the generic HeadFeature token via "eye").
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


JOINT_NAME_EMBEDDING_SCHEMA_VERSION = 12

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


# Adjacent canonical tokens that name one anatomical part together. Applied to
# the raw tokens, before the per-token substitutions in
# _refine_joint_embedding_tokens -- those rewrite "arm", which would otherwise
# hide every <modifier>+arm pair from this table.
_EMBED_TEXT_TOKEN_PAIR_MERGES = {
    ('upper', 'leg'): 'Thigh',
    ('up', 'leg'): 'Thigh',
    ('fore', 'arm'): 'Forearm',
    ('fore', 'leg'): 'Foreleg',
    ('upper', 'arm'): 'UpperArm',
    ('lower', 'arm'): 'Forearm',
    # The counterpart of ('upper', 'leg'): the segment below the upper leg. On a
    # quadruped's *fore* limb that segment is really a forearm, but the pair
    # above already reads that rig's "UpperLeg" as a Thigh, so decoding both
    # halves the same way at least keeps one limb in one family instead of
    # splitting it across two.
    ('lower', 'leg'): 'Calf',
    # Same three segments as spelled by the L -> R mirrored names above.
    ('rower', 'arm'): 'Forearm',
    ('upper', 'reg'): 'Thigh',
    ('rower', 'reg'): 'Calf',
    ('lower', 'reg'): 'Calf',
    # 3ds Max Biped's extra digitigrade leg link, exported by 33 species here.
    # It sits Thigh -> Calf -> HorseLink -> Foot in every one of them, which is
    # the ankle/hock; merging the pair both names it correctly and keeps the
    # creature word "horse" out of the embedding text of a Cat, a Lion and a
    # Chicken. No species carries both a HorseLink and an Ankle, so nothing
    # collides.
    ('horse', 'link'): 'Ankle',
    # CamelCase splits "EyeLid"/"EyeLids" into two tokens, and a bare 'eye' also
    # emits the shared HeadFeature category -- merging the pair keeps an eyelid
    # an eyelid instead of "Eye HeadFeature Lid". The glued lowercase spelling is
    # already protected by _COMPOUND_SPLIT_PROTECTED_TOKENS.
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


def _refine_joint_embedding_tokens(clean_token, bare_arm_is_upper_arm=False,
                                   quadrant_codes_name_a_limb=False):
    """Map one canonical token to the embedding token(s) it contributes."""
    limb_code_token = _EMBED_TEXT_LIMB_CODE_TOKENS.get(clean_token)
    if limb_code_token is not None:
        return [limb_code_token]
    if quadrant_codes_name_a_limb:
        quadrant_token = _EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS.get(clean_token)
        if quadrant_token is not None:
            return [quadrant_token]
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


def _refine_joint_embedding_name(name, bare_arm_is_upper_arm=False, additional_prefixes=()):
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

    # Gate for the ambiguous fore/hind codes: only a name that also spells a limb
    # gets them decoded, so a mouth corner keeps its "Bl" and a leg does not.
    quadrant_codes_name_a_limb = bool(
        set(clean_tokens) & _EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS
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

    return deduped_tokens or canonical_name.split()


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


def build_joint_embedding_texts(object_cond):
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
    refined_tokens_per_joint = [
        _refine_joint_embedding_name(
            joint_name,
            bare_arm_flags[joint_index],
            additional_prefixes=species_prefixes,
        )
        for joint_index, joint_name in enumerate(base_joint_names)
    ]
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
        if joint_index in contact_joints:
            flag_tokens.append('Contact')
        if joint_index in end_effector_joints:
            flag_tokens.append('EndEffector')
        flag_tokens_per_joint.append(flag_tokens)

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


# Side half of a quadruped limb code, dropped from the symmetry signature; the
# fore/hind half is kept as a bare 'f'/'b' so LfLeg01 can only ever pair with
# RfLeg01. Erasing the code outright would put a fore and a hind leg in one
# group and leave the mirror test to tell them apart. The swapped spellings
# (Fl/Fr, Bl/Br) and the hexapod's middle pair (Lm/Rm) need the same treatment
# -- once detect_joint_side reads them, a front and a hind leg would otherwise
# share the signature "leg 1" and could cross-pair.
_LIMB_CODE_SIGNATURE_TOKENS = {
    'lf': 'f', 'rf': 'f', 'lb': 'b', 'rb': 'b',
    'fl': 'f', 'fr': 'f', 'bl': 'b', 'br': 'b',
    'lm': 'm', 'rm': 'm',
}


# Spelling-only repairs, applied to the symmetry signature. The pack that ran a
# global L -> R replace over its mirrored bone names corrupted the words too, so
# the two halves of one limb no longer share a signature: "UpperLegLeft" against
# "UpperRegRight", "LowerArmLeft" against "RowerArmRight". These undo the letter
# swap so the pair can form; the embedding text has its own entries for the same
# spellings. "Lwing"/"Rwing" is the other spelling problem in the same class: the
# side letter is glued to the word with no case boundary, so the two sides read
# as two different parts. Deliberately not the full synonym table -- a signature
# is a spelling key, and folding synonyms into it would regroup every existing
# rig.
_SIGNATURE_SPELLING_TOKENS = {
    'rower': 'lower',
    'reg': 'leg',
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
        ' npc r',
        ' bip01 r',
        ' bn r',
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
        ' npc l',
        ' bip01 l',
        ' bn l',
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
    # middle pair (Lm/Rm). Read only next to a limb word, for the same reason
    # _EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS is gated: a mouth corner named
    # "MouthBL" is a bottom-left corner, not a back-left leg. Without this a
    # whole quadruped ("FlLeg1".."BrLegFoot2") came back 'center' and formed no
    # symmetry pairs at all.
    if tokens & _EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS:
        right_quadrant = tokens & {'fr', 'br', 'rm'}
        left_quadrant = tokens & {'fl', 'bl', 'lm'}
        if right_quadrant and not left_quadrant:
            return 'right'
        if left_quadrant and not right_quadrant:
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


def _infer_symmetry_metadata(joint_names, parents, rest_positions, return_details=False):
    depths = _joint_depths(parents)
    joint_side_labels = []
    grouped_indices = {}

    for joint_index, joint_name in enumerate(joint_names):
        side = detect_joint_side(joint_name)
        if side is None:
            side = detect_joint_side(_canonicalize_joint_name(joint_name))
        side = side if side in ('left', 'right') else 'center'
        joint_side_labels.append(side)

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
