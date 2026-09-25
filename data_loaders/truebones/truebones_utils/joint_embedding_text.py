"""Per-joint T5 sentences built from canonical joint names.

The token tables that fold a canonical name into body-part words (filler,
non-anatomical markers, limb and face codes, synonyms, pair merges) and
``build_joint_embedding_texts``, which turns each joint into "side + body part".
``JOINT_NAME_EMBEDDING_SCHEMA_VERSION`` keys the stored embeddings to this text.
``attach_t5_embeddings_to_cond`` encodes those texts with T5 (or reuses a
reference cond's vectors) and bakes them into each cond entry.
"""

import json
import numpy as np
import os
from os.path import join as pjoin
import re

from data_loaders.truebones.truebones_utils.dataset_tags import (
    assert_species_tags_cover,
)
from .joint_name_canonical import (
    EMBED_TEXT_CREATURE_TOKENS,
    EMBED_TEXT_HEAD_FEATURE_TOKENS,
    build_joint_name_inspection_rows,
    canonicalize_joint_name,
    infer_species_joint_name_prefixes,
    normalize_joint_name,
    refresh_joint_metadata_in_object_cond,
    write_joint_name_collision_report,
)
from .joint_struct_features import child_lists


# Quadruped limb codes (Lf/Rf/Lb/Rb = left/right fore/hind), decoded to their
# fore/hind half; the side comes from the geometry label.
EMBED_TEXT_LIMB_CODE_TOKENS = {
    'lf': 'Front',
    'rf': 'Front',
    'lb': 'Back',
    'rb': 'Back',
}
# The same codes spelled fore/hind first (Fl/Br) plus a hexapod's middle pair
# (Lm/Rm, Ml/Mr). Ambiguous alone ("MouthBL" is bottom-left), so read only when
# the name also carries a limb word. A wing is a limb here: a cicada's hind wing
# is "wingBL".
EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS = {
    'fl': 'Front',
    'fr': 'Front',
    'bl': 'Back',
    'br': 'Back',
    'lm': 'Mid',
    'rm': 'Mid',
    'ml': 'Mid',
    'mr': 'Mid',
}
EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS = frozenset({'arm', 'leg', 'wing'})
# Top/bottom x left/right corners of a face part ("RigMouthTL" .. "RigMouthBR").
# Read only next to a face word, which is what tells "Bl" apart from a hind limb.
EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS = {
    'tl': 'Upper',
    'tr': 'Upper',
    'bl': 'Lower',
    'br': 'Lower',
}
EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS = frozenset({
    'mouth', 'lip', 'jaw', 'beak', 'eye', 'eyelid', 'brow', 'cheek',
})
# Lm/Rm on a head joint ("Head_LM01") is a mouth corner, not a middle limb. A limb
# word wins when a name carries both.
EMBED_TEXT_HEAD_SIDE_CODE_TOKENS = {
    'lm': 'Mouth',
    'rm': 'Mouth',
}
EMBED_TEXT_HEAD_SIDE_CODE_CONTEXT_TOKENS = frozenset({'head'})

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


# Bump whenever the text build_joint_embedding_texts produces for a joint can
# change: the token tables here or in joint_name_canonical, name canonicalization
# or refinement, or what goes into the sentence. Stored name embeddings are keyed
# by this version, so a bump makes the loader reject stale cond files until
# preprocessing re-runs.
JOINT_NAME_EMBEDDING_SCHEMA_VERSION = 17


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
    # Reverse-index sweep: a child always has a higher index than its parent in
    # these rigs.
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

    children = child_lists(parents)
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
    limb_code_token = EMBED_TEXT_LIMB_CODE_TOKENS.get(clean_token)
    if limb_code_token is not None:
        return [limb_code_token]
    if quadrant_codes_name_a_limb:
        quadrant_token = EMBED_TEXT_QUADRANT_LIMB_CODE_TOKENS.get(clean_token)
        if quadrant_token is not None:
            return [quadrant_token]
    else:
        if side_codes_name_a_head:
            head_code_token = EMBED_TEXT_HEAD_SIDE_CODE_TOKENS.get(clean_token)
            if head_code_token is not None:
                return [head_code_token]
        if quadrant_codes_name_a_face:
            face_code_token = EMBED_TEXT_FACE_QUADRANT_CODE_TOKENS.get(clean_token)
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
    if clean_token in EMBED_TEXT_HEAD_FEATURE_TOKENS:
        # Emit the specific word *and* the shared category. The category token
        # keeps every head appendage close together in T5 space (the point of
        # the grouping), while the specific word stops Jaguar's Eye, Ear and
        # Beard from collapsing onto one identical "HeadFeature Right".
        return [clean_token.capitalize(), 'HeadFeature']
    return [clean_token.capitalize()]


def clean_embedding_token(token):
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
        clean_embedding_token(token)
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
    canonical_name = canonicalize_joint_name(name, additional_prefixes=additional_prefixes)
    clean_tokens = []
    for token in canonical_name.split():
        clean_token = clean_embedding_token(token)
        if not clean_token or clean_token.isdigit() or clean_token in _EMBED_TEXT_SKIP_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            continue
        if clean_token in _EMBED_TEXT_SIDE_TOKENS:
            continue
        if clean_token in EMBED_TEXT_CREATURE_TOKENS:
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
        set(clean_tokens) & EMBED_TEXT_QUADRANT_LIMB_CONTEXT_TOKENS
    )
    side_codes_name_a_head = bool(set(clean_tokens) & EMBED_TEXT_HEAD_SIDE_CODE_CONTEXT_TOKENS)
    quadrant_codes_name_a_face = bool(set(clean_tokens) & EMBED_TEXT_FACE_QUADRANT_CONTEXT_TOKENS)
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
        if clean_embedding_token(token) not in _EMBED_TEXT_SIDE_TOKENS
    ]
    return fallback_tokens or canonical_name.split()


def build_joint_embedding_texts(object_cond):
    """Per-joint T5 sentences: side + body part.

    Structure-derived tokens (chain position, sibling numbering, contact and
    end-effector flags) belong to the structural channel, never to the text.
    Repeated siblings (a centipede's leg pairs, a bat's wing fingers) therefore
    share one text on purpose: telling them apart is what sib_rank / fore_aft_n
    / lateral_signed do, comparably across species.
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
    texts = []
    for joint_index, joint_name in enumerate(base_joint_names):
        refined_tokens = _refine_joint_embedding_name(
            joint_name,
            bare_arm_flags[joint_index],
            additional_prefixes=species_prefixes,
            bare_leg_is_calf=bare_leg_flags[joint_index],
        )
        # Same cleaning the table lookups use. It matters on the fallback path:
        # a name made *only* of markers comes back as the raw canonical tokens
        # ("BN_P", "Bip01"), which a bare .lower() cannot match against the set.
        lowered_tokens = {clean_embedding_token(token) for token in refined_tokens}
        if lowered_tokens & _EMBED_TEXT_NON_ANATOMICAL_TOKENS:
            texts.append('')
            continue

        # Side leads, so the text opens as a plain English noun phrase
        # ("Right Finger") -- a construction T5 saw in pretraining.
        side = joint_side_labels[joint_index] if joint_index < len(joint_side_labels) else 'center'
        side_tokens = [side.capitalize()] if side in ('left', 'right') else []
        texts.append(' '.join([*side_tokens, *refined_tokens]))
    return texts


# Texts per T5 forward pass; bounds padding memory on a full-dataset encode.
_T5_ENCODE_BATCH = 256


def _build_t5_text_cache(cache_cond, t5_name):
    """Map every joint-name text already baked into *cache_cond* to its T5 vector.

    The encoder is a masked mean over one text's own tokens, so a text encoded
    under the same T5 model yields the same vector whichever cond it was baked
    into.
    """
    cache = {}
    if not isinstance(cache_cond, dict):
        return cache
    for entry in cache_cond.values():
        if not isinstance(entry, dict):
            continue
        joint_meta = entry.get('joints_names_embs_meta')
        joint_embs = entry.get('joints_names_embs')
        if (isinstance(joint_meta, dict) and joint_embs is not None
                and str(joint_meta.get('t5_name') or '') == t5_name):
            texts = list(joint_meta.get('embedding_texts') or ())
            joint_embs = np.asarray(joint_embs, dtype=np.float32)
            if joint_embs.ndim == 2 and joint_embs.shape[0] == len(texts):
                for text, emb in zip(texts, joint_embs):
                    cache.setdefault(str(text), emb)
    return cache


def _reference_joint_texts(reference_cond, t5_name):
    """Every joint-name text the reference cond encodes under *t5_name*.

    Membership is only meaningful when the reference was built by the same text
    builder, so an entry under another joint-name schema is a hard error rather
    than a source of false "unseen" verdicts.
    """
    texts = set()
    if not isinstance(reference_cond, dict):
        raise ValueError('blanking unseen joint names needs the reference cond.npy')
    for object_type, entry in reference_cond.items():
        meta = entry.get('joints_names_embs_meta') if isinstance(entry, dict) else None
        if not isinstance(meta, dict) or str(meta.get('t5_name') or '') != t5_name:
            continue
        schema_version = meta.get('schema_version')
        if schema_version is None or int(schema_version) != JOINT_NAME_EMBEDDING_SCHEMA_VERSION:
            raise ValueError(
                f"reference cond entry '{object_type}' was encoded under joint-name schema "
                f"{schema_version}, this code is at {JOINT_NAME_EMBEDDING_SCHEMA_VERSION}; "
                f"which names it covers cannot be judged across schemas"
            )
        texts.update(str(text) for text in meta.get('embedding_texts') or ())
    if not texts:
        raise ValueError(f'the reference cond holds no joint-name texts encoded with {t5_name}')
    return texts


def _blank_unseen_joint_texts(embedding_texts_by_object, known_texts):
    """Swap every joint text the reference never encodes for the blank text.

    The model is trained to read an all-zero name row as "name unknown"
    (``--joint_name_drop_prob``), and the blank text encodes to exactly that row.
    A text the checkpoint was never trained on would instead be encoded into a
    point of T5 space the model has no prior for. Returns
    ``{object_type: {joint_index: original_text}}``.
    """
    blanked_by_object = {}
    for object_type, texts in embedding_texts_by_object.items():
        blanked = {
            index: text for index, text in enumerate(texts)
            if str(text).strip() and text not in known_texts
        }
        embedding_texts_by_object[object_type] = [
            '' if index in blanked else text for index, text in enumerate(texts)
        ]
        blanked_by_object[object_type] = blanked
        if blanked:
            named = sum(1 for text in texts if str(text).strip())
            print(f'[{object_type}] blanked {len(blanked)}/{named} named joint(s) whose '
                  f'text the reference cond never encodes:')
            for index, text in sorted(blanked.items()):
                print(f'  - joint {index}: {text!r}')
    return blanked_by_object


def attach_t5_embeddings_to_cond(cond, save_dir, t5_name='t5-base', write_collision_report=True,
                                  t5_conditioner=None, embedding_cache_cond=None,
                                  blank_unseen_joint_names=False):
    """Bake joint-name T5 embeddings into every entry of *cond*.

    No species vector is baked: ``species_emb`` is bound from the species
    descriptor table at training / generation time, so a stale one is dropped.

    ``embedding_cache_cond`` is an already-encoded cond (e.g. the checkpoint's
    reference cond.npy): any text it holds under the same T5 model is reused
    verbatim, and T5 is loaded only when some text is missing from it.

    ``blank_unseen_joint_names`` gives every joint whose text that reference
    never encodes the blank (all-zero) name instead of a fresh T5 vector; the
    originals are kept in ``joints_names_embs_meta['blanked_unseen_texts']``.
    Every text is then either cached or blank, so T5 is never loaded -- the
    inference path (process_new_skeleton) relies on that.
    """
    # Imported here so the text builders above load without torch.
    import torch

    if not cond:
        return

    inspection_dir = pjoin(save_dir, 'joint_name_inspection')
    os.makedirs(inspection_dir, exist_ok=True)

    embedding_texts_by_object = {}
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        refresh_joint_metadata_in_object_cond(object_cond)
        embedding_texts = build_joint_embedding_texts(object_cond)
        embedding_texts_by_object[object_type] = embedding_texts

    blanked_by_object = {}
    if blank_unseen_joint_names:
        blanked_by_object = _blank_unseen_joint_texts(
            embedding_texts_by_object, _reference_joint_texts(embedding_cache_cond, t5_name)
        )

    object_types_to_encode = sorted(cond)
    joint_count = len(object_types_to_encode)

    if t5_conditioner is None:
        # Fast-fail before any encoding: the per-species descriptor has no fallback,
        # so a species missing from species_tags.jsonl must surface here.
        assert_species_tags_cover(cond.keys())

    # Every text either comes from the cache or is encoded once, in one batch.
    text_cache = _build_t5_text_cache(embedding_cache_cond, t5_name)
    wanted = []
    for object_type in object_types_to_encode:
        wanted.extend(embedding_texts_by_object[object_type])
    # The blank name is the all-zero row (T5Conditioner masks an empty text out
    # entirely), so it never needs the encoder.
    if '' in wanted and '' not in text_cache and text_cache:
        text_cache[''] = np.zeros_like(next(iter(text_cache.values())))
    missing = list(dict.fromkeys(text for text in wanted if text not in text_cache))
    if text_cache:
        print(f'Reusing cached T5 embeddings for {len(set(wanted)) - len(missing)}/'
              f'{len(set(wanted))} texts.')

    if missing and blank_unseen_joint_names:
        raise RuntimeError(
            f'joint-name texts neither cached nor blanked: {missing}; the unseen-name '
            'path must not need T5'
        )
    if missing:
        if t5_conditioner is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f'Loading T5 model {t5_name} on {device.upper()} ...')
            from model.conditioners import T5Conditioner
            t5_conditioner = T5Conditioner(
                name=t5_name,
                finetune=False,
                word_dropout=0.0,
                normalize_text=False,
                device=device,
                autocast_dtype=None,
                local_files_only=True,
            )
        if text_cache:
            print(f'Texts not in the cache: {missing}')
        print(f'Encoding {len(missing)} texts via T5 ...')
        with torch.no_grad():
            for start in range(0, len(missing), _T5_ENCODE_BATCH):
                chunk = missing[start:start + _T5_ENCODE_BATCH]
                tokens = t5_conditioner.tokenize_entries(chunk)
                embs = t5_conditioner(tokens).detach().cpu().numpy().astype(np.float32, copy=False)
                text_cache.update(zip(chunk, embs))
    else:
        print('All embedding texts cached; skipped T5.')

    print(f'Attaching T5 embeddings for {joint_count} object types ...')
    for object_type in object_types_to_encode:
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        embs = np.stack([text_cache[text] for text in embedding_texts]).astype(np.float32, copy=False)
        object_cond['joints_names_embs'] = embs
        object_cond['joints_names_embs_meta'] = {
            't5_name': t5_name,
            'schema_version': JOINT_NAME_EMBEDDING_SCHEMA_VERSION,
            'embedding_dim': int(embs.shape[1]) if embs.ndim == 2 else 0,
            'embedding_texts': list(embedding_texts),
            'blanked_unseen_texts': dict(blanked_by_object.get(object_type, {})),
        }

        object_cond.pop('species_emb', None)
        object_cond.pop('species_emb_meta', None)

    # cond keys are '<namespace>/<species>', which cannot go into a filename;
    # the file token degrades to the plain species name whenever it is unique.
    from .dataset_sources import build_species_file_tokens
    file_tokens = build_species_file_tokens(cond)
    for object_type in sorted(cond):
        object_cond = cond[object_type]
        embedding_texts = embedding_texts_by_object[object_type]
        inspection_path = pjoin(inspection_dir, f'{file_tokens[object_type]}.json')
        with open(inspection_path, 'w', encoding='utf-8') as inspection_file:
            json.dump(build_joint_name_inspection_rows(object_cond, embedding_texts), inspection_file, indent=2)

    if write_collision_report:
        write_joint_name_collision_report(cond, save_dir)
