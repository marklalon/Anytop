"""Per-joint body parts: what each joint is, and whether it stands on the ground.

:func:`prefill_joint_parts` proposes ``{part, contact, src, why}`` for every
joint of a skeleton from names first, then inheritance down the tree, then
geometry, and takes the contact flags from :func:`prefill_contacts`. Limbs are
labelled by fore/hind position, not by use: a quadruped's foreleg is
``arm``/``hand``, a multiped's walking legs are all ``leg``/``foot``; standing on
a limb is what the separate ``contact`` bit says.

Nothing here is stored with a dataset. motion_edit runs the prefill as the
default of its per-species part overrides; grounding steps (retarget, native
GLB export) read the ``foot`` chains and the contact ``hand`` chains off it.
"""

from __future__ import annotations

import numpy as np

from .joint_embedding_text import (
    _refine_joint_embedding_name,
    build_joint_embedding_texts,
    clean_embedding_token,
    joint_name_is_helper_node,
)
from .joint_name_canonical import infer_species_joint_name_prefixes, normalize_joint_name
from .physics_joint_annotation import (
    _joint_semantic_text,
    _text_matches_keywords,
    build_semantic_metadata,
    infer_symmetry_metadata,
    rest_positions_from_offsets,
)
from .joint_struct_features import child_lists

JOINT_PARTS = (
    'trunk',
    'neck',
    'head',
    'arm',
    'hand',
    'leg',
    'foot',
    'wing',
    'tail',
    'fin',
    'soft',
)
# A definite "not a body part" verdict (IK targets, props).
HELPER_PART = 'helper'
ALL_PART_LABELS = JOINT_PARTS + (HELPER_PART,)

# Provenance of a joint's part: the prefill pass that decided it, or a person's edit.
PART_SOURCES = ('name', 'inherit', 'geometry', 'manual')


def skeleton_entry(joint_names, parents, rest_positions, species_name=None) -> dict:
    """A cond-like entry for a rig outside every dataset (an external GLB/FBX),
    enough for :func:`prefill_joint_parts`. ``rest_positions`` are world
    positions, Y up."""
    parents = np.asarray(parents, dtype=np.int64)
    rest = np.asarray(rest_positions, dtype=np.float64)
    offsets = rest - np.where(parents[:, None] >= 0, rest[np.maximum(parents, 0)], 0.0)
    names = [str(name) for name in joint_names]
    entry = {'joints_names': names, 'parents': parents, 'offsets': offsets}
    if species_name:
        entry['species_name'] = str(species_name)
    entry.update(build_semantic_metadata(names, parents, offsets, rest_positions=rest,
                                         species_name=species_name))
    return entry


def foot_chains(parents, parts, contact=None) -> list[list[int]]:
    """Joints of each foot a rig stands on, one sorted list per foot: every
    connected run of ``foot`` joints, plus each connected run of ``hand``
    joints holding a ``contact`` joint (a quadruped's fore hooves). ``parts``
    is one part name per joint, ``contact`` one flag per joint (None: no hand
    stands)."""
    parents = np.asarray(parents, dtype=np.int64)
    chains: dict[int, list[int]] = {}
    top_of: dict[int, int] = {}
    for index in _topological_order(parents)[0]:
        if parts[index] not in ('foot', 'hand'):
            continue
        parent = int(parents[index])
        same_run = parent >= 0 and parent in top_of and parts[parent] == parts[index]
        top = top_of[parent] if same_run else index
        top_of[index] = top
        chains.setdefault(top, []).append(int(index))
    return [
        sorted(chain) for top, chain in sorted(chains.items())
        if parts[top] == 'foot' or (contact is not None and any(contact[j] for j in chain))
    ]


def prefill_foot_chains(cond_entry) -> list[list[int]]:
    """:func:`foot_chains` of the prefilled parts and contacts of ``cond_entry``."""
    prefill = prefill_joint_parts(cond_entry)
    rows = [prefill[str(name)] for name in cond_entry['joints_names']]
    return foot_chains(cond_entry['parents'], [row['part'] for row in rows],
                       [bool(row['contact']) for row in rows])


def ground_floors(lowest_heights, chains) -> np.ndarray:
    """Floor height of each foot chain: its lowest joint's ``lowest_heights``
    entry. Without foot chains, the single lowest joint stands in."""
    lowest_heights = np.asarray(lowest_heights, dtype=np.float64)
    if not chains:
        return np.array([float(lowest_heights.min())])
    return np.array([float(lowest_heights[chain].min()) for chain in chains])


# ---------------------------------------------------------------------------
# Prefill rules
# ---------------------------------------------------------------------------

# Slim embedding-text tokens (see build_joint_embedding_texts) -> part. When a
# text holds tokens of several parts, the first part in _NAME_PART_PRIORITY
# wins: "Tail Hair" is soft, "Wing Claw" is wing, "Neck Hair" is soft.
_NAME_PART_TOKENS = {
    'soft': {
        'hair', 'fur', 'mane', 'beard', 'mustache', 'whisker', 'bang', 'pony',
        'skirt', 'cape', 'cloak', 'ribbon', 'tassle', 'pendant', 'necklace', 'bell',
        'hat', 'helmet', 'glasses', 'jiggle', 'wiggle', 'muscle', 'fat', 'flap',
        'leaf', 'leaves', 'petal', 'flower', 'rose', 'bud',
        # Antennae and whisker-like feelers trail the head's motion.
        'feeler', 'feelers', 'shall',
    },
    'wing': {'wing', 'wings'},
    'fin': {'fin', 'pectoral', 'dorsal', 'caudal', 'anal', 'pelvic', 'gill'},
    'tail': {'tail'},
    'head': {
        'head', 'headfeature', 'face', 'ear', 'eye', 'eyelid', 'eyebrow', 'eyeball',
        'pupil', 'mascara', 'jaw', 'mouth', 'tongue', 'lip', 'cheek', 'corner', 'nose',
        'nostril', 'horn', 'horns', 'beak', 'fang', 'fangs', 'tooth', 'teeth',
        'lure', 'wattle', 'crest', 'crown',
        # An elephant's trunk is a nose; the torso is spelled Spine/Chest/Body.
        'trunk',
    },
    'neck': {'neck', 'apple'},
    'hand': {'hand', 'finger', 'thumb', 'thumbs', 'wrist', 'manus', 'palm', 'pincers'},
    'foot': {'foot', 'toe', 'ankle', 'heel', 'ball', 'hoof'},
    'arm': {'arm', 'upperarm', 'forearm', 'clavicle', 'shoulder', 'elbow'},
    'leg': {'leg', 'upperleg', 'thigh', 'calf', 'knee'},
    'trunk': {'spine', 'pelvis', 'hips', 'chest', 'body', 'belly', 'stomach', 'breast', 'butt', 'shell', 'root'},
}
_NAME_PART_PRIORITY = ('soft', 'wing', 'fin', 'tail', 'head', 'neck', 'hand', 'foot', 'arm', 'leg', 'trunk')

# Tokens whose part is the limb they hang from: a claw on a hand is a hand, on a
# foot a foot. Resolved from the inherited part instead of the name.
_DISTAL_TOKENS = {'claw', 'pad', 'nail'}
# A tentacle hanging from the head is a feeler; one from the body is a limb
# (an octopus arm), or passive on a drifting species (a jellyfish).
_TENTACLE_TOKENS = {'tentacle', 'tentacles'}
# Fore/hind words in a limb's own name.
_FORE_TOKENS = {'front', 'fore'}
_HIND_TOKENS = {'hind', 'back', 'rear'}
# Body plans whose limbs split into fore (arm/hand) and hind (leg/foot).
_TETRAPOD_PLANS = ('biped', 'quadruped', 'winged')
# A feather belongs to the wing it grows on; anywhere else it is passive.
_FEATHER_TOKENS = {'feather', 'feathers', 'prima'}
# Words the slim text blanks as non-anatomical, read back from the raw name:
# a carried object or an attachment socket is a helper, a hanging cloth or
# ponytail is soft. Any other blanked name ("Bone02", "joint7") takes its part
# from the tree.
_PROP_TOKENS = {
    'arrow', 'backpack', 'bag', 'barrel', 'blade', 'bolt', 'bow', 'effects', 'fan', 'gun',
    'halo', 'halter', 'handle', 'magic', 'mount', 'passenger', 'projectile', 'prop', 'quiver',
    'reins', 'saddle', 'shield', 'spear', 'staff', 'stick', 'sword', 'trajectory', 'weapon',
    'wood',
}
_BLANKED_SOFT_TOKENS = {'ponytail', 'ponitail', 'robe', 'headband'}
# A sided "Hip" is the thigh's root; a centred one is the pelvis.
_SIDED_HIP_TOKEN = 'hip'

# Raw-name words the slim text folds into a distal word: a Biped "HorseLink"
# (the extra segment of a digitigrade leg) is spelled "Ankle" for T5, but it is
# still the leg, above the foot.
_RAW_LEG_SEGMENTS = ('horselink',)

# Words of the joint a hand or foot hangs from: not a contact when it has
# children (they are the toes or fingers that touch down).
_NON_CONTACT_ROOT_TOKENS = {'foot', 'hand'}

# Proximal limb words: a contact joint carrying one is still the limb, not its end.
_PROXIMAL_LIMB_TOKENS = {
    'thigh', 'calf', 'knee', 'upperleg', 'upperarm', 'forearm', 'elbow', 'shoulder', 'clavicle',
}

_LIMB_PARTS = ('arm', 'hand', 'leg', 'foot')
_FORE_OF = {'arm': 'arm', 'hand': 'hand', 'leg': 'arm', 'foot': 'hand'}
_HIND_OF = {'arm': 'leg', 'hand': 'foot', 'leg': 'leg', 'foot': 'foot'}
_DISTAL_OF = {'arm': 'hand', 'leg': 'foot'}

# Geometric fore/hind split, for a tetrapod whose forelegs carry no fore word:
# the gap between limb attachments along the trunk axis must exceed this share
# of the body's length along it, or the limbs are one group.
_FORE_HIND_MIN_GAP_RATIO = 0.15
# Geometry fallback: a lateral branch whose lowest joint is within this share of
# the body height above the ground is a leg.
_GROUND_BAND_RATIO = 0.12


def _text_tokens(text):
    return [clean_embedding_token(token) for token in str(text).split() if clean_embedding_token(token)]


def _name_part(tokens, side):
    """(part, matched token) from a joint's slim text, or (None, None)."""
    token_set = set(tokens)
    if _SIDED_HIP_TOKEN in token_set:
        token_set.discard(_SIDED_HIP_TOKEN)
        token_set.add('thigh' if side != 'center' else 'hips')
    for part in _NAME_PART_PRIORITY:
        hit = token_set & _NAME_PART_TOKENS[part]
        if hit:
            return part, sorted(hit)[0]
    return None, None


def _blanked_name_part(raw_name, prefixes):
    """(part, word) for a name the slim text blanked: a prop or a passive cloth."""
    tokens = {
        clean_embedding_token(token)
        for token in _refine_joint_embedding_name(raw_name, additional_prefixes=prefixes)
    }
    if tokens & _BLANKED_SOFT_TOKENS:
        return 'soft', sorted(tokens & _BLANKED_SOFT_TOKENS)[0]
    if tokens & _PROP_TOKENS:
        return HELPER_PART, sorted(tokens & _PROP_TOKENS)[0]
    return None, None


def _topological_order(parents):
    children = child_lists(parents)
    order = [index for index, parent in enumerate(parents) if parent < 0]
    for index in order:
        order.extend(children[index])
    return order, children


def _limb_components(parents, parts, order):
    """Connected runs of limb-labelled joints, each as (root, members) in tree order."""
    components = {}
    root_of = {}
    for index in order:
        if parts[index] not in _LIMB_PARTS:
            continue
        parent = int(parents[index])
        root = root_of[parent] if parent >= 0 and parent in root_of else index
        root_of[index] = root
        components.setdefault(root, []).append(index)
    return components


def _trunk_axis(rest, parts, root_index, body_plan):
    head = [index for index, part in enumerate(parts) if part in ('head', 'neck')]
    if head:
        axis = rest[head].mean(axis=0) - rest[root_index]
        if np.linalg.norm(axis) > 1e-6:
            return axis / np.linalg.norm(axis)
    return np.array([0.0, 1.0, 0.0]) if body_plan == 'biped' else np.array([0.0, 0.0, 1.0])


def prefill_joint_parts(cond_entry) -> dict[str, dict]:
    """Proposed ``{joint name: {part, contact, src, why}}`` for one cond skeleton."""
    raw_names = [str(name) for name in cond_entry['joints_names']]
    parents = np.asarray(cond_entry['parents'], dtype=np.int64)
    joint_count = len(raw_names)
    rest = rest_positions_from_offsets(cond_entry['offsets'], parents)
    sides = list(cond_entry.get('joint_side_labels') or ['center'] * joint_count)
    tags = [str(tag).lower() for tag in (cond_entry.get('species_tags') or ())]
    body_plan = tags[0] if tags else ''
    prefixes = infer_species_joint_name_prefixes(
        raw_names, cond_entry.get('species_name') or cond_entry.get('object_type'))
    texts = build_joint_embedding_texts(cond_entry)
    tokens = [_text_tokens(text) for text in texts]
    order, children = _topological_order(parents)

    parts = [None] * joint_count
    src = [None] * joint_count
    why = [''] * joint_count

    # Helpers: IK/FX nodes and named props.
    for index in range(joint_count):
        if joint_name_is_helper_node(raw_names[index], additional_prefixes=prefixes):
            parts[index], src[index], why[index] = HELPER_PART, 'name', 'helper node'
        elif not tokens[index]:
            part, word = _blanked_name_part(raw_names[index], prefixes)
            if part is not None:
                parts[index], src[index] = part, 'name'
                why[index] = f'prop ({word})' if part == HELPER_PART else word
    # The root is the trunk whatever it is called ("Armature", "Bip01", a
    # locator, a hub): the whole body hangs from it.
    for index in np.flatnonzero(parents < 0):
        parts[index], src[index], why[index] = 'trunk', 'name', 'root'

    # Pass 1: names.
    proximal = [False] * joint_count
    distal = [False] * joint_count
    feather = [False] * joint_count
    tentacle = [False] * joint_count
    for index in range(joint_count):
        if parts[index] is not None:
            continue
        token_set = set(tokens[index])
        compact = ''.join(ch for ch in raw_names[index].lower() if ch.isalnum())
        segment = next((word for word in _RAW_LEG_SEGMENTS if word in compact), None)
        part, hit = _name_part(tokens[index], sides[index])
        if segment is not None:
            parts[index], src[index] = 'leg', 'name'
            why[index] = f'{texts[index] or raw_names[index]}; {segment} is a leg segment'
            proximal[index] = True
        elif part is not None:
            parts[index], src[index], why[index] = part, 'name', texts[index]
        elif token_set & _TENTACLE_TOKENS:
            tentacle[index] = True
        elif token_set & _FEATHER_TOKENS:
            feather[index] = True
        elif token_set & _DISTAL_TOKENS:
            distal[index] = True

    # Pass 2: inheritance down the tree.
    for index in order:
        if parts[index] is not None:
            continue
        parent = int(parents[index])
        inherited = parts[parent] if parent >= 0 else None
        if inherited is None or inherited == HELPER_PART:
            continue
        label = texts[index] or raw_names[index]
        if tentacle[index]:
            part = inherited if inherited in ('head', 'soft') else (
                'soft' if body_plan == 'drifting' else 'arm')
        elif feather[index]:
            part = 'wing' if inherited == 'wing' else 'soft'
        elif distal[index]:
            part = 'foot' if inherited in ('leg', 'foot') else 'wing' if inherited == 'wing' else (
                inherited if inherited in ('head', 'tail', 'fin') else 'hand')
        else:
            part = inherited
        parts[index], src[index] = part, 'inherit'
        why[index] = f'{label}: under {raw_names[parent]} ({inherited})'

    # A winged species whose wings are spelled as arms (a bat's Arm/Finger chain).
    if body_plan == 'winged' and 'wing' not in parts:
        for index in range(joint_count):
            if parts[index] in ('arm', 'hand'):
                why[index] = f'{why[index]}; winged species, arm chain is the wing'
                parts[index] = 'wing'

    # Contact ends of a limb are its hand/foot even when only spelled "Leg".
    contact_joints = prefill_contacts(raw_names, parents, rest)
    contact = np.zeros(joint_count, dtype=bool)
    contact[[int(index) for index in contact_joints]] = True
    for index in range(joint_count):
        if (contact[index] and parts[index] in _DISTAL_OF and not proximal[index]
                and not set(tokens[index]) & _PROXIMAL_LIMB_TOKENS):
            parts[index] = _DISTAL_OF[parts[index]]
            why[index] = f'{why[index]}; ground contact, limb end'

    # A limb chain with no hand/foot gets one at its end: below its last fork,
    # or its leaf when it never forks. Tentacles are limbs without an end.
    def limb_components():
        return {
            root: members
            for root, members in _limb_components(parents, parts, order).items()
            if not any(tentacle[index] for index in members)
        }

    components = limb_components()
    for root, members in components.items():
        if any(parts[index] in ('hand', 'foot') for index in members):
            continue
        member_set = set(members)
        forks = [index for index in members if sum(child in member_set for child in children[index]) >= 2]
        start = forks[-1] if forks else None
        for index in members:
            below_fork = start is not None and _is_descendant(index, start, parents)
            is_leaf = not any(child in member_set for child in children[index])
            if below_fork or (start is None and is_leaf):
                parts[index] = _DISTAL_OF[parts[index]]
                why[index] = f'{why[index]}; end of limb chain'

    # A limb chain that carries a wing is the wing's own skeleton
    # (Shoulder/Elbow/Wrist ending in "Wing" feathers).
    for root, members in limb_components().items():
        if any(parts[child] == 'wing' for index in members for child in children[index]):
            for index in members:
                why[index] = f'{why[index]}; carries the wing'
                parts[index] = 'wing'

    # A multiped walks on every leg: an arm chain that reaches the ground is a
    # leg, whatever its rig calls it; pincers and palps never touch down.
    if body_plan == 'multiped':
        for root, members in limb_components().items():
            if any(contact[index] for index in members):
                for index in members:
                    new_part = _HIND_OF[parts[index]]
                    if new_part != parts[index]:
                        why[index] = f'{why[index]}; multiped walking limb'
                        parts[index] = new_part

    # Fore/hind on a tetrapod: forelimbs are arm/hand, hindlimbs leg/foot. The
    # limb's own Front/Hind word decides first; geometry only when no limb is
    # an arm by any name (forelegs spelled plain "Leg"): the limbs attached
    # ahead along the trunk, past the widest gap, are the fore group.
    if body_plan in _TETRAPOD_PLANS:
        components = limb_components()
        position = {}
        for root, members in components.items():
            words = set().union(*(set(tokens[index]) for index in members))
            fore, hind = bool(words & _FORE_TOKENS), bool(words & _HIND_TOKENS)
            if fore != hind:
                position[root] = 'fore' if fore else 'hind'
        has_fore = 'fore' in position.values() or any(
            parts[index] in ('arm', 'hand') for members in components.values() for index in members
        )
        unplaced = [root for root in components if root not in position]
        if has_fore:
            # The rig spells its limbs: a chain is the side its top joint names
            # (an "Arm Ball" under an UpperArm is a hand, not a foot).
            for root in unplaced:
                position[root] = 'fore' if parts[root] in ('arm', 'hand') else 'hind'
        elif len(unplaced) >= 2:
            root_index = int(cond_entry.get('translation_root_index', 0) or 0)
            projection = rest @ _trunk_axis(rest, parts, root_index, body_plan)
            roots = sorted(unplaced, key=lambda root: projection[root])
            gaps = [projection[roots[k + 1]] - projection[roots[k]] for k in range(len(roots) - 1)]
            split = int(np.argmax(gaps))
            body_length = max(float(np.ptp(projection)), 1e-6)
            if gaps[split] > _FORE_HIND_MIN_GAP_RATIO * body_length:
                for rank, root in enumerate(roots):
                    position[root] = 'fore' if rank > split else 'hind'
        for root, side in position.items():
            mapping = _FORE_OF if side == 'fore' else _HIND_OF
            for index in components[root]:
                new_part = mapping[parts[index]]
                if new_part != parts[index]:
                    why[index] = f'{why[index]}; {side} limb'
                    parts[index] = new_part

    # Pass 3: geometry, for subtrees no name reaches.
    height = rest[:, 1]
    ground = float(height.min())
    body_height = max(float(np.ptp(height)), 1e-6)
    root_index = int(cond_entry.get('translation_root_index', 0) or 0)
    axis = _trunk_axis(rest, parts, root_index, body_plan)
    lateral_span = max(float(np.ptp(rest[:, 0])), 1e-6)
    for index in order:
        if parts[index] is not None:
            continue
        subtree = [index] + [other for other in range(joint_count) if _is_descendant(other, index, parents)]
        is_lateral = abs(float(rest[subtree, 0].mean() - rest[root_index, 0])) > 0.1 * lateral_span
        reaches_ground = float(height[subtree].min()) - ground < _GROUND_BAND_RATIO * body_height
        ahead = float((rest[index] - rest[root_index]) @ axis) > 0.0
        if is_lateral and reaches_ground:
            part, reason = 'leg', 'lateral branch reaching the ground'
        elif is_lateral and body_plan == 'winged':
            part, reason = 'wing', 'lateral branch of a winged species'
        elif is_lateral and body_plan in ('aquatic', 'drifting'):
            part, reason = 'fin', 'lateral branch of a swimming species'
        elif is_lateral:
            part, reason = 'soft', 'lateral branch, nothing else fits'
        elif ahead:
            part, reason = 'head', 'midline, ahead of the root'
        else:
            part, reason = 'tail', 'midline, behind the root'
        for member in subtree:
            if parts[member] is not None:
                continue
            member_part = part
            if part == 'leg' and contact[member] or (part == 'leg' and not children[member]):
                member_part = 'foot'
            parts[member], src[member], why[member] = member_part, 'geometry', reason

    # A joint spelled Foot or Hand with toes or fingers below it is not where
    # the limb touches down; its toes and fingers keep their contact. Helper
    # children (IK targets, sockets) do not count.
    for index in range(joint_count):
        if set(tokens[index]) & _NON_CONTACT_ROOT_TOKENS and any(
            parts[child] != HELPER_PART for child in children[index]
        ):
            contact[index] = False

    result = {}
    for index, name in enumerate(raw_names):
        is_contact = bool(contact[index]) and parts[index] != HELPER_PART
        result[name] = {
            'part': parts[index],
            'contact': int(is_contact),
            'src': src[index],
            'why': why[index],
        }
    return result


def _is_descendant(index, ancestor, parents):
    current = int(parents[index])
    while current >= 0:
        if current == ancestor:
            return True
        current = int(parents[current])
    return False


# ---------------------------------------------------------------------------
# Contact prefill (heuristic)
# ---------------------------------------------------------------------------

# Leaf name tokens that keep a leaf out of the geometric contact candidates.
_CONTACT_EXCLUDE_TOKENS = (
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
    'container',
)

# Contact joint detection tokens
_CONTACT_JOINT_KEYWORDS = (
    'toe',
    'foot',
    'feet',
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
    'feet',
    'ball',
    'wrist',
    'ankle',
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
    'feet',
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


def _select_grounded_contact_leaves(candidate_indices, joint_names, parents, rest_positions):
    if len(candidate_indices) == 0:
        return []

    candidate_indices = sorted({int(joint_index) for joint_index in candidate_indices})
    body_height = max(float(np.ptp(rest_positions[:, 1])), 1e-6)
    pair_height_margin = max(body_height * 0.24, 1e-3)
    single_height_margin = max(body_height * 0.18, 1e-3)

    _, symmetry_partner_indices, _ = infer_symmetry_metadata(joint_names, parents, rest_positions)
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


def _infer_contact_leaf_candidates(parents, joint_names):
    """Leaves named as a distal limb part: the pool the geometric contact prefill grounds."""
    children = child_lists(parents)
    candidates = []
    for joint_index, child_indices in enumerate(children):
        if child_indices or not _is_informative_joint_name(joint_names[joint_index]):
            continue
        semantic_text = _joint_semantic_text(joint_names[joint_index])
        if _text_matches_keywords(semantic_text, _CONTACT_EXCLUDE_TOKENS):
            continue
        if _text_matches_keywords(semantic_text, _CONTACT_GEOMETRY_DISTAL_TOKENS):
            candidates.append(joint_index)
    return candidates


def _infer_contact_joints_from_names(joint_names, parents, rest_positions):
    strong_candidates = []
    weak_candidates = []
    children = child_lists(parents)

    for joint_index, joint_name in enumerate(joint_names):
        semantic_text = _joint_semantic_text(joint_name)
        family_text = _joint_family_semantic_text(joint_index, joint_names, parents, max_depth=3)
        has_upper_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_UPPER_LIMB_TOKENS)
        has_lower_limb_context = _text_matches_keywords(family_text, _CONTACT_JOINT_CONTEXT_KEYWORDS)

        is_strong_contact = _text_matches_keywords(semantic_text, _CONTACT_JOINT_KEYWORDS)
        is_ball_contact = _text_matches_keywords(semantic_text, ('ball',)) and has_lower_limb_context and not has_upper_limb_context
        is_claw_contact = _text_matches_keywords(semantic_text, ('claw',)) and has_lower_limb_context and not has_upper_limb_context
        is_end_site_contact = (
            _text_matches_keywords(semantic_text, ('nub', 'end site'))
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

    candidates = _infer_contact_leaf_candidates(parents, joint_names)
    if not candidates:
        return []

    grounded_leaves = _select_grounded_contact_leaves(candidates, joint_names, parents, rest_positions)
    if not grounded_leaves:
        return []

    return _expand_contact_chain_from_leaves(grounded_leaves, joint_names, parents, rest_positions)


def prefill_contacts(joint_names, parents, rest_positions) -> list[int]:
    """Heuristic ground-contact joints of a skeleton: geometry first, then names.
    The contact prefill of :func:`prefill_joint_parts`."""
    contact_joints = _infer_contact_joints_from_geometry(joint_names, rest_positions, parents)
    if contact_joints:
        return contact_joints
    return _infer_contact_joints_from_names(joint_names, parents, rest_positions)
