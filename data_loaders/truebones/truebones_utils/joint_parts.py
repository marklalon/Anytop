"""Per-joint body-part annotation: what each joint is, and whether it stands on the ground.

The annotation is a training *target*, never a condition: the model is asked to
predict it from the names, geometry and motion it is already given. It lives in
one sidecar per dataset, ``<processed>/joint_parts.jsonl``, and a generation run
writes the same file next to its ``.npy`` output, so a reader resolves both
through :func:`load_joint_parts` and never through ``cond.npy``.

Rows are keyed by species and their joints by **name**, not index: a joint set
that changes (cropping, prop-socket removal, leaf cleanup) would silently shift
every index after the edit. ``skeleton_sig`` pins the skeleton a row was written
for; a row whose signature no longer matches the cond skeleton is stale and is
refused until it is reviewed again.

:func:`prefill_joint_parts` proposes a row from names first, then inheritance
down the tree, then geometry, and takes the contact flags from
:func:`infer_contact_joints`. Limbs are labelled by fore/hind position, not by
use: a quadruped's foreleg is ``arm``/``hand``, a multiped's walking legs are all
``leg``/``foot``; standing on a limb is what the separate ``contact`` bit says.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .joint_embedding_text import (
    _refine_joint_embedding_name,
    build_joint_embedding_texts,
    clean_embedding_token,
    joint_name_is_helper_node,
)
from .joint_name_canonical import infer_species_joint_name_prefixes
from .physics_joint_annotation import infer_contact_joints, rest_positions_from_offsets
from .joint_struct_features import child_lists

# Bumped whenever a part is added, removed, renamed or reordered: part ids are
# the classifier's output channels.
JOINT_PART_SCHEMA_VERSION = 1

# Order is the class-id order of the auxiliary head.
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
# A definite "not a body part" verdict (wrapper roots, IK targets, props). Kept
# out of the class ids: it is shown and reviewed, never trained on.
HELPER_PART = 'helper'
HELPER_PART_ID = 255
PART_IDS = {name: index for index, name in enumerate(JOINT_PARTS)}
PART_IDS[HELPER_PART] = HELPER_PART_ID
ALL_PART_LABELS = JOINT_PARTS + (HELPER_PART,)

# Provenance of a joint's entry. ``manual`` is a person's edit and is never
# overwritten by a later prefill.
PART_SOURCES = ('name', 'inherit', 'geometry', 'manual')

JOINT_PARTS_FILE = 'joint_parts.jsonl'


class JointPartsError(ValueError):
    """A joint-parts sidecar is malformed or does not fit its skeleton."""


# ---------------------------------------------------------------------------
# Sidecar I/O and binding
# ---------------------------------------------------------------------------

def skeleton_signature(joint_names, parents) -> str:
    """Stable fingerprint of a skeleton's joint names and topology."""
    payload = json.dumps(
        [[str(name) for name in joint_names], [int(parent) for parent in parents]],
        separators=(',', ':'),
    )
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def species_of(cond_entry) -> str:
    """The sidecar key of a cond entry: its bare species name."""
    name = cond_entry.get('species_name') or str(cond_entry.get('object_type', '')).rsplit('/', 1)[-1]
    if not name:
        raise JointPartsError('cond entry has neither species_name nor object_type.')
    return str(name)


def _validate_row(row, where):
    species = row.get('species')
    if not isinstance(species, str) or not species.strip():
        raise JointPartsError(f'{where}: missing species.')
    if not isinstance(row.get('skeleton_sig'), str):
        raise JointPartsError(f'{where} ({species}): missing skeleton_sig.')
    if not isinstance(row.get('reviewed'), bool):
        raise JointPartsError(f'{where} ({species}): reviewed must be true or false.')
    joints = row.get('joints')
    if not isinstance(joints, dict) or not joints:
        raise JointPartsError(f'{where} ({species}): joints must be a non-empty object.')
    for joint_name, entry in joints.items():
        part = entry.get('part')
        if part not in ALL_PART_LABELS:
            raise JointPartsError(f'{where} ({species}/{joint_name}): unknown part {part!r}.')
        if entry.get('contact') not in (0, 1):
            raise JointPartsError(f'{where} ({species}/{joint_name}): contact must be 0 or 1.')
        if part == HELPER_PART and entry['contact']:
            raise JointPartsError(f'{where} ({species}/{joint_name}): a helper joint cannot be a contact.')
        if entry.get('src') not in PART_SOURCES:
            raise JointPartsError(f'{where} ({species}/{joint_name}): unknown src {entry.get("src")!r}.')


def read_joint_parts_sidecar(path) -> dict[str, dict]:
    """``{species: row}`` from a ``joint_parts.jsonl``; empty when the file is absent."""
    path = Path(path)
    if not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        where = f'{path.name}:{line_no}'
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise JointPartsError(f'{where} is not valid JSON: {exc}') from exc
        _validate_row(row, where)
        if row['species'] in rows:
            raise JointPartsError(f'{where} duplicates species {row["species"]!r}.')
        rows[row['species']] = row
    return rows


def write_joint_parts_sidecar(path, rows) -> None:
    """Atomically rewrite ``path`` with ``rows`` (an iterable of row dicts), one per line."""
    path = Path(path)
    lines = []
    for row in rows:
        _validate_row(row, path.name)
        lines.append(json.dumps(row, ensure_ascii=False, separators=(', ', ': ')))
    tmp_path = path.with_name(path.name + '.tmp')
    tmp_path.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
    os.replace(tmp_path, path)


@dataclass(frozen=True)
class BoundJointParts:
    """One sidecar row bound onto a skeleton, in cond joint order."""
    part_ids: np.ndarray   # (J,) int16; HELPER_PART_ID for helpers
    contact: np.ndarray    # (J,) bool
    reviewed: bool
    source: str            # the row's ``source`` when it has one ("annotation" / "model")

    @property
    def contact_joints(self) -> list[int]:
        return [int(index) for index in np.flatnonzero(self.contact)]


def row_is_stale(row, joint_names, parents) -> bool:
    return row['skeleton_sig'] != skeleton_signature(joint_names, parents)


def bind_joint_parts(row, joint_names, parents) -> BoundJointParts:
    """Arrays for ``row`` on the skeleton ``joint_names`` / ``parents``.

    Refuses a stale row and a row that misses any joint: a partial binding would
    hand the loss or a grounding step a joint whose label nobody wrote.
    """
    species = row['species']
    if row_is_stale(row, joint_names, parents):
        raise JointPartsError(
            f"joint_parts row for {species!r} was written for another skeleton "
            f"(skeleton_sig {row['skeleton_sig']} != {skeleton_signature(joint_names, parents)}); "
            f"re-run tools/prefill_joint_parts.py and review it."
        )
    joints = row['joints']
    missing = [name for name in joint_names if name not in joints]
    if missing:
        raise JointPartsError(f'joint_parts row for {species!r} has no entry for {missing[:8]}.')
    part_ids = np.array([PART_IDS[joints[name]['part']] for name in joint_names], dtype=np.int16)
    contact = np.array([bool(joints[name]['contact']) for name in joint_names], dtype=bool)
    return BoundJointParts(part_ids, contact, bool(row['reviewed']), str(row.get('source', 'annotation')))


def load_joint_parts(sidecar_dir, species, cond_entry) -> BoundJointParts:
    """The parts of ``species`` from ``<sidecar_dir>/joint_parts.jsonl``, bound to ``cond_entry``.

    ``sidecar_dir`` is a processed dataset for its own clips, or a generation
    output directory for the motions written there.
    """
    path = Path(sidecar_dir) / JOINT_PARTS_FILE
    rows = read_joint_parts_sidecar(path)
    if species not in rows:
        raise JointPartsError(
            f'{path} has no row for {species!r}; run tools/prefill_joint_parts.py first.'
        )
    return bind_joint_parts(rows[species], list(cond_entry['joints_names']), cond_entry['parents'])


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
    },
    'wing': {'wing', 'wings'},
    'fin': {'fin', 'pectoral', 'dorsal', 'caudal', 'anal', 'pelvic', 'gill'},
    'tail': {'tail'},
    'head': {
        'head', 'headfeature', 'face', 'ear', 'eye', 'eyelid', 'eyebrow', 'eyeball',
        'pupil', 'mascara', 'jaw', 'mouth', 'tongue', 'lip', 'cheek', 'corner', 'nose',
        'nostril', 'horn', 'horns', 'beak', 'fang', 'fangs', 'tooth', 'teeth',
        'feeler', 'feelers', 'lure', 'wattle', 'crest', 'crown',
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
    prefixes = infer_species_joint_name_prefixes(raw_names, species_of(cond_entry))
    texts = build_joint_embedding_texts(cond_entry)
    tokens = [_text_tokens(text) for text in texts]
    order, children = _topological_order(parents)

    parts = [None] * joint_count
    src = [None] * joint_count
    why = [''] * joint_count

    # Helpers: IK/FX nodes, named props, and the wrapper chain (no body-part
    # word: "Armature", "Bip01", a bare locator) above the trunk's first part.
    for index in range(joint_count):
        if joint_name_is_helper_node(raw_names[index], additional_prefixes=prefixes):
            parts[index], src[index], why[index] = HELPER_PART, 'name', 'helper node'
        elif not tokens[index]:
            part, word = _blanked_name_part(raw_names[index], prefixes)
            if part is not None:
                parts[index], src[index] = part, 'name'
                why[index] = f'prop ({word})' if part == HELPER_PART else word
    for index in order:
        parent = int(parents[index])
        on_wrapper_chain = parent < 0 or (parts[parent] == HELPER_PART and why[parent] == 'wrapper')
        names_no_part = _name_part(tokens[index], sides[index])[0] is None
        if on_wrapper_chain and names_no_part and parts[index] is None and children[index]:
            parts[index], src[index], why[index] = HELPER_PART, 'name', 'wrapper'

    # Pass 1: names.
    distal = [False] * joint_count
    feather = [False] * joint_count
    tentacle = [False] * joint_count
    for index in range(joint_count):
        if parts[index] is not None:
            continue
        token_set = set(tokens[index])
        part, hit = _name_part(tokens[index], sides[index])
        if part is not None:
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
    contact_joints, _ = infer_contact_joints(raw_names, parents, rest)
    contact = np.zeros(joint_count, dtype=bool)
    contact[[int(index) for index in contact_joints]] = True
    for index in range(joint_count):
        if contact[index] and parts[index] in _DISTAL_OF and not set(tokens[index]) & _PROXIMAL_LIMB_TOKENS:
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


def merge_prefill(existing_row, prefill, species, skeleton_sig):
    """The row a prefill run writes, and the joint names whose entry changed.

    Kept from ``existing_row``: every ``manual`` joint, and every joint of a
    reviewed, non-stale row. A stale row loses its ``reviewed`` mark: its
    skeleton changed under it.
    """
    old_joints = dict(existing_row['joints']) if existing_row else {}
    stale = bool(existing_row) and existing_row['skeleton_sig'] != skeleton_sig
    keep_all = bool(existing_row) and existing_row['reviewed'] and not stale
    joints = {}
    changed = []
    for name, proposal in prefill.items():
        old = old_joints.get(name)
        if old is not None and (keep_all or old['src'] == 'manual'):
            joints[name] = old
            continue
        joints[name] = proposal
        if old != proposal:
            changed.append(name)
    changed.extend(name for name in old_joints if name not in prefill)
    reviewed = bool(existing_row) and existing_row['reviewed'] and not stale
    row = {'species': species, 'skeleton_sig': skeleton_sig, 'reviewed': reviewed, 'joints': joints}
    return row, changed
