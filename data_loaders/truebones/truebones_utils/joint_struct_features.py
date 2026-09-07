"""Per-joint structural descriptors: where a joint sits in its skeleton.

The joint-name embedding is the only per-joint identity signal the token
carries, so a rename can move a knee. ``graph_dist`` / ``joint_relations`` do
not fill that gap -- they are *pairwise* attention relations and never tell a
token "I am the third link of a limb that ends on the ground". This module
computes that missing per-joint signal from geometry and topology alone, so the
name is free to carry body-part semantics and nothing else.

Read ONLY from ``parents``, the physical rest positions and the contact
annotation. Deliberately *not* from any joint name: the whole point is a channel
that is bit-identical when a rig is renamed. Deliberately not from the canonical
rest token handed to ``InputProcess`` either -- that token is standardized per
species and no longer holds per-joint rest positions.

The features are pure functions of the cond entry, so they are recomputed on
both sides rather than written into cond.npy: training and generation MUST call
this one builder (``build_joint_struct_features``), never a local
re-derivation, or the two drift into different conditions for the same skeleton.
"""

from __future__ import annotations

import numpy as np

# Bumped whenever the descriptor's meaning, order or dimensionality changes.
# Recorded in args.json at train time and re-checked at resume/generation: a
# checkpoint fitted on schema N cannot read schema N+1 features, and a pure
# reordering would leave every shape lining up.
JOINT_STRUCT_FEATURE_SCHEMA_VERSION = 1

# Order is the channel order of the returned array; changing it is a schema bump.
JOINT_STRUCT_FEATURE_NAMES = (
    'depth_norm',        # (k + 1) / L      -- position along its branch-free run
    'run_len_inv',       # 1 / L            -- bounded run length, pairs with depth_norm
    'run_ends_contact',  # the run terminates on (or just above) a ground contact
    'is_contact',        # this joint is a ground contact
    'contact_known',     # the species HAS a contact annotation at all
    'is_leaf',           # no children
    'height_n',          # (y - y_min) / y_span
    'lateral_signed',    # (x - x_root) / scale   -- signed, so left != right
    'fore_aft_n',        # (z - z_mean) / scale
    'attach_h_n',        # height_n of the run's first joint -- where the limb attaches
    'subtree_n',         # subtree size / J
    'sib_rank',          # rank among same-parent siblings, in [0, 1]
    'sib_n_inv',         # 1 / (number of same-parent siblings)
)

JOINT_STRUCT_DIM = len(JOINT_STRUCT_FEATURE_NAMES)

_EPS = 1e-6


class JointStructFeatureError(ValueError):
    """A cond entry cannot produce structural features."""


def _validated_parents(object_cond, source):
    raw_parents = object_cond.get('parents')
    if raw_parents is None:
        raise JointStructFeatureError(f"{source}: cond entry has no 'parents'.")
    parents = np.asarray(raw_parents, dtype=np.int64).reshape(-1)
    joint_count = int(parents.shape[0])
    if joint_count == 0:
        raise JointStructFeatureError(f"{source}: 'parents' is empty.")

    root_indices = np.flatnonzero(parents < 0)
    if root_indices.size != 1:
        raise JointStructFeatureError(
            f"{source}: expected exactly one root (parents < 0), found "
            f"{root_indices.size} at {root_indices.tolist()}."
        )
    if int(root_indices[0]) != 0:
        # Every downstream consumer -- the collapsed joint order, the root
        # feature head, slot 0 of the padding mask -- assumes joint 0 is the root.
        raise JointStructFeatureError(
            f"{source}: the root is joint {int(root_indices[0])}, not joint 0."
        )

    non_root = np.flatnonzero(parents >= 0)
    if non_root.size and np.any(parents[non_root] >= joint_count):
        raise JointStructFeatureError(
            f"{source}: 'parents' holds an index >= the joint count {joint_count}."
        )
    if non_root.size and np.any(parents[non_root] == non_root):
        raise JointStructFeatureError(f"{source}: a joint is its own parent.")

    # Cycle check: walk every joint to the root under a hard step bound.
    for joint_index in range(joint_count):
        cursor = joint_index
        for _ in range(joint_count):
            if parents[cursor] < 0:
                break
            cursor = int(parents[cursor])
        else:
            raise JointStructFeatureError(
                f"{source}: 'parents' contains a cycle reachable from joint {joint_index}."
            )
    return parents


def _validated_rest_positions(object_cond, joint_count, source):
    """Physical rest positions, ``[J, 3]``, y-up.

    ``rest_pos_ric_hml`` is ``FK(offsets)`` with identity rotations; the
    ``rest_pose`` fallback carries the same three numbers in its first columns.
    """
    rest_pos = object_cond.get('rest_pos_ric_hml')
    if rest_pos is None:
        rest_pose = object_cond.get('rest_pose')
        if rest_pose is None:
            raise JointStructFeatureError(
                f"{source}: cond entry has neither 'rest_pos_ric_hml' nor 'rest_pose'."
            )
        rest_pos = np.asarray(rest_pose, dtype=np.float64)
    rest_pos = np.asarray(rest_pos, dtype=np.float64)
    if rest_pos.ndim != 2 or rest_pos.shape[1] < 3:
        raise JointStructFeatureError(
            f"{source}: rest positions have shape {rest_pos.shape}, expected [J, >=3]."
        )
    rest_pos = rest_pos[:, 0:3]
    if rest_pos.shape[0] != joint_count:
        raise JointStructFeatureError(
            f"{source}: {rest_pos.shape[0]} rest positions for {joint_count} joints."
        )
    if not np.all(np.isfinite(rest_pos)):
        raise JointStructFeatureError(f"{source}: rest positions contain non-finite values.")
    return rest_pos


def _child_lists(parents):
    children = [[] for _ in range(len(parents))]
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            children[int(parent_index)].append(joint_index)
    return children


def _branch_free_runs(parents, children, source):
    """Partition the joints into non-overlapping branch-free runs.

    The root is a run of its own whatever its degree, every child of the root or
    of a branch point opens a new run, and a run walks its single child until it
    reaches a leaf or the next branch point -- which belongs to the *upstream*
    run, its own children opening new ones.
    """
    joint_count = len(parents)
    run_of = np.full(joint_count, -1, dtype=np.int64)
    position_in_run = np.zeros(joint_count, dtype=np.int64)
    run_members = []

    pending = [0]
    while pending:
        start = pending.pop()
        members = [start]
        cursor = start
        if start != 0:
            while len(children[cursor]) == 1:
                cursor = children[cursor][0]
                members.append(cursor)
        run_index = len(run_members)
        for offset, joint_index in enumerate(members):
            if run_of[joint_index] >= 0:
                raise JointStructFeatureError(
                    f"{source}: joint {joint_index} lands in two branch-free runs; "
                    "'parents' is not a tree."
                )
            run_of[joint_index] = run_index
            position_in_run[joint_index] = offset
        run_members.append(members)
        pending.extend(reversed(children[cursor]))

    if np.any(run_of < 0):
        unreached = np.flatnonzero(run_of < 0).tolist()
        raise JointStructFeatureError(
            f"{source}: joints {unreached[:8]} are not reachable from the root; "
            "'parents' is not one tree."
        )
    return run_of, position_in_run, run_members


def _subtree_sizes(parents, children):
    """Node count of each joint's subtree, accumulated leaves-first."""
    joint_count = len(parents)
    sizes = np.ones(joint_count, dtype=np.float64)
    order = []
    stack = [0]
    while stack:
        joint_index = stack.pop()
        order.append(joint_index)
        stack.extend(children[joint_index])
    for joint_index in reversed(order):
        parent_index = int(parents[joint_index])
        if parent_index >= 0:
            sizes[parent_index] += sizes[joint_index]
    return sizes


def _contact_flags(object_cond, joint_count, source):
    contact_source = str(object_cond.get('contact_joint_source') or 'none')
    is_contact = np.zeros(joint_count, dtype=bool)
    for raw_index in list(object_cond.get('contact_joints') or []):
        contact_index = int(raw_index)
        if not 0 <= contact_index < joint_count:
            raise JointStructFeatureError(
                f"{source}: contact joint index {contact_index} is out of range for "
                f"{joint_count} joints."
            )
        is_contact[contact_index] = True
    return is_contact, contact_source != 'none'


def build_joint_struct_features(object_cond, source='cond entry'):
    """``[J, JOINT_STRUCT_DIM]`` float32 structural descriptors for one species.

    Deterministic and name-invariant: two cond entries with the same ``parents``,
    rest positions and contact annotation produce bit-identical output whatever
    their joints are called.
    """
    parents = _validated_parents(object_cond, source)
    joint_count = int(parents.shape[0])
    rest_pos = _validated_rest_positions(object_cond, joint_count, source)
    children = _child_lists(parents)
    _run_of, position_in_run, run_members = _branch_free_runs(parents, children, source)
    is_contact, contact_known = _contact_flags(object_cond, joint_count, source)

    spans = rest_pos.max(axis=0) - rest_pos.min(axis=0)
    # One isotropic scale for the two signed axes, so a limb's lateral offset is
    # not rescaled by how flat or how tall this particular species happens to be.
    # A degenerate rig (every joint at one point) floors at _EPS rather than
    # dividing by zero, and lands on the all-equal features that describes.
    scale = float(max(spans.max(), _EPS))
    y_min = float(rest_pos[:, 1].min())
    y_span = float(max(spans[1], _EPS))
    height_n = (rest_pos[:, 1] - y_min) / y_span
    lateral_signed = (rest_pos[:, 0] - float(rest_pos[0, 0])) / scale
    fore_aft_n = (rest_pos[:, 2] - float(rest_pos[:, 2].mean())) / scale

    subtree_sizes = _subtree_sizes(parents, children)

    # Sibling rank: a stable order over same-parent children, so a centipede's
    # repeated leg pairs are told apart by where they sit rather than by whatever
    # array order they were exported in. Ranked on the scale-normalized
    # coordinates -- rounded, so float noise cannot reorder a mirrored pair --
    # with the array index as the final tiebreak.
    sib_rank = np.zeros(joint_count, dtype=np.float64)
    sib_n_inv = np.ones(joint_count, dtype=np.float64)
    for parent_index in range(joint_count):
        siblings = children[parent_index]
        sibling_count = len(siblings)
        if sibling_count == 0:
            continue
        ordered = sorted(
            siblings,
            key=lambda joint_index: (
                round(float(rest_pos[joint_index, 2]) / scale, 6),
                round(float(rest_pos[joint_index, 0]) / scale, 6),
                joint_index,
            ),
        )
        for rank, joint_index in enumerate(ordered):
            sib_rank[joint_index] = 0.0 if sibling_count == 1 else rank / float(sibling_count - 1)
            sib_n_inv[joint_index] = 1.0 / float(sibling_count)

    run_length = np.zeros(joint_count, dtype=np.float64)
    run_start_index = np.zeros(joint_count, dtype=np.int64)
    run_ends_contact = np.zeros(joint_count, dtype=np.float64)
    for members in run_members:
        last_index = members[-1]
        # "or its direct child": a foot's contact is often annotated one joint
        # further down, on the toe that opens the next run.
        ends_contact = bool(is_contact[last_index]) or any(
            bool(is_contact[child_index]) for child_index in children[last_index]
        )
        for joint_index in members:
            run_length[joint_index] = float(len(members))
            run_start_index[joint_index] = members[0]
            run_ends_contact[joint_index] = 1.0 if ends_contact else 0.0

    is_leaf = np.asarray(
        [0.0 if children[joint_index] else 1.0 for joint_index in range(joint_count)],
        dtype=np.float64,
    )
    features = np.stack(
        [
            (position_in_run.astype(np.float64) + 1.0) / run_length,
            1.0 / run_length,
            run_ends_contact,
            is_contact.astype(np.float64),
            np.full(joint_count, 1.0 if contact_known else 0.0),
            is_leaf,
            height_n,
            lateral_signed,
            fore_aft_n,
            height_n[run_start_index],
            subtree_sizes / float(joint_count),
            sib_rank,
            sib_n_inv,
        ],
        axis=1,
    )
    if features.shape != (joint_count, JOINT_STRUCT_DIM):
        raise JointStructFeatureError(
            f"{source}: built {features.shape} structural features, expected "
            f"{(joint_count, JOINT_STRUCT_DIM)}."
        )
    if not np.all(np.isfinite(features)):
        bad = np.flatnonzero(~np.isfinite(features).all(axis=1)).tolist()
        raise JointStructFeatureError(
            f"{source}: structural features are non-finite at joints {bad[:8]}."
        )
    return np.ascontiguousarray(features, dtype=np.float32)


def build_joint_struct_features_for_cond(cond_dict):
    """``{object_type: [J, D] float32}`` for a whole cond dict."""
    return {
        str(object_type): build_joint_struct_features(entry, source=str(object_type))
        for object_type, entry in cond_dict.items()
    }
