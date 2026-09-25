"""Animation processing & joint metadata utilities.

Lowest layer of the motion-processing pipeline. Handles skeleton editing
(cropping, prop-socket / end-site removal, DFS reorder), BVH export
preparation, scaling and FK helpers. Joint-name canonicalization lives in
``joint_name_canonical`` and root motion in ``root_motion``; both are
re-exported here.
"""

from motion_lib import Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global
import numpy as np
import re
from data_loaders.truebones.truebones_utils.param_utils import (
    DEGENERATE_BONE_LENGTH_RATIO,
    HML_REF_AXIAL_BONE_LENGTH,
    HML_REF_MAX_SPAN,
    MAX_JOINTS,
    PROP_SOCKET_BONE_LENGTH_RATIO,
    PROP_SOCKET_MAX_SUBTREE_JOINTS,
    SCALE_BODY_SPAN_BLEND_WEIGHT,
)
from data_loaders.truebones.truebones_utils.skeleton_cropping import (
    select_cropped_joint_indices,
)
from .joint_embedding_text import joint_name_is_non_anatomical


################## Constants #####################

ANSI_YELLOW = '\033[93m'
ANSI_RESET = '\033[0m'


def _warn(msg: str):
    """Print a warning message in yellow."""
    print(f'{ANSI_YELLOW}[WARN] {msg}{ANSI_RESET}')


from .joint_name_canonical import (  # noqa: F401  re-exported
    canonical_name_for_bvh,
    build_joint_name_inspection_rows,
    _remove_token_counts,
    _joint_disambiguation_tokens,
    _display_disambiguation_tokens,
    _disambiguate_duplicate_canonical_names,
    assign_canonical_joint_names,
    collect_joint_name_collision_groups,
    write_joint_name_collision_report,
    refresh_joint_metadata_in_object_cond,
    refresh_joint_metadata_in_cond_dict,
)
from .joint_embedding_text import (  # noqa: F401  re-exported
    _T5_ENCODE_BATCH,
    _build_t5_text_cache,
    attach_t5_embeddings_to_cond,
)
from .root_motion import (  # noqa: F401  re-exported
    ROOT_XZ_DRIFT_THRESHOLD,
    ROOT_Y_DRIFT_THRESHOLD,
    ROOT_XZ_SOFT_CLAMP_KNEE,
    ROOT_XZ_SOFT_CLAMP_LIMIT,
    ROOT_XZ_LOCOMOTION_KNEE,
    ROOT_XZ_LOCOMOTION_LIMIT,
    LOOP_DETECTION_GAP_RATIO,
    LOOP_DETECTION_STEP_MIN,
    LOOP_DETECTION_STEP_MAX,
    LOOP_DETECTION_ROOT_XZ_TOLERANCE,
    max_joint_span,
    compute_motion_loop_diagnostics,
    detect_motion_loop,
    detect_loop_from_features,
    _translation_root_candidate_chain,
    find_translation_root,
    ROOT_TRANSPORT_CARRIER_SHARE,
    ROOT_TRANSPORT_MIN_TRAVEL,
    chain_xz_travel,
    select_transport_carrier,
    rest_pose_animation,
    _get_reference_body_length,
    _excursion_scale,
    _compress_positive_excursion,
    _compress_negative_excursion,
    _compress_below_negative_band,
    _soft_clamp_min_height,
    _VERTICAL_BAND_SUBSETS,
    vertical_band_inputs,
    clamp_vertical_height_track,
    clamp_vertical_trajectory,
    _coerce_root_xz_center,
    _get_translation_root_initial_xz,
    move_xz_to_origin,
    xz_locomotion_extent,
    root_xz_trajectory,
    root_xz_heading,
    _xz_rotation,
    _detrend_frame,
    _frame_correction,
    root_xz_drift_correction,
    flatten_root_xz_drift,
    root_y_drift_correction,
    flatten_root_y_drift,
    soft_clamp_extent,
    soft_clamp_root_xz,
    scale_root_xz_extent,
    TRANSPORT_CARRIER_EPS,
    translation_root_ancestor_chain,
    collapse_translation_root_chain,
    _transport_carrier_index,
    _set_translation_root_axes,
    set_translation_root_xz,
    set_translation_root_y,
    resolve_detected_translation_root_index,
)


################## BVH Export Utilities #####################

def needs_bvh_position_channels(anim, tol=1e-4):
    """Return True when BVH export must write non-root position channels.

    This repo's internal FK uses ``anim.positions`` directly as each joint's local
    translation. Exporting with ``positions=False`` makes BVH viewers reconstruct
    every non-root joint from the static rest ``offsets`` instead, so any non-root
    local position that differs from those offsets must be written explicitly.
    """
    if anim.positions.shape[1] <= 1:
        return False

    nonroot_positions = np.asarray(anim.positions[:, 1:, :], dtype=np.float64)
    rest_offsets = np.asarray(anim.offsets[1:], dtype=np.float64)[None, :, :]
    return bool(np.any(np.abs(nonroot_positions - rest_offsets) > tol))


def reindex_animation_to_kept_joints(anim, names, keep_indices):
    """Rebuild ``anim``/``names`` over ``keep_indices``, remapping the parents.

    ``keep_indices`` must be ascending and closed under parenthood -- a kept
    joint's parent is kept too -- which is what makes the remapped hierarchy a
    valid tree. Both callers guarantee it: cropping only ever peels current
    leaves, and the prop-socket filter drops whole subtrees.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    n = int(parents.shape[0])
    keep_set = set(int(index) for index in keep_indices)
    for old_index in keep_indices:
        parent_index = int(parents[old_index])
        # A dropped parent would remap to -1 and silently promote its child to a
        # second root, so reject the caller's keep-set instead of the tree.
        if parent_index >= 0 and parent_index not in keep_set:
            raise ValueError(
                f"joint {old_index} is kept but its parent {parent_index} was dropped"
            )

    old_to_new = -np.ones((n,), dtype=np.int64)
    for new_index, old_index in enumerate(keep_indices):
        old_to_new[old_index] = new_index
    parent_dtype = getattr(anim.parents, 'dtype', np.int32)
    new_parents = np.array(
        [
            int(old_to_new[parents[old_index]]) if parents[old_index] >= 0 else -1
            for old_index in keep_indices
        ],
        dtype=parent_dtype,
    )

    keep_arr = np.asarray(keep_indices, dtype=np.int64)
    orient_count = len(anim.orients)
    if orient_count not in (0, n):
        raise ValueError(
            f"Expected 0 or {n} joint orients to reindex skeleton, got {orient_count}"
        )
    new_orients = anim.orients.copy() if orient_count == 0 else anim.orients[keep_arr].copy()

    reindexed = Animation(
        anim.rotations[:, keep_arr].copy(),
        anim.positions[:, keep_arr].copy(),
        new_orients,
        anim.offsets[keep_arr].copy(),
        new_parents,
    )
    return reindexed, [names[old_index] for old_index in keep_indices]


def crop_animation_to_max_joints(anim, names, max_joints=MAX_JOINTS, *, context=None):
    """Crop ``anim``/``names`` to at most ``max_joints`` joints.

    Cropping always removes current leaves first. Deeper leaves are removed
    before shallower ones; at the same depth, shorter bones are trimmed before
    longer ones. Non-root joints whose rest-offset length exceeds the mean
    non-root bone length are preserved whenever possible and are only dropped as
    a last resort when the cap still cannot be met.

    Returns ``(anim, names, keep_indices)``. When no cropping is needed the
    inputs are returned unchanged with ``keep_indices=None``. When cropping
    happens a yellow [WARN] log line names the dropped joints so the affected
    clip/skeleton is easy to spot. The keep-set is a deterministic function of
    the parent topology and rest-pose offsets, so the rest-pose skeleton and
    every motion clip of the same character crop to the identical joint set.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    n = int(parents.shape[0])
    names = list(names)
    if len(names) != n:
        raise ValueError(
            f"Expected {n} joint names to crop skeleton, got {len(names)}"
        )

    selection = select_cropped_joint_indices(parents, max_joints, offsets=anim.offsets)
    if selection is None:
        return anim, names, None
    keep_indices, removed_order = selection

    cropped, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)

    removed_names = [names[old_index] for old_index in removed_order]
    label = f' for {context}' if context else ''
    _warn(f'skeleton exceeds MAX_JOINTS ({n} > {max_joints}){label}')
    return cropped, new_names, keep_indices


def promote_translation_root_to_hierarchy_root(anim, names, depth, *, context=None):
    """Drop ``depth`` wrapper joints above the effective root, folding them into it.

    The joints between the hierarchy root and the measured transport carrier are
    inert control nodes -- ``Cg``, ``Ctrl``, ``All``, and rigs whose wrapper is
    merely NAMED ``Hips`` / ``Root`` / ``Body``. Their offset from the real root
    carries no motion (that is what the carrier measurement establishes), so all
    they contribute to the model is a joint token whose whole 12-dim feature
    vector is a per-species constant, plus a second meaning for "joint 0".

    Each step folds the dropped joint's rotation, offset and animated translation
    into its child (``promote_root_once``), so every remaining joint keeps its
    world transform exactly and the character's yaw -- which 9 of the 34 wrapper
    species author on the wrapper itself, ``SabreToothTiger_180RIght`` and
    ``KI_Human_RunTurn02Right`` among them -- survives on the real root.

    Deliberately NOT done in the FBX/BVH loader. That layer is a conservative
    fallback that has to guess from names and offsets, and it cannot know where
    the transport is: the species root is only decided after phase 1 has measured
    every clip. Returns ``(anim, names, keep_indices)``, inputs unchanged and
    ``keep_indices=None`` when ``depth`` is 0.
    """
    depth = int(depth)
    if depth <= 0:
        return anim, list(names), None

    names = list(names)
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to promote the translation root, "
            f"got {len(names)}"
        )
    if depth >= joint_count:
        raise ValueError(
            f"Cannot promote past the whole skeleton: depth {depth} of {joint_count} joints"
        )

    from motion_lib.root_collapse import promote_root_once

    working_names = names
    working_parents = np.asarray(anim.parents, dtype=np.int32)
    working_offsets = np.asarray(anim.offsets, dtype=np.float64)
    working_rotations = np.asarray(anim.rotations.qs, dtype=np.float64)
    working_positions = np.asarray(anim.positions, dtype=np.float64)
    working_orients = anim.orients

    for step in range(depth):
        if int(np.count_nonzero(working_parents == 0)) != 1:
            where = f" for {context}" if context else ""
            raise ValueError(
                f"Cannot promote the translation root{where}: joint "
                f"'{working_names[0]}' has more than one child, so dropping it "
                f"would move something other than the root chain"
            )
        (
            working_names,
            working_parents,
            working_offsets,
            working_rotations,
            working_positions,
            working_orients,
        ) = promote_root_once(
            working_names,
            working_parents,
            working_offsets,
            working_rotations,
            working_positions,
            working_orients,
        )

    promoted = Animation(
        Quaternions(working_rotations),
        working_positions,
        working_orients,
        working_offsets,
        np.asarray(working_parents, dtype=anim.parents.dtype),
    )
    return promoted, working_names, list(range(depth, joint_count))


def drop_prop_socket_joints(anim, names, *, drop_names=None, context=None):
    """Drop parked prop/weapon socket subtrees from a loaded skeleton.

    A held weapon parked away from the body (MLH_Archer's ``Bow``/``Arrow``,
    RMW_Orc's ``Weapon01``) is rig furniture, not anatomy: it carries no body
    motion to learn and spends a joint slot in every window that samples it.
    ``find_prop_socket_joints`` picks the sockets; this removes them and their
    subtrees, then remaps the hierarchy.

    Run this BEFORE ``crop_animation_to_max_joints`` so the MAX_JOINTS budget is
    spent on anatomy, and so every downstream rest-pose artifact (face joints,
    contact joints, offsets) is inferred on the filtered skeleton.

    ``drop_names`` makes the removal follow an explicit name list instead of
    re-detecting. The rest pose and each motion clip are *different files*, so
    detecting independently on both risks the two disagreeing; the rest pose
    decides once and every clip of that character follows the same names. A name
    the clip's rig does not carry is an error, not a silent skip. Detection logs
    a [WARN] naming the dropped joints (once per character); following an
    explicit list stays quiet, since it only mirrors a decision already logged.

    Returns ``(anim, names, keep_indices)``, with ``keep_indices=None`` and the
    inputs unchanged when nothing is dropped.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to drop prop sockets, got {len(names)}"
        )
    label = f' for {context}' if context else ''

    if drop_names is None:
        prop_joints = find_prop_socket_joints(anim.offsets, parents, names)
    else:
        wanted = set(drop_names)
        missing = wanted.difference(names)
        if missing:
            raise ValueError(
                f"prop-socket joints {sorted(missing)} named by the rest pose are "
                f"missing from this skeleton{label}"
            )
        prop_joints = {
            joint_index
            for joint_index in range(joint_count)
            if names[joint_index] in wanted
        }
    if 0 in prop_joints:
        raise ValueError(f"prop-socket filter would drop the root joint{label}")
    # Sweep the sockets' descendants in, so the removed set is always whole
    # subtrees. Parents always precede their children, so one ascending pass
    # carries a socket down its whole chain.
    for joint_index in range(1, joint_count):
        if int(parents[joint_index]) in prop_joints:
            prop_joints.add(joint_index)
    if not prop_joints:
        return anim, names, None

    keep_indices = [
        joint_index for joint_index in range(joint_count) if joint_index not in prop_joints
    ]
    filtered, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)
    if drop_names is None:
        dropped = [names[joint_index] for joint_index in sorted(prop_joints)]
        _warn(f'dropping {len(dropped)} prop-socket joint(s){label}: {dropped}')
    return filtered, new_names, keep_indices


# A BVH "End Site" re-imported as a real bone: a leaf whose name is its parent's
# plus an ``end``/``end site`` terminator ("Tail10_end", "L Toe01_end_site",
# "Hand End"). The separator is required so real anatomy never matches -- "Bend"
# and "Legend" are names, not terminators.
_END_SITE_NAME_RE = re.compile(r'[_\s.-](end|end[_\s.-]?site)$', re.IGNORECASE)


def find_end_site_joints(parents, names):
    """Indices of the End-Site terminator leaves in a skeleton.

    A joint qualifies only when it is a leaf AND its name carries an End-Site
    suffix, so a real bone that happens to sit at the end of a chain is never
    touched. Terminators that stack ("Foo_end" carrying a "Foo_end_site") are
    peeled repeatedly, since removing the outer one turns the inner one into a
    leaf.
    """
    parents = np.asarray(parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    dropped = set()
    while True:
        has_child = {
            int(parents[joint_index])
            for joint_index in range(joint_count)
            if joint_index not in dropped and int(parents[joint_index]) >= 0
        }
        newly = {
            joint_index
            for joint_index in range(1, joint_count)
            if joint_index not in dropped
            and joint_index not in has_child
            and _END_SITE_NAME_RE.search(str(names[joint_index]))
        }
        if not newly:
            return dropped
        dropped |= newly


def drop_end_site_joints(anim, names, *, drop_names=None, context=None):
    """Drop BVH End-Site terminator leaves from a loaded skeleton.

    A tpose that round-tripped through BVH comes back with every End Site
    materialised as a real bone. They carry no motion, and each one steals the
    ``EndEffector``/``ChainEnd`` marker from the joint it hangs off -- which
    changes that joint's canonical name and so its T5 conditioning vector, and
    lengthens every ``Segment N Of M`` count up its chain. The same character
    loaded from FBX and from BVH would then condition differently, so the
    terminators are removed rather than cropped: this is not a joint-budget
    decision and it runs whether or not ``crop_animation_to_max_joints`` does.

    Run it AFTER ``drop_prop_socket_joints`` (a parked weapon's terminator goes
    out with the weapon) and BEFORE ``crop_animation_to_max_joints``, so the crop
    spends its budget on anatomy instead of punctuation.

    ``drop_names`` mirrors the prop-socket contract: the rest pose detects once
    and every motion clip of that character follows the same explicit list, so
    the two can never disagree about the joint set. A name the clip's rig does
    not carry is an error, not a silent skip.

    Returns ``(anim, names, keep_indices)``, with ``keep_indices=None`` and the
    inputs unchanged when nothing is dropped.
    """
    parents = np.asarray(anim.parents, dtype=np.int64)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names to drop end sites, got {len(names)}"
        )
    label = f' for {context}' if context else ''

    if drop_names is None:
        end_site_joints = find_end_site_joints(parents, names)
    else:
        wanted = set(drop_names)
        missing = wanted.difference(names)
        if missing:
            raise ValueError(
                f"end-site joints {sorted(missing)} named by the rest pose are "
                f"missing from this skeleton{label}"
            )
        end_site_joints = {
            joint_index
            for joint_index in range(joint_count)
            if names[joint_index] in wanted
        }
    if 0 in end_site_joints:
        raise ValueError(f"end-site filter would drop the root joint{label}")
    if not end_site_joints:
        return anim, names, None

    keep_indices = [
        joint_index for joint_index in range(joint_count) if joint_index not in end_site_joints
    ]
    # reindex_animation_to_kept_joints rejects a keep-set that orphans a child,
    # which is the guard that an explicit drop_names list really did name leaves.
    filtered, new_names = reindex_animation_to_kept_joints(anim, names, keep_indices)
    if drop_names is None:
        dropped = [names[joint_index] for joint_index in sorted(end_site_joints)]
        _warn(f'dropping {len(dropped)} BVH end-site joint(s){label}: {dropped}')
    return filtered, new_names, keep_indices


def reorder_animation_to_dfs(anim, names):
    """Reindex an Animation into true DFS order for BVH export.

    Helper joints are appended at the tail of the joint arrays during
    preprocessing, which keeps parent-before-child ordering but no longer matches
    the BVH hierarchy traversal order. ``motion_lib.BVH.save`` writes its
    HIERARCHY recursively in DFS order while the MOTION block is emitted in array
    index order, so helper-augmented animations must be remapped so both orders
    agree.
    """
    parents = np.asarray(anim.parents, dtype=np.int32)
    joint_count = int(parents.shape[0])
    names = list(names)
    if len(names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names for BVH export, got {len(names)}"
        )
    if joint_count <= 1:
        return anim, names

    children = [[] for _ in range(joint_count)]
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        if parent_index >= joint_count:
            raise ValueError(
                f"Joint {joint_index} has invalid parent {parent_index} for {joint_count} joints"
            )
        children[int(parent_index)].append(joint_index)

    roots = np.flatnonzero(parents < 0).tolist()
    if not roots:
        raise ValueError("BVH export requires at least one root joint")

    dfs_order = []
    stack = list(reversed(roots))
    while stack:
        joint_index = stack.pop()
        dfs_order.append(joint_index)
        stack.extend(reversed(children[joint_index]))

    if len(dfs_order) != joint_count:
        raise ValueError(
            f"DFS traversal covered {len(dfs_order)} of {joint_count} joints during BVH export"
        )

    if dfs_order == list(range(joint_count)):
        return anim, names

    old_to_new = np.empty((joint_count,), dtype=np.int32)
    for new_index, old_index in enumerate(dfs_order):
        old_to_new[old_index] = new_index

    reordered_parents = np.array([
        old_to_new[parent_index] if parent_index >= 0 else -1
        for parent_index in parents[dfs_order]
    ], dtype=np.int32)

    orient_count = len(anim.orients)
    if orient_count not in (0, joint_count):
        raise ValueError(
            f"Expected 0 or {joint_count} joint orients for BVH export, got {orient_count}"
        )
    reordered_orients = anim.orients.copy() if orient_count == 0 else anim.orients[dfs_order].copy()

    reordered_anim = Animation(
        anim.rotations[:, dfs_order].copy(),
        anim.positions[:, dfs_order].copy(),
        reordered_orients,
        anim.offsets[dfs_order].copy(),
        reordered_parents,
    )
    reordered_names = [names[joint_index] for joint_index in dfs_order]
    return reordered_anim, reordered_names


################## Scaling Utilities #####################

# 3ds Max Biped's held-prop bone ("Bip001-Prop1", "Bip01 Prop2"). The Biped
# system reserves the name for props, so it is trusted without the length gate:
# a rig can park its prop close to the body in the T-pose (Pet_Hamperor's
# sceptre sits under the gate while it rides the right hand in every clip).
_BIPED_PROP_BONE_PATTERN = re.compile(r'(?<![a-z0-9])bip\d+[\s_\-]?prop\d+$', re.IGNORECASE)


def _is_biped_prop_bone(name):
    return bool(_BIPED_PROP_BONE_PATTERN.search(str(name)))


def find_prop_socket_joints(
    offsets,
    parents,
    joint_names,
    length_ratio=PROP_SOCKET_BONE_LENGTH_RATIO,
    max_subtree_joints=PROP_SOCKET_MAX_SUBTREE_JOINTS,
):
    """Joints held by a detached prop/weapon socket bone, as an index set.

    Unity character packs park a held weapon in its own bone far from the body
    (MLH_Archer's ``Bow``/``Arrow`` at (+/-3, 1, 0) on a 1.03-tall archer), which
    dominates the rest-pose size statistics and shrinks the character ~3x. A
    socket is a joint satisfying all three of:

    * its name carries no body part (``joint_name_is_non_anatomical``),
    * it carries at most ``max_subtree_joints`` joints below it,
    * its bone is ``length_ratio`` times the skeleton's 90th-percentile bone
      (median as floor for near-degenerate rigs).

    A 3ds Max Biped prop bone (``Bip001-Prop1``) skips the length test; the
    subtree cap still applies. That name is the Biped system's own reservation
    for props, unlike the free-form words the name test reads elsewhere.

    The socket and its whole subtree are returned. Name and geometry fail in
    opposite directions and are exact only together: 194 dataset joints are
    non-anatomically named but sit on the body (armor, saddles), while
    geometry alone cannot tell a parked staff from a single-bone tail.

    The scan repeats on the skeleton left after each pass's removal, until a pass
    finds nothing. The props sit in the top decile they are measured against, so
    a rig carrying several of them inflates its own reference: PetKikiA's stick
    and first fire bone lift the P90 far enough to hide the other two fire bones
    until they are gone. A rig with no socket stops after the first pass, so only
    rigs that already had one can change -- the result is always exactly what
    ``drop_prop_socket_joints`` followed by a fresh scan would leave, which keeps
    the drop idempotent.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    parents = np.asarray(parents)
    joint_count = len(parents)
    if joint_count < 2:
        return set()
    if len(joint_names) != joint_count:
        raise ValueError(
            f"Expected {joint_count} joint names for the socket-name guard, got {len(joint_names)}"
        )

    # The root's own offset is rig placement, not a bone: keep it out of both
    # the reference and the scan, the same way every other bone statistic does.
    lengths = np.linalg.norm(offsets, axis=1)
    non_anatomical = {}

    def _name_is_non_anatomical(joint_index):
        if joint_index not in non_anatomical:
            non_anatomical[joint_index] = joint_name_is_non_anatomical(joint_names[joint_index])
        return non_anatomical[joint_index]

    prop_joints = set()
    while True:
        remaining = [joint_index for joint_index in range(1, joint_count) if joint_index not in prop_joints]
        if not remaining:
            break
        remaining_lengths = lengths[remaining]
        reference = max(
            float(np.percentile(remaining_lengths, 90)),
            float(np.median(remaining_lengths)),
        )
        if reference <= 0.0:
            break

        subtree_size = np.ones(joint_count, dtype=np.int64)
        for joint_index in range(joint_count - 1, 0, -1):
            parent_index = int(parents[joint_index])
            if parent_index >= 0 and joint_index not in prop_joints:
                subtree_size[parent_index] += subtree_size[joint_index]

        found = {
            joint_index
            for joint_index in remaining
            if subtree_size[joint_index] <= max_subtree_joints
            and (
                _is_biped_prop_bone(joint_names[joint_index])
                or (
                    lengths[joint_index] > length_ratio * reference
                    and _name_is_non_anatomical(joint_index)
                )
            )
        }
        if not found:
            break
        prop_joints |= found
        # Sweep the sockets' descendants in. Parents always precede their
        # children, so one ascending pass carries a socket down its whole chain.
        for joint_index in range(1, joint_count):
            if int(parents[joint_index]) in prop_joints:
                prop_joints.add(joint_index)
    return prop_joints


def get_average_axial_bone_length(offsets, parents, joint_side_labels, joint_names):
    """Compute the mean bone length of axial (center-labeled) bones, excluding root.

    Falls back to the mean bone length across *all* non-root bones when no
    center-labeled bones exist, so the return value is always a positive float.

    Degenerate bones -- rig helpers that sit exactly on their parent, e.g. the
    zero-length ``RigSpine`` of MU04_Pollen -- are dropped from the mean. They
    carry no size information but drag the average down, which inflates the
    character's scale factor (Pollen: 2.95 instead of 4.43, a 1.22x oversize).
    Prop-socket bones (see ``find_prop_socket_joints``) are dropped for the
    mirror-image reason: MLH_Archer's 3.05-long ``Bow``/``Arrow`` lift the 0.31
    body mean to 0.55 and shrink the archer 1.8x. The branch is still chosen by
    the *unfiltered* center-bone count, so filtering never flips skeletons
    between branches.
    """
    prop_joints = find_prop_socket_joints(offsets, parents, joint_names)
    all_lengths = [
        float(np.linalg.norm(offsets[j]))
        for j in range(1, len(parents))
        if j not in prop_joints
    ]
    if not all_lengths:
        return 0.1  # ultimate fallback for single-bone skeletons
    min_bone_length = DEGENERATE_BONE_LENGTH_RATIO * (sum(all_lengths) / len(all_lengths))

    center_count = 0
    axial_lengths = []
    for joint_index in range(1, len(parents)):  # skip root (no parent bone)
        if joint_index < len(joint_side_labels) and joint_side_labels[joint_index] == 'center':
            center_count += 1
            if joint_index not in prop_joints:
                axial_lengths.append(float(np.linalg.norm(offsets[joint_index])))
    if center_count >= 10 and axial_lengths:
        return _mean_excluding_degenerate(axial_lengths, min_bone_length)
    # Fallback: average bone length across all non-root bones.
    return _mean_excluding_degenerate(all_lengths, min_bone_length)


def _mean_excluding_degenerate(lengths, min_bone_length):
    """Mean of ``lengths`` ignoring degenerate ones; unfiltered mean if all are."""
    kept = [length for length in lengths if length >= min_bone_length]
    if not kept:
        return sum(lengths) / len(lengths)
    return sum(kept) / len(kept)


def rest_positions_from_offsets(offsets, parents):
    """Accumulate parent-to-child ``offsets`` into rest-pose joint positions.

    The offsets must already be rest-pose deltas (``rest_positions[j] -
    rest_positions[parent[j]]``). Bone-local ``Animation.offsets`` are stated in
    the parent bone's frame and sum to a straightened-out skeleton, so FK a
    loaded animation instead of passing ``anim.offsets`` here.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    rest_positions = np.zeros_like(offsets)
    for joint_index, parent_index in enumerate(parents):
        if parent_index >= 0:
            rest_positions[joint_index] = rest_positions[parent_index] + offsets[joint_index]
        else:
            rest_positions[joint_index] = offsets[joint_index]
    return rest_positions


def get_rest_body_max_span(offsets, parents, exclude_joints=()):
    """``max_joint_span`` of the rest pose accumulated from ``offsets``."""
    return max_joint_span(rest_positions_from_offsets(offsets, parents), exclude_joints)


def get_scale_reference_extent(rest_positions, parents, joint_names):
    """Character extent used for scale normalization.

    This is the rest-pose joint span widened to the root's own elevation above
    the rig origin. A handful of rigs (hovering/drifting creatures, effect rigs)
    seat their root far above the origin while their bones stay tiny; that
    elevation is part of the character's size but never shows up in a
    joint-to-joint span, so a purely bone-driven scale blows the normalized root
    height up into an outlier -- MU04_Pollen ends up with its root at 14.7 joint
    spans (8.6 units against a dataset median of 0.45). Folding the elevation in
    re-anchors those characters and is a no-op for every rig whose root already
    sits inside its own joint span, which is all 104 truebones/zoo species.

    Only the vertical component counts: the root offset's XZ is authoring
    placement rather than size, and is exactly 0.0 for all 260 dataset species.

    Prop-socket joints (see ``find_prop_socket_joints``) are left out of the
    span for the same reason: a parked weapon sets the span from where the rig
    authored it, not from the body. ``rest_positions`` must be the FK'd rest
    pose, not hand-accumulated offsets: see ``rest_positions_from_offsets``.
    """
    rest_positions = np.asarray(rest_positions, dtype=np.float64)
    parents = np.asarray(parents)
    # Parent-to-child deltas. Their lengths are the bone lengths the socket
    # filter needs, and are identical to the bone-local offsets' lengths.
    bone_deltas = rest_positions.copy()
    has_parent = parents >= 0
    bone_deltas[has_parent] -= rest_positions[parents[has_parent]]
    prop_joints = find_prop_socket_joints(bone_deltas, parents, joint_names)
    joint_span = max_joint_span(rest_positions, exclude_joints=prop_joints)
    root_elevation = abs(float(rest_positions[0][1]))
    return max(joint_span, root_elevation)


def compute_scale_factor(axial_avg_len, body_max_span=None, *, span_blend_weight=SCALE_BODY_SPAN_BLEND_WEIGHT):
    """Blend axial and whole-body span scaling to reduce size outliers symmetrically.

    Axial mean bone length remains the primary normalization signal because it
    tracks body thickness better than raw max span. A secondary max-span term is
    blended in log-space so compact skeletons scale up and wide/long skeletons
    scale down without letting tails or wings dominate as aggressively as pure
    max-span scaling.
    """
    if axial_avg_len <= 0:
        raise ValueError(f"Expected positive axial_avg_len, got {axial_avg_len}.")
    if not 0.0 <= span_blend_weight <= 1.0:
        raise ValueError(f"Expected span_blend_weight in [0, 1], got {span_blend_weight}.")

    axial_scale_factor = HML_REF_AXIAL_BONE_LENGTH / axial_avg_len
    if body_max_span is None or span_blend_weight == 0.0:
        return float(axial_scale_factor)
    if body_max_span <= 0:
        raise ValueError(f"Expected positive body_max_span, got {body_max_span}.")

    span_scale_factor = HML_REF_MAX_SPAN / body_max_span
    return float(
        (axial_scale_factor ** (1.0 - span_blend_weight))
        * (span_scale_factor ** span_blend_weight)
    )


def scale_anim(anim, scale_factor):
    if scale_factor is None:
        raise ValueError("scale_factor must be precomputed once per character and passed explicitly.")
    new_anim = Animation(
        anim.rotations.copy(),
        anim.positions * scale_factor,
        anim.orients.copy(),
        anim.offsets * scale_factor,
        anim.parents.copy(),
    )
    return new_anim


################## Leaf Rotation Helpers #####################

def _reference_clip_needs_local_position_rebuild(anim, tol=1e-4):
    """Return the max absolute error between first-frame local positions and rest offsets.

    Returns 0.0 when the clip is already aligned (error <= tol) or when there are
    insufficient joints to compare. Callers can truthily check the return value
    to decide whether repair is needed.
    """
    if len(anim) == 0 or anim.positions.shape[1] <= 1:
        return 0.0

    root_candidates = np.where(np.asarray(anim.parents) < 0)[0]
    if root_candidates.size == 0:
        return 0.0

    nonroot_indices = np.delete(np.arange(anim.positions.shape[1]), int(root_candidates[0]))
    if nonroot_indices.size == 0:
        return 0.0

    local_positions = np.asarray(anim.positions[0, nonroot_indices], dtype=np.float64)
    rest_offsets = np.asarray(anim.offsets[nonroot_indices], dtype=np.float64)
    error = float(np.max(np.abs(local_positions - rest_offsets)))
    return error if error > tol else 0.0


################## FK Helpers #####################

def coerce_single_orientation_quat(orientation_quat):
    if orientation_quat is None:
        raise ValueError(
            "orientation_quat must be precomputed from the reference rest pose and provided to downstream motion processing"
        )

    orientation_qs = getattr(orientation_quat, 'qs', orientation_quat)
    orientation_qs = np.asarray(orientation_qs, dtype=np.float64)
    if orientation_qs.ndim > 1:
        orientation_qs = orientation_qs[0]
    return Quaternions(orientation_qs.reshape(1, 4)).normalized()


def compute_rots_from_tpos(tpos_quats, dest_quats, parents):
    new_rots = dest_quats.copy()
    new_rots[:, 0] = new_rots[:, 0] * -tpos_quats[:, 0]
    cum_rots = tpos_quats.copy()
    for j, p in enumerate(parents[1:], start=1):
        cum_rots[:, j] = cum_rots[:, p] * tpos_quats[:, j]
        new_rots[:, j] = cum_rots[:, p] * dest_quats[:, j] * -tpos_quats[:, j] * -cum_rots[:, p]
    return new_rots


def solve_local_positions_for_target_global(
    rotations,
    target_global_positions,
    offsets,
    parents,
    orients,
    initial_positions=None,
    position_match_threshold=1e-5,
    max_passes=2,
):
    frames_num = target_global_positions.shape[0]
    if initial_positions is None:
        local_positions = offsets.copy()[None, :].repeat(frames_num, axis=0)
    else:
        local_positions = initial_positions.copy()

    # Local translations can be recovered directly from the target parent/child
    # global positions because global joint rotations depend only on `rotations`
    # and the hierarchy, not on the local translations we are solving for.
    global_rots = rotations_global(Animation(rotations, local_positions, orients, offsets, parents))
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx < 0:
            local_positions[:, joint_idx] = target_global_positions[:, joint_idx]
            continue

        local_positions[:, joint_idx] = (
            -global_rots[:, parent_idx]
        ) * (target_global_positions[:, joint_idx] - target_global_positions[:, parent_idx])

    direct_anim = Animation(rotations, local_positions, orients, offsets, parents)
    direct_global_pos = positions_global(direct_anim)
    direct_error = np.max(np.abs(target_global_positions - direct_global_pos))
    if direct_error <= position_match_threshold:
        return local_positions

    for _ in range(max_passes):
        temp_anim = Animation(rotations, local_positions, orients, offsets, parents)
        temp_global_pos = positions_global(temp_anim)
        per_joint_err = np.max(np.abs(target_global_positions - temp_global_pos), axis=(0, 2))
        joints_to_fix = [
            joint_idx
            for joint_idx in range(len(parents))
            if per_joint_err[joint_idx] > position_match_threshold
        ]
        if not joints_to_fix:
            break

        for joint_idx in joints_to_fix:
            if parents[joint_idx] < 0:
                local_positions[:, joint_idx] = target_global_positions[:, joint_idx]
                continue

            parent_idx = parents[joint_idx]
            local_positions[:, joint_idx] = (
                -global_rots[:, parent_idx]
            ) * (target_global_positions[:, joint_idx] - temp_global_pos[:, parent_idx])

    return local_positions
