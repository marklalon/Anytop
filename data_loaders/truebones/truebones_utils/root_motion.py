"""Root motion: loop detection, translation root and root trajectory policy.

Finds the joint that carries a clip's translation, detects loops, clamps the
vertical band, and detrends / soft-clamps the root XZ and Y trajectories.
Re-exported by ``animation_utils``.
"""

from motion_lib import Animation, Quaternions
from motion_lib.Animation import positions_global, rotations_global
import numpy as np
from data_loaders.truebones.truebones_utils.param_utils import (
    ROOT_Y_MIN_HEIGHT,
    ROOT_Y_SOFT_CLAMP_KNEE,
    VERTICAL_CLAMP_MIN_RATIO,
    VERTICAL_CLAMP_MAX_RATIO,
)
from data_loaders.truebones.truebones_utils.dataset_tags import (
    dataset_tags,
)


################## Constants #####################


# Sustained one-directional translation-root XZ displacement (in HML-normalised
# units, where a body span is 1.389) above which a clip is re-seated in place.
#
# Measured on the TRANSPORT the clip carries in its own heading frame, never on
# its raw extent. An out-and-back excursion has no net transport however far it
# reaches, so a lunge, a dodge or a swing keeps its root motion verbatim; a gait
# that travels is nothing but transport and gets flattened however small each
# step is. Clips are centred on the effective translation root's initial XZ
# position before this is evaluated.
ROOT_XZ_DRIFT_THRESHOLD = 0.08

# Net VERTICAL displacement of the translation root (same HML-normalised units)
# above which the detrend extends from the XZ plane to all three axes.
#
# Y is not like XZ: the XZ origin is arbitrary (clips are centred on it), but the
# Y origin is the FLOOR, so a clip's absolute height is meaningful and removing a
# ramp moves the character relative to the ground. The threshold is therefore
# several times looser than the XZ one, selecting only deliberate climbs. It sits
# between one rig's vertical gaits so they answer alike: MB_TigerDrago's FlyDown
# nets well above it while FlyUp barely clears it, and a threshold between them
# would flatten the dive but keep the climb on the same skeleton.
#
# Measured on the net endpoint offset, like the XZ threshold: a hop or jump that
# comes back down nets out and keeps its arc, while a clip that ends elsewhere
# went there and stayed.
ROOT_Y_DRIFT_THRESHOLD = 0.25

# Root XZ soft clamp, in HML-normalised units (a body span is 1.389).
#
# Travel is bounded, not gated: inside the knee nothing is touched at all, past
# it the excess is compressed smoothly, and the ceiling is an asymptote the path
# approaches but never reaches. This is what keeps a lunge, a dodge or a death
# slide recognisable at a magnitude the representation can carry -- the old
# extent gate answered the same question by zeroing the whole trajectory.
ROOT_XZ_SOFT_CLAMP_KNEE = 0.3
ROOT_XZ_SOFT_CLAMP_LIMIT = 0.5

# Locomotion's own extent bound, tighter than the soft clamp above and applied to
# EVERY clip the root-XZ policy selects -- all locomotion plus transition clips
# whose is_loop is true -- not only the ones that travelled. That is what makes it
# an invariant of the selected set rather than of whether a clip happened to move.
# The knee sits inside a normal gait's range (post-detrend extent runs p50 0.013,
# p90 0.110, p95 0.159), so unlike the 0.6 ceiling it is touched routinely and must
# leave the cycle's shape alone: it scales the whole clip by ONE factor, not each
# frame's radius (see ``scale_root_xz_extent``).
ROOT_XZ_LOCOMOTION_KNEE = 0.1
ROOT_XZ_LOCOMOTION_LIMIT = 0.2


# Loop detection judges the wrap-around gap (last frame -> first frame) against
# the clip's own frame-to-frame motion distribution, read over the frames near
# the two boundaries. A high percentile gives a compact robust envelope without
# tying tolerance to skeleton size.
#
# A clip loops only when the endpoint gap fits inside
# ``clamp(GAP_RATIO * envelope, STEP_MIN, STEP_MAX)`` and the translation root's
# accumulated XZ displacement returns to the start.
#
# The constants are tightened for PRECISION, because the two errors are not
# symmetric: a clip wrongly called a loop is tiled into training data that hitches
# every cycle, while one wrongly refused merely loses that augmentation.
LOOP_DETECTION_GAP_RATIO = 2
LOOP_DETECTION_STEP_MIN = 0.03
LOOP_DETECTION_STEP_MAX = 0.08

# The translation root's accumulated XZ displacement must also return to the
# start, or the clip transports and the second cycle begins somewhere else.
LOOP_DETECTION_ROOT_XZ_TOLERANCE = 0.05


################## Rest Pose Span #####################

def max_joint_span(positions, exclude_joints=()):
    """Largest joint-to-joint distance among ``positions``.

    ``exclude_joints`` drops joints from the measurement only -- the positions
    are already resolved, so excluding one never moves another. The unfiltered
    span is returned when the exclusions would leave fewer than two joints.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if exclude_joints:
        kept = [j for j in range(len(positions)) if j not in exclude_joints]
        if len(kept) >= 2:
            positions = positions[kept]
    joint_deltas = positions[:, None, :] - positions[None, :, :]
    max_span = np.linalg.norm(joint_deltas, axis=-1).max()
    return max(float(max_span), 1e-8)



################## Animation Transform Utilities #####################

def compute_motion_loop_diagnostics(positions, root_xz_velocity=None,
                                    translation_root_index=0):
    """Return loop diagnostics on the exact runtime boundary used by detect_motion_loop.

    The endpoint gap is compared against a robust upper envelope of the clip's
    own per-frame motion, taken over the frames near the two boundaries. When
    ``root_xz_velocity`` is provided, the translation root's accumulated XZ
    displacement must also close for the clip to count as a loop.

    ``loop_margin`` is the gap as a fraction of its tolerance, so <= 1 passes and
    the value says by how much; ``term_margins`` carries the same number under the
    name of the term that produced it, so the runtime and the report share one
    vocabulary even though this rule has a single term.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] < 3:
        return {
            'wrap_gap': 0.0, 'transition_envelope': 0.0,
            'effective_tolerance': 0.0, 'loop_margin': float('inf'),
            'term_margins': {}, 'root_xz_total_disp': 0.0,
            'root_xz_is_closed': True, 'is_closed': False, 'is_loop': False,
        }

    # wrap_gap: p80 of per-joint endpoint distance -- robust against a single
    # outlier joint while still capturing the bulk of the discontinuity.
    wrap_gap = float(np.percentile(np.linalg.norm(positions[-1] - positions[0], axis=-1), 80))

    # Use only frames near the clip boundaries to estimate the "normal transition"
    # envelope. The wrap_gap measures the jump from last frame -> first frame, so
    # it should be compared against the typical motion amplitude at the clip edges
    # rather than the peak motion in the middle (e.g., a fast swing or stride).
    # A fixed 5 frames per end keeps the window short and predictable across clip
    # lengths; halved so the two windows never overlap on very short clips.
    frame_steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)  # (T-1, J)
    boundary_count = min(5, frame_steps.shape[0] // 2)  # a fixed 5 frames at each end
    boundary_steps = np.concatenate([
        frame_steps[:boundary_count],
        frame_steps[-boundary_count:],
    ], axis=0)
    transition_envelope = float(np.percentile(boundary_steps, 65.0))  # p65 boundary envelope
    effective_tolerance = min(
        max(
            LOOP_DETECTION_GAP_RATIO * transition_envelope,
            LOOP_DETECTION_STEP_MIN,
        ),
        LOOP_DETECTION_STEP_MAX,
    )
    loop_margin = wrap_gap / effective_tolerance if effective_tolerance > 0.0 else float('inf')
    term_margins = {'position_wrap': loop_margin}

    is_closed = bool(wrap_gap <= effective_tolerance)

    root_xz_total_disp = 0.0
    root_xz_is_closed = True

    if root_xz_velocity is not None:
        velocity = np.asarray(root_xz_velocity, dtype=np.float64)
        if velocity.ndim != 3 or velocity.shape[2] < 3:
            raise ValueError(
                f"root_xz_velocity must have shape (T, J, 3), got {velocity.shape}"
            )

        translation_root_index = int(translation_root_index)
        if not 0 <= translation_root_index < velocity.shape[1]:
            raise ValueError(
                f"translation_root_index {translation_root_index} is out of bounds for {velocity.shape[1]} joints"
            )

        root_velocity = velocity[:, translation_root_index, :]
        if velocity.shape[0] == positions.shape[0]:
            # The stored terminal row is the wrap delta a PREVIOUS loop verdict
            # wrote, so reading it here would let that verdict decide this one.
            root_velocity = root_velocity[:-1]
        elif velocity.shape[0] != positions.shape[0] - 1:
            raise ValueError(
                f"root_xz_velocity frame count must be T or T-1 relative to positions, "
                f"got {velocity.shape[0]} vs positions T={positions.shape[0]}"
            )

        root_xz_total_disp = float(np.linalg.norm(np.sum(root_velocity[:, [0, 2]], axis=0)))
        root_xz_is_closed = bool(root_xz_total_disp <= LOOP_DETECTION_ROOT_XZ_TOLERANCE)

    return {
        'wrap_gap': float(wrap_gap),
        'transition_envelope': float(transition_envelope),
        'effective_tolerance': float(effective_tolerance),
        'loop_margin': float(loop_margin),
        'term_margins': {name: float(value) for name, value in term_margins.items()},
        'root_xz_total_disp': float(root_xz_total_disp),
        'root_xz_is_closed': bool(root_xz_is_closed),
        'is_closed': bool(is_closed),
        'is_loop': bool(is_closed and root_xz_is_closed),
    }


def detect_motion_loop(positions, root_xz_velocity=None, translation_root_index=0):
    return compute_motion_loop_diagnostics(
        positions,
        root_xz_velocity=root_xz_velocity,
        translation_root_index=translation_root_index,
    )['is_loop']


def detect_loop_from_features(features, translation_root_index=0):
    """The detector's verdict on a STORED (T, J, 12) clip.

    The same rule extraction applies to a fresh clip, read off the tensor it
    wrote: RIC positions are channels 0:3 and local velocity 9:12, and the
    diagnostics drop the terminal velocity row themselves so an earlier
    verdict's wrap delta cannot vote.
    """
    features = np.asarray(features)
    return detect_motion_loop(
        features[..., 0:3],
        root_xz_velocity=features[..., 9:12],
        translation_root_index=translation_root_index,
    )


def _translation_root_candidate_chain(parents, max_depth=5):
    parents = np.asarray(parents, dtype=np.int32).reshape(-1)
    if parents.size == 0:
        return [0]

    root_candidates = np.flatnonzero(parents < 0)
    if root_candidates.size == 0:
        return [0]

    current = int(root_candidates[0])
    chain = [current]
    for _depth in range(max(int(max_depth), 0)):
        children = np.flatnonzero(parents == current)
        if children.size != 1:
            break
        current = int(children[0])
        chain.append(current)

    return chain


def find_translation_root(anim, max_depth=5):
    """Return the first near-root joint with significant local position animation.

    Most skeletons carry root motion on joint 0, but some Truebones rigs
    (Horse, Bear, Camel, Trex, etc.) use intermediate bones like Bip01 or
    jt_Cog_C. Detecting this allows the rest of the pipeline to centre,
    strip trajectory, and export BVH correctly.

    Detection is intentionally limited to the unbranched chain directly below
    the hierarchy root. Once the chain branches, deeper descendants are treated
    as limb / local motion rather than transport candidates. Search also stops
    after ``max_depth`` descendants to avoid latching onto deep noisy joints.

    Returns the hierarchy root (normally joint 0) when no candidate carries
    significant local position animation. This answers "where is THIS clip's
    root motion", which is not the same question as "which joint is this
    species' transport control" -- see :func:`select_transport_carrier`.
    """
    frame_count = int(anim.positions.shape[0]) if anim.positions.ndim >= 3 else 0
    min_active_frames = max(3, int(np.ceil(frame_count * 0.2))) if frame_count > 0 else 0
    candidate_chain = _translation_root_candidate_chain(anim.parents, max_depth=max_depth)
    first_candidate = int(candidate_chain[0]) if candidate_chain else 0
    fallback_joint = first_candidate

    for j in candidate_chain:
        joint_positions = np.asarray(anim.positions[:, j], dtype=np.float64)
        ptp = np.ptp(joint_positions, axis=0)
        if not np.any(ptp > 5e-3):
            continue

        delta_from_first = np.linalg.norm(joint_positions - joint_positions[0:1], axis=-1)
        active_frames = int(np.count_nonzero(delta_from_first > 5e-3))

        if active_frames >= min_active_frames:
            return j

        # Only consider as fallback if there are genuinely active frames,
        # not just a single-frame PTP spike from numerical noise.
        if fallback_joint == first_candidate and j != first_candidate and active_frames >= 2:
            fallback_joint = j

    return int(fallback_joint)


# A chain joint counts as carrying a clip's transport when its own horizontal
# travel is at least this share of the largest travel anywhere on that chain.
# Rigs stack several near-root control joints and animators do not always use the
# same one, so the test has to be "is this where the motion is", not "does this
# move at all": Crow's Spine drifts 0.026 while its Pelvis flies 0.809, and a
# rule that counted any motion would read the whole species' trajectory off a
# body joint.
ROOT_TRANSPORT_CARRIER_SHARE = 0.5

# ...and at least this much travel outright, in HML-normalised units. The same
# magnitude as the flatten threshold, for the same reason: below it nothing
# downstream reacts to the travel at all, so it cannot be worth re-rooting a
# species over. Idle sway and vertical bob on a spine joint live down here.
ROOT_TRANSPORT_MIN_TRAVEL = 0.08


def chain_xz_travel(anim, max_depth=5):
    """Return ``{joint: horizontal travel}`` for the translation-root candidates.

    Travel is measured the way :func:`find_translation_root` measures activity --
    local position, distance from the first frame -- but in XZ only, because the
    root trajectory, the gait flattener and the soft clamp are all horizontal.
    A joint that only bobs vertically carries no transport.
    """
    chain = _translation_root_candidate_chain(anim.parents, max_depth=max_depth)
    positions = np.asarray(anim.positions, dtype=np.float64)
    travel = {}
    for joint in chain:
        xz = positions[:, joint][:, [0, 2]]
        travel[int(joint)] = (
            float(np.linalg.norm(xz - xz[0:1], axis=1).max()) if xz.shape[0] else 0.0
        )
    return travel


def select_transport_carrier(chain_travel,
                             share=ROOT_TRANSPORT_CARRIER_SHARE,
                             min_travel=ROOT_TRANSPORT_MIN_TRAVEL):
    """Return the deepest chain joint that carries the horizontal travel, or None.

    ``None`` means this clip never goes anywhere and has no opinion about where
    its species keeps transport.

    DEEPEST, not most active: a joint's global position already contains every
    ancestor's translation, so a root at or below the carrier sees all of them,
    while a root above it sees none of that clip's travel -- it lands in a
    descendant's RIC channel instead, where the flattener, the clamp and both
    validators are blind to it. That is why the choice is forced rather than a
    vote, and why the two filters above it have to do the work of deciding what
    counts as travel in the first place.
    """
    if not chain_travel:
        return None
    peak = max(chain_travel.values())
    if peak < min_travel:
        return None
    threshold = max(share * peak, min_travel)
    carriers = [joint for joint, travel in chain_travel.items() if travel >= threshold]
    return max(carriers) if carriers else None


def rest_pose_animation(anim):
    """The one-frame rest pose of ``anim``: its orients seated on its offsets.

    ``positions_global`` reads only the per-frame ``rotations``/``positions``, so
    FK'ing a multi-frame clip never yields its rest pose -- reduce it here first.
    """
    rest_rotations = np.asarray(anim.orients.qs, dtype=np.float64)
    offsets = np.asarray(anim.offsets, dtype=np.float64)
    if rest_rotations.shape[0] != offsets.shape[0]:
        raise ValueError(
            f"Animation has {offsets.shape[0]} offsets but "
            f"{rest_rotations.shape[0]} rest rotations"
        )
    return Animation(
        Quaternions(rest_rotations[None].copy()),
        offsets[None].copy(),
        anim.orients.copy(),
        offsets.copy(),
        np.asarray(anim.parents, dtype=np.int32).copy(),
    )


def _get_reference_body_length(anim):
    """Character size from the rest-pose joint span.

    ``anim`` is a multi-frame clip; reduce it to its rest pose first so the band
    this feeds is a species property, not tied to the pose the clip opens with.
    """
    return max_joint_span(positions_global(rest_pose_animation(anim))[0])


def _excursion_scale(extent, min_h, max_h):
    """The factor an excursion reaching ``extent`` is scaled by.

    ``min_h`` is the knee and ``max_h`` the asymptote of the shared hyperbola, so
    the peak lands on ``soft_clamp_extent`` and everything below the knee is left
    where it is. ``max_h`` used to be the target rather than the asymptote, which
    made it the reported height of EVERY clip that reached it -- 304 of the 582
    winged clips, so a bird's hop and a dragon's climb came out at the same
    number. Ordering across clips is what that cost, and what this returns.

    The excess drops out of the ratio as ``w / (e + w)``, so a clip that barely
    clears the knee is scaled by ~1 and the map is continuous there. Cancellation
    in the numerator only bites once ``e`` is down around 1e-13, and the shift it
    could cause is itself bounded by ``e`` -- there is nothing left to protect.
    """
    excess = extent - min_h
    return (float(soft_clamp_extent(extent, min_h, max_h)) - min_h) / excess


def _compress_positive_excursion(values, min_h, max_h):
    """Scale the part of ``values`` above ``min_h`` so its peak approaches ``max_h``.

    ONE factor for the whole clip, taken from the clip's own peak, exactly as
    ``scale_root_xz_extent`` does and for the same reason: past the knee the
    excursion IS the flight, so a per-frame map would bend the top of every arc
    harder than its base and change the shape of the climb rather than its size.
    The knee is reached routinely here -- 71.3% of winged clips peak above it.
    """
    peak = float(values.max())
    if peak <= min_h or max_h <= min_h:
        return values, False

    scale = _excursion_scale(peak, min_h, max_h)
    compressed = values.copy()
    mask = compressed > min_h
    compressed[mask] = min_h + (compressed[mask] - min_h) * scale
    return compressed, True


def _compress_negative_excursion(values, min_h, max_h):
    """The mirror of :func:`_compress_positive_excursion` on downward depth."""
    depth = -float(values.min())
    if depth <= min_h or max_h <= min_h:
        return values, False

    scale = _excursion_scale(depth, min_h, max_h)
    compressed = values.copy()
    mask = compressed < -min_h
    compressed[mask] = -min_h - ((-compressed[mask]) - min_h) * scale
    return compressed, True


def _compress_below_negative_band(values, min_h, max_h):
    """Compress values below negative thresholds derived from positive ratios."""
    return _compress_negative_excursion(values, abs(min_h), abs(max_h))


def _soft_clamp_min_height(values, knee, min_height):
    """Bound the root's downward excursion by ``min_height`` without a hard floor.

    The root-XZ hyperbola (:func:`soft_clamp_extent`) mirrored onto depth: heights
    above ``knee`` come back bit-for-bit, the descent past it is compressed with a
    continuous value and slope, order is preserved, and ``min_height`` is an
    asymptote rather than a value.

    A hard ``np.maximum`` floor has none of the last three. It mapped every frame
    past -0.5 onto exactly -0.5, so what a dive, a fall or a burrow left in the
    data was a pinned plateau, not a descent -- 86 shipped clips carried one, 19 of
    them for 8+ consecutive frames, and Pirrana_MidSwim was a dead constant for all
    97 of its frames. It also fought the aquatic band directly: that band puts
    swim depth in [-minH, -maxH], which at the HML reference span is
    [-0.417, -0.694], and the floor then sheared the deeper half of it onto one
    height.
    """
    knee_depth = abs(float(knee))
    if float(values.min()) >= -knee_depth:
        return values, False
    # soft_clamp_extent is identity at and below its knee, negative radii included,
    # so every frame shallower than the knee (and every positive height) survives
    # the round trip through the negation exactly.
    return -soft_clamp_extent(-values, knee_depth, abs(float(min_height))), True


# The object_subsets whose vertical band needs the character's size. Everything
# else takes the root-Y lower bound alone, which is an absolute height.
_VERTICAL_BAND_SUBSETS = ('winged', 'aquatic')


def vertical_band_inputs(anim, object_type):
    """Return ``(object_subset, body_length)`` for the vertical clamp.

    ``body_length`` is ``None`` for the subsets whose band does not need it, so
    the rest-pose FK behind :func:`_get_reference_body_length` is only paid by the
    species that use it -- the same laziness the single-function version had.

    Split out so the two callers agree by construction: preprocessing reads these
    off the pre-detrend anim and clamps the trajectory afterwards, while a rest
    pose clamps the anim in place (:func:`clamp_vertical_trajectory`).
    """
    # object_subset_for accepts a bare species name or a canonical
    # '<namespace>/<species>' key, unlike a raw subset_members membership test.
    object_subset = dataset_tags().object_subset_for(object_type)
    body_length = (
        _get_reference_body_length(anim)
        if object_subset in _VERTICAL_BAND_SUBSETS else None
    )
    return object_subset, body_length


def clamp_vertical_height_track(
    world_y,
    object_subset,
    body_length,
    min_ratio=VERTICAL_CLAMP_MIN_RATIO,
    max_ratio=VERTICAL_CLAMP_MAX_RATIO,
    root_y_min_height=ROOT_Y_MIN_HEIGHT,
    root_y_soft_clamp_knee=ROOT_Y_SOFT_CLAMP_KNEE,
):
    """Return ``(clamped_height, changed)`` for a ``(T,)`` root height track.

    The whole vertical policy as a function of the track alone, so it can be
    applied at the point in the pipeline where the track is final. That point is
    AFTER the vertical detrend, and the order matters more than it looks: the
    bands are a per-value map with a kink at ``min_ratio``, so running them on a
    trajectory that still carries transport leaves behind an artifact of where
    the climb crossed that kink rather than the motion. Measured on the shipped
    flight clips the residual came out anywhere between 0.42x and 2.43x of the
    true one, in both directions. Once the transport is gone the band is idle for
    nearly every one of them, and the root-Y lower bound -- which runs last --
    becomes a bound on what actually ships instead of on an intermediate.

    ``object_subset`` and ``body_length`` come from :func:`vertical_band_inputs`.
    """
    world_y = np.asarray(world_y, dtype=np.float64).reshape(-1)
    clamped_world_y = world_y.copy()
    changed = False
    if object_subset == 'winged':
        min_h = body_length * min_ratio
        max_h = body_length * max_ratio
        clamped_world_y, changed = _compress_positive_excursion(clamped_world_y, min_h, max_h)
    elif object_subset == 'aquatic':
        positive_min_h = body_length * min_ratio
        positive_max_h = body_length * max_ratio
        clamped_world_y, changed_pos = _compress_positive_excursion(
            clamped_world_y,
            positive_min_h,
            positive_max_h,
        )
        negative_min_h = -positive_min_h
        negative_max_h = -positive_max_h
        clamped_world_y, changed_neg = _compress_below_negative_band(
            clamped_world_y,
            negative_min_h,
            negative_max_h,
        )
        changed = changed_pos or changed_neg
    else:
        clamped_world_y = world_y.copy()

    clamped_world_y, changed_floor = _soft_clamp_min_height(
        clamped_world_y,
        root_y_soft_clamp_knee,
        root_y_min_height,
    )
    return clamped_world_y, bool(changed or changed_floor)


def clamp_vertical_trajectory(
    processed_anim,
    object_type,
    min_ratio=VERTICAL_CLAMP_MIN_RATIO,
    max_ratio=VERTICAL_CLAMP_MAX_RATIO,
    root_y_min_height=ROOT_Y_MIN_HEIGHT,
    root_y_soft_clamp_knee=ROOT_Y_SOFT_CLAMP_KNEE,
    translation_root_index=None,
):
    """Constrain the processed translation-root vertical trajectory.

    What this compresses is the root's ABSOLUTE height, not its excursion: once
    the peak passes ``min_ratio`` of the body span, everything above ``min_ratio``
    is scaled by one factor that sends the peak toward ``max_ratio`` without ever
    reaching it. The band is calibrated on flight, where an unbounded climb is the
    thing worth bounding.

    ``drifting`` deliberately does NOT take this branch, though it too leaves the
    ground. A drifting species holds a near-constant hover height that is a trait
    of the species, and that height routinely sits above ``max_ratio`` (a hover
    robot's root rides at 2-3 body spans while a walking biped's sits at 0.1-0.4),
    so the flight band would crush the hover back down to walking height -- and by
    a factor that depends on each clip's own peak, which would make one species'
    hover height differ from clip to clip. Its vertical range in level travel is
    already as small as level flight's, so there is nothing here worth clamping;
    it keeps only the root-Y lower bound, the same as a ground species.

    Aquatic species use the same positive height clamp and also apply the same
    ratios with a negative sign so their downward swim depth is compressed into
    [-maxH, -minH]. Every species then gets the root-Y lower bound, which is a
    soft clamp, not a floor: it leaves everything above its knee alone and
    compresses the descent below it into ``(root_y_min_height, knee]`` (see
    :func:`_soft_clamp_min_height`). It runs last, so an aquatic dive whose band
    reaches past the bound comes out as a monotone descent -- compressed further
    than the band left it, but never the flat plateau a hard floor cut.

    ``translation_root_index`` is the species' frozen root. Without it this falls
    back to per-clip detection, which is right for a bare rest pose or a raw
    retarget source but wrong inside preprocessing: a species whose clips author
    transport on different joints (see :func:`select_transport_carrier`) would
    get its height read off one joint and its trajectory off another, and the
    two differ by the bone between them plus whatever that bone animates.
    """
    if translation_root_index is None:
        trans_root = find_translation_root(processed_anim)
    else:
        trans_root = int(translation_root_index)
    global_pos = positions_global(processed_anim)
    world_y = global_pos[:, trans_root, 1]

    object_subset, body_length = vertical_band_inputs(processed_anim, object_type)
    clamped_world_y, changed = clamp_vertical_height_track(
        world_y,
        object_subset,
        body_length,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        root_y_min_height=root_y_min_height,
        root_y_soft_clamp_knee=root_y_soft_clamp_knee,
    )

    if not changed:
        return processed_anim

    new_positions = processed_anim.positions.copy()
    if trans_root == 0 or processed_anim.parents[trans_root] < 0:
        new_positions[:, trans_root, 1] += clamped_world_y - world_y
    else:
        global_rots = rotations_global(processed_anim)
        parent_index = processed_anim.parents[trans_root]
        parent_global_pos = global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        desired_global = global_pos[:, trans_root].copy()
        desired_global[:, 1] = clamped_world_y
        new_positions[:, trans_root] = (-parent_global_rots) * (desired_global - parent_global_pos)

    return Animation(
        processed_anim.rotations.copy(),
        new_positions,
        processed_anim.orients.copy(),
        processed_anim.offsets.copy(),
        processed_anim.parents.copy(),
    )


def _coerce_root_xz_center(root_xz_center):
    root_xz_center = np.asarray(root_xz_center, dtype=np.float64).reshape(-1)
    if root_xz_center.size == 3:
        return root_xz_center
    if root_xz_center.size == 2:
        return np.array([root_xz_center[0], 0.0, root_xz_center[1]], dtype=np.float64)
    raise ValueError(f"root_xz_center must have shape (2,) or (3,), got {root_xz_center.shape}")


def _get_translation_root_initial_xz(anim, translation_root_index=None):
    """Return the effective translation root's initial XZ position in global space."""
    if translation_root_index is None:
        translation_root_index = find_translation_root(anim)

    global_pos = positions_global(anim)
    root_xz = np.asarray(global_pos[0, translation_root_index, [0, 2]], dtype=np.float64)
    return np.array([root_xz[0], 0.0, root_xz[1]], dtype=np.float64)


""" move motion s.t the effective translation root's initial XZ is centred at the origin.

For most skeletons joint 0 carries the root motion, but some rigs store it
on an intermediate bone (e.g. Bip01 for Horse).  We detect the effective
root via its global position and apply the shift to joint 0 (whose local
position equals its global position), so the entire skeleton moves via FK.
"""
def move_xz_to_origin(anim, root_xz_center=None, translation_root_index=None):
    if root_xz_center is None:
        root_xz_center = _get_translation_root_initial_xz(
            anim,
            translation_root_index=translation_root_index,
        )
    else:
        root_xz_center = _coerce_root_xz_center(root_xz_center)
    new_positions = anim.positions.copy()
    new_positions[:, 0] -= root_xz_center
    new_offsets = anim.offsets.copy()
    new_offsets[0] -= root_xz_center
    new_anim = Animation(anim.rotations.copy(), new_positions, anim.orients.copy(), new_offsets, anim.parents.copy())
    return new_anim, root_xz_center


def xz_locomotion_extent(anim, translation_root_index):
    """Return the maximum translation-root XZ distance from the current origin."""
    global_pos = positions_global(anim)
    root_xz = global_pos[:, translation_root_index, [0, 2]]
    return float(np.linalg.norm(root_xz, axis=1).max())


def root_xz_trajectory(anim, translation_root_index):
    """Return the effective translation root's ``(T, 2)`` world XZ path."""
    global_pos = positions_global(anim)
    return np.asarray(global_pos[:, translation_root_index][:, [0, 2]], dtype=np.float64)


def root_xz_heading(anim, translation_root_index):
    """Return the translation root's per-frame world heading, unwrapped, in radians.

    The angle of the root's own forward axis in the XZ plane. ``anim`` is
    canonicalised to face +Z by ``rotate_to_hml_orientation`` before it ever gets
    here, so the forward axis is +Z and no per-species orientation enters.

    Unwrapped, so a clip that turns past +/-pi reads as one continuous ramp
    rather than a jump -- the ramp is the whole point, it is what a straight-line
    fit can follow and a raw angle cannot.

    ``anim`` is expected to have been through ``collapse_translation_root_chain``
    already, which seats the inert wrapper nodes onto the root; the root's world
    rotation then carries the wrapper's rotation too, which is where some rigs
    keep the turn.
    """
    rotations = rotations_global(anim)[:, int(translation_root_index)]
    forward = rotations * np.array([0.0, 0.0, 1.0])
    return np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))


def _xz_rotation(angle):
    """Return ``(T, 2, 2)`` rotations by ``angle`` about +Y, acting on ``(x, z)``."""
    angle = np.asarray(angle, dtype=np.float64)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack([np.stack([cos, sin], -1), np.stack([-sin, cos], -1)], -2)


def _detrend_frame(heading):
    """Return the per-frame frame angle to detrend in.

    A straight ramp between the heading's endpoints, so the frame turns by the
    clip's NET turn and nothing else: a yaw that wanders out and comes back lands
    where it started and cannot bend it, however that wander was shaped, while a
    clip that really turns bends it in full. A least-squares line through the
    whole heading cannot say that -- it weights every frame, so an asymmetric
    wobble tilts it (MB_TigerDrago_RunJump travels dead straight under 2.68 rad of
    pelvis yaw for a net of 0.21, and the fitted line invented 0.42 of lateral
    travel).

    The heading is the right signal and the path is not: a 180-degree turning gait
    and an out-and-back lunge trace nearly the same path, and only the body's
    orientation says which is which. Fitting to the path's velocity direction
    invents 0.76 of transport on a lunge whose net displacement is zero.

    The frame reads the heading alone, never the path, which is what lets the
    dataset validator re-derive it from the stored features: a constant offset
    between the two headings shifts both endpoints alike and cancels. See
    docs/root_xz_motion_refactor.md §2.1 for the measured comparison against the
    constant-frame and least-squares alternatives.
    """
    heading = np.asarray(heading, dtype=np.float64).reshape(-1)
    n_frames = heading.shape[0]
    if n_frames < 2:
        return heading.copy()
    steps = np.arange(n_frames, dtype=np.float64) / (n_frames - 1)
    return heading[0] + (heading[-1] - heading[0]) * steps


def _frame_correction(traj, frame):
    """Return the accumulated transport of ``traj`` measured in ``frame``."""
    local_step = np.einsum('tij,tj->ti', _xz_rotation(-frame), np.diff(traj, axis=0))
    transport = np.einsum('tij,j->ti', _xz_rotation(frame), local_step.mean(axis=0))
    return np.concatenate([np.zeros((1, 2)), np.cumsum(transport, axis=0)], axis=0)


def root_xz_drift_correction(traj, heading):
    """Return ``(correction, drift)`` for a ``(T, 2)`` root XZ path.

    The travel is removed IN THE ROOT'S OWN HEADING FRAME, and that frame is the
    whole of it. In world space a turning gait's velocity direction rotates with
    the character, so there is no straight trend to subtract and no low-order
    curve that fits one either: an end-to-end line fit leaves the arc's sagitta
    behind, which on the shipped data reached 0.371 for MB_Unka_GlideLeft and
    0.732 for IAC_Cavewoman_RunTurnRight -- 27% and 53% of a body span of
    residual travel, turning as it went.

    Rotated into the heading frame the same motion is a near-constant forward
    speed with the within-cycle surge and sway riding on it, and removing a
    constant is exactly what the old world-space line fit was already doing --
    just in the wrong frame. So the arithmetic is: rotate the frame-to-frame
    displacement by the negated frame angle, subtract its mean, rotate back,
    re-integrate from frame 0.

    The root's heading is only a proxy for the travel direction, so the frame is
    read from its NET turn alone (``_detrend_frame``): a pelvis that yaws through
    the stride and comes back cannot bend it.

    ``correction`` is the accumulated transport, so ``traj - correction`` keeps
    frame 0 where the pipeline centred it and keeps everything the transport
    could not explain -- the surge and sway that make a gait read as a gait
    rather than as a rig welded to the floor.

    ``drift`` is where that transport ENDS UP, not how far it ever reached. Only
    the endpoint separates "went and stayed" from "went and came back", and both
    of those spend most of the clip away from the origin, so no statistic over
    time can tell them apart. An out-and-back lunge reverses in the heading frame
    too and nets out to nothing, so it is kept whatever its amplitude.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] < 2:
        return np.zeros_like(traj), 0.0
    frame = _detrend_frame(np.asarray(heading, dtype=np.float64)[:-1])
    correction = _frame_correction(traj, frame)
    return correction, float(np.linalg.norm(correction[-1]))


def flatten_root_xz_drift(traj, heading):
    """Return ``(flattened_traj, drift)`` for a root XZ path.

    The single entry point the pipeline and the dataset validator share, so both
    answer "does this clip travel" with the same arithmetic. ``heading`` is the
    translation root's per-frame world yaw, from ``root_xz_heading``; the frame
    it is detrended in turns by that heading's net turn (see
    ``root_xz_drift_correction``).

    It removes the travel and nothing else. Bounding what is left is a separate
    step the caller owns -- ``scale_root_xz_extent`` for a locomotion clip, the
    dataset-wide ``soft_clamp_root_xz`` for every clip -- and it stays out of here
    because the validator calls this to MEASURE a stored clip.
    """
    correction, drift = root_xz_drift_correction(traj, heading)
    return np.asarray(traj, dtype=np.float64) - correction, drift


def root_y_drift_correction(traj_y):
    """Return ``(correction, drift)`` for a ``(T,)`` root height track.

    The vertical counterpart of :func:`root_xz_drift_correction`, and simpler
    than it for one reason: there is no frame to rotate into. The XZ detrend has
    to work in the root's heading frame because a turning gait's travel
    direction rotates with the body; "up" does not rotate, so the transport is
    already expressed on its own axis and removing it is removing a straight
    ramp.

    That ramp is the mean per-frame step re-integrated, which is exactly the line
    through the track's two ENDPOINTS -- the same quantity ``_detrend_frame``
    reads off the heading, and the same reason: only the endpoints separate a
    climb from a flap that goes up and comes back. A statistic over time cannot,
    because both spend the clip away from where they started.

    ``correction`` starts at zero, so ``traj_y - correction`` keeps frame 0 at
    its authored height -- the height the vertical clamp and the floor bound
    already agreed on -- and keeps every wingbeat, bob and surge that the
    transport could not explain. ``drift`` is the signed travel's magnitude.
    """
    traj_y = np.asarray(traj_y, dtype=np.float64).reshape(-1)
    n_frames = traj_y.shape[0]
    if n_frames < 2:
        return np.zeros_like(traj_y), 0.0
    net = float(traj_y[-1] - traj_y[0])
    steps = np.arange(n_frames, dtype=np.float64) / (n_frames - 1)
    return net * steps, abs(net)


def flatten_root_y_drift(traj_y):
    """Return ``(flattened_height, drift)`` for a root height track.

    The entry point preprocessing and the dataset validator share for the
    vertical channel, mirroring :func:`flatten_root_xz_drift`. It removes the
    travel and nothing else; bounding what is left is the caller's business (the
    vertical clamp has already run by the time preprocessing gets here).
    """
    correction, drift = root_y_drift_correction(traj_y)
    return np.asarray(traj_y, dtype=np.float64).reshape(-1) - correction, drift


def soft_clamp_extent(radius, knee=ROOT_XZ_SOFT_CLAMP_KNEE,
                      limit=ROOT_XZ_SOFT_CLAMP_LIMIT):
    """Return ``radius`` compressed into ``[0, limit)`` past ``knee``.

    ``g(r) = r`` up to the knee, then ``limit - w**2 / (r - knee + w)`` with
    ``w = limit - knee``. Four things are needed at once and this hyperbola is
    the simplest form that has all of them:

    * identity below the knee, so a clip that already stays near the origin is
      bit-for-bit untouched;
    * continuous VALUE AND SLOPE at the knee (``g'(knee) = 1``), so nothing in
      the dataset has a velocity step at 0.6 for the model to learn as a feature;
    * strictly increasing, so ordering survives -- a bigger lunge still reads as
      a bigger lunge, which a hard clamp destroys by mapping every excursion past
      the ceiling onto the same value;
    * ``limit`` as an asymptote rather than a value, so the bound is strict and
      the far tail decelerates smoothly instead of hitting a wall.

    An exponential knee has the same four properties on paper and loses the third
    one in float64: past about 34 knee-widths ``limit - w * exp(...)`` rounds to
    ``limit`` exactly, and the deepest excursions in the shipped data (up to 4.79)
    would land inside a 1e-3 band of each other. The hyperbola decays as ``1/r``
    and keeps them apart.

    With the shipped knee/limit: ``g(0.8) = 0.700``, ``g(1.2) = 0.750``,
    ``g(3.0) = 0.785``, ``g(10.0) = 0.796``.
    """
    radius = np.asarray(radius, dtype=np.float64)
    width = float(limit) - float(knee)
    if width <= 0.0:
        return np.minimum(radius, float(limit))
    excess = np.maximum(radius - float(knee), 0.0) + width
    return np.where(radius > knee, float(limit) - width * width / excess, radius)


def soft_clamp_root_xz(traj, knee=ROOT_XZ_SOFT_CLAMP_KNEE,
                       limit=ROOT_XZ_SOFT_CLAMP_LIMIT):
    """Return a ``(T, 2)`` root XZ path with its distance from the origin bounded.

    Applied per frame on the radius, so the map is a fixed function of position:
    the parts of the clip that stay within the knee are preserved exactly, and
    only the frames that reach past it are compressed. Scaling the whole
    trajectory by one factor instead would shrink a clip's near-origin footwork
    in proportion to an excursion elsewhere in the clip, and would let a single
    outlier frame resize everything around it.

    Direction is preserved frame by frame, so a closed path stays closed and
    frame 0 -- which the pipeline centred on the origin -- stays there.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] == 0:
        return traj.copy()
    radius = np.linalg.norm(traj, axis=1)
    scale = np.ones_like(radius)
    over = radius > float(knee)
    if np.any(over):
        scale[over] = soft_clamp_extent(radius[over], knee, limit) / radius[over]
    return traj * scale[:, None]


def scale_root_xz_extent(traj, knee=ROOT_XZ_LOCOMOTION_KNEE,
                         limit=ROOT_XZ_LOCOMOTION_LIMIT):
    """Return a ``(T, 2)`` root XZ path scaled by ONE factor into ``[0, limit)``.

    The bound every selected clip is held to -- all locomotion, plus transition
    clips marked loop -- applied after the detrend. It reuses the hyperbola of
    ``soft_clamp_extent`` -- identity below the knee, a strict asymptote at the
    limit, order-preserving in between -- but evaluates it once on the clip's own
    extent and scales the whole path by that ratio.

    The single factor is the deliberate difference from ``soft_clamp_root_xz``.
    What a detrend leaves behind IS the gait cycle, spread over the whole clip, so
    a per-frame radial map would compress the far half of every stride harder than
    the near half and change the SHAPE of the surge -- and this knee, unlike the
    0.6 ceiling, is reached routinely. One factor changes only its size.

    Frame 0 sits at the origin, so scaling keeps it there and keeps a closed path
    closed.
    """
    traj = np.asarray(traj, dtype=np.float64)
    if traj.shape[0] == 0:
        return traj.copy()
    extent = float(np.linalg.norm(traj, axis=1).max())
    if extent <= float(knee):
        return traj.copy()
    return traj * (float(soft_clamp_extent(extent, knee, limit)) / extent)


# A joint counts as carrying transport when its world XZ moves at all. In
# HML-normalised units (body span 1.389) this is 0.36% of a span -- authoring
# noise on a joint that is meant to sit still stays well under it.
TRANSPORT_CARRIER_EPS = 5e-3


def translation_root_ancestor_chain(parents, translation_root_index):
    """Return the joints from the hierarchy root down to the translation root."""
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    chain = []
    joint = int(translation_root_index)
    seen = set()
    while 0 <= joint < parents.shape[0] and joint not in seen:
        seen.add(joint)
        chain.append(joint)
        joint = int(parents[joint])
    chain.reverse()
    return chain


def collapse_translation_root_chain(anim, translation_root_index):
    """Seat the joints above the translation root rigidly onto it.

    The joints between the hierarchy root and the effective root are inert
    control nodes -- ``Cg``, ``Ctrl``, ``All``, a bare ``Root`` -- that exist to
    hold the character, not to move relative to it. Nothing hangs off them but
    the chain itself, and the transport was measured to live at or below the
    effective root, so their offset from it carries no motion of its own.

    Left alone it carries two artifacts instead, and both land in RIC channels
    the model has to account for:

    * an arbitrary per-clip CONSTANT. Centring subtracts the effective root's
      initial XZ from joint 0, so a wrapper ends up at minus wherever the
      animator happened to park the character: ``MB_TigerDrago_DogdeLeftG``
      seats its ``Cg`` 1.546 from the origin, ``Run`` 0.193, ``Idle`` 0.102,
      ``GetHitR`` 0.000 -- same rig, same species, four different answers.
    * the NEGATED root trajectory, whenever the wrapper stands still while the
      root walks away from it (``Horse_Attack`` 0.718, ``Pirrana_Jump2`` 0.783).

    This rewrites the chain so every joint on it sits on its rest offset and the
    hierarchy root carries the whole trajectory. The effective root and its
    entire subtree keep their world positions EXACTLY -- only the inert joints
    move -- so pose, foot contacts and the root trajectory are untouched, and
    what is left in the wrapper's RIC is the rest geometry rather than an
    accident of authoring. Idempotent: a second pass finds the chain already
    seated and reproduces the same positions.
    """
    chain = translation_root_ancestor_chain(anim.parents, translation_root_index)
    if len(chain) <= 1:
        return anim

    parents = np.asarray(anim.parents, dtype=np.int64).reshape(-1)
    # Only safe while nothing but the chain hangs off these joints. Every root
    # this pipeline picks comes off the unbranched candidate chain, so this is a
    # guard against a hand-set or stale index, not a case that occurs.
    for joint in chain[:-1]:
        if int(np.count_nonzero(parents == joint)) != 1:
            return anim

    target = np.asarray(
        positions_global(anim)[:, translation_root_index], dtype=np.float64
    )

    seated_positions = anim.positions.copy()
    for joint in chain[1:]:
        seated_positions[:, joint] = anim.offsets[joint]
    seated_positions[:, chain[0]] = 0.0

    # Joint 0's local position IS its world position and adds straight through FK
    # to every descendant, so one probe gives the constant to solve against.
    probe = Animation(
        anim.rotations, seated_positions, anim.orients, anim.offsets, anim.parents
    )
    base = np.asarray(
        positions_global(probe)[:, translation_root_index], dtype=np.float64
    )
    seated_positions[:, chain[0]] = target - base

    return Animation(
        anim.rotations.copy(),
        seated_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )


def _transport_carrier_index(anim, translation_root_index, global_pos=None,
                             eps=TRANSPORT_CARRIER_EPS, axes=(0, 2)):
    """Return the joint a root correction on ``axes`` has to be applied to.

    The highest joint on the hierarchy-root-to-translation-root chain that is not
    static on those world axes. Everything below it -- the translation root
    included -- then moves rigidly with it, so the correction lands exactly on
    the translation root without inventing relative motion between joints that
    travelled together in the source.

    ``axes`` is asked per correction rather than once for the whole edit, because
    a chain can be static on one axis and not another: a wrapper that holds a
    flier at a fixed spot while the root climbs away from it must not absorb the
    vertical correction, and a wrapper that carries the walk must absorb the
    horizontal one. Answering both with one carrier would apply at least one of
    them to a joint that does not move on that axis, which is the exact failure
    the per-axis test below exists to avoid.

    Both shapes occur in the data and they need opposite answers:

    * Horse, Jaws, Bear, Crow, Pirrana: a wrapper joint sits still at the origin
      while the effective root walks away from it. Shifting the wrapper would
      make a static joint travel, so the correction belongs on the root itself.
    * Dog, Dog-2: joint 0 (Hips) carries the travel and the effective root
      Spine0 rides along rigidly. Re-seating only Spine0 leaves Hips travelling
      and tears the two apart -- measured at 5.27 and 8.68 of fabricated
      divergence on Dog_Running and Dog-2_RunFast, all of it landing in Hips'
      RIC channel, which is what "the body is in place but Hips still slides"
      looks like from the outside.
    """
    chain = translation_root_ancestor_chain(anim.parents, translation_root_index)
    if len(chain) <= 1:
        return int(translation_root_index)
    if global_pos is None:
        global_pos = positions_global(anim)
    axes = list(axes)
    for joint in chain:
        track = np.asarray(global_pos[:, joint][:, axes], dtype=np.float64)
        if float(np.ptp(track, axis=0).max()) > eps:
            return int(joint)
    return int(translation_root_index)


def _set_translation_root_axes(anim, translation_root_index, target, axes,
                               target_name='target'):
    """Return the animation with the root's world ``axes`` set to ``target``.

    The shared body of :func:`set_translation_root_xz` and
    :func:`set_translation_root_y`. The edit is applied to whichever joint
    actually carries the transport on those axes (see
    :func:`_transport_carrier_index`), never unconditionally to joint 0 and never
    unconditionally to the translation root. Pushing every rig's correction up to
    joint 0 would make a static wrapper slide backwards; keeping every rig's on
    the translation root tears a travelling ancestor away from it.
    """
    axes = list(axes)
    target = np.asarray(target, dtype=np.float64)
    global_pos = positions_global(anim)
    root_track = np.asarray(global_pos[:, translation_root_index][:, axes], dtype=np.float64)
    if target.shape != root_track.shape:
        raise ValueError(
            f"{target_name} must have shape {root_track.shape}, got {target.shape}"
        )
    delta = target - root_track
    if np.max(np.abs(delta)) <= 1e-8:
        return anim

    carrier = _transport_carrier_index(
        anim, translation_root_index, global_pos=global_pos, axes=axes,
    )

    new_positions = anim.positions.copy()
    if anim.parents[carrier] < 0:
        for column, axis in enumerate(axes):
            new_positions[:, carrier, axis] += delta[:, column]
    else:
        global_rots = rotations_global(anim)
        parent_index = anim.parents[carrier]
        parent_global_pos = global_pos[:, parent_index]
        parent_global_rots = global_rots[:, parent_index]
        # The carrier takes the same delta the translation root needs, so the
        # root lands on the target exactly and the two stay rigidly linked.
        desired_global = global_pos[:, carrier].copy()
        for column, axis in enumerate(axes):
            desired_global[:, axis] += delta[:, column]
        new_positions[:, carrier] = (-parent_global_rots) * (desired_global - parent_global_pos)

    return Animation(
        anim.rotations.copy(),
        new_positions,
        anim.orients.copy(),
        anim.offsets.copy(),
        anim.parents.copy(),
    )


def set_translation_root_xz(anim, translation_root_index, target_xz):
    """Return the animation with the effective root's world XZ set to ``target_xz``."""
    return _set_translation_root_axes(
        anim, translation_root_index, target_xz, (0, 2), target_name='target_xz',
    )


def set_translation_root_y(anim, translation_root_index, target_y):
    """Return the animation with the effective root's world height set to ``target_y``.

    The vertical twin of :func:`set_translation_root_xz`, kept a separate call
    rather than folded into a single XYZ setter so each correction reaches the
    joint that carries it on ITS axes (see :func:`_transport_carrier_index`) --
    a rig whose wrapper walks but does not climb needs two different answers.
    ``target_y`` is a ``(T,)`` track.
    """
    target_y = np.asarray(target_y, dtype=np.float64).reshape(-1, 1)
    return _set_translation_root_axes(
        anim, translation_root_index, target_y, (1,), target_name='target_y',
    )


def resolve_detected_translation_root_index(aligned_index, export_index, object_type):
    valid_indices = sorted({int(index) for index in (aligned_index, export_index) if int(index) >= 0})
    if len(valid_indices) > 1:
        raise ValueError(
            f"{object_type}: inconsistent translation_root_index between aligned and export animations: "
            f"{aligned_index} vs {export_index}"
        )
    if valid_indices:
        return valid_indices[0]
    return -1
