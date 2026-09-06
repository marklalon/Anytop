"""Re-seat a species' rest pose onto the geometry its own clips actually use.

The canonical position channel is a residual FROM REST, so a joint whose rest
position disagrees with where every clip holds it hands the model a per-joint DC
constant with no signal in it -- and those joints are tokens the model has to
predict like any other. The offenders are helper geometry: nub / locator /
expression-control bones an exporter gave a rest offset while the animation
collapses them onto their parent, ears and horns and tongues seated somewhere the
animator never uses.

This runs as a step of ``regenerate_dataset_artifacts`` -- which
``preprocess_and_validate`` invokes itself -- so it is applied to every dataset
build rather than being a patch someone has to remember to re-apply. It is
idempotent by construction: once a joint's rest agrees with its clips the
mismatch is zero, below the threshold, and later passes leave it alone.

What moves is the rest bone's LENGTH, keeping its direction: length is the
rotation-invariant part of the disagreement (the bone vector turns with the
parent's frame every frame, so there is no single "clip bone offset" to copy).

Four guards decide a joint (see :func:`reseat_candidates`); the load-bearing one
is that the joint must be a LEAF. ``rest_pos_ric_hml`` is exactly ``FK(offsets)``
with identity rotations, so moving a joint means moving its parent-relative
offset too -- and for an interior joint every descendant offset would need a
compensating shift, turning one bone's error into its subtree's. Leaves cost one
offset each.
"""

from __future__ import annotations

import numpy as np

from .animation_utils import find_prop_socket_joints

# A bone counts as disagreeing when its clip length differs from its rest length
# by more than this fraction.
RESEAT_MISMATCH = 0.10
# ... and the clips are only trusted as ground truth when they hold the bone
# rigidly (this much relative spread or less). An animated bone's rest is a
# legitimate neutral, not an error.
RESEAT_SELF_RIGID = 0.02
# Rest bones shorter than this carry no meaningful length ratio. The floor is
# scale-relative: a collapsed rest bone (a wrapper root, a zero-length nub) can
# be 1e-22 long, and dividing by it turns a normal clip into a
# 3.5-million-percent "error".
MIN_REST_FRAC_OF_SPAN = 1e-3
MIN_REST_LEN_ABS = 1e-6


def _measured_bones(entry):
    """``(children, parents, rest_pos, rest_len, min_len)`` for the measurable bones.

    ``children`` indexes the child joint of each bone (the root has no bone).
    Degenerate rest bones are dropped, and ``min_len`` -- the scale-relative floor
    that dropped them -- is returned because the same floor decides later whether
    a *clip* length is long enough to carry a relative spread at all.

    ``None`` when the entry has no rest geometry to measure against (a minimal
    synthetic fixture, an entry built before rest geometry existed): nothing to
    compare, so nothing to re-seat.
    """
    if entry is None or "parents" not in entry or "rest_pos_ric_hml" not in entry:
        return None
    parents = np.asarray(entry["parents"], dtype=np.int64)
    rest_pos = np.asarray(entry["rest_pos_ric_hml"], dtype=np.float64)
    children = np.flatnonzero(parents >= 0)
    if children.size == 0 or rest_pos.ndim != 2 or rest_pos.shape[0] != parents.size:
        return None
    rest_len = np.linalg.norm(rest_pos[children] - rest_pos[parents[children]], axis=-1)
    span = float(np.linalg.norm(rest_pos.max(axis=0) - rest_pos.min(axis=0)))
    min_len = max(MIN_REST_LEN_ABS, MIN_REST_FRAC_OF_SPAN * span)
    keep = rest_len > min_len
    if not keep.any():
        return None
    return children[keep], parents, rest_pos, rest_len[keep], min_len


def accumulate_rest_vs_clip(entry, motion, acc=None):
    """Fold one physical clip into a rest-vs-clip accumulator.

    ``motion`` is a physical ``[T, J, F]`` array as stored under ``motions/``.
    Pass the same ``acc`` for every clip of the species. Returns ``acc``
    unchanged (never raises) when the clip cannot be measured against this
    skeleton. Streaming: nothing scales with the clip count.
    """
    measured = _measured_bones(entry)
    if measured is None:
        return acc
    children, parents, rest_pos, rest_len, _min_len = measured
    del rest_pos  # only the parent-relative lengths matter here

    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim != 3 or motion.shape[1] < int(parents.size) or motion.shape[-1] < 3:
        return acc
    pos = motion[:, : parents.size, 0:3]
    lengths = np.linalg.norm(pos[:, children] - pos[:, parents[children]], axis=-1)
    if lengths.size == 0:
        return acc

    if acc is None:
        acc = {
            "sum": np.zeros(children.size),
            "sumsq": np.zeros(children.size),
            "min": np.full(children.size, np.inf),
            "max": np.full(children.size, -np.inf),
            "frames": 0,
            "clips": 0,
        }
    acc["sum"] += lengths.sum(axis=0)
    acc["sumsq"] += (lengths ** 2).sum(axis=0)
    acc["min"] = np.minimum(acc["min"], lengths.min(axis=0))
    acc["max"] = np.maximum(acc["max"], lengths.max(axis=0))
    acc["frames"] += int(lengths.shape[0])
    acc["clips"] += 1
    return acc


def finalize_rest_vs_clip(entry, acc):
    """Turn an accumulator into the per-bone report, or ``None`` if it is empty.

    ``vs_rest_pct`` / ``self_pct`` are the two species-level RMS percentages that
    separate "rest disagrees with rigid clips" from "the clip is simply animated
    non-rigidly" -- only the first is fixable here.
    """
    if acc is None or acc["frames"] <= 0:
        return None
    measured = _measured_bones(entry)
    if measured is None:
        return None
    children, _parents, _rest_pos, rest_len, min_len = measured
    if children.size != acc["sum"].shape[0]:
        return None

    frames = float(acc["frames"])
    clip_len = acc["sum"] / frames

    # A bone the clips collapse onto its parent has no length to take a RATIO of:
    # dividing by ~1e-9 turns a bone that sits rigidly at zero into an infinite
    # relative spread and would reject exactly the joints this exists to fix
    # (Dog's Ponytail3Nub, Deer_Buck's horns). Below the floor the bone is rigid
    # at zero by definition, so its relative spread is 0.
    long_enough = clip_len > min_len
    safe_len = np.where(long_enough, clip_len, 1.0)
    # E[(l/m - 1)^2] = E[l^2]/m^2 - 1, so the RMS needs no second pass.
    self_var = np.where(
        long_enough, np.maximum(acc["sumsq"] / frames / (safe_len ** 2) - 1.0, 0.0), 0.0
    )
    spread = np.where(
        long_enough,
        np.maximum(np.abs(acc["max"] / safe_len - 1.0),
                   np.abs(acc["min"] / safe_len - 1.0)),
        0.0,
    )

    vs_rest_sq = acc["sumsq"] / (rest_len ** 2) - 2.0 * acc["sum"] / rest_len + frames
    return {
        "children": children,
        "rest_len": rest_len,
        "clip_len": clip_len,
        "self_spread": spread,
        "vs_rest_pct": float(100.0 * np.sqrt(max(vs_rest_sq.mean() / frames, 0.0))),
        "self_pct": float(100.0 * np.sqrt(self_var.mean())),
        "n_clips": int(acc["clips"]),
    }


def reseat_candidates(entry, report):
    """Bones whose rest should move onto the position their clips agree on.

    Four guards, all required:

    * the disagreement is real (> ``RESEAT_MISMATCH`` relative length),
    * the clips hold the bone RIGIDLY (<= ``RESEAT_SELF_RIGID`` spread), so there
      genuinely is one place to move the rest to rather than an animated bone
      whose rest is a legitimate neutral,
    * the joint is a LEAF, so only its own ``offsets`` entry moves and no
      descendant needs a compensating shift, and
    * the joint is not a prop socket. ``find_prop_socket_joints`` is the
      repo's calibrated predicate for a held weapon parked away from the body in
      the T-pose (it flags Bow / Arrow / Sword / Shield across 14 unitybundles
      species and nothing in truebones). Those are already excluded from the
      SCALE statistics and deliberately left in the rest geometry: re-seating one
      onto the hand would change the rest span, which is the ``L`` that
      ``_length_scale_from_rest`` divides every species by. Same exclusion, same
      reason, one predicate.
    """
    if report is None:
        return []
    parents = np.asarray(entry["parents"], dtype=np.int64)
    has_children = np.zeros(parents.size, dtype=bool)
    for child, parent in enumerate(parents):
        if parent >= 0:
            has_children[parent] = True

    joint_names = [str(name) for name in entry.get("joints_names", [])]
    prop_joints = set()
    if len(joint_names) == parents.size:
        prop_joints = find_prop_socket_joints(
            np.asarray(entry["offsets"], dtype=np.float64), parents, joint_names
        )

    candidates = []
    for i, joint in enumerate(report["children"]):
        joint = int(joint)
        ratio = report["clip_len"][i] / report["rest_len"][i]
        if abs(ratio - 1.0) <= RESEAT_MISMATCH:
            continue
        if report["self_spread"][i] > RESEAT_SELF_RIGID:
            continue
        if has_children[joint] or joint in prop_joints:
            continue
        candidates.append({
            "joint": joint,
            "name": joint_names[joint] if joint < len(joint_names) else str(joint),
            "rest_len": float(report["rest_len"][i]),
            "clip_len": float(report["clip_len"][i]),
            "ratio": float(ratio),
        })
    return candidates


def apply_reseat(entry, candidates):
    """Scale each candidate leaf's rest bone to the length its clips hold it at.

    The rest offset keeps its DIRECTION and takes the clips' length. Length is the
    rotation-invariant part of the disagreement: the bone vector itself turns with
    the parent's frame every frame, so there is no "the clip's bone offset" to
    copy, while there is exactly one length. A nub the animation collapses onto
    its parent (ratio 0) therefore collapses in the rest pose too, which is what
    every frame already does.

    ``rest_pos_ric_hml`` is ``FK(offsets)`` with identity rotations, so the
    absolute rest position is rebuilt from the parent's, which never moves --
    candidates are leaves, so no candidate can be another's parent and the order
    is irrelevant. Returns the count applied.

    Idempotent: afterwards the rest length IS the clip length, so the ratio is 1
    and :func:`reseat_candidates` no longer selects the joint. (A bone re-seated
    to ~0 drops out of the measurable set entirely, which converges too.)
    """
    if not candidates:
        return 0
    parents = np.asarray(entry["parents"], dtype=np.int64)
    rest_pose = np.asarray(entry["rest_pose"], dtype=np.float32).copy()
    offsets = np.asarray(entry["offsets"], dtype=np.float32).copy()
    for candidate in candidates:
        joint = candidate["joint"]
        parent = int(parents[joint])
        offsets[joint] = offsets[joint] * np.float32(candidate["ratio"])
        rest_pose[joint, 0:3] = rest_pose[parent, 0:3] + offsets[joint]
    entry["rest_pose"] = rest_pose
    entry["offsets"] = offsets
    entry["rest_pos_ric_hml"] = rest_pose[:, 0:3].astype(np.float32, copy=True)
    return len(candidates)
