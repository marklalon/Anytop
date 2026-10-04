"""Per-frame contact intervals for a given set of contact joints.

The joint set is never detected here: it is cond's ``contact_joints`` with the
user's overrides applied.  Only *when* each of those joints is planted is
decided from the motion, by the rule of section 4.1 step 2 of
``docs/skeleton_profile_and_motion_edit_runtime.md``.  In-place locomotion
slides a planted foot backwards at the implied ground speed ``v_g``, so a
plant is "low, vertically still, and moving with the other planted feet", not
"not moving".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


@dataclass(frozen=True)
class ContactParams:
    """Thresholds, in leg lengths (and leg lengths per second for speeds)."""

    height: float = 0.05            # above the joint's own floor
    joint_floor_reach: float = 0.3  # a joint whose floor sits higher than this above the
    #                                 ground is never planted (tucked feet, a hovering clip)
    vertical_speed: float = 0.5
    slip_absolute: float = 0.3      # |foot velocity + v_g| below max(absolute, relative * |v_g|)
    slip_relative: float = 0.3
    hysteresis: float = 1.5         # the loose (exit) thresholds, as a multiple of the strict ones
    floor_quantile: float = 0.05
    min_frames: int = 2
    position_median: int = 3        # frames; 1 disables the median pre-filter
    velocity_sigma: float = 1.0     # frames of Gaussian smoothing before differentiation
    ground_speed_sigma: float = 1.0  # frames of Gaussian smoothing on v_g


@dataclass
class ContactResult:
    mask: np.ndarray        # (F, K) bool, column k is contact_joints[k]
    ground_velocity: np.ndarray  # (F, 2) implied ground velocity v_g in XZ, world units / s
    floor: float            # ground height the rule measured against (world Y)


def _mode(periodic: bool) -> str:
    return "wrap" if periodic else "nearest"


def _derivative(x: np.ndarray, fps: float, periodic: bool) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    if periodic:
        return (np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)) * (0.5 * fps)
    return np.gradient(x, axis=0) * fps


def _still_speed(x: np.ndarray, fps: float, periodic: bool) -> np.ndarray:
    """|dx/dt| from the stiller of the backward and forward one-frame differences.

    A short stance has no frame whose both neighbours are planted: a centred or
    smoothed difference mixes in the touchdown and lift-off motion and never
    reads still, while a planted frame is still on at least one side.
    """
    if x.shape[0] < 2:
        return np.zeros_like(x)
    if periodic:
        step = np.abs(np.roll(x, -1, axis=0) - x)          # step[f] = |x[f+1] - x[f]|
        return np.minimum(step, np.roll(step, 1, axis=0)) * fps
    step = np.abs(np.diff(x, axis=0))
    forward = np.concatenate([step, step[-1:]], axis=0)
    backward = np.concatenate([step[:1], step], axis=0)
    return np.minimum(forward, backward) * fps


def _fill_nan_rows(v: np.ndarray, periodic: bool) -> np.ndarray:
    """Linearly interpolate the all-NaN rows of ``v`` (F, C) from the valid ones."""
    valid = ~np.isnan(v).any(axis=1)
    if valid.all():
        return v
    if not valid.any():
        return np.zeros_like(v)
    frames = np.arange(v.shape[0], dtype=np.float64)
    out = v.copy()
    for c in range(v.shape[1]):
        if periodic:
            out[:, c] = np.interp(frames, frames[valid], v[valid, c], period=float(v.shape[0]))
        else:
            out[:, c] = np.interp(frames, frames[valid], v[valid, c])
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` runs of True in a 1-D mask."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _hysteresis(strict: np.ndarray, loose: np.ndarray, min_frames: int) -> np.ndarray:
    out = np.zeros_like(strict)
    for start, end in _runs(loose):
        if end - start >= min_frames and strict[start:end].any():
            out[start:end] = True
    return out


def detect_contacts(
    positions: np.ndarray,
    contact_joints,
    leg_length: float,
    fps: float,
    *,
    periodic: bool = False,
    ground: float = 0.0,
    params: ContactParams = ContactParams(),
) -> ContactResult:
    """Contact mask of ``contact_joints`` over the clip ``positions`` (F, J, 3), Y up.

    ``periodic`` treats frame ``F - 1`` as followed by frame 0 (a loop clip
    without its closing key), so a plant that spans the seam is one interval.
    ``ground`` is the world height of the ground: a joint that never comes
    near it is never planted, however still it hangs.
    """
    contact_joints = [int(j) for j in contact_joints]
    frame_count = int(positions.shape[0])
    k = len(contact_joints)
    if k == 0 or frame_count < 2 or leg_length <= 0:
        return ContactResult(np.zeros((frame_count, k), dtype=bool),
                             np.zeros((frame_count, 2)), float(ground))

    p = np.asarray(positions, dtype=np.float64)[:, contact_joints]
    mode = _mode(periodic)
    if params.position_median > 1:
        p = median_filter(p, size=(params.position_median, 1, 1), mode=mode)
    if params.velocity_sigma > 0:
        p_smooth = gaussian_filter1d(p, params.velocity_sigma, axis=0, mode=mode)
    else:
        p_smooth = p

    height = p[..., 1]
    # Each joint is measured against its own floor (an ankle plants higher than
    # the toe below it), but only if that floor is near the ground.
    joint_floor = np.quantile(height, params.floor_quantile, axis=0)
    reachable = joint_floor - ground <= params.joint_floor_reach * leg_length
    v_vertical = _still_speed(height, fps, periodic)
    v_horizontal = _derivative(p_smooth, fps, periodic)[..., [0, 2]]

    def candidates(scale):
        return (reachable[None]
                & (height < joint_floor[None] + scale * params.height * leg_length)
                & (v_vertical < scale * params.vertical_speed * leg_length))

    # v_g from the strict candidates only: a swinging foot skimming just above
    # the floor passes the loose height test and would drag the median off.
    masked = np.where(candidates(1.0)[..., None], v_horizontal, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN frames
        ground_velocity = -np.nanmedian(masked, axis=1)
    ground_velocity = _fill_nan_rows(ground_velocity, periodic)
    if params.ground_speed_sigma > 0:
        ground_velocity = gaussian_filter1d(ground_velocity, params.ground_speed_sigma, axis=0, mode=mode)

    slip = np.linalg.norm(v_horizontal + ground_velocity[:, None, :], axis=-1)
    slip_limit = np.maximum(params.slip_absolute * leg_length,
                            params.slip_relative * np.linalg.norm(ground_velocity, axis=-1))[:, None]

    def confirmed(scale):
        return candidates(scale) & (slip < scale * slip_limit)

    strict = confirmed(1.0)
    loose = confirmed(params.hysteresis)

    mask = np.zeros((frame_count, k), dtype=bool)
    for c in range(k):
        if periodic:
            tiled = _hysteresis(np.tile(strict[:, c], 3), np.tile(loose[:, c], 3), params.min_frames)
            mask[:, c] = tiled[frame_count:2 * frame_count]
        else:
            mask[:, c] = _hysteresis(strict[:, c], loose[:, c], params.min_frames)
    return ContactResult(mask, ground_velocity, float(ground))


def ground_velocity_from_mask(
    positions: np.ndarray,
    contact_joints,
    mask: np.ndarray,
    fps: float,
    *,
    periodic: bool = False,
    params: ContactParams = ContactParams(),
) -> np.ndarray:
    """Implied ground velocity ``v_g`` (F, 2) of a given contact mask (a hand-edited one).

    The same estimate :func:`detect_contacts` makes from its strict candidates,
    over the joints the mask plants instead.
    """
    contact_joints = [int(j) for j in contact_joints]
    frame_count = int(positions.shape[0])
    if not contact_joints or frame_count < 2:
        return np.zeros((frame_count, 2))
    p = np.asarray(positions, dtype=np.float64)[:, contact_joints]
    mode = _mode(periodic)
    if params.position_median > 1:
        p = median_filter(p, size=(params.position_median, 1, 1), mode=mode)
    if params.velocity_sigma > 0:
        p = gaussian_filter1d(p, params.velocity_sigma, axis=0, mode=mode)
    v_horizontal = _derivative(p, fps, periodic)[..., [0, 2]]
    masked = np.where(np.asarray(mask, dtype=bool)[..., None], v_horizontal, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ground = -np.nanmedian(masked, axis=1)
    ground = _fill_nan_rows(ground, periodic)
    if params.ground_speed_sigma > 0:
        ground = gaussian_filter1d(ground, params.ground_speed_sigma, axis=0, mode=mode)
    return ground


def contact_intervals(mask_column: np.ndarray, periodic: bool = False) -> list[tuple[int, int]]:
    """Half-open intervals of one mask column; a periodic seam-spanning plant is one
    interval whose ``end`` exceeds the frame count (``end - F`` frames wrap to the start)."""
    runs = _runs(np.asarray(mask_column, dtype=bool))
    frame_count = len(mask_column)
    if periodic and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == frame_count:
        first = runs.pop(0)
        last = runs.pop()
        runs.append((last[0], frame_count + first[1]))
    return runs
