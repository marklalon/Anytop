"""Locomotion and vertical statistics (section 3.6)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from motion_edit.contacts import contact_intervals, detect_contacts

MIN_SPEED = 0.05             # leg lengths / s; slower loops do not inform the stride fit
MIN_FIT_CLIPS = 3
DEFAULT_STRIDE_EXPONENT = 0.5
VERTICAL_NET = 0.25          # leg lengths of net height change ...
VERTICAL_PEAK = 0.15         # ... or of the lowest joint's lift, to count as vertical
# A limb has stepped only if its foot rose this share of its swing height (its
# clip-wide q95 above its floor) between two plants; a lower gap is a flicker
# inside one stance.  Relative, because a walking toe clears the floor by little.
LIFTOFF_SHARE = 0.25
# Median relative deviation of the touchdown spacings from their median above which
# a clip's steps do not repeat: feet resettling in place, not a gait.
GAIT_SPACING_SPREAD = 0.2


@dataclass
class ClipGait:
    clip: str
    action_label: str
    period: float            # frames per cycle
    duty: float
    speed: float             # |v_g|, leg lengths / s
    stride: float            # leg lengths per cycle
    phase: dict              # limb foot joint -> touchdown phase relative to the first limb


def clip_contacts(clip, contacts: list[int], leg_length: float):
    return detect_contacts(clip.global_positions, contacts, leg_length, clip.fps,
                           periodic=clip.is_loop)


def _fill_unlifted_gaps(mask: np.ndarray, height: np.ndarray, periodic: bool) -> np.ndarray:
    """Close the gaps between plants during which ``height`` stayed below
    ``LIFTOFF_SHARE`` of its swing height."""
    if mask.all() or not mask.any():
        return mask
    if periodic:
        shift = int(np.flatnonzero(mask)[0])    # start on a plant so no gap wraps
        filled = _fill_unlifted_gaps(np.roll(mask, -shift), np.roll(height, -shift), False)
        return np.roll(filled, shift)
    floor, top = np.quantile(height, (0.05, 0.95))
    lift = LIFTOFF_SHARE * (top - floor)
    out = mask.copy()
    planted = np.flatnonzero(mask)
    for a, b in zip(planted[:-1], planted[1:]):
        if b - a > 1 and height[a + 1:b].max() - floor < lift:
            out[a + 1:b] = True
    return out


def limb_masks(result, contacts: list[int], limbs: dict[int, list[int]],
               positions: np.ndarray, periodic: bool) -> dict[int, np.ndarray]:
    """Per limb (keyed by its foot joint): planted when any of its contact joints is,
    from touchdown to the next lift-off (see ``LIFTOFF_SHARE``)."""
    column = {j: k for k, j in enumerate(contacts)}
    out = {}
    for joints in limbs.values():
        mask = result.mask[:, [column[j] for j in joints]].any(axis=1)
        height = positions[:, joints, 1].min(axis=1)
        out[joints[0]] = _fill_unlifted_gaps(mask, height, periodic)
    return out


def _touchdowns(mask: np.ndarray, periodic: bool) -> list[int]:
    return sorted(s for s, _ in contact_intervals(mask, periodic=periodic))


def _periods(starts: list[int], frames: int, periodic: bool) -> list[float]:
    """Touchdown-to-touchdown spacings of one limb (a loop also counts the wrap)."""
    gaps = [float(b - a) for a, b in zip(starts, starts[1:])]
    if periodic and starts:
        gaps.append(float(frames - starts[-1] + starts[0]))
    return gaps


def clip_gait(clip, contacts: list[int], limbs: dict[int, list[int]], leg_length: float,
              result=None) -> Optional[ClipGait]:
    """Gait of one locomotion clip, or None when it shows no repeated step."""
    if not contacts or leg_length <= 0:
        return None
    result = result if result is not None else clip_contacts(clip, contacts, leg_length)
    masks = limb_masks(result, contacts, limbs, clip.global_positions, clip.is_loop)
    starts = {foot: _touchdowns(m, clip.is_loop) for foot, m in masks.items()}
    starts = {foot: s for foot, s in starts.items() if s}
    periods = [g for s in starts.values() for g in _periods(s, clip.frame_count, clip.is_loop)]
    if not periods:
        return None
    period = float(np.median(periods))
    if period < 2:
        return None
    if np.median(np.abs(np.asarray(periods) - period)) > GAIT_SPACING_SPREAD * period:
        return None
    duty = float(np.mean([masks[f].mean() for f in starts]))
    stance = np.stack([masks[f] for f in starts], axis=1).any(axis=1)
    speed = float(np.linalg.norm(result.ground_velocity[stance].mean(axis=0))) / leg_length
    reference = next(iter(starts.values()))[0]
    phase = {str(f): round(float(((s[0] - reference) % period) / period), 4)
             for f, s in starts.items()}
    return ClipGait(clip.name, clip.action_label, period, duty, speed,
                    speed * period / clip.fps, phase)


def _circular_mean(phases: list[float]) -> float:
    angles = 2 * np.pi * np.asarray(phases)
    mean = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    return round(float((mean / (2 * np.pi)) % 1.0), 4)


def aggregate_gait(gaits: list[ClipGait]) -> dict:
    by_label = defaultdict(list)
    for g in gaits:
        by_label[g.action_label].append(g)
    out = {}
    for label, items in sorted(by_label.items()):
        joints = sorted({j for g in items for j in g.phase}, key=int)
        out[label] = {
            "clips": len(items),
            "period": round(float(np.median([g.period for g in items])), 3),
            "duty": round(float(np.median([g.duty for g in items])), 4),
            "v_g": round(float(np.median([g.speed for g in items])), 4),
            "stride": round(float(np.median([g.stride for g in items])), 4),
            "phase": {j: _circular_mean([g.phase[j] for g in items if j in g.phase]) for j in joints},
        }
    return out


def stride_speed_fit(gaits: list[ClipGait]) -> dict:
    """``stride = a · v_g^b`` over the species' locomotion clips, loop or not; b fixed when
    the data is thin."""
    points = [(g.speed, g.stride) for g in gaits if g.speed > MIN_SPEED and g.stride > 0]
    if len(points) >= MIN_FIT_CLIPS and len({round(p[0], 3) for p in points}) >= 2:
        logv = np.log([p[0] for p in points])
        logs = np.log([p[1] for p in points])
        b, loga = np.polyfit(logv, logs, 1)
        return {"a": round(float(np.exp(loga)), 4), "b": round(float(b), 4),
                "n_clips": len(points), "fitted": True}
    if points:
        a = float(np.median([s / v ** DEFAULT_STRIDE_EXPONENT for v, s in points]))
    else:
        a = None
    return {"a": None if a is None else round(a, 4), "b": DEFAULT_STRIDE_EXPONENT,
            "n_clips": len(points), "fitted": False}


def clip_vertical(clip, root: int, leg_length: float) -> Optional[dict]:
    """Net root-height change and the lowest joint's lift, in leg lengths, if either is real."""
    if leg_length <= 0:
        return None
    y = clip.global_positions[..., 1]
    root_y = y[:, root]
    lowest = y.min(axis=1)
    net = float(root_y[-1] - root_y[0]) / leg_length
    ground = min(lowest[0], lowest[-1])
    peak = float(lowest.max() - ground) / leg_length
    if abs(net) < VERTICAL_NET and peak < VERTICAL_PEAK:
        return None
    return {"clip": clip.name, "action_label": clip.action_label, "net": net, "peak": peak}


def aggregate_vertical(rows: list[dict]) -> dict:
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["action_label"]].append(r)
    out = {}
    for label, items in sorted(by_label.items()):
        net = [r["net"] for r in items]
        peak = [r["peak"] for r in items]
        out[label] = {
            "clips": len(items),
            "net": [round(float(np.min(net)), 4), round(float(np.median(net)), 4), round(float(np.max(net)), 4)],
            "peak": [round(float(np.min(peak)), 4), round(float(np.median(peak)), 4), round(float(np.max(peak)), 4)],
        }
    return out
