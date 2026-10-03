"""EditRuntime: Edit Package + parameters -> Animation (section 5).

One fixed, versioned composition shared by the tuning UI and
``motion_edit.apply_edit``: identical package and parameters give
bit-identical output.  Every layer whose parameters sit at their defaults is
skipped, so the all-default result *is* the decoded generation (the strict
replay of section 1, principle 4).

Order (section 5.2):

1. time: every output frame samples the clip at a source time ``s``; a loop
   keeps a whole number of frames per period and wraps its samples; a
   one-shot's windup / strike / recover speeds warp time through its events (PCHIP);
2. amplitude: unwrapped chain offsets scaled per group (joints winding a whole
   turn per loop held at gain 1); a hanging part's (the tail, the passive joints)
   by its ``*_weight`` only up to 1;  the strike body (active chain, the trunk it
   hangs on, the root) drawn further from its contact pose in the windup
   (windup_depth) and pushed past it after contact (overshoot), with a lean
   and a shift of the body against / along the strike; the paired arms / legs
   turned at their roots so their tips move out or in (spread);
3. root: oscillation (sway, bounce), airborne arcs (jump_height), posture;
4. plants + limb IK: planted feet follow their targets, swing feet carry the
   correction between them (``motion_edit.ik``);
5. secondary motion: past 1, ``tail_weight`` / ``passive_weight`` add that fraction of
   their part's spring response (``*_stiffness`` scales the spring's frequency) to the body's motion on top of its own curves (each
   part hinges at its parent outside it, so its first joint swings too);
6. ground: a planted foot IK could not bring down lowers the body instead.

Step 5 runs after step 6: no passive joint lies on a limb IK solves, and the
springs then read the body as it is finally placed.

The runtime imports neither torch nor the decode path; everything it needs
is in the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import expm

from motion_edit.ik import LimbSolver, build_limbs, world_to_local
from motion_edit.package import CHAIN_GROUPS, EditPackage
from data_loaders.truebones.truebones_utils.physics_joint_annotation import joint_name_matches_keywords
from motion_edit.profile.skeleton import CENTER as CENTER_SIDE
from motion_edit.profile.spring import default_spring, drive_terms
from motion_edit.rotations import (
    nearest_rotvec_branch,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    quat_rotate,
    quat_slerp,
    rotation_between,
    rotvec_from_quat,
)
from motion_lib.Animation import Animation, positions_global
from motion_lib.Quaternions import Quaternions


@dataclass(frozen=True)
class ParamSpec:
    default: float | bool
    low: float = 0.0
    high: float = 1.0
    group: str = ""
    toggle: bool = False


AMP_GROUPS = tuple(g for g in CHAIN_GROUPS if g not in ("root", "other", "tail"))
_AMP = {f"amp.{g}": ParamSpec(1.0, 0.0, 2.0, "amp") for g in AMP_GROUPS}
# Hanging parts have one treatment and a channel each: the tail (its joints that can swing:
# no support joint below them) and the passive joints.  Up to 1 a channel's weight scales
# the part's own curves (0 holds it rigid), past 1 it adds that fraction of the part's
# spring swing, hinged at its non-passive parent (step 5); its stiffness multiplies the
# spring's natural frequency at a constant damping ratio (stiffer: a smaller, quicker swing).
SWING_CHANNELS = ("tail", "passive")
_SWING = {f"{c}_{knob}": ParamSpec(1.0, low, 2.0, "secondary")
          for c in SWING_CHANNELS for knob, low in (("weight", 0.0), ("stiffness", 0.5))}
# spread.{group}: the tip of each sided limb of the group moves away from the body's midline
# (toward it below 0), level with the ground, by the slider value times the group's fraction
# of the limb's length.  The limb turns at its root; a planted foot is moved with it.
SPREAD_REACH = {"arms": 1.0, "legs": 0.3}
SPREAD_GROUPS = tuple(SPREAD_REACH)
_SPREAD = {f"spread.{g}": ParamSpec(0.0, -1.0, 1.0, "spread") for g in SPREAD_GROUPS}
# A limb whose first bone is shorter than this fraction of the limb (a clavicle, a hip
# stub), or whose first joint is named as a shoulder girdle, turns at the joint below it.
SPREAD_SHORT_ROOT = 0.3
SPREAD_GIRDLE_KEYWORDS = ("clavicle", "collar", "scapula")
# The slider scaling each group's chain offsets; tail joints that cannot swing follow the
# tail's weight as a plain gain.
GROUP_GAIN = {**{g: f"amp.{g}" for g in AMP_GROUPS}, "tail": "tail_weight"}

# The v1 parameter set (section 5.1), in panel order (grouped).
PARAM_SPECS: dict[str, ParamSpec] = {
    "tempo": ParamSpec(1.0, 0.5, 2.0, "time"),
    "stride": ParamSpec(1.0, 0.6, 1.6, "locomotion"),
    "posture": ParamSpec(0.0, -0.3, 0.2, "root"),
    "jump_height": ParamSpec(1.0, 0.5, 1.8, "root"),
    "bounce": ParamSpec(1.0, 0.0, 2.0, "root"),
    "sway": ParamSpec(1.0, 0.0, 2.0, "root"),
    **_AMP,
    **_SPREAD,
    "force": ParamSpec(1.0, 0.5, 2.0, "force"),
    "windup_depth": ParamSpec(1.0, 0.0, 2.0, "force"),
    "overshoot": ParamSpec(1.0, 0.0, 2.0, "force"),
    "windup_speed": ParamSpec(1.0, 0.5, 2.0, "force"),
    "strike_speed": ParamSpec(1.0, 0.5, 2.0, "force"),
    "recover_speed": ParamSpec(1.0, 0.5, 2.0, "force"),
    **_SWING,
    # the springs' gravity term, both channels: off, a swing ignores which way is down
    "gravity": ParamSpec(True, group="secondary", toggle=True),
    "foot_lock": ParamSpec(False, group="contact", toggle=True),
    "soft_stretch": ParamSpec(0.1, 0.0, 0.2, "contact"),
}

# Parameters whose non-default values this runtime version composes.  A
# non-default value of any other parameter is refused, never ignored.
IMPLEMENTED: frozenset[str] = frozenset(
    ["tempo", "stride", "bounce", "jump_height", "sway", "posture", "foot_lock", "soft_stretch",
     *_AMP, *_SPREAD, *(name for name, spec in PARAM_SPECS.items() if spec.group in ("force", "secondary"))])

# Parameters that move the body or a support chain: any of them away from its
# default re-solves the planted limbs.  The timing ones (tempo and the strike's
# segment speeds) only re-time the clip.
_IK_PARAMS = ("stride", "bounce", "jump_height", "sway", "posture", *GROUP_GAIN.values(), "force",
              "windup_depth", "overshoot", "spread.legs")
# force is a preset over every other strike parameter: those here are multiplied by it,
# FORCE_SLOWED divided by force ** FORCE_SLOW_EXPONENT (a harder strike winds up and
# recovers somewhat more slowly; the strike itself carries most of the change).
FORCE_PARAMS = ("windup_depth", "strike_speed", "overshoot")
FORCE_SLOWED = ("windup_speed", "recover_speed")
FORCE_SLOW_EXPONENT = 0.5
# Speed of each strike segment: start -> windup, windup -> impact, impact -> recover.
SEGMENT_SPEEDS = ("windup_speed", "strike_speed", "recover_speed")
# Trunk lean (rad) and body shift (x leg length) per unit of windup_depth / overshoot
# off 1: back against the strike in the windup, forward along it past contact.  They
# give the edit a whole-body weight shift the clip's own trunk motion may not carry.
STRIKE_LEAN = np.radians(20.0)
STRIKE_SHIFT = 0.15
# A strike whose horizontal travel is less than this fraction of its travel (a smash
# straight down) leans along the character's facing instead.
STRIKE_HORIZONTAL = 0.3
# Strike events a request may move (source frames); it may also pick another chain.
EVENT_KEYS = ("windup", "impact", "recover")

# Foot error after IK reported as unreachable; a planted foot left higher than
# this above its target lowers the body instead.
REACH_TOLERANCE = 1e-4    # x leg length
# Rise and fall of that lowering around the frames that need it.
GROUND_RAMP = 0.2         # seconds
# soft_stretch: a leg starts to lengthen once its target asks for more than this
# extension (reach / chain length), or more than the pose's own if that is higher,
# and to shorten once it asks for less than this fraction of the pose's own.
SOFT_STRETCH_START = 0.96
SOFT_COMPRESS_START = 0.9


# Action-label word that offers jump_height on a clip with airborne spans, loop or not.
JUMP_WORD = "jump"


def label_has_jump(action_label: str) -> bool:
    return JUMP_WORD in {w.strip() for w in str(action_label or "").split(",")}


# profile_spring columns (``decompose.SPRING_FIELDS``).
SPRING_K, SPRING_C, SPRING_G_ANG, SPRING_G_LIN, SPRING_G_GRAV = range(5)


class UnsupportedParameterError(ValueError):
    pass


def available_params(package_facts: dict) -> list[str]:
    """Parameters that mean something for this clip (the UI hides the rest).

    ``package_facts``: ``is_loop``, ``locomotion``, ``has_plants``, ``turning``,
    ``airborne``, ``jump`` (the action label names a jump), ``has_passive``, ``strike``
    (one-shot events found) and ``chain_groups`` (groups with joints).
    """
    out = ["tempo", "bounce", "sway"]
    if package_facts["locomotion"] and package_facts["has_plants"] and not package_facts["turning"]:
        out.append("stride")
    if package_facts["airborne"] and package_facts.get("jump"):
        out.append("jump_height")
    out += [f"amp.{g}" for g in AMP_GROUPS if g in package_facts["chain_groups"]]
    # which limbs are arms or legs depends on the clip's plants: EditRuntime keeps a spread
    # only when its group has a left / right pair
    if {"arms", "legs"} & set(package_facts["chain_groups"]):
        out += [f"spread.{g}" for g in SPREAD_GROUPS]
    if "tail" in package_facts["chain_groups"]:
        out += ["tail_weight", "tail_stiffness"]
    if package_facts["has_plants"]:
        out.append("posture")
    if package_facts.get("strike"):
        out += [name for name, spec in PARAM_SPECS.items() if spec.group == "force"]
    if package_facts["has_passive"]:
        out += ["passive_weight", "passive_stiffness"]
    if package_facts["has_passive"] or "tail" in package_facts["chain_groups"]:
        out.append("gravity")
    if package_facts["has_plants"]:
        out += ["foot_lock", "soft_stretch"]
    return [name for name in PARAM_SPECS if name in out]


def param_manifest(available: list[str]) -> dict:
    return {
        name: {
            "default": spec.default,
            "min": None if spec.toggle else spec.low,
            "max": None if spec.toggle else spec.high,
            "group": spec.group,
            "toggle": spec.toggle,
            "available": name in available,
            "implemented": name in IMPLEMENTED,
        }
        for name, spec in PARAM_SPECS.items()
    }


# ── math helpers ─────────────────────────────────────────────────────────────

def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(yaw, dtype=np.float64)
    zeros = np.zeros_like(half)
    return np.stack([np.cos(half), zeros, np.sin(half), zeros], axis=-1)


def ground_displacement(ground_velocity: np.ndarray, fps: float, periodic: bool) -> np.ndarray:
    """Cumulative ``v_g`` displacement (XZ), one entry per frame, two periods for a loop."""
    v = np.tile(ground_velocity, (2, 1)) if periodic else ground_velocity
    step = v / fps
    return np.concatenate([np.zeros((1, 2)), np.cumsum(step[:-1], axis=0)], axis=0)


def _bracket(times: np.ndarray, count: int, periodic: bool):
    lo = np.floor(times).astype(np.int64)
    alpha = times - lo
    if periodic:
        lo = lo % count
        hi = (lo + 1) % count
    else:
        lo = np.clip(lo, 0, count - 1)
        hi = np.minimum(lo + 1, count - 1)
        alpha = np.where(lo >= count - 1, 0.0, alpha)
    return lo, hi, alpha


def sample_linear(values: np.ndarray, times: np.ndarray, periodic: bool) -> np.ndarray:
    """``values`` (F, ...) at fractional frame ``times``; integer times return the keys exactly."""
    lo, hi, alpha = _bracket(np.asarray(times, dtype=np.float64), values.shape[0], periodic)
    a = alpha.reshape(alpha.shape + (1,) * (values.ndim - 1))
    return np.where(a == 0.0, values[lo], values[lo] * (1.0 - a) + values[hi] * a)


def sample_angle(values: np.ndarray, times: np.ndarray, periodic: bool) -> np.ndarray:
    lo, hi, alpha = _bracket(np.asarray(times, dtype=np.float64), values.shape[0], periodic)
    delta = np.angle(np.exp(1j * (values[hi] - values[lo])))
    return np.where(alpha == 0.0, values[lo], values[lo] + alpha * delta)


def sample_quat(values: np.ndarray, times: np.ndarray, periodic: bool) -> np.ndarray:
    lo, hi, alpha = _bracket(np.asarray(times, dtype=np.float64), values.shape[0], periodic)
    a = np.broadcast_to(alpha.reshape(alpha.shape + (1,) * (values.ndim - 2)), values[lo].shape[:-1])
    return quat_slerp(values[lo], values[hi], a)


def forward_kinematics(parents, rotations: np.ndarray, local_positions: np.ndarray):
    """Global rotations and positions; parents precede children."""
    g = np.empty_like(rotations)
    p = np.empty_like(local_positions)
    for j, parent in enumerate(parents):
        if parent < 0:
            g[:, j] = rotations[:, j]
            p[:, j] = local_positions[:, j]
        else:
            g[:, j] = quat_mul(g[:, parent], rotations[:, j])
            p[:, j] = p[:, parent] + quat_rotate(g[:, parent], local_positions[:, j])
    return g, p


def _refresh_fk(parents, rotations, local_positions, g, p, joints) -> None:
    """``forward_kinematics`` redone in place for ``joints`` (index order, none the root),
    their parents' globals already current."""
    for j in joints:
        parent = parents[j]
        g[:, j] = quat_mul(g[:, parent], rotations[:, j])
        p[:, j] = p[:, parent] + quat_rotate(g[:, parent], local_positions[:, j])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _stance_runs(mask: np.ndarray, periodic: bool) -> list[np.ndarray]:
    """Frame indices of each run of True; a loop's run across the seam starts negative."""
    runs = _runs(mask)
    count = len(mask)
    if periodic and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == count:
        first, last = runs.pop(0), runs.pop()
        runs.append((last[0] - count, first[1]))
    return [np.arange(a, b) for a, b in runs]


def blend_between_keys(values: np.ndarray, keyed: np.ndarray, periodic: bool) -> np.ndarray:
    """Fill the unkeyed rows of ``values`` (F, C) by linear interpolation between keyed rows
    (wrapping for a loop, held flat past the ends of a one-shot)."""
    keys = np.flatnonzero(keyed)
    if keys.size == 0:
        return np.zeros_like(values)
    if keys.size == values.shape[0]:
        return values
    frames = np.arange(values.shape[0], dtype=np.float64)
    out = values.copy()
    for c in range(values.shape[1]):
        if periodic:
            out[:, c] = np.interp(frames, keys, values[keys, c], period=float(values.shape[0]))
        else:
            out[:, c] = np.interp(frames, keys, values[keys, c])
    out[keyed] = values[keyed]
    return out


def pchip(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Monotone piecewise-cubic (Fritsch-Carlson) interpolation of increasing nodes ``x``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.diff(x)
    delta = np.diff(y) / h
    slope = np.zeros_like(x)
    if len(x) == 2:
        slope[:] = delta[0]
    else:
        for k in range(1, len(x) - 1):
            if delta[k - 1] * delta[k] > 0.0:
                w1, w2 = 2.0 * h[k] + h[k - 1], h[k] + 2.0 * h[k - 1]
                slope[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        for end, (h0, h1, d0, d1) in ((0, (h[0], h[1], delta[0], delta[1])),
                                      (-1, (h[-1], h[-2], delta[-1], delta[-2]))):
            m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if np.sign(m) != np.sign(d0):
                m = 0.0
            elif np.sign(d0) != np.sign(d1) and abs(m) > 3.0 * abs(d0):
                m = 3.0 * d0
            slope[end] = m
    query = np.asarray(query, dtype=np.float64)
    k = np.clip(np.searchsorted(x, query, side="right") - 1, 0, len(x) - 2)
    t = (query - x[k]) / h[k]
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h[k] * slope[k]
            + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h[k] * slope[k + 1])


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return (values >= low).astype(np.float64)
    x = np.clip((values - low) / (high - low), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def strike_shapes(times: np.ndarray, windup: int, impact: int, recover: int,
                  period: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """The force layers' time shapes at source ``times``, both in [0, 1].

    ``hold`` weighs the windup: eased in over one windup-to-impact length
    before windup and out over the strike, so the body is back at its contact
    pose by impact.
    ``push`` weighs the drive past contact: eased in from halfway through the
    strike to a crest a third of the way from impact to recover, out by recover.
    A loop's (``period``) shapes wrap: the ease-in never reaches back past the
    previous period's recover, and both shapes are zero where the period joins.
    """
    start = windup - (impact - windup)
    if period:
        start = max(start, recover - period)
        times = start + np.mod(np.asarray(times, dtype=np.float64) - start, period)
    else:
        start = max(0.0, start)
    hold = _smoothstep(times, start, windup) * (1.0 - _smoothstep(times, windup, impact))
    crest = impact + (recover - impact) / 3.0
    push = _smoothstep(times, 0.5 * (windup + impact), crest) * (1.0 - _smoothstep(times, crest, recover))
    return hold, push


def soft_scale(need: np.ndarray, own: np.ndarray, limit: float) -> np.ndarray:
    """Leg length factor for a target that asks for extension ``need`` of a pose whose own
    extension is ``own`` (both reach / chain length): 1 inside the comfortable band; past
    either edge, the factor that would hold the edge, eased so it never exceeds ``1 +- limit``."""
    if limit <= 0.0:
        return np.ones_like(need)
    high = np.maximum(SOFT_STRETCH_START, own)
    low = SOFT_COMPRESS_START * own
    excess = np.where(need > high, need / high - 1.0,
                      np.where(need < low, need / np.maximum(low, 1e-12) - 1.0, 0.0))
    return 1.0 + limit * np.tanh(excess / limit)


def ramped_envelope(values: np.ndarray, width: int, periodic: bool) -> np.ndarray:
    """Upper envelope of ``values`` (F,) >= 0 that rises and falls over ``width`` frames:
    each frame is the peak of a raised-cosine bump of that half-width (wrapping for a loop)."""
    out = np.array(values, dtype=np.float64, copy=True)
    count = out.shape[0]
    for offset in range(1, min(width, count - 1) + 1):
        weight = 0.5 * (1.0 + np.cos(np.pi * offset / (width + 1)))
        if periodic:
            later, earlier = np.roll(values, offset), np.roll(values, -offset)
        else:
            later = np.concatenate([np.zeros(offset), values[:-offset]])
            earlier = np.concatenate([values[offset:], np.zeros(offset)])
        out = np.maximum(out, weight * np.maximum(later, earlier))
    return out


def spring_response(drive: np.ndarray, k: float, c: float, fps: float, periodic: bool) -> np.ndarray:
    """``θ`` (F, 3) of ``θ̈ + c θ̇ + k θ = drive`` on every frame, the three axes alike.

    Each step is the oscillator's exact (matrix-exponential) transition under
    the drive held at its mid-step value, so it is stable at any stiffness.  A
    one-shot starts at rest; a loop starts on its periodic steady state, so the
    response closes over the period.
    """
    system = np.zeros((3, 3))
    system[0, 1] = 1.0
    system[1] = [-k, -c, 1.0]
    step = expm(system / fps)
    phi, gamma = step[:2, :2], step[:2, 2]
    frames = drive.shape[0]
    ahead = np.roll(drive, -1, axis=0) if periodic else np.concatenate([drive[1:], drive[-1:]])
    mid = 0.5 * (drive + ahead)

    def run(state):
        out = np.empty((frames, 3))
        for i in range(frames):
            out[i] = state[0]
            state = phi @ state + gamma[:, None] * mid[i][None]
        return out, state

    if not periodic:
        return run(np.zeros((2, 3)))[0]
    _, end = run(np.zeros((2, 3)))
    # x_F = Φ^F x_0 + (the response from rest) = x_0
    start = np.linalg.lstsq(np.eye(2) - np.linalg.matrix_power(phi, frames), end, rcond=None)[0]
    return run(start)[0]


# ── result ───────────────────────────────────────────────────────────────────

@dataclass
class EditResult:
    animation: Animation
    fps: float
    global_positions: np.ndarray        # (F', J, 3)
    ground_velocity: np.ndarray         # (F', 2) the result's implied ground velocity v_g'
    composed: bool                      # False: the all-default replay shortcut
    params: dict
    source_time: np.ndarray             # (F',) source frame each output frame samples
    plant_id: np.ndarray                # (F', K) package plant interval planted on, -1 off
    plant_time: np.ndarray              # (F', K) unwrapped source time inside that interval
    plant_target: np.ndarray            # (F', K, 3) where a planted joint was sent (NaN off)
    pivot: np.ndarray                   # (F', K) the joint IK placed exactly
    unreached: np.ndarray               # (F', K) pivots IK could not bring to their target
    stride_factor: float                # S: v_g' = S * (time rate) * v_g
    strike: Optional[dict] = None       # strike events used (source frames) and the active chain
    diagnostics: list = field(default_factory=list)


@dataclass
class SpreadLimb:
    group: str              # "arms" / "legs"
    pivot: int              # the joint the limb turns at
    tip: int                # the joint whose outward move the slider sets
    sign: float             # +1 on the left, -1 on the right
    length: float           # bone length from the pivot down to the tip
    columns: list = field(default_factory=list)   # a leg's contact columns


@dataclass
class _Timeline:
    times: np.ndarray       # source frame per output frame
    rate: float             # source frames per output frame (mean, for a warped one-shot)
    stride: float           # S
    notes: list
    speed: Optional[np.ndarray] = None   # (F',) local rate where time is warped
    unwrapped: Optional[np.ndarray] = None   # (F',) a warped loop's source time before wrapping

    def local_rate(self) -> np.ndarray:
        return self.speed if self.speed is not None else np.full(len(self.times), self.rate)

    def unwrapped_times(self) -> np.ndarray:
        return self.unwrapped if self.unwrapped is not None else self.times


class EditRuntime:
    def __init__(self, package: EditPackage):
        self.package = package
        # from the package facts, so a package built before a parameter existed still offers it
        facts = package.manifest.get("facts")
        if facts and "jump" not in facts:
            facts = {**facts, "jump": label_has_jump(package.manifest.get("action_label", ""))}
        self.available = set(available_params(facts) if facts else package.manifest.get("available_params", []))
        self._original = None
        self._limbs = None
        self._spread_limbs = None
        self._lateral = {}
        self._springs = None
        self._channels = None
        self._subtrees = None
        for group in SPREAD_GROUPS:
            if {l.sign for l in self.spread_limbs() if l.group == group} != {1.0, -1.0}:
                self.available.discard(f"spread.{group}")

    # ── parameters ───────────────────────────────────────────────────────
    def defaults(self) -> dict:
        return {name: spec.default for name, spec in PARAM_SPECS.items()}

    def resolve_params(self, params: dict | None) -> tuple[dict, list]:
        """Defaults filled in, values clamped to range; refuses what cannot be honoured."""
        out = self.defaults()
        diagnostics = []
        for name, value in (params or {}).items():
            spec = PARAM_SPECS.get(name)
            if spec is None:
                raise ValueError(f"unknown parameter '{name}'")
            if spec.toggle:
                value = bool(value)
            else:
                value = float(value)
                clamped = float(np.clip(value, spec.low, spec.high))
                if clamped != value:
                    diagnostics.append({"kind": "clamped", "param": name,
                                        "message": f"{name}={value:g} clamped to {clamped:g}"})
                value = clamped
            out[name] = value
        for name, value in out.items():
            if value == PARAM_SPECS[name].default:
                continue
            if name not in self.available:
                raise UnsupportedParameterError(f"'{name}' does not apply to this clip")
            if name not in IMPLEMENTED:
                raise UnsupportedParameterError(
                    f"'{name}' is not implemented by this runtime version")
        return out, diagnostics

    # ── package-derived data ─────────────────────────────────────────────
    @property
    def leg(self) -> float:
        leg = float(self.package.manifest.get("leg_length") or 0.0)
        return leg if leg > 0 else 1.0

    def original_fk(self) -> tuple[np.ndarray, np.ndarray]:
        """Global rotations and positions of the package's own (unedited) clip."""
        if self._original is None:
            self._original = forward_kinematics(
                self.package["parents"], np.asarray(self.package["base_rot"], dtype=np.float64),
                np.asarray(self.package["base_pos"], dtype=np.float64))
        return self._original

    def original_positions(self) -> np.ndarray:
        return self.original_fk()[1]

    # ── composition ──────────────────────────────────────────────────────
    def strike(self, events: dict | None = None) -> Optional[dict]:
        """The package's strike events with ``events`` (moved ``windup`` / ``impact`` /
        ``recover`` frames, another ``chain`` from the candidates) applied; ``None``
        without strike events.  ``joints`` are the active chain's (its subtree
        without the chains nested in it) and ``effector`` its own leaf, ``moved``
        the events the request moved."""
        pkg = self.package
        base = pkg.manifest["events"].get("strike")
        events = {k: v for k, v in (events or {}).items() if v is not None}
        if base is None:
            if events:
                raise ValueError("this clip has no strike events to move")
            return None
        unknown = set(events) - set(EVENT_KEYS) - {"chain"}
        if unknown:
            raise ValueError(f"unknown event(s) {sorted(unknown)}")
        out = dict(base)
        for key in (*EVENT_KEYS, "chain"):
            if key in events:
                out[key] = int(events[key])
        frames = pkg.frame_count
        if pkg.is_loop:
            # unwrapped about impact, within one period
            lap = out["impact"] // frames
            for key in EVENT_KEYS:
                out[key] -= lap * frames
            if not out["windup"] < out["impact"] < out["recover"] <= out["windup"] + frames - 1:
                raise ValueError("strike events must keep windup < impact < recover within one loop period")
        elif not 0 <= out["windup"] < out["impact"] < out["recover"] <= frames - 1:
            raise ValueError("strike events must keep 0 <= windup < impact < recover <= last frame")
        chains = {c["joint"]: c for c in base["candidates"]}
        if out["chain"] not in chains:
            raise ValueError(f"joint {out['chain']} is not a strike chain candidate")
        chain = chains[out["chain"]]
        out["joints"] = [int(j) for j in chain["joints"]]
        out["chain_name"], out["effector"], out["effector_name"] = (
            chain["name"], int(chain["effector"]), chain["effector_name"])
        out["moved"] = sorted(k for k in events if events[k] != base.get(k))
        return out

    def strike_body(self, strike: dict) -> tuple[list[int], list[int]]:
        """``(trunk, body)`` of a strike: the joints between the root and the active
        chain, and those with the chain's own (gain-locked joints left out)."""
        pkg = self.package
        parents = np.asarray(pkg["parents"])
        locked = np.asarray(pkg["chain_gain_locked"], dtype=bool)
        trunk, j = [], int(parents[strike["chain"]])
        while j >= 0 and j != pkg.root:
            trunk.append(j)
            j = int(parents[j])
        body = sorted(k for k in set(trunk) | set(strike["joints"]) if not locked[k] and k != pkg.root)
        return trunk[::-1], body

    def strike_direction(self, strike: dict) -> np.ndarray:
        """Horizontal unit direction (X, Z) of the strike: the active chain joint that
        travels furthest from windup to impact, or the facing when it strikes downward."""
        pkg = self.package
        original = self.original_positions()
        frames = pkg.frame_count
        impact = strike["impact"] % frames
        travel = original[impact, strike["joints"]] - original[strike["windup"] % frames, strike["joints"]]
        move = travel[int(np.argmax(np.linalg.norm(travel, axis=-1)))]
        flat = move[[0, 2]]
        if np.linalg.norm(flat) >= STRIKE_HORIZONTAL * max(np.linalg.norm(move), 1e-12) > 0.0:
            return flat / np.linalg.norm(flat)
        front, base = pkg.manifest.get("forward_joint_index"), pkg.manifest.get("forward_base_joint_index")
        if front is not None and base is not None:
            facing = (original[impact, front] - original[impact, base])[[0, 2]]
            if np.linalg.norm(facing) > 1e-9:
                return facing / np.linalg.norm(facing)
        yaw = float(np.asarray(pkg["root_yaw"])[impact])
        return np.array([np.sin(yaw), np.cos(yaw)])

    def strike_lean_axis(self, strike: dict, direction: np.ndarray) -> np.ndarray:
        """World axis the body leans about, scaled to how far a turn about it carries the
        effector along the strike: the lever from the root to the effector at contact
        swung toward ``direction``.  An upright trunk under a punch leans fully, a long
        neck already reaching along the strike barely (its weight shift does the work)."""
        original = self.original_positions()
        impact = strike["impact"] % self.package.frame_count
        lever = original[impact, strike["effector"]] - original[impact, self.package.root]
        toward = np.array([direction[0], 0.0, direction[1]])
        return np.cross(lever, toward) / max(float(np.linalg.norm(lever)), 1e-12)

    @staticmethod
    def _force(p: dict) -> tuple[dict, list]:
        """The strike parameters with the force preset applied, each clamped to its own range."""
        out, notes = {}, []
        for name in FORCE_PARAMS + FORCE_SLOWED:
            spec = PARAM_SPECS[name]
            slowed = name in FORCE_SLOWED
            value = p[name] / p["force"] ** FORCE_SLOW_EXPONENT if slowed else p[name] * p["force"]
            clamped = float(np.clip(value, spec.low, spec.high))
            if clamped != value:
                notes.append({"kind": "clamped", "param": name, "message":
                              f"{name} with force {p['force']:g} = {value:g} clamped to {clamped:g}"})
            out[name] = clamped
        return out, notes

    def apply(self, params: dict | None = None, *, compose: bool = False,
              events: dict | None = None) -> EditResult:
        """Compose the clip under ``params``.

        ``compose=True`` disables the all-default shortcut and rebuilds the
        clip from its layers with unit gains (T1b: it exercises the
        decomposition the shortcut never reads).  ``events`` moves the strike
        events the force parameters key on (``strike``).
        """
        resolved, diagnostics = self.resolve_params(params)
        strike = self.strike(events)
        is_default = all(resolved[n] == PARAM_SPECS[n].default for n in PARAM_SPECS)
        pkg = self.package
        if is_default and not compose:
            timeline = _Timeline(np.arange(pkg.frame_count, dtype=np.float64), 1.0, 1.0, [])
            pid, ptime = self._plant_timeline(timeline.times)
            return self._result(
                np.asarray(pkg["base_rot"]), np.asarray(pkg["base_pos"]), resolved, False, timeline,
                pid, ptime, self._stance_targets(pid, ptime, timeline, 1.0, False),
                np.zeros(pid.shape, dtype=bool), diagnostics, strike=strike)

        force, notes = self._force(resolved)
        diagnostics += notes
        timeline = self._timeline(resolved, strike, force)
        diagnostics += [{"kind": "tempo", "message": m} for m in timeline.notes]
        rotations, positions, held, notes = self._compose_layers(resolved, timeline.times, strike, force)
        diagnostics += notes
        if held:
            diagnostics.append({"kind": "gain_locked", "message":
                                f"{held} joint(s) wind a whole turn per loop; gain held at 1"})
        pid, ptime = self._plant_timeline(timeline.times)
        pivot = np.zeros(pid.shape, dtype=bool)
        unreached = None
        # no contact joints: nothing to plant, the composed pose stands
        if pid.shape[1] and (any(resolved[n] != PARAM_SPECS[n].default for n in _IK_PARAMS)
                             or resolved["foot_lock"]):
            targets = self._stance_targets(pid, ptime, timeline, timeline.stride,
                                           bool(resolved["foot_lock"]),
                                           {g: resolved[f"spread.{g}"] for g in SPREAD_GROUPS})
            rotations, positions, pivot, unreached, ik_notes = self._solve_plants(
                rotations, positions, pid, targets, resolved["soft_stretch"])
            diagnostics += ik_notes
        else:
            targets = self._stance_targets(pid, ptime, timeline, 1.0, False)
        # 5. secondary motion: the swing past the hanging parts' own curves
        weights = {c: resolved[f"{c}_weight"] - 1.0 for c in SWING_CHANNELS if resolved[f"{c}_weight"] > 1.0}
        if weights:
            stiffness = {c: resolved[f"{c}_stiffness"] for c in weights}
            rotations, positions, notes = self._secondary(weights, stiffness, bool(resolved["gravity"]),
                                                          rotations, positions)
            diagnostics += notes
        return self._result(rotations, positions, resolved, True, timeline, pid, ptime, targets, pivot,
                            diagnostics, unreached, strike=strike)

    def _result(self, rotations, positions, resolved, composed, timeline, pid, ptime, targets, pivot,
                diagnostics, unreached=None, strike=None) -> EditResult:
        pkg = self.package
        animation = Animation(
            Quaternions(np.array(rotations, copy=True)),
            np.array(positions, copy=True),
            Quaternions(np.array(pkg["orients"], copy=True)),
            np.array(pkg["anim_offsets"], copy=True),
            np.array(pkg["parents"], copy=True),
        )
        ground = sample_linear(np.asarray(pkg["ground_velocity"], dtype=np.float64), timeline.times, pkg.is_loop)
        return EditResult(
            animation=animation,
            fps=pkg.fps,
            global_positions=np.asarray(positions_global(animation)),
            ground_velocity=ground * timeline.stride * timeline.local_rate()[:, None],
            composed=composed,
            params=resolved,
            source_time=timeline.times,
            plant_id=pid,
            plant_time=ptime,
            plant_target=targets,
            pivot=pivot,
            unreached=unreached if unreached is not None else np.zeros(pid.shape, dtype=bool),
            stride_factor=timeline.stride,
            strike=strike,
            diagnostics=diagnostics,
        )

    # ── 1. time ──────────────────────────────────────────────────────────
    def _timeline(self, p: dict, strike: Optional[dict], force: dict) -> _Timeline:
        pkg = self.package
        frames = pkg.frame_count
        rate = p["tempo"]
        stride = p["stride"]
        notes = []
        if strike is not None and any(force[name] != 1.0 for name in SEGMENT_SPEEDS):
            if pkg.is_loop:
                return self._loop_strike_timeline(rate, stride, strike, force)
            return self._strike_timeline(rate, stride, strike, force)
        if pkg.is_loop:
            count = max(2, int(round(frames / rate)))
            times = np.arange(count, dtype=np.float64) * (frames / count)
            effective = frames / count
            if count != frames and abs(effective - rate) > 1e-9:
                notes.append(f"time rate {rate:.3f} -> {count} frames per loop (effective {effective:.3f})")
            rate = effective
        else:
            count = int(np.floor((frames - 1) / rate + 1e-9)) + 1
            times = np.arange(count, dtype=np.float64) * rate
        return _Timeline(times, float(rate), float(stride), notes)

    def _strike_timeline(self, rate: float, stride: float, strike: dict, force: dict) -> _Timeline:
        """One-shot time warped through its events: start -> windup, windup -> impact and
        impact -> recover each run at their own speed, the rest at its pace; ``tempo``
        scales it all.  Source time per output frame is a monotone cubic (PCHIP) through
        the event nodes."""
        last = self.package.frame_count - 1
        windup, impact, recover = strike["windup"], strike["impact"], strike["recover"]
        source = np.array(sorted({0, windup, impact, recover, last}), dtype=np.float64)
        length = np.diff(source)
        for begin, end, name in ((0, windup, "windup_speed"), (windup, impact, "strike_speed"),
                                 (impact, recover, "recover_speed")):
            if end > begin:                     # each one runs between two adjacent nodes
                length[int(np.flatnonzero(source == begin)[0])] /= force[name]
        out = np.concatenate([[0.0], np.cumsum(length)]) / rate
        count = int(np.floor(out[-1] + 1e-9)) + 1
        times = np.clip(pchip(out, source, np.arange(count, dtype=np.float64)), 0.0, float(last))
        speed = np.gradient(times) if count > 1 else np.ones(1)
        return _Timeline(times, float(last / max(out[-1], 1e-12)), float(stride), [], speed)

    def _loop_strike_timeline(self, rate: float, stride: float, strike: dict, force: dict) -> _Timeline:
        """A loop's period warped through its strike: windup -> impact at ``strike_speed``,
        impact -> recover at ``recover_speed``, recover -> the next windup at
        ``windup_speed``; ``tempo`` scales it all, and the period keeps a whole number of
        frames (as tempo alone does).  Output frame 0 samples source frame 0.  The PCHIP
        runs through the nodes of one period plus a neighbour on each side, so its slope
        is the same where consecutive periods meet."""
        frames = self.package.frame_count
        windup, impact, recover = strike["windup"], strike["impact"], strike["recover"]
        zero = windup + (-windup) % frames                      # source frame 0 in [windup, windup + F)
        source = np.array(sorted({windup, impact, recover, zero, windup + frames}), dtype=np.float64)
        length = np.diff(source)
        for k, begin in enumerate(source[:-1]):
            name = ("strike_speed" if windup <= begin < impact else
                    "recover_speed" if impact <= begin < recover else "windup_speed")
            length[k] /= force[name]
        out = np.concatenate([[0.0], np.cumsum(length)]) / rate
        count = max(2, int(round(out[-1])))
        notes = []
        if abs(count - out[-1]) > 1e-9:
            notes.append(f"strike speeds -> {count} frames per loop (from {out[-1]:.2f})")
        out *= count / out[-1]
        period = float(count)
        i_at = lambda value: out[int(np.flatnonzero(source == value)[0])]
        ext_source = np.concatenate([[recover - frames], source, [impact + frames]])
        ext_out = np.concatenate([[i_at(recover) - period], out, [i_at(impact) + period]])
        raw = i_at(zero) + np.arange(count, dtype=np.float64)
        lap = np.floor((raw - out[0]) / period)
        unwrapped = pchip(ext_out, ext_source, raw - lap * period) + lap * frames - zero
        speed = np.gradient(np.concatenate([unwrapped[-1:] - frames, unwrapped, unwrapped[:1] + frames]))[1:-1]
        return _Timeline(np.mod(unwrapped, frames), frames / count, float(stride), notes, speed, unwrapped)

    # ── 2 + 3. layers, amplitude, root ───────────────────────────────────
    def _root_track(self, p: dict) -> np.ndarray:
        """Edited root translation on the source frames."""
        pkg = self.package
        trend = np.asarray(pkg["root_trend"], dtype=np.float64)
        osc = np.asarray(pkg["root_osc"], dtype=np.float64)
        root = trend + osc * np.array([p["sway"], p["bounce"], p["sway"]])
        if p["jump_height"] != 1.0:
            self._scale_airborne(root, trend[:, 1] + osc[:, 1], p["jump_height"])
        root[:, 1] += p["posture"] * self.leg
        return root

    def _scale_airborne(self, root: np.ndarray, source_y: np.ndarray, k: float) -> None:
        """Scale each airborne span's lift off the ground by ``k``, in place (a loop's span
        may wrap the seam).  The lift is the lowest contact joint's height above its floor
        (the lowest height it is planted at in the clip, or the ground when it never is),
        above the line between its values at the last and first planted frames (and above
        the floor); the root moves by ``(k - 1)`` times it.  The lowest joint then stays on
        or above that line and the floor, so lowering never sinks a foot, and the span's
        ends do not move.  Heights are the
        source pose's, carried by the edited root."""
        pkg = self.package
        count = pkg.frame_count
        contacts = np.asarray(pkg["contact_joints"], dtype=np.int64)
        heights = self.original_positions()[:, contacts, 1] + (root[:, 1] - source_y)[:, None]
        mask = np.asarray(pkg["contact_mask"], dtype=bool)
        ground = float(pkg.manifest.get("ground_height", 0.0))
        floor = np.array([heights[mask[:, c], c].min() if mask[:, c].any() else ground
                          for c in range(len(contacts))])
        clearance = (heights - floor[None]).min(axis=1)
        for start, end in pkg.manifest["events"].get("airborne", []):
            take, land = start - 1, end            # last planted frame, first planted frame
            if not pkg.is_loop and (take < 0 or land >= count):
                continue
            frames = np.arange(take, land + 1) % count
            base = np.interp(np.arange(take, land + 1), [take, land],
                             [clearance[frames[0]], clearance[frames[-1]]])
            lift = np.maximum(clearance[frames] - np.maximum(base, 0.0), 0.0)
            root[frames, 1] += (k - 1.0) * lift

    def _compose_layers(self, p: dict, times: np.ndarray, strike: Optional[dict], force: dict):
        pkg = self.package
        periodic = pkg.is_loop
        root = pkg.root
        reference = np.asarray(pkg["chain_reference"], dtype=np.float64)
        rotations = sample_quat(np.asarray(pkg["base_rot"], dtype=np.float64), times, periodic)
        # the sampled rotation's offset, on the branch of the stored unwrapped offsets
        offsets = nearest_rotvec_branch(
            rotvec_from_quat(quat_mul(rotations, quat_inv(reference)[None])),
            sample_linear(np.asarray(pkg["chain_offsets"], dtype=np.float64), times, periodic))
        locked = np.asarray(pkg["chain_gain_locked"], dtype=bool)
        gain = np.ones(pkg.joint_count)
        groups = [str(g) for g in pkg["chain_group"]]
        channels = self.channels()
        for j, group in enumerate(groups):
            if locked[j]:
                continue
            if channels[j]:
                # 0 holds the joint rigid at its reference pose, 1 keeps its own curve
                gain[j] = min(p[f"{channels[j]}_weight"], 1.0)
            elif group in GROUP_GAIN:
                gain[j] = p[GROUP_GAIN[group]]
        offsets = offsets * gain[None, :, None]
        root_track = self._root_track(p)
        tilt = sample_quat(np.asarray(pkg["root_tilt"], dtype=np.float64), times, periodic)
        root_pos = sample_linear(root_track, times, periodic)
        hit = strike is not None and (force["windup_depth"] != 1.0 or force["overshoot"] != 1.0)
        if hit:
            windup, impact = strike["windup"], strike["impact"]
            hold, push = strike_shapes(times, windup, impact, strike["recover"],
                                       pkg.frame_count if periodic else None)
            windup, impact = windup % pkg.frame_count, impact % pkg.frame_count
            g = 1.0 + (force["windup_depth"] - 1.0) * hold         # distance from the contact pose
            b = (force["overshoot"] - 1.0) * push                  # drive past it, along the strike
            trunk, body = self.strike_body(strike)
            # the strike body's own motion: the chain, the trunk it hangs on and the root are
            # drawn further from their contact pose in the windup and driven on past it
            stored = np.asarray(pkg["chain_offsets"], dtype=np.float64)[:, body] * gain[body, None]
            contact, drive = stored[impact], stored[impact] - stored[windup]
            offsets[:, body] = (contact[None] + g[:, None, None] * (offsets[:, body] - contact[None])
                                + b[:, None, None] * drive[None])
            source_tilt = np.asarray(pkg["root_tilt"], dtype=np.float64)
            off_contact = rotvec_from_quat(quat_mul(tilt, quat_inv(source_tilt[impact])[None]))
            tilt_drive = rotvec_from_quat(quat_mul(source_tilt[impact], quat_inv(source_tilt[windup])))
            tilt = quat_mul(quat_from_rotvec(g[:, None] * off_contact + b[:, None] * tilt_drive[None]),
                            source_tilt[impact][None])
            root_pos = (root_track[impact] + g[:, None] * (root_pos - root_track[impact])
                        + b[:, None] * (root_track[impact] - root_track[windup]))
            # and the body's weight: back against the strike in the windup, forward past contact
            lean = -(force["windup_depth"] - 1.0) * hold + b               # units of STRIKE_LEAN / SHIFT
            direction = self.strike_direction(strike)
            root_pos[:, [0, 2]] += (STRIKE_SHIFT * self.leg * lean)[:, None] * direction[None]
        rotations = quat_mul(quat_from_rotvec(offsets), reference[None])
        yaw = sample_angle(np.asarray(pkg["root_yaw"], dtype=np.float64), times, periodic)
        rotations[:, root] = quat_mul(yaw_quat(yaw), tilt)

        positions = sample_linear(np.asarray(pkg["base_pos"], dtype=np.float64), times, periodic)
        positions[:, root] = root_pos
        if hit:
            self._lean(rotations, positions, [root] + trunk, STRIKE_LEAN * lean,
                       self.strike_lean_axis(strike, direction))
        notes = []
        for group in SPREAD_GROUPS:
            if p[f"spread.{group}"] != 0.0:
                notes += self._spread(rotations, positions, group, p[f"spread.{group}"], times)
        held = int(sum(locked[j] and p.get(GROUP_GAIN.get(g, ""), 1.0) != 1.0 for j, g in enumerate(groups)))
        return rotations, positions, held, notes

    def _lean(self, rotations, positions, joints: list[int], angle: np.ndarray, axis: np.ndarray):
        """Turn the body by ``angle`` (F,) rad times ``axis`` (a world rotation vector per
        radian), shared equally by ``joints`` (the root and the trunk above it), in place.
        Each turns about the same world axis, so the turns commute and the trunk ends
        ``angle`` over."""
        parents = np.asarray(self.package["parents"])
        turn = quat_from_rotvec((angle / len(joints))[:, None] * axis[None])
        glob, _ = forward_kinematics(parents, rotations, positions)
        for j in joints:
            parent = int(parents[j])
            if parent < 0:
                rotations[:, j] = quat_mul(turn, rotations[:, j])
            else:
                rotations[:, j] = quat_mul(quat_mul(quat_inv(glob[:, parent]), quat_mul(turn, glob[:, parent])),
                                           rotations[:, j])

    def spread_limbs(self) -> list[SpreadLimb]:
        """The sided limbs spread turns: each IK leg, and each arm hanging straight off the
        trunk (an arms-group joint on a center parent; a sided swing joint on a leg is a
        dewclaw, not an arm)."""
        if self._spread_limbs is None:
            pkg = self.package
            parents = np.asarray(pkg["parents"])
            sides = [str(s) for s in pkg["sides"]]
            groups = [str(g) for g in pkg["chain_group"]]
            names = [str(n) for n in pkg["names"]]
            bone = np.linalg.norm(np.asarray(pkg["anim_offsets"], dtype=np.float64), axis=-1)
            children = [[] for _ in parents]
            for j, parent in enumerate(parents):
                if parent >= 0:
                    children[int(parent)].append(j)

            def path(top: int, tip: int) -> list[int]:
                out = [tip]
                while out[-1] != top:
                    out.append(int(parents[out[-1]]))
                return out[::-1]

            def limb(group, top, tip, columns=()):
                if sides[top] == CENTER_SIDE:
                    return None
                chain = path(top, tip)
                if len(chain) > 2 and (bone[chain[1]] < SPREAD_SHORT_ROOT * bone[chain[1:]].sum()
                                       or joint_name_matches_keywords(names[chain[0]], SPREAD_GIRDLE_KEYWORDS)):
                    chain = chain[1:]
                return SpreadLimb(group, chain[0], tip, 1.0 if sides[top] == "left" else -1.0,
                                  float(bone[chain[1:]].sum()), list(columns))

            out = [limb("legs", leg.root, leg.foot, leg.columns) for leg in self.limbs()[0]]
            for j, group in enumerate(groups):
                parent = int(parents[j])
                if group != "arms" or parent < 0 or sides[parent] != CENTER_SIDE:
                    continue
                arm, stack = [], [j]
                while stack:
                    k = stack.pop()
                    arm.append(k)
                    stack += [c for c in children[k] if groups[c] == "arms"]
                reach = {k: bone[path(j, k)[1:]].sum() for k in arm}
                out.append(limb("arms", j, max(arm, key=lambda k: (reach[k], -k))))
            self._spread_limbs = [l for l in out if l is not None and l.length > 0.0]
        return self._spread_limbs

    def _spread_lateral(self, group: str, times: np.ndarray) -> Optional[np.ndarray]:
        """(F', 3) unit horizontal direction from the group's right limb roots to its left
        ones in the original clip at source ``times``; ``None`` without a pair."""
        if group not in self._lateral:
            limbs = [l for l in self.spread_limbs() if l.group == group]
            left = [l.pivot for l in limbs if l.sign > 0]
            right = [l.pivot for l in limbs if l.sign < 0]
            lateral = None
            if left and right:
                pos = self.original_positions()
                lateral = pos[:, left].mean(axis=1) - pos[:, right].mean(axis=1)
                lateral[:, 1] = 0.0
            self._lateral[group] = lateral
        lateral = self._lateral[group]
        if lateral is None:
            return None
        lateral = sample_linear(lateral, times, self.package.is_loop)
        return lateral / np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-12)

    def _spread(self, rotations, positions, group: str, value: float, times: np.ndarray) -> list[dict]:
        """Turn the group's limbs at their pivots, in place, so each tip moves
        ``value * SPREAD_REACH[group] * length`` along its outward direction.  The turn is
        about the axis normal to the limb and the outward direction, so it swings the tip
        straight toward it; a limb pointing straight out or in moves as far as it can."""
        pkg = self.package
        parents = np.asarray(pkg["parents"])
        name = f"spread.{group}"
        lateral = self._spread_lateral(group, times)
        if lateral is None:
            return [{"kind": "spread", "param": name,
                     "message": f"{name}: no left / right pair of limbs; nothing spread"}]
        limbs = [l for l in self.spread_limbs() if l.group == group]
        glob_rot, glob_pos = forward_kinematics(parents, rotations, positions)
        middle = glob_pos[:, [l.pivot for l in limbs]].mean(axis=1)
        notes = []
        for limb in limbs:
            out = limb.sign * lateral
            v = glob_pos[:, limb.tip] - glob_pos[:, limb.pivot]
            r = np.linalg.norm(v, axis=-1)
            along = np.sum(v * out, axis=-1)
            w = v - along[:, None] * out
            rw = np.linalg.norm(w, axis=-1)
            phi = np.arctan2(along, rw)
            want = (along + value * SPREAD_REACH[group] * limb.length) / np.maximum(r, 1e-12)
            theta = np.where(rw > 1e-9 * limb.length, np.arcsin(np.clip(want, -1.0, 1.0)) - phi, 0.0)
            axis = np.cross(w / np.maximum(rw, 1e-12)[:, None], out)
            turn = quat_from_rotvec(theta[:, None] * axis)
            parent = int(parents[limb.pivot])
            rotations[:, limb.pivot] = quat_mul(
                quat_mul(quat_inv(glob_rot[:, parent]), quat_mul(turn, glob_rot[:, parent])),
                rotations[:, limb.pivot])
            if group == "legs":
                # a foot keeps its world orientation: it stays as flat on the ground as it was
                above = int(parents[limb.tip])
                rotations[:, limb.tip] = quat_mul(quat_inv(quat_mul(turn, glob_rot[:, above])),
                                                  glob_rot[:, limb.tip])
            joint = str(pkg["names"][limb.tip])
            capped = np.abs(want) > 1.0
            if capped.any():
                way = "out" if value > 0 else "in"
                notes.append({"kind": "spread", "param": name, "joint": joint, "frame": int(np.argmax(capped)),
                              "message": f"{name}: {joint} points straight {way} on {int(capped.sum())} "
                                         f"frame(s); spread less than asked"})
            side = np.sum((glob_pos[:, limb.pivot] - middle) * out, axis=-1)
            before, after = side + along, side + np.sum(quat_rotate(turn, v) * out, axis=-1)
            crossed = (after < 0.0) & (before >= 0.0)
            if crossed.any():
                notes.append({"kind": "spread", "param": name, "joint": joint, "frame": int(np.argmax(crossed)),
                              "message": f"{name}: {joint} crosses the body's midline on "
                                         f"{int(crossed.sum())} frame(s)"})
        return notes

    # ── 4. plants ────────────────────────────────────────────────────────
    def _plant_timeline(self, times: np.ndarray):
        """Plant interval and unwrapped source time of every output frame and contact column.

        An output frame between two source frames is planted only when both
        are, on the same interval: a sample half a frame before touchdown is
        still the landing foot.
        """
        pkg = self.package
        frames = pkg.frame_count
        lo, hi, alpha = _bracket(times, frames, pkg.is_loop)
        source = np.asarray(pkg["plant_id"])
        pid = np.where((alpha[:, None] == 0.0) | (source[lo] == source[hi]), source[lo], -1)
        ptime = np.full(pid.shape, np.nan)
        intervals = np.asarray(pkg["plant_intervals"])
        for k in range(pid.shape[1]):
            on = pid[:, k] >= 0
            if not on.any():
                continue
            start = intervals[pid[on, k], 1].astype(np.float64)
            u = times[on].copy()
            if pkg.is_loop:
                u = np.where(u < start - 0.5, u + frames, u)
            ptime[on, k] = u
        return pid, ptime

    def limbs(self):
        """IK limbs of the package's contact set, and notes on the joints none can carry."""
        if self._limbs is None:
            pkg = self.package
            self._limbs = build_limbs(np.asarray(pkg["parents"]), [str(s) for s in pkg["sides"]],
                                      np.asarray(pkg["contact_joints"]), pkg.root)
        return self._limbs

    def _foot_groups(self) -> list[list[int]]:
        """Contact columns that plant together: each limb's, and every unsolvable column alone."""
        limbs, _ = self.limbs()
        groups = [list(limb.columns) for limb in limbs]
        covered = {c for g in groups for c in g}
        groups += [[c] for c in range(len(self.package["contact_joints"])) if c not in covered]
        return groups

    def _column_depth(self) -> np.ndarray:
        """Depth of each contact column below its limb's foot (0 for a column no limb carries)."""
        depth = np.zeros(len(self.package["contact_joints"]), dtype=np.int64)
        for limb in self.limbs()[0]:
            for joint, column in zip(limb.contacts, limb.columns):
                depth[column] = limb.depth[joint]
        return depth

    def _reference_columns(self, pid: np.ndarray) -> np.ndarray:
        """(F, K): the column each planted column's foot is pinned by on that frame, the
        foot's deepest planted contact (the toe a foot rolls off over); -1 off a plant."""
        depth = self._column_depth()
        out = np.full(pid.shape, -1, dtype=np.int64)
        for columns in self._foot_groups():
            columns = np.asarray(columns)
            planted = pid[:, columns] >= 0
            order = np.where(planted, depth[columns][None, :], -1)
            ref = columns[np.argmax(order, axis=1)]
            out[:, columns] = np.where(planted, ref[:, None], -1)
        return out

    def _continuous_displacement(self, times: np.ndarray) -> np.ndarray:
        """``D`` at unbounded source times: a loop keeps accumulating one period's travel per lap."""
        pkg = self.package
        velocity = np.asarray(pkg["ground_velocity"], dtype=np.float64)
        frames = pkg.frame_count
        steps = np.concatenate([np.zeros((1, 2)), np.cumsum(velocity / pkg.fps, axis=0)], axis=0)
        times = np.asarray(times, dtype=np.float64)
        if not pkg.is_loop:
            return sample_linear(steps[:frames], times, False)
        laps = np.floor(times / frames)
        return sample_linear(steps, times - laps * frames, False) + laps[:, None] * steps[frames]

    def _stance_targets(self, pid, ptime, timeline: _Timeline, stride: float, lock: bool,
                        spread: Optional[dict] = None) -> np.ndarray:
        """Where each planted contact joint goes (NaN off a plant).

        ``W = P(u) - (S - 1) (D(U) - D(U_mid)) - lock * residual(u)``: the
        original position at source time ``u``, slid by the extra ground
        travel of stride ``S`` about the middle of the foot's stance, and with
        foot_lock without the slip the original carried.  ``U`` is source
        time unwrapped along the output timeline, and every contact of one
        foot shares the stance middle, so the foot keeps its shape.  In the
        new ground frame (``W + S D``) a plant is its original ground-frame
        position plus a constant: it drifts no more than it did.

        The lock removes the slip of the foot's reference contact from all of
        its planted contacts: a heel peeling up about a planted toe keeps
        the peel and only loses the toe's slip.

        ``spread`` (group -> spread value) moves every contact of a spread limb
        outward, level with the ground, as far as the limb's turn moves its tip, along the outward
        direction at the middle of the stance, held for all of it so the foot does
        not slide.
        """
        pkg = self.package
        periodic = pkg.is_loop
        original = self.original_positions()
        disp = ground_displacement(np.asarray(pkg["ground_velocity"], dtype=np.float64), pkg.fps, periodic)
        anchors = np.asarray(pkg["plant_anchor"], dtype=np.float64)
        contacts = np.asarray(pkg["contact_joints"])
        count = pid.shape[0]
        out = np.full(pid.shape + (3,), np.nan)
        shift = np.zeros(pid.shape + (2,))
        if stride != 1.0:
            for columns in self._foot_groups():
                stance = (pid[:, columns] >= 0).any(axis=1)
                for index in _stance_runs(stance, periodic):
                    frames = index % count
                    # a loop's seam run starts at a negative index; either may be warped
                    unwrapped = (timeline.unwrapped_times()[index % count] + pkg.frame_count * (index // count)
                                 if periodic else timeline.times[index])
                    mid = 0.5 * (unwrapped[0] + unwrapped[-1])
                    d = self._continuous_displacement(unwrapped) - self._continuous_displacement(
                        np.array([mid]))
                    shift[frames[:, None], np.asarray(columns)[None, :]] = (stride - 1.0) * d[:, None, :]
        residual = np.zeros(pid.shape + (3,))
        for k, joint in enumerate(contacts):
            on = pid[:, k] >= 0
            if not on.any():
                continue
            u = ptime[on, k]
            pos = sample_linear(original[:, joint], u, periodic)
            out[on, k] = pos
            out[on, k, 0] -= shift[on, k, 0]
            out[on, k, 2] -= shift[on, k, 1]
            slip = pos - anchors[pid[on, k]]
            slip[:, [0, 2]] += sample_linear(disp, u, False)
            residual[on, k] = slip
        if lock:
            out -= self._lock_slip(pid, residual)
        for group, value in (spread or {}).items():
            lateral = self._spread_lateral(group, timeline.times) if value != 0.0 else None
            if lateral is None:
                continue
            for limb in (l for l in self.spread_limbs() if l.group == group and l.columns):
                stance = (pid[:, limb.columns] >= 0).any(axis=1)
                for index in _stance_runs(stance, periodic):
                    frames = index % count
                    move = (value * SPREAD_REACH[group] * limb.length * limb.sign
                            * lateral[frames[len(frames) // 2]])
                    out[frames[:, None], np.asarray(limb.columns)[None, :]] += move
        return out

    def _lock_slip(self, pid, residual) -> np.ndarray:
        """(F, K, 3) slip foot_lock removes from each planted column.

        Per stance of a foot: its reference contact's residual on the first
        frame, then accumulated frame to frame from the change of the deepest
        contact planted on the same interval on both frames.  The foot keeps
        its original shape and motion about that contact, and a reference
        that hands over (a toe landing, an interval broken for a frame) does
        not jump.
        """
        periodic = self.package.is_loop
        count = pid.shape[0]
        depth = self._column_depth()
        ref = self._reference_columns(pid)
        out = np.zeros(pid.shape + (3,))
        for columns in self._foot_groups():
            columns = np.asarray(columns)
            planted = pid[:, columns] >= 0
            for index in _stance_runs(planted.any(axis=1), periodic):
                frames = index % count
                first = columns[np.argmax(planted[frames[0]])]
                slip = residual[frames[0], ref[frames[0], first]].copy()
                for i, f in enumerate(frames):
                    if i:
                        prev = frames[i - 1]
                        same = planted[f] & (pid[f, columns] == pid[prev, columns])
                        if same.any():
                            c = columns[np.argmax(np.where(same, depth[columns], -1))]
                            slip += residual[f, c] - residual[prev, c]
                    out[f, columns[planted[f]]] = slip
        return out

    def _solve_plants(self, rotations, positions, pid, targets, stretch: float):
        pkg = self.package
        parents = np.asarray(pkg["parents"])
        contacts = np.asarray(pkg["contact_joints"])
        limbs, notes = self.limbs()
        diagnostics = [{"kind": "ik", "message": n} for n in notes]
        profile = {k: pkg.arrays.get(k) for k in ("profile_dof", "profile_axes", "profile_confidence",
                                                   "profile_flex_sign")}
        solvers = [LimbSolver(limb, parents, profile["profile_dof"], profile["profile_axes"],
                              profile["profile_confidence"], profile["profile_flex_sign"]) for limb in limbs]
        positions = np.array(positions, copy=True)
        composed = rotations
        rotations, solved, scale, gaps, pivot = self._solve_limbs(
            solvers, composed, positions, pid, targets, stretch)
        # 6. ground: a planted foot left above its target lowers the body
        planted = np.isfinite(gaps)
        lift = np.where(planted, np.maximum(gaps, 0.0), 0.0)
        if planted.any() and lift.max() > REACH_TOLERANCE * self.leg:
            # ramped so the body eases down and back up rather than dropping on the
            # frames themselves; held across the frames with nothing planted
            shift = np.maximum(ramped_envelope(lift, int(round(GROUND_RAMP * pkg.fps)), pkg.is_loop),
                               blend_between_keys(lift[:, None], planted, pkg.is_loop)[:, 0])
            positions[:, pkg.root, 1] -= shift
            diagnostics.append({"kind": "ground", "frame": int(np.argmax(lift)), "message":
                                f"body lowered by up to {shift.max() / self.leg:.3f} leg so planted feet reach the ground"})
            # re-solved from the composed pose: the first pass's stance misses must not
            # be blended into swing frames that never had them.  The legs keep the first
            # pass's scale: the lowering was measured with it, and a lower body would
            # otherwise ask for less stretch and fall short again
            rotations, solved, scale, gaps, pivot = self._solve_limbs(
                solvers, composed, positions, pid, targets, stretch, scale)
        if scale.size and np.abs(scale - 1.0).max() > 1e-3:
            diagnostics.append({"kind": "stretch", "frame": int(np.argmax(np.abs(scale - 1.0).max(axis=1))),
                                "message": f"leg length scaled {scale.min():.3f} to {scale.max():.3f} (soft_stretch)"})

        _, glob = forward_kinematics(parents, rotations, solved)
        miss = np.where(pivot, np.linalg.norm(glob[:, contacts] - np.nan_to_num(targets), axis=-1), 0.0)
        bad = miss > REACH_TOLERANCE * self.leg
        for k in np.flatnonzero(bad.any(axis=0)):
            frames = np.flatnonzero(bad[:, k])
            diagnostics.append({"kind": "reach", "joint": str(pkg["names"][contacts[k]]),
                                "frame": int(frames[0]), "message":
                                f"{pkg['names'][contacts[k]]}: target out of reach on {len(frames)} frame(s), "
                                f"up to {miss[:, k].max() / self.leg:.3f} leg"})
        return rotations, solved, pivot, bad, diagnostics

    def _solve_limbs(self, solvers, rotations, positions, pid, targets, stretch: float, scale=None):
        """One IK pass over every limb; returns rotations, local positions with the chains
        scaled by soft_stretch (by ``scale`` (F, limbs) when given), the scale, per-frame
        vertical gaps of the planted pivots (NaN where nothing is planted) and the pivot mask."""
        pkg = self.package
        parents = np.asarray(pkg["parents"])
        frames = rotations.shape[0]
        pivot = np.zeros(pid.shape, dtype=bool)
        gaps = np.full(frames, np.nan)
        rotations = np.array(rotations, copy=True)
        positions = np.array(positions, copy=True)
        fixed = scale is not None
        scale = np.array(scale, copy=True) if fixed else np.ones((frames, len(solvers)))
        for index, solver in enumerate(solvers):
            limb = solver.limb
            glob_rot, glob_pos = forward_kinematics(parents, rotations, positions)
            planted = pid[:, limb.columns] >= 0                            # (F, n)
            if not planted.any():
                continue
            foot_rot = glob_rot[:, limb.foot]
            foot_pos = glob_pos[:, limb.foot]
            depth = np.array([limb.depth[j] for j in limb.contacts])
            # pivot: the deepest planted contact, the one foot_lock pins (_reference_columns);
            # distal: the planted one farthest from it
            order = np.where(planted, depth[None, :], -1)
            p_idx = np.argmax(order, axis=1)
            stance = planted.any(axis=1)
            rows = np.arange(frames)
            p_joint = np.asarray(limb.contacts)[p_idx]
            p_col = np.asarray(limb.columns)[p_idx]
            p_edit = glob_pos[rows, p_joint]
            p_target = np.where(stance[:, None], targets[rows, p_col], p_edit)
            spread = np.where(planted[..., None],
                              glob_pos[:, limb.contacts] - p_edit[:, None], 0.0)
            d_idx = np.argmax(np.linalg.norm(spread, axis=-1), axis=1)
            d_joint = np.asarray(limb.contacts)[d_idx]
            d_col = np.asarray(limb.columns)[d_idx]
            d_target = targets[rows, d_col]
            has_distal = stance & (d_idx != p_idx) & np.isfinite(d_target).all(axis=1)
            align = rotation_between(glob_pos[rows, d_joint] - p_edit,
                                     np.where(has_distal[:, None], d_target - p_target, 0.0))
            align = np.where(has_distal[:, None], align, np.array([1.0, 0.0, 0.0, 0.0]))

            # correction keyed on stance frames, blended across the swing between them
            shift = p_target - quat_rotate(align, p_edit - foot_pos) - foot_pos
            turn = rotvec_from_quat(align)
            correction = blend_between_keys(np.concatenate([shift, turn], axis=1), stance, pkg.is_loop)
            if np.abs(correction).max() <= 1e-12:
                scale[:, index] = 1.0
                pivot[stance, p_col[stance]] = True
                gaps[stance] = np.fmax(gaps[stance], 0.0)
                continue
            foot_target = foot_pos + correction[:, :3]
            foot_world = quat_mul(quat_from_rotvec(correction[:, 3:]), foot_rot)

            chain = limb.chain
            parent = int(parents[chain[0]])
            # soft_stretch: the bones below the limb root lengthen toward a target the leg
            # could only reach straight, and shorten toward one it could only reach deeply
            # bent; keyed on stance frames, blended across the swing like the correction, and
            # eased in and out like the ground lowering (a leg longer or shorter than it must be
            # still reaches: the knee takes up the difference)
            bones = chain[1:] + [limb.foot]
            if not fixed:
                length = np.maximum(np.linalg.norm(positions[:, bones], axis=-1).sum(axis=1), 1e-12)
                hip = glob_pos[:, chain[0]]
                keyed = soft_scale(np.linalg.norm(foot_target - hip, axis=-1) / length,
                                   np.linalg.norm(foot_pos - hip, axis=-1) / length, stretch)
                keyed = blend_between_keys(keyed[:, None], stance, pkg.is_loop)[:, 0] - 1.0
                ramp = int(round(GROUND_RAMP * pkg.fps))
                scale[:, index] = (1.0 + ramped_envelope(np.maximum(keyed, 0.0), ramp, pkg.is_loop)
                                   - ramped_envelope(np.maximum(-keyed, 0.0), ramp, pkg.is_loop))
            positions[:, bones] *= scale[:, index, None, None]
            chain_rot, _ = solver.solve(glob_rot[:, parent], glob_pos[:, parent], rotations[:, chain],
                                        positions[:, chain + [limb.foot]], foot_target, self.leg)
            rotations[:, chain] = chain_rot
            glob_rot, glob_pos = forward_kinematics(parents, rotations, positions)
            rotations[:, limb.foot] = world_to_local(glob_rot[:, int(parents[limb.foot])], foot_world)
            glob_rot, glob_pos = forward_kinematics(parents, rotations, positions)
            pivot[stance, p_col[stance]] = True
            gap = glob_pos[rows, p_joint, 1] - p_target[:, 1]
            gaps[stance] = np.fmax(gaps[stance], gap[stance])
        return rotations, positions, scale, gaps, pivot

    # ── 5. secondary motion ──────────────────────────────────────────────
    def channels(self) -> list[str]:
        """Per joint, the channel of the hanging part it belongs to ("" for none): passive
        joints ``passive``; the tail's joints that can swing (secondary-motion candidates,
        closed downward) ``tail``."""
        if self._channels is None:
            pkg = self.package
            parents = np.asarray(pkg["parents"])
            passive = np.asarray(pkg.arrays.get("profile_passive", np.zeros(pkg.joint_count, dtype=bool)),
                                 dtype=bool)
            candidates = set((pkg.manifest.get("passive") or {}).get("candidates", []))
            groups = [str(g) for g in pkg["chain_group"]]
            out = [""] * pkg.joint_count
            for j in range(pkg.joint_count):          # parents precede children
                if passive[j]:
                    out[j] = "passive"
                elif j in candidates and (groups[j] == "tail" or (parents[j] >= 0 and out[parents[j]] == "tail")):
                    out[j] = "tail"
            self._channels = out
        return self._channels

    def springs(self) -> tuple[list[dict], list[str]]:
        """The swinging bones the runtime simulates, parents first, and notes on what it cannot.

        A joint ``X`` of a hanging part is the end of a bone that swings, so ``X`` itself moves:
        * ``virtual``: the top of a part, whose parent is outside it, hinges on a virtual
          joint at that parent: the swing turns ``X``'s local translation and rotation about
          the parent (default spring: a rod from the parent to the part's tip);
        * otherwise a joint of the part with children turns its own rotation, swinging them
          (the profile's spring for the joint).
        A leaf has no bone below it to swing; the bone above it does.  Each entry: ``joint``,
        ``channel`` (its part's channel), ``virtual``, ``bone`` (the lever in the hinge frame) and
        ``spring`` (``profile_spring`` layout)."""
        if self._springs is None:
            pkg = self.package
            channels = self.channels()
            table = pkg.arrays.get("profile_spring")
            parents = np.asarray(pkg["parents"])
            names = pkg["names"]
            reference = np.asarray(pkg["chain_reference"], dtype=np.float64)
            mean_offset = np.asarray(pkg["base_pos"], dtype=np.float64).mean(axis=0)
            length = np.linalg.norm(mean_offset, axis=-1)
            hang = np.zeros(pkg.joint_count)           # longest rest path below each joint
            for j in range(pkg.joint_count - 1, -1, -1):
                if parents[j] >= 0:
                    hang[parents[j]] = max(hang[parents[j]], length[j] + hang[j])

            def default(rod: float) -> np.ndarray:
                spring = default_spring(rod, self.leg)
                return np.array([spring[k] for k in ("k", "c", "g_ang", "g_lin", "g_grav")])

            out, notes = [], []
            for j, channel in enumerate(channels):
                if not channel or j == pkg.root:
                    continue
                if channels[parents[j]] != channel:
                    if length[j] <= 1e-9:
                        notes.append(f"{names[j]}: sits on its parent; its part does not swing there")
                    else:
                        out.append({"joint": j, "channel": channel, "virtual": True,
                                    "bone": mean_offset[j] / length[j], "spring": default(length[j] + hang[j])})
                children = np.flatnonzero(parents == j)
                if not children.size or length[children].max() <= 1e-9:
                    continue
                spring = None if table is None else np.asarray(table[j], dtype=np.float64)
                if spring is None or not np.isfinite(spring).all() or spring[SPRING_K] <= 0.0:
                    spring = default(hang[j])
                bone = quat_rotate(reference[j], mean_offset[children[int(np.argmax(length[children]))]])
                out.append({"joint": j, "channel": channel, "virtual": False,
                            "bone": bone / np.linalg.norm(bone), "spring": spring})
            self._springs = (out, notes)
        return self._springs

    @staticmethod
    def _spring_drive(spring: dict, hinge_rot, hinge_pos, fps: float, periodic: bool,
                      gravity: bool) -> np.ndarray:
        """The right-hand side of the spring model (its constant term left out; without
        ``gravity`` its gravity term too)."""
        alpha, lin, grav = drive_terms(hinge_rot, hinge_pos, spring["bone"], fps, periodic)
        gains = spring["spring"]
        drive = gains[SPRING_G_ANG] * alpha + gains[SPRING_G_LIN] * lin
        if gravity:
            drive = drive + gains[SPRING_G_GRAV] * grav
        return -drive

    def _secondary(self, weights: dict, stiffness: dict, gravity: bool, rotations, positions):
        """Every swinging bone keeps its own curve and swings its channel's ``weights`` times
        its spring's response to the motion of its hinge, as edited (a loop's response is
        periodic); the channel's ``stiffness`` multiplies the spring's natural frequency, its
        damping ratio kept.  A part whose weight is not past 1 does not swing.  ``gravity``
        keeps the springs' gravity term (a fitted spring's sag as its hinge tilts).

        A virtual hinge sits at the part's parent and moves with it; a joint's hinge is the
        joint, in its parent frame (turned by the virtual swing above it, if any).  Bones run
        parents first, so one further down is driven by the full swing above it; only then
        is every swing scaled by its weight, so the result grows in proportion to the slider
        rather than compounding down the chain."""
        pkg = self.package
        parents = np.asarray(pkg["parents"])
        springs, notes = self.springs()
        diagnostics = [{"kind": "secondary", "message": n} for n in notes]
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        full_rot, full_pos = np.array(rotations, copy=True), np.array(positions, copy=True)
        out_rot, out_pos = np.array(rotations, copy=True), np.array(positions, copy=True)
        glob_rot, glob_pos = forward_kinematics(parents, full_rot, full_pos)
        virtual_full, virtual_out = {}, {}
        count, peak = {}, {}
        for spring in springs:
            channel = spring["channel"]
            weight = weights.get(channel, 0.0)
            if weight <= 0.0:
                continue
            j = spring["joint"]
            parent = int(parents[j])
            if spring["virtual"]:
                hinge_rot, hinge_pos = glob_rot[:, parent], glob_pos[:, parent]
            else:
                hinge_rot = quat_mul(glob_rot[:, parent], virtual_full.get(j, identity))
                hinge_pos = glob_pos[:, j]
            drive = self._spring_drive(spring, hinge_rot, hinge_pos, pkg.fps, pkg.is_loop, gravity)
            hard = stiffness[spring["channel"]]
            response = spring_response(drive, hard * hard * spring["spring"][SPRING_K],
                                       hard * spring["spring"][SPRING_C], pkg.fps, pkg.is_loop)
            for scale, rot, pos, virtual in ((1.0, full_rot, full_pos, virtual_full),
                                             (weight, out_rot, out_pos, virtual_out)):
                turn = quat_from_rotvec(scale * response)
                if spring["virtual"]:
                    # the virtual joint at the parent turns this joint's bone and frame
                    pos[:, j] = quat_rotate(turn, pos[:, j])
                    rot[:, j] = quat_mul(turn, rot[:, j])
                    virtual[j] = turn
                else:
                    # in the hinge frame: inside the virtual turn this joint already took
                    above = virtual.get(j, identity)
                    rot[:, j] = quat_mul(quat_mul(above, quat_mul(turn, quat_inv(above))), rot[:, j])
            # only j's subtree moved: the hinges further down read it
            _refresh_fk(parents, full_rot, full_pos, glob_rot, glob_pos, self._subtree(j))
            angle = np.linalg.norm(weight * response, axis=-1)
            count[channel] = count.get(channel, 0) + 1
            if channel not in peak or angle.max() > peak[channel][0]:
                peak[channel] = (float(angle.max()), j, int(np.argmax(angle)))
        for channel, (best, j, frame) in peak.items():
            name = str(pkg["names"][j])
            diagnostics.append({"kind": "secondary", "joint": name, "frame": frame, "message":
                                f"{channel}_weight: {count[channel]} swinging bone(s), up to "
                                f"{np.degrees(best):.1f} deg added ({name})"})
        return out_rot, out_pos, diagnostics

    def _subtree(self, j: int) -> list[int]:
        """``j`` and its descendants in index order (parents precede children)."""
        if self._subtrees is None:
            parents = np.asarray(self.package["parents"])
            below = [[k] for k in range(len(parents))]
            for k in range(len(parents) - 1, 0, -1):
                if parents[k] >= 0:
                    below[parents[k]] += below[k]
            self._subtrees = [sorted(s) for s in below]
        return self._subtrees[j]
