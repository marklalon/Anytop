"""EditRuntime: Edit Package + parameters -> Animation (section 5).

One fixed, versioned composition shared by the tuning UI and
``motion_edit.apply_edit``: identical package and parameters give
bit-identical output.  Every layer whose parameters sit at their defaults is
skipped, so the all-default result *is* the decoded generation (the strict
replay of section 1, principle 4).

Order (section 5.2):

1. time: every output frame samples the clip at a source time ``s``; a loop
   keeps a whole number of frames per period and wraps its samples;
2. amplitude: chain offsets scaled per group (frames near pi held at gain 1);
3. root: oscillation (sway, bounce), airborne arcs (jump_height), posture;
4. plants + limb IK: planted feet follow their targets, swing feet carry the
   correction between them (``motion_edit.ik``);
6. ground: a planted foot IK could not bring down lowers the body instead.

The runtime imports neither torch nor the decode path; everything it needs
is in the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from motion_edit.ik import LimbSolver, build_limbs, world_to_local
from motion_edit.package import CHAIN_GROUPS, NEAR_PI, EditPackage
from motion_edit.rotations import (
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


AMP_GROUPS = tuple(g for g in CHAIN_GROUPS if g not in ("root", "other"))
_AMP = {f"amp.{g}": ParamSpec(1.0, 0.0, 2.0, "amp") for g in AMP_GROUPS}

# The v1 parameter set (section 5.1), in panel order (grouped).
PARAM_SPECS: dict[str, ParamSpec] = {
    "tempo": ParamSpec(1.0, 0.5, 2.0, "time"),
    "stride": ParamSpec(1.0, 0.6, 1.6, "locomotion"),
    "bounce": ParamSpec(1.0, 0.0, 2.0, "root"),
    "jump_height": ParamSpec(1.0, 0.5, 1.8, "root"),
    "sway": ParamSpec(1.0, 0.0, 2.0, "root"),
    "posture": ParamSpec(0.0, -0.3, 0.2, "root"),
    **_AMP,
    "force": ParamSpec(1.0, 0.5, 2.0, "force"),
    "windup_depth": ParamSpec(1.0, 0.0, 2.0, "force"),
    "strike_speed": ParamSpec(1.0, 0.5, 2.0, "force"),
    "overshoot": ParamSpec(1.0, 0.0, 2.0, "force"),
    "impact_shift": ParamSpec(0.0, -0.3, 0.3, "force"),
    "secondary.stiffness": ParamSpec(1.0, 0.25, 4.0, "secondary"),
    "secondary.damping": ParamSpec(1.0, 0.25, 4.0, "secondary"),
    "foot_lock": ParamSpec(False, group="contact", toggle=True),
    "soft_stretch": ParamSpec(0.1, 0.0, 0.2, "contact"),
}

# Parameters whose non-default values this runtime version composes.  A
# non-default value of any other parameter is refused, never ignored.
IMPLEMENTED: frozenset[str] = frozenset(
    ["tempo", "stride", "bounce", "jump_height", "sway", "posture", "foot_lock", "soft_stretch",
     *_AMP])

# Parameters that move the body or a support chain: any of them away from its
# default re-solves the planted limbs.  tempo alone only re-times the clip.
_IK_PARAMS = ("stride", "bounce", "jump_height", "sway", "posture", *_AMP)

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


class UnsupportedParameterError(ValueError):
    pass


def available_params(package_facts: dict) -> list[str]:
    """Parameters that mean something for this clip (the UI hides the rest).

    ``package_facts``: ``is_loop``, ``locomotion``, ``has_plants``, ``turning``,
    ``airborne``, ``has_passive`` and ``chain_groups`` (groups with joints).
    """
    out = ["tempo", "bounce", "sway"]
    if package_facts["locomotion"] and package_facts["has_plants"] and not package_facts["turning"]:
        out.append("stride")
    if package_facts["airborne"] and not package_facts["is_loop"]:
        out.append("jump_height")
    out += [f"amp.{g}" for g in AMP_GROUPS if g in package_facts["chain_groups"]]
    if package_facts["has_plants"]:
        out.append("posture")
    if not package_facts["is_loop"]:
        out += ["force", "windup_depth", "strike_speed", "overshoot", "impact_shift"]
    if package_facts["has_passive"]:
        out += ["secondary.stiffness", "secondary.damping"]
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
    diagnostics: list = field(default_factory=list)


@dataclass
class _Timeline:
    times: np.ndarray       # source frame per output frame
    rate: float             # source frames per output frame
    stride: float           # S
    notes: list


class EditRuntime:
    def __init__(self, package: EditPackage):
        self.package = package
        # from the package facts, so a package built before a parameter existed still offers it
        facts = package.manifest.get("facts")
        self.available = set(available_params(facts) if facts else package.manifest.get("available_params", []))
        self._original = None
        self._limbs = None

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

    def original_positions(self) -> np.ndarray:
        if self._original is None:
            _, self._original = forward_kinematics(
                self.package["parents"], np.asarray(self.package["base_rot"], dtype=np.float64),
                np.asarray(self.package["base_pos"], dtype=np.float64))
        return self._original

    # ── composition ──────────────────────────────────────────────────────
    def apply(self, params: dict | None = None, *, compose: bool = False) -> EditResult:
        """Compose the clip under ``params``.

        ``compose=True`` disables the all-default shortcut and rebuilds the
        clip from its layers with unit gains (T1b: it exercises the
        decomposition the shortcut never reads).
        """
        resolved, diagnostics = self.resolve_params(params)
        is_default = all(resolved[n] == PARAM_SPECS[n].default for n in PARAM_SPECS)
        pkg = self.package
        if is_default and not compose:
            timeline = _Timeline(np.arange(pkg.frame_count, dtype=np.float64), 1.0, 1.0, [])
            pid, ptime = self._plant_timeline(timeline.times)
            return self._result(
                np.asarray(pkg["base_rot"]), np.asarray(pkg["base_pos"]), resolved, False, timeline,
                pid, ptime, self._stance_targets(pid, ptime, 1.0, 1.0, False),
                np.zeros(pid.shape, dtype=bool), diagnostics)

        timeline = self._timeline(resolved)
        diagnostics += [{"kind": "tempo", "message": m} for m in timeline.notes]
        rotations, positions, near_pi = self._compose_layers(resolved, timeline.times)
        if near_pi:
            diagnostics.append({"kind": "near_pi", "message":
                                f"chain offset near pi on {near_pi} joint-frame(s); gain held at 1"})
        pid, ptime = self._plant_timeline(timeline.times)
        pivot = np.zeros(pid.shape, dtype=bool)
        unreached = None
        if any(resolved[n] != PARAM_SPECS[n].default for n in _IK_PARAMS) or resolved["foot_lock"]:
            targets = self._stance_targets(pid, ptime, timeline.rate, timeline.stride,
                                           bool(resolved["foot_lock"]))
            rotations, positions, pivot, unreached, ik_notes = self._solve_plants(
                rotations, positions, pid, targets, resolved["soft_stretch"])
            diagnostics += ik_notes
        else:
            targets = self._stance_targets(pid, ptime, timeline.rate, 1.0, False)
        return self._result(rotations, positions, resolved, True, timeline, pid, ptime, targets, pivot,
                            diagnostics, unreached)

    def _result(self, rotations, positions, resolved, composed, timeline, pid, ptime, targets, pivot,
                diagnostics, unreached=None) -> EditResult:
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
            ground_velocity=ground * timeline.stride * timeline.rate,
            composed=composed,
            params=resolved,
            source_time=timeline.times,
            plant_id=pid,
            plant_time=ptime,
            plant_target=targets,
            pivot=pivot,
            unreached=unreached if unreached is not None else np.zeros(pid.shape, dtype=bool),
            stride_factor=timeline.stride,
            diagnostics=diagnostics,
        )

    # ── 1. time ──────────────────────────────────────────────────────────
    def _timeline(self, p: dict) -> _Timeline:
        pkg = self.package
        frames = pkg.frame_count
        rate = p["tempo"]
        stride = p["stride"]
        notes = []
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

    # ── 2 + 3. layers, amplitude, root ───────────────────────────────────
    def _root_track(self, p: dict) -> np.ndarray:
        """Edited root translation on the source frames."""
        pkg = self.package
        trend = np.asarray(pkg["root_trend"], dtype=np.float64)
        osc = np.asarray(pkg["root_osc"], dtype=np.float64)
        root = trend + osc * np.array([p["sway"], p["bounce"], p["sway"]])
        if p["jump_height"] != 1.0:
            for start, end in pkg.manifest["events"].get("airborne", []):
                take, land = start - 1, end            # last planted frame, first planted frame
                if take < 0 or land >= pkg.frame_count:
                    continue
                frames = np.arange(take, land + 1)
                base = np.interp(frames, [take, land], [root[take, 1], root[land, 1]])
                root[frames, 1] = base + p["jump_height"] * (root[frames, 1] - base)
        root[:, 1] += p["posture"] * self.leg
        return root

    def _compose_layers(self, p: dict, times: np.ndarray):
        pkg = self.package
        periodic = pkg.is_loop
        root = pkg.root
        reference = np.asarray(pkg["chain_reference"], dtype=np.float64)
        rotations = sample_quat(np.asarray(pkg["base_rot"], dtype=np.float64), times, periodic)
        offsets = rotvec_from_quat(quat_mul(rotations, quat_inv(reference)[None]))
        near_pi = np.linalg.norm(offsets, axis=-1) > NEAR_PI
        gain = np.ones(pkg.joint_count)
        groups = [str(g) for g in pkg["chain_group"]]
        for j, group in enumerate(groups):
            if group in AMP_GROUPS:
                gain[j] = p[f"amp.{group}"]
        gain = np.where(near_pi, 1.0, gain[None, :])
        rotations = quat_mul(quat_from_rotvec(offsets * gain[..., None]), reference[None])
        yaw = sample_angle(np.asarray(pkg["root_yaw"], dtype=np.float64), times, periodic)
        tilt = sample_quat(np.asarray(pkg["root_tilt"], dtype=np.float64), times, periodic)
        rotations[:, root] = quat_mul(yaw_quat(yaw), tilt)

        positions = sample_linear(np.asarray(pkg["base_pos"], dtype=np.float64), times, periodic)
        positions[:, root] = sample_linear(self._root_track(p), times, periodic)
        flagged = int((near_pi & (gain == 1.0) & (np.asarray(
            [p.get(f"amp.{g}", 1.0) for g in groups])[None] != 1.0)).sum())
        return rotations, positions, flagged

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

    def _stance_targets(self, pid, ptime, rate: float, stride: float, lock: bool) -> np.ndarray:
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
                    unwrapped = index * rate
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
        if np.abs(scale - 1.0).max() > 1e-3:
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
