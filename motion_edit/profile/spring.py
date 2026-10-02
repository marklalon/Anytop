"""Secondary-motion springs (section 3.5): the data fit and the default spring.

Model, shared across the three axes of the joint's parent frame::

    θ̈ + c·θ̇ + k·θ = −g_ang·α_p − g_lin·(b × a_p) − g_grav·(b × ĝ_p) + bias

``θ`` is the joint's rotation vector away from its mean pose, ``α_p`` / ``a_p``
the parent's world angular / linear acceleration and ``ĝ_p`` world down, all
expressed in the parent frame, ``b`` the joint's mean bone direction.  An
undriven oscillator fits any sinusoid, so the fit is used only when the
driving terms buy a clear, significant gain in held-out R²; otherwise the
joint gets :func:`default_spring`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from motion_edit.rotations import quat_inv, quat_mul, quat_rotate, rotvec_from_quat

SG_WINDOW = 9
SG_ORDER = 3
HOLDOUT = 0.2               # trailing fraction of every clip
MIN_R2 = 0.6
MIN_DELTA_R2 = 0.2
MIN_T = 3.0
FREQ_RANGE_HZ = (0.2, 8.0)  # natural frequency sqrt(k) / 2π

# Default spring of a confirmed joint whose fit did not pass: a hanging chain
# swings slower the longer it is (pendulum, f ~ 1/sqrt(length)).
DEFAULT_HZ_AT_LEG_LENGTH = 1.5   # natural frequency of a chain one leg length long
DEFAULT_HZ_RANGE = (0.5, 4.0)
DEFAULT_DAMPING_RATIO = 0.3


@dataclass
class SpringFit:
    k: float
    c: float
    g_ang: float
    g_lin: float
    g_grav: float
    r2: float
    delta_r2: float
    max_drive_t: float
    natural_hz: float
    frames: int
    passed: bool
    reason: str

    def as_json(self) -> dict:
        return {key: (round(v, 6) if isinstance(v, float) else v)
                for key, v in self.__dict__.items()}


def default_spring(swing_length: float, leg_length: float) -> dict:
    """Generic spring for a chain of ``swing_length`` hanging below the joint.

    The drive gains are those of a uniform rod pivoting at the joint: it lags
    its parent's rotation one to one (``g_ang = 1``) and a sideways push at the
    pivot turns it by ``3 / (2 L)`` (``g_lin``).  Gravity is left out: the
    keyed pose already holds the chain where the animator wanted it, and the
    runtime only adds the response to an edit.
    """
    length = max(float(swing_length), 1e-6)
    ratio = length / leg_length if leg_length > 0 else 1.0
    hz = float(np.clip(DEFAULT_HZ_AT_LEG_LENGTH / np.sqrt(ratio), *DEFAULT_HZ_RANGE))
    k = (2.0 * np.pi * hz) ** 2
    return {
        "source": "default",
        "k": round(k, 6),
        "c": round(2.0 * DEFAULT_DAMPING_RATIO * np.sqrt(k), 6),
        "g_ang": 1.0,
        "g_lin": round(1.5 / length, 6),
        "g_grav": 0.0,
        "natural_hz": round(hz, 4),
        "damping_ratio": DEFAULT_DAMPING_RATIO,
    }


def fitted_spring(fit: "SpringFit") -> dict:
    """The fit's parameters in the runtime's layout."""
    damping_ratio = fit.c / (2.0 * np.sqrt(fit.k)) if fit.k > 0 else 0.0
    return {
        "source": "fit",
        "k": round(fit.k, 6), "c": round(fit.c, 6),
        "g_ang": round(fit.g_ang, 6), "g_lin": round(fit.g_lin, 6), "g_grav": round(fit.g_grav, 6),
        "natural_hz": round(fit.natural_hz, 4), "damping_ratio": round(float(damping_ratio), 4),
    }


def _sg(x: np.ndarray, deriv: int, fps: float, periodic: bool) -> np.ndarray:
    return savgol_filter(x, SG_WINDOW, SG_ORDER, deriv=deriv, delta=1.0 / fps,
                         axis=0, mode="wrap" if periodic else "interp")


def _angular_velocity(rot: np.ndarray, fps: float, periodic: bool) -> np.ndarray:
    """World angular velocity (F, 3) from global rotations by central differences."""
    if periodic:
        nxt, prv = np.roll(rot, -1, axis=0), np.roll(rot, 1, axis=0)
        return rotvec_from_quat(quat_mul(nxt, quat_inv(prv))) * (0.5 * fps)
    omega = np.zeros(rot.shape[:-1] + (3,))
    omega[1:-1] = rotvec_from_quat(quat_mul(rot[2:], quat_inv(rot[:-2]))) * (0.5 * fps)
    omega[0] = rotvec_from_quat(quat_mul(rot[1:2], quat_inv(rot[:1])))[0] * fps
    omega[-1] = rotvec_from_quat(quat_mul(rot[-1:], quat_inv(rot[-2:-1])))[0] * fps
    return omega


def clip_rows(theta: np.ndarray, parent_rot: np.ndarray, parent_pos: np.ndarray,
              bone_dir: np.ndarray, fps: float, periodic: bool):
    """Regression rows ``(y, X)`` of one clip, axes stacked; None if too short."""
    frames = theta.shape[0]
    if frames < 2 * SG_WINDOW:
        return None
    theta_s = _sg(theta, 0, fps, periodic)
    theta_d = _sg(theta, 1, fps, periodic)
    theta_dd = _sg(theta, 2, fps, periodic)
    inv_parent = quat_inv(parent_rot)
    alpha = _sg(_angular_velocity(parent_rot, fps, periodic), 1, fps, periodic)
    alpha_local = quat_rotate(inv_parent, alpha)
    accel_local = quat_rotate(inv_parent, _sg(parent_pos, 2, fps, periodic))
    down_local = quat_rotate(inv_parent, np.array([0.0, -1.0, 0.0]))
    lin = np.cross(bone_dir, accel_local)
    grav = np.cross(bone_dir, down_local)
    cols = [theta_s, theta_d, alpha_local, lin, grav]
    y = theta_dd.T.reshape(-1)                                   # axis-major stacking
    x = np.stack([c.T.reshape(-1) for c in cols] + [np.ones(3 * frames)], axis=1)
    split = np.tile(np.arange(frames) >= int(round(frames * (1.0 - HOLDOUT))), 3)
    return y, x, split


def _r2(y, pred) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / sst if sst > 0 else 0.0


def fit_spring(rows: list) -> Optional[SpringFit]:
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    y = np.concatenate([r[0] for r in rows])
    x = np.concatenate([r[1] for r in rows])
    test = np.concatenate([r[2] for r in rows])
    train = ~test
    if train.sum() < 50 or test.sum() < 10:
        return None

    full = [0, 1, 2, 3, 4, 5]
    reduced = [0, 1, 5]
    coef, *_ = np.linalg.lstsq(x[train][:, full], y[train], rcond=None)
    coef_r, *_ = np.linalg.lstsq(x[train][:, reduced], y[train], rcond=None)
    r2 = _r2(y[test], x[test][:, full] @ coef)
    r2_reduced = _r2(y[test], x[test][:, reduced] @ coef_r)

    resid = y[train] - x[train][:, full] @ coef
    dof = max(1, int(train.sum()) - len(full))
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(x[train][:, full].T @ x[train][:, full])
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 1e-30))
    t = np.abs(coef / se)

    k, c = -coef[0], -coef[1]
    natural = float(np.sqrt(k) / (2 * np.pi)) if k > 0 else 0.0
    delta = r2 - r2_reduced
    max_t = float(t[2:5].max())
    reasons = []
    if r2 < MIN_R2:
        reasons.append(f"R2 {r2:.2f} < {MIN_R2}")
    if delta < MIN_DELTA_R2:
        reasons.append(f"drive gain {delta:.2f} < {MIN_DELTA_R2}")
    if max_t < MIN_T:
        reasons.append(f"drive |t| {max_t:.1f} < {MIN_T}")
    if not (FREQ_RANGE_HZ[0] <= natural <= FREQ_RANGE_HZ[1]):
        reasons.append(f"natural {natural:.2f} Hz outside {FREQ_RANGE_HZ}")
    if c < 0:
        reasons.append("negative damping")
    return SpringFit(
        k=float(k), c=float(c), g_ang=float(-coef[2]), g_lin=float(-coef[3]),
        g_grav=float(-coef[4]), r2=float(r2), delta_r2=float(delta), max_drive_t=max_t,
        natural_hz=natural, frames=int(len(y) // 3), passed=not reasons,
        reason="; ".join(reasons) if reasons else "ok",
    )
