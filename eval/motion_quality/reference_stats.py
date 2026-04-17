"""
Channel layout constants and low-level motion utilities.

Motion format  (T × J × 13  float32, normalised):
    ch 0-2  : local RIC position
    ch 3-8  : 6-D rotation  (two packed 3-D unit-norm vectors)
    ch 9-11 : linear velocity
    ch 12   : foot-contact flag
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Channel layout (13 features per joint)
# ---------------------------------------------------------------------------
CH_POS   = slice(0, 3)
CH_ROT   = slice(3, 9)
CH_ROT_A = slice(3, 6)   # first 6D basis vector
CH_ROT_B = slice(6, 9)   # second 6D basis vector
CH_VEL   = slice(9, 12)
CH_CONT  = 12

_ROOT_JERK_WEIGHT = 5.0   # root joint upweight in jerk aggregation


def _compute_normalised_jerk(motion: np.ndarray, ch: slice) -> float:
    """Root-weighted jerk normalised by per-joint activity (variance).

    Dividing per-joint jerk by per-joint variance produces a scale-invariant
    *relative jitter* metric: a clip with large legitimate motion is not
    penalised more than one with tiny motion and tiny jerk.

    For near-static joints (variance ≈ 0) the normalisation clamps to a safe
    value so idle joints do not dominate the score.
    """
    r = motion[:, :, ch]                              # (T, J, C)
    a = np.diff(np.diff(r, axis=0), axis=0)
    j = np.diff(a, axis=0)                            # (T-3, J, C)

    per_joint_jerk = (j ** 2).mean(axis=(0, 2))       # (J,)
    per_joint_var  = r.var(axis=0).mean(axis=-1)      # (J,)  activity level

    normalised = per_joint_jerk / (per_joint_var + 1e-10)

    J = normalised.shape[0]
    weights = np.ones(J, dtype=np.float64)
    weights[0] = _ROOT_JERK_WEIGHT
    weights /= weights.sum()
    return float((normalised * weights).sum())
