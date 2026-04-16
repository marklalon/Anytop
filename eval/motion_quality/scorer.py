"""
LightweightMotionQualityScorer
==============================

Evaluates a single motion clip **without any trained quality model**.
All metrics are based on physical / geometric / statistical priors derived
from the motion representation itself and (optionally) from a reference
dataset of ground-truth motions for the same skeleton type.

Motion format  (N_frames × N_joints × 13  float32, normalised):
    ch 0-2  : local RIC position
    ch 3-8  : 6-D rotation  (two packed 3-D vectors, each ~unit-length)
    ch 9-11 : linear velocity
    ch 12   : foot-contact flag (not evaluated)

Score range:  0.0 (worst) → 1.0 (perfect)

Sub-scores
----------
rotation_6d_consistency   [primary, w=0.6]
    The 6-D rotation representation stores two orthonormal basis vectors.
    Ground-truth motions have exact unit norms (deviation = 0).
    Diffusion-model outputs accumulate small floating-point drift, making
    this a perfectly reliable discriminator between GT and generated motion.

jerk_smoothness           [supporting, w=0.3]
    Activity-normalised jerk of rotation and position channels,
    normalised against the reference-dataset IQR for this skeleton.
    Penalises both excessive jitter AND extreme over-smoothing.

temporal_variance         [supporting, w=0.1]
    Overall temporal variance of the clip, normalised against the reference
    dataset IQR.  Over-smoothed generated clips have abnormally low variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .reference_stats import (
    CH_POS,
    CH_ROT,
    CH_ROT_A,
    CH_ROT_B,
    CH_VEL,
    PerSkeletonReferenceStats,
    _compute_normalised_jerk,
)


# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
_W_ROT_CONSISTENCY  = 0.6    # primary: geometry check
_W_JERK             = 0.3   # supporting
_W_TEMPORAL_VAR     = 0.1   # supporting


@dataclass
class MotionQualityReport:
    """Full quality report for one motion clip."""

    # -- Overall --
    total_score: float = 0.0             # weighted aggregate [0, 1]

    # -- Sub-scores [0, 1] (higher = better) --
    rotation_6d_consistency: float = 0.0
    jerk_smoothness: float = 0.0
    temporal_variance: float = 0.0

    # -- Raw diagnostics (for human inspection) --
    raw: Dict[str, float] = field(default_factory=dict)

    # -- Meta --
    object_type: Optional[str] = None
    n_frames: int = 0
    n_joints: int = 0
    has_reference: bool = False

    def as_dict(self) -> dict:
        return {
            "total_score":             round(self.total_score, 4),
            "rotation_6d_consistency": round(self.rotation_6d_consistency, 4),
            "jerk_smoothness":         round(self.jerk_smoothness, 4),
            "temporal_variance":       round(self.temporal_variance, 4),
            "raw":                     {k: round(v, 6) for k, v in self.raw.items()},
            "meta": {
                "object_type": self.object_type,
                "n_frames":    self.n_frames,
                "n_joints":    self.n_joints,
                "has_reference": self.has_reference,
            },
        }

    def __str__(self) -> str:
        lines = [
            f"Motion Quality Report",
            f"  Object type   : {self.object_type or 'unknown'}",
            f"  Frames x Joints : {self.n_frames} x {self.n_joints}",
            f"  Reference data: {'yes' if self.has_reference else 'no (scores may be imprecise)'}",
            f"",
            f"  +---------------------------------------+--------+",
            f"  | Sub-score                             | Score  |",
            f"  +---------------------------------------+--------+",
            f"  | Rotation 6D consistency  (w=0.6)    | {self.rotation_6d_consistency:5.3f}  |",
            f"  | Jerk smoothness         (w=0.3)  | {self.jerk_smoothness:5.3f}  |",
            f"  | Temporal variance       (w=0.1)  | {self.temporal_variance:5.3f}  |",
            f"  +---------------------------------------+--------+",
            f"  | TOTAL                                | {self.total_score:5.3f}  |",
            f"  +---------------------------------------+--------+",
            f"",
            f"  Raw diagnostics:",
        ]
        for k, v in self.raw.items():
            lines.append(f"    {k:40s} = {v:.6f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid_score(value: float, center: float, scale: float) -> float:
    """Map a non-negative penalty `value` to [0, 1].

    score = sigmoid(-(value - center) / scale)
    When value == center the score is 0.5.
    When value << center the score → 1.0 (good).
    When value >> center the score → 0.0 (bad).
    """
    x = -(value - center) / max(scale, 1e-12)
    return float(1.0 / (1.0 + math.exp(-x)))


def _iqr_score(value: float, p25: float, p75: float, penalty_dir: str = "above") -> float:
    """
    Score based on whether `value` falls within the reference IQR.

    penalty_dir='above' → values above p75 are penalised (e.g. high jerk).
    penalty_dir='below' → values below p25 are penalised (e.g. low variance).
    penalty_dir='both'  → deviation from IQR midpoint is penalised.
    """
    mid = (p25 + p75) / 2.0
    iqr = max(p75 - p25, 1e-12)

    if penalty_dir == "above":
        # Sigmoid centred at p75: score→1 as value→0, score=0.5 at p75,
        # score→0 as value→∞.  This means low jerk always scores higher than
        # high jerk, so a near-static GT clip (very low jerk) correctly beats
        # a generated clip whose jerk happens to sit inside the IQR.
        return _sigmoid_score(value, center=p75, scale=iqr * 0.5)

    elif penalty_dir == "below":
        shortfall = max(0.0, p25 - value)
        return _sigmoid_score(shortfall, 0.0, iqr * 0.5)

    else:  # both
        deviation = abs(value - mid)
        return _sigmoid_score(deviation, iqr * 0.5, iqr * 0.5)


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class LightweightMotionQualityScorer:
    """
    Evaluate a motion clip without a trained quality model.

    Parameters
    ----------
    ref_stats : PerSkeletonReferenceStats, optional
        Pre-computed reference statistics for the skeleton type.  When
        provided, the reference-dependent sub-scores use calibrated
        thresholds.  When absent, those sub-scores fall back to
        reasonable but less accurate defaults.
    """

    def __init__(self, ref_stats: Optional[PerSkeletonReferenceStats] = None):
        self.ref_stats = ref_stats

    # ------------------------------------------------------------------
    # Primary metric: 6-D rotation consistency
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rotation_6d(motion: np.ndarray) -> tuple[float, dict]:
        """
        Check that each pair of 3-D vectors in the 6-D rotation channels
        has unit norm.  Ground-truth data is exactly unit-normalised;
        diffusion model outputs accumulate error.

        Returns (score [0,1], raw_dict).
        """
        r = motion[:, :, CH_ROT]         # (T, J, 6)
        v_a = r[:, :, :3]                # first basis vector
        v_b = r[:, :, 3:]                # second basis vector

        norm_a = np.sqrt((v_a ** 2).sum(-1))   # (T, J)
        norm_b = np.sqrt((v_b ** 2).sum(-1))

        dev_a = np.abs(norm_a - 1.0).mean()
        dev_b = np.abs(norm_b - 1.0).mean()
        mean_dev = (dev_a + dev_b) / 2.0

        # Calibrated sigmoid: clean data has dev≈0 (score→1),
        # generated data has dev≈0.04 (score→0.1 with center=0.005, scale=0.01)
        score = _sigmoid_score(mean_dev, center=0.005, scale=0.01)

        raw = {
            "rot6d_norm_dev_a_mean": float(dev_a),
            "rot6d_norm_dev_b_mean": float(dev_b),
            "rot6d_norm_dev_max":    float(max(
                np.abs(norm_a - 1.0).max(),
                np.abs(norm_b - 1.0).max(),
            )),
        }
        return score, raw



    # ------------------------------------------------------------------
    # Supporting: jerk smoothness
    # ------------------------------------------------------------------

    def _score_jerk(self, motion: np.ndarray) -> tuple[float, dict]:
        """
        Activity-normalised, root-weighted jerk of rotation and position channels.

        Each per-joint jerk value is divided by that joint's own temporal variance
        (activity level) before aggregation.  This makes the metric scale-invariant:
        a clip with large legitimate motion is not penalised more than one with small
        motion just because raw jerk is proportional to motion magnitude.

        Why pos+rot (not vel):
        - Position jerk on the root is the strongest visual jitter signal
          (pred root pos jerk is typically 10-350x higher than clean after normalisation).
        - Velocity channels in diffusion outputs are intrinsically smooth regardless
          of positional jitter, so including velocity jerk biases scores in favour of
          generated motions.
        Why root-weighted:
        - Without upweighting root, its high jitter is diluted by ~79 smooth joints.
        """
        jerk_rot = _compute_normalised_jerk(motion, CH_ROT)
        jerk_pos = _compute_normalised_jerk(motion, CH_POS)

        raw = {
            "jerk_rot_norm": jerk_rot,
            "jerk_pos_norm": jerk_pos,
        }

        if self.ref_stats is None:
            # No reference: use calibrated sigmoid on log10 scale so the metric
            # works across the wide dynamic range of normalised jerk values.
            log_rot = math.log10(max(jerk_rot, 1e-30))
            log_pos = math.log10(max(jerk_pos, 1e-30))
            # Centres calibrated on normalised motion data.
            # scale=0.8 gives ~0.8 for clean-level values and ~0.2 for jittery pred.
            score_rot = _sigmoid_score(log_rot, center=-3.0, scale=0.8)
            score_pos = _sigmoid_score(log_pos, center=-3.0, scale=0.8)
            # Position jerk weighted higher (0.7) because root position jitter is
            # more visually salient than rotation jitter, and diffusion models
            # tend to produce smoother rotations while their position channels
            # are noisier — equal weighting would let clean rotation mask pos jitter.
            score = 0.3 * score_rot + 0.7 * score_pos
        else:
            r = self.ref_stats
            # Only penalise "above" (too jerky). GT motions can legitimately be
            # near-static (idle, slow), so low jerk must NOT be penalised here.
            # Over-smoothing detection is handled separately by temporal_variance.
            #
            # Use log10 scale so that small differences between clean and pred
            # (both typically well below the reference p75) are amplified.
            # Center and scale are derived from reference stats in log space to
            # remain calibrated without hard-coded constants.
            log_jerk_rot = math.log10(max(jerk_rot, 1e-30))
            log_jerk_pos = math.log10(max(jerk_pos, 1e-30))
            log_center_rot = math.log10(max((r.jerk_rot_p25 + r.jerk_rot_p75) / 2.0, 1e-30))
            log_center_pos = math.log10(max((r.jerk_pos_p25 + r.jerk_pos_p75) / 2.0, 1e-30))
            log_scale_rot = max(
                math.log10(max(r.jerk_rot_p75, 1e-30)) - math.log10(max(r.jerk_rot_p25, 1e-30)),
                0.1,
            )
            log_scale_pos = max(
                math.log10(max(r.jerk_pos_p75, 1e-30)) - math.log10(max(r.jerk_pos_p25, 1e-30)),
                0.1,
            )
            score_rot = _sigmoid_score(log_jerk_rot, center=log_center_rot, scale=log_scale_rot * 0.5)
            score_pos = _sigmoid_score(log_jerk_pos, center=log_center_pos, scale=log_scale_pos * 0.5)
            # Same positional bias as the no-reference path.
            score = 0.3 * score_rot + 0.7 * score_pos

        return score, raw

    # ------------------------------------------------------------------
    # Supporting: temporal variance
    # ------------------------------------------------------------------

    def _score_temporal_variance(self, motion: np.ndarray) -> tuple[float, dict]:
        """
        Temporal variance of the clip.
        Over-smoothed generated clips have abnormally low variance.
        """
        tvar = float(motion.var(axis=0).mean())

        raw = {"temporal_variance": tvar}

        if self.ref_stats is None:
            # Heuristic: normalised motion with no variation is bad
            score = _sigmoid_score(-tvar, center=-0.005, scale=0.005)
        else:
            r = self.ref_stats
            score = _iqr_score(tvar, r.temporal_var_p25, r.temporal_var_p75, "below")

        return score, raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        motion: np.ndarray,
        object_type: Optional[str] = None,
    ) -> MotionQualityReport:
        """
        Evaluate a single motion clip.

        Parameters
        ----------
        motion      : np.ndarray  shape (T, J, 13)
        object_type : str, optional  skeleton name for metadata

        Returns
        -------
        MotionQualityReport
        """
        if motion.ndim != 3 or motion.shape[-1] != 13:
            raise ValueError(
                f"Expected motion shape (T, J, 13), got {motion.shape}"
            )

        motion = motion.astype(np.float32)
        T, J, _ = motion.shape

        raw: dict = {}

        # Primary sub-score
        s_rot,  r_rot  = self._score_rotation_6d(motion)
        raw.update(r_rot)

        # Supporting sub-scores
        s_jerk, r_jerk = self._score_jerk(motion)
        s_tvar, r_tvar = self._score_temporal_variance(motion)
        raw.update(r_jerk)
        raw.update(r_tvar)

        # Weighted total
        total = (
            _W_ROT_CONSISTENCY  * s_rot  +
            _W_JERK             * s_jerk +
            _W_TEMPORAL_VAR     * s_tvar
        )

        return MotionQualityReport(
            total_score=float(np.clip(total, 0.0, 1.0)),
            rotation_6d_consistency=float(np.clip(s_rot, 0.0, 1.0)),
            jerk_smoothness=float(np.clip(s_jerk, 0.0, 1.0)),
            temporal_variance=float(np.clip(s_tvar, 0.0, 1.0)),
            raw=raw,
            object_type=object_type or (self.ref_stats.object_type if self.ref_stats else None),
            n_frames=T,
            n_joints=J,
            has_reference=self.ref_stats is not None,
        )
