from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_TESTS_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN
from eval.motion_quality import physics, prior
from eval.motion_quality.physics import PhysicsProfile, PhysicsReport, Tolerance, clip_statistics, rest_span
from eval.motion_quality.prior import PriorProfile, PriorReport, clip_features, deviation, label_match_weight
from eval.motion_quality.quality import MotionQualityScorer


# ── Synthetic skeletons ──────────────────────────────────────────────────────
def _chain_cond(joint_count: int = 4, bone: float = 1.0) -> dict:
    """A straight vertical chain: root at the bottom, one bone per joint."""
    offsets = np.zeros((joint_count, 3), dtype=np.float64)
    offsets[1:, 1] = bone
    return {
        "parents": np.arange(-1, joint_count - 1, dtype=np.int32),
        "offsets": offsets,
        "joints_names": ["Pelvis", "Spine", "Neck", "Head"][:joint_count],
        "joint_side_labels": ["center"] * joint_count,
        "species_tags": ("Serpentine",),
    }


def _chain_motion(cond: dict, frames: int, sway: float = 0.3, period: float = 20.0) -> np.ndarray:
    """RIC features of a rigid chain swaying about its root."""
    bone = float(np.linalg.norm(cond["offsets"][1]))
    angle = sway * np.sin(2 * np.pi * np.arange(frames) / period)
    motion = np.zeros((frames, len(cond["parents"]), FEATS_LEN), dtype=np.float32)
    for joint in range(len(cond["parents"])):
        motion[:, joint, 0] = joint * bone * np.sin(angle)
        motion[:, joint, 1] = joint * bone * np.cos(angle)
    return motion


_BIPED_NAMES = ["Hips", "Spine", "Head", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]
_BIPED_PARENTS = [-1, 0, 1, 0, 3, 4, 0, 6, 7]
_BIPED_REST = np.array([
    [0, 1, 0], [0, 1.3, 0], [0, 1.6, 0.1],
    [0.2, 1, 0], [0.2, 0.5, 0], [0.2, 0.05, 0.05],
    [-0.2, 1, 0], [-0.2, 0.5, 0], [-0.2, 0.05, 0.05],
], dtype=np.float64)
_LEFT_LEG, _RIGHT_LEG = (3, 4, 5), (6, 7, 8)


def _biped_cond() -> dict:
    parents = np.asarray(_BIPED_PARENTS)
    offsets = _BIPED_REST - np.where(parents[:, None] >= 0, _BIPED_REST[np.maximum(parents, 0)], 0.0)
    return {
        "joints_names": list(_BIPED_NAMES),
        "parents": parents,
        "offsets": offsets,
        "joint_side_labels": ["center"] * 3 + ["left"] * 3 + ["right"] * 3,
        "symmetry_partner_indices": [0, 1, 2, 6, 7, 8, 3, 4, 5],
        "species_tags": ("Biped", "Walking"),
    }


def _gait(frames: int = 64, period: float = 16.0, right_phase: float = np.pi,
          right_period: float | None = None, right_gain: float = 1.0) -> np.ndarray:
    """RIC features of legs swinging fore-aft: the swing grows down the leg."""
    t = np.arange(frames)
    motion = np.zeros((frames, len(_BIPED_NAMES), FEATS_LEN), dtype=np.float32)
    motion[:, :, :3] = _BIPED_REST
    left = np.sin(2 * np.pi * t / period)
    right = right_gain * np.sin(2 * np.pi * t / (right_period or period) + right_phase)
    for leg, swing in ((_LEFT_LEG, left), (_RIGHT_LEG, right)):
        for depth, joint in enumerate(leg):
            motion[:, joint, 2] += 0.15 * depth * swing
    motion[:, 0, 11] = 0.02
    return motion


# ── Physics ──────────────────────────────────────────────────────────────────
def test_tolerance_is_one_up_to_the_median_and_halves_per_doubling_past_the_tail() -> None:
    tol = Tolerance.from_quantiles([1.0, 2.0, 4.0, 8.0], floor=0.5, layer="t")
    assert tol.score(0.5) == 1.0
    assert tol.score(1.0) == 1.0
    assert tol.score(8.0) == pytest.approx(physics.SCORE_AT_QUANTILE[-1][1])
    assert tol.score(16.0) == pytest.approx(physics.SCORE_AT_QUANTILE[-1][1] / 2)
    values = [tol.score(v) for v in np.linspace(1.0, 30.0, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_tolerance_anchors_are_floored_and_strictly_increasing() -> None:
    tol = Tolerance.from_quantiles([0.0, 0.0, 0.0, 0.0], floor=0.5, layer="t")
    assert tol.anchors[0] == pytest.approx(0.5)
    assert np.all(np.diff(tol.anchors) > 0)


def test_rigid_clip_has_no_stretch_and_a_stretched_one_does() -> None:
    cond = _chain_cond()
    profile = PhysicsProfile(cond)
    rigid = _chain_motion(cond, 40)
    assert clip_statistics(rigid, profile)["stretch_mean"] == pytest.approx(0.0, abs=1e-6)
    stretched = rigid.copy()
    stretched[..., :3] *= 1.2
    stats = clip_statistics(stretched, profile)
    assert stats["stretch_mean"] == pytest.approx(0.2, abs=1e-4)
    assert stats["stretch_long"] == pytest.approx(0.2, abs=1e-4)


def test_spike_rises_for_skipped_frames_and_for_jitter() -> None:
    cond = _chain_cond()
    profile = PhysicsProfile(cond)
    clean = _chain_motion(cond, 60)
    skipped = np.concatenate([clean[:30], clean[38:]])
    jittered = clean.copy()
    jittered[..., :3] += np.random.default_rng(0).normal(0, 0.02, jittered[..., :3].shape)
    base = clip_statistics(clean, profile)["spike"]
    assert clip_statistics(skipped, profile)["spike"] > base
    # The low-passed denominator is the deliberate motion's level, so jitter
    # cannot hide a jump by raising it.
    assert clip_statistics(jittered, profile)["spike"] > base


def test_jitter_raises_jerk_relative_to_a_smooth_clip() -> None:
    cond = _chain_cond()
    profile = PhysicsProfile(cond)
    clean = _chain_motion(cond, 60)
    jittered = clean.copy()
    jittered[..., :3] += np.random.default_rng(1).normal(0, 0.02, jittered[..., :3].shape)
    part = next(iter(profile.parts))
    assert clip_statistics(jittered, profile)[f"jerk:{part}"] > clip_statistics(clean, profile)[f"jerk:{part}"]


def test_rest_span_rejects_a_degenerate_rest_pose() -> None:
    cond = _chain_cond()
    cond["offsets"] = np.zeros_like(cond["offsets"])
    with pytest.raises(ValueError, match="degenerate"):
        rest_span(cond)


# ── Prior ────────────────────────────────────────────────────────────────────
def test_biped_profile_finds_sided_parts_and_mirror_legs() -> None:
    profile = PriorProfile(_biped_cond())
    assert set(profile.sided) >= {"leg", "foot"}
    assert len(profile.limbs["legs"]) == 2
    assert profile.limb_pairs["legs"] == [(0, 1)]


def test_symmetric_gait_has_no_asymmetry_and_a_damped_side_does() -> None:
    profile = PriorProfile(_biped_cond())
    sym = clip_features(_gait(), profile)
    damped = clip_features(_gait(right_gain=0.5), profile)
    assert sym["asym:leg"] == pytest.approx(0.0, abs=1e-3)
    assert damped["asym:leg"] > 0.3


def test_legs_in_antiphase_are_locked_and_opposite() -> None:
    profile = PriorProfile(_biped_cond())
    feats = clip_features(_gait(right_phase=np.pi), profile)
    assert feats["agree:legs"] == 1.0
    assert feats["lock:legs"] > 0.95
    assert feats["lr_phase:legs"] > 0.9
    in_step = clip_features(_gait(right_phase=0.0), profile)
    assert in_step["lr_phase:legs"] < 0.1


def test_a_leg_at_another_frequency_breaks_agreement_and_lock() -> None:
    profile = PriorProfile(_biped_cond())
    steady = clip_features(_gait(), profile)
    chaotic = clip_features(_gait(right_period=9.0), profile)
    assert chaotic["agree:legs"] < steady["agree:legs"]
    assert chaotic["lock:legs"] < steady["lock:legs"] - 0.3


def test_tempo_is_the_motion_frequency_and_barely_moves_under_jitter() -> None:
    profile = PriorProfile(_biped_cond())
    clean = _gait(frames=96, period=16.0)
    tempo = clip_features(clean, profile)["tempo:foot"]
    assert tempo == pytest.approx(1 / 16, abs=0.01)
    jittered = clean.copy()
    jittered[..., :3] += np.random.default_rng(2).normal(0, 0.003, jittered[..., :3].shape)
    assert abs(clip_features(jittered, profile)["tempo:foot"] - tempo) < 0.02


def test_deviation_counts_the_bad_side_only_for_one_sided_features() -> None:
    layers = [{"asym:leg": (0.2, 0.1), "lock:legs": (0.9, 0.05), "tempo:leg": (0.1, 0.01)}]
    assert deviation("asym:leg", 0.1, layers) == 0.0          # more symmetric than typical
    assert deviation("asym:leg", 0.5, layers) == pytest.approx(3.0)
    assert deviation("lock:legs", 1.0, layers) == 0.0         # more in step than typical
    assert deviation("lock:legs", 0.8, layers) == pytest.approx(2.0)
    assert deviation("tempo:leg", 0.08, layers) == pytest.approx(2.0)
    assert deviation("tempo:arm", 0.1, layers) is None


def test_deviation_takes_the_layer_the_value_fits_best() -> None:
    layers = [{"tempo:leg": (0.1, 0.01)}, {"tempo:leg": (0.15, 0.01)}]
    assert deviation("tempo:leg", 0.15, layers) == 0.0


def test_label_match_weight_tiers() -> None:
    exact, heads, partial = prior.LABEL_MATCH_WEIGHTS
    assert label_match_weight(("run", "forward"), ("run", "forward")) == exact
    assert label_match_weight(("run", "forward"), ("run", "left")) == heads
    # Hover and bite are modifiers: both labels have the same attack head.
    assert label_match_weight(("attack", "hover"), ("attack", "bite")) == heads
    assert label_match_weight(("attack", "jump"), ("attack", "bite")) == partial
    assert label_match_weight(("run",), ("walk",)) == 0.0


# ── Combined ─────────────────────────────────────────────────────────────────
class _Stub:
    def __init__(self, report):
        self.report = report

    def evaluate(self, motions, object_type, *args, names=None):
        return self.report


def test_quality_is_the_geometric_mean_of_the_dimensions_per_clip() -> None:
    physics_report = PhysicsReport(
        object_type="x", subset="biped",
        clips=[
            {"file": "a", "score": 0.81, "terms": {"rigidity": 0.81}, "sub": {"stretch_mean": 0.81}},
            {"file": "b", "score": 1.0, "terms": {"rigidity": 1.0}, "sub": {"stretch_mean": 1.0}},
            {"file": "c", "score": 0.25, "terms": {"rigidity": 0.25}, "sub": {"stretch_mean": 0.25}},
        ],
        score=0.81, terms={"rigidity": 0.81},
    )
    prior_report = PriorReport(
        object_type="x", action_label="run", subset="biped",
        clips=[
            {"file": "a", "score": 1.0, "terms": {"coordination": 1.0}, "sub": {"lock:legs": 1.0}},
            {"file": "b", "score": 0.64, "terms": {"coordination": 0.64}, "sub": {"lock:legs": 0.64}},
            {"file": "c", "score": 1.0, "terms": {"coordination": 1.0}, "sub": {"lock:legs": 1.0}},
        ],
        score=1.0, terms={"coordination": 1.0},
    )
    scorer = object.__new__(MotionQualityScorer)
    scorer.physics = _Stub(physics_report)
    scorer.prior = _Stub(prior_report)
    report = scorer.evaluate([np.zeros((8, 1, FEATS_LEN))] * 3, "x", "run", names=["a", "b", "c"])
    assert [c["score"] for c in report.clips] == pytest.approx([0.9, 0.8, 0.5])
    assert report.score == pytest.approx(0.8)
    assert report.dimensions == {"physics": pytest.approx(0.81), "prior": pytest.approx(1.0)}
    assert report.worst[0] == ("physics/stretch_mean", pytest.approx(0.81))
    payload = report.as_dict()
    assert payload["score"] == pytest.approx(0.8)
    assert payload["physics"] == pytest.approx(0.81)
    assert payload["meta"]["n_clips"] == 3
