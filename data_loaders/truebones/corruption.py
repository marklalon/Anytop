from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


POS_CHANNELS = slice(0, 3)
ROT_CHANNELS = slice(3, 9)
VEL_CHANNELS = slice(9, 12)
CONTACT_CHANNEL = 12
FREEZE_RECOVERY_FRAMES = 2


@dataclass(frozen=True)
class CorruptionPreset:
    name: str
    joint_dropout_segments: tuple[int, int]
    joint_dropout_length: tuple[int, int]
    per_frame_dropout_prob: tuple[float, float]
    jitter_std: tuple[float, float]
    drift_scale: tuple[float, float]
    flicker_segments: tuple[int, int]
    flicker_length: tuple[int, int]
    limb_corruption_segments: tuple[int, int]
    terminal_freeze_segments: tuple[int, int]
    terminal_freeze_depth: tuple[int, int]
    root_noise_std: tuple[float, float]


CURRICULUM_PRESETS = {
    1: CorruptionPreset(
        name="stage1",
        joint_dropout_segments=(0, 2),
        joint_dropout_length=(2, 10),
        per_frame_dropout_prob=(0.00, 0.04),
        jitter_std=(0.005, 0.02),
        drift_scale=(0.005, 0.02),
        flicker_segments=(0, 2),
        flicker_length=(1, 5),
        limb_corruption_segments=(0, 2),
        terminal_freeze_segments=(2, 3),
        terminal_freeze_depth=(2, 3),
        root_noise_std=(0.01, 0.03),
    ),
    2: CorruptionPreset(
        name="stage2",
        joint_dropout_segments=(0, 2),
        joint_dropout_length=(2, 10),
        per_frame_dropout_prob=(0.01, 0.05),
        jitter_std=(0.008, 0.025),
        drift_scale=(0.008, 0.025),
        flicker_segments=(0, 2),
        flicker_length=(1, 5),
        limb_corruption_segments=(0, 2),
        terminal_freeze_segments=(4, 6),
        terminal_freeze_depth=(3, 4),
        root_noise_std=(0.015, 0.04),
    ),
}


def _sample_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    if high <= low:
        return float(low)
    return float(rng.uniform(low, high))


def _sample_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    low, high = bounds
    if high <= low:
        return int(low)
    return int(rng.integers(low, high + 1))


def _low_frequency_noise(
    rng: np.random.Generator,
    length: int,
    dims: int,
    scale: float,
    num_knots: int = 4,
) -> np.ndarray:
    if length <= 1 or scale <= 0.0:
        return np.zeros((length, dims), dtype=np.float32)
    knot_count = max(2, min(length, num_knots))
    knot_x = np.linspace(0, length - 1, knot_count)
    knot_y = rng.normal(loc=0.0, scale=scale, size=(knot_count, dims)).astype(np.float32)
    xs = np.arange(length, dtype=np.float32)
    smoothed = np.stack(
        [np.interp(xs, knot_x, knot_y[:, dim]) for dim in range(dims)],
        axis=-1,
    )
    return smoothed.astype(np.float32)


def _extract_chain_joints(kinematic_chains: Any, rng: np.random.Generator, n_joints: int) -> np.ndarray:
    if isinstance(kinematic_chains, np.ndarray):
        kinematic_chains = kinematic_chains.tolist()
    if not isinstance(kinematic_chains, list) or not kinematic_chains:
        candidate_joints = np.arange(1, n_joints, dtype=np.int64)
        if candidate_joints.size == 0:
            return np.empty(0, dtype=np.int64)
        joint_count = min(max(2, n_joints // 5), candidate_joints.size)
        return np.sort(rng.choice(candidate_joints, size=joint_count, replace=False))
    chain_index = int(rng.integers(0, len(kinematic_chains)))
    chain = kinematic_chains[chain_index]
    chain = np.asarray(chain, dtype=np.int64)
    chain = chain[(chain >= 0) & (chain < n_joints)]
    if chain.size == 0:
        return np.asarray([0], dtype=np.int64)
    return np.unique(chain)


def _normalize_kinematic_chains(kinematic_chains: Any, n_joints: int) -> list[np.ndarray]:
    if isinstance(kinematic_chains, np.ndarray):
        kinematic_chains = kinematic_chains.tolist()
    if not isinstance(kinematic_chains, list):
        return []

    normalized: list[np.ndarray] = []
    for chain in kinematic_chains:
        chain_array = np.asarray(chain, dtype=np.int64)
        chain_array = chain_array[(chain_array >= 0) & (chain_array < n_joints)]
        if chain_array.size > 1:
            ordered_unique = np.asarray(list(dict.fromkeys(chain_array.tolist())), dtype=np.int64)
            if ordered_unique.size > 1:
                normalized.append(ordered_unique)
    return normalized


def _build_parent_lookup(kinematic_chains: Any, n_joints: int) -> np.ndarray:
    parent_lookup = np.full(n_joints, -1, dtype=np.int64)
    for chain in _normalize_kinematic_chains(kinematic_chains, n_joints):
        for parent_joint, child_joint in zip(chain[:-1], chain[1:]):
            child_index = int(child_joint)
            parent_index = int(parent_joint)
            if parent_lookup[child_index] < 0:
                parent_lookup[child_index] = parent_index
    return parent_lookup


def _order_joints_for_freeze(joints: np.ndarray, parent_lookup: np.ndarray | None) -> np.ndarray:
    ordered_unique = np.asarray(list(dict.fromkeys(np.asarray(joints, dtype=np.int64).tolist())), dtype=np.int64)
    if ordered_unique.size <= 1 or parent_lookup is None:
        return ordered_unique

    joint_set = {int(joint) for joint in ordered_unique}
    depth_cache: dict[int, int] = {}

    def depth(joint: int, active: set[int] | None = None) -> int:
        cached = depth_cache.get(joint)
        if cached is not None:
            return cached

        if active is None:
            active = set()
        if joint in active:
            return 0

        active.add(joint)
        parent_joint = int(parent_lookup[joint])
        if parent_joint < 0 or parent_joint not in joint_set:
            joint_depth = 0
        else:
            joint_depth = depth(parent_joint, active) + 1
        active.remove(joint)
        depth_cache[joint] = joint_depth
        return joint_depth

    return np.asarray(sorted(ordered_unique.tolist(), key=lambda joint: (depth(int(joint)), int(joint))), dtype=np.int64)


def _select_terminal_chain_segments(
    kinematic_chains: Any,
    rng: np.random.Generator,
    n_joints: int,
    segment_bounds: tuple[int, int],
    depth_bounds: tuple[int, int],
) -> list[tuple[int, np.ndarray]]:
    chains = _normalize_kinematic_chains(kinematic_chains, n_joints)
    if not chains:
        return []

    num_segments = min(_sample_int(rng, segment_bounds), len(chains))
    if num_segments <= 0:
        return []

    selected_indices = rng.choice(np.arange(len(chains)), size=num_segments, replace=False)
    segments: list[tuple[int, np.ndarray]] = []
    for chain_index in np.atleast_1d(selected_indices):
        chain = chains[int(chain_index)]
        max_depth = min(chain.size - 1, depth_bounds[1])
        if max_depth <= 0:
            continue
        min_depth = min(max_depth, max(1, depth_bounds[0]))
        depth = _sample_int(rng, (min_depth, max_depth))
        anchor_joint = int(chain[-(depth + 1)]) if chain.size > depth else 0
        segment = chain[-depth:]
        segment = segment[segment > 0]
        if segment.size > 0:
            segments.append((anchor_joint, segment.astype(np.int64, copy=False)))
    return segments


def _reduce_confidence(
    confidence: np.ndarray,
    start: int,
    end: int,
    joints: np.ndarray,
    value: float,
) -> None:
    confidence[start:end, joints, 0] = np.minimum(confidence[start:end, joints, 0], value)
    blend_end = min(confidence.shape[0], end + FREEZE_RECOVERY_FRAMES)
    if blend_end > end:
        confidence[end:blend_end, joints, 0] = np.minimum(confidence[end:blend_end, joints, 0], max(value, 0.35))


def _freeze_joint_window(
    reference: np.ndarray,
    source_motion: np.ndarray,
    start: int,
    end: int,
    joints: np.ndarray,
    anchor_frame: int,
    parent_lookup: np.ndarray | None = None,
) -> None:
    if end <= start or joints.size == 0:
        return
    anchor_frame = int(np.clip(anchor_frame, 0, source_motion.shape[0] - 1))
    frozen_pose = source_motion[anchor_frame].copy()
    ordered_joints = _order_joints_for_freeze(joints, parent_lookup)
    for joint in ordered_joints:
        parent_joint = -1 if parent_lookup is None else int(parent_lookup[joint])
        if parent_joint >= 0:
            local_offset = frozen_pose[joint, POS_CHANNELS] - frozen_pose[parent_joint, POS_CHANNELS]
            reference[start:end, joint, POS_CHANNELS] = reference[start:end, parent_joint, POS_CHANNELS] + local_offset[None, :]
        else:
            reference[start:end, joint, POS_CHANNELS] = frozen_pose[joint, POS_CHANNELS]
        reference[start:end, joint, ROT_CHANNELS] = frozen_pose[joint, ROT_CHANNELS]
        reference[start:end, joint, VEL_CHANNELS] = 0.0
        if reference.shape[2] > CONTACT_CHANNEL:
            reference[start:end, joint, CONTACT_CHANNEL] = frozen_pose[joint, CONTACT_CHANNEL]


def _apply_recovery_blend(
    reference: np.ndarray,
    source_motion: np.ndarray,
    start: int,
    joints: np.ndarray,
    blend_frames: int = FREEZE_RECOVERY_FRAMES,
) -> None:
    if start <= 0 or blend_frames <= 0 or joints.size == 0 or start >= reference.shape[0]:
        return

    blend_length = min(blend_frames, reference.shape[0] - start)
    if blend_length <= 0:
        return

    joints = np.asarray(joints, dtype=np.int64)
    frozen_pose = reference[start - 1, joints, :].copy()
    target_pose = source_motion[start:start + blend_length, joints, :]
    alphas = np.linspace(
        1.0 / (blend_length + 1),
        blend_length / (blend_length + 1),
        blend_length,
        dtype=np.float32,
    )
    for blend_index, alpha in enumerate(alphas):
        frame = start + blend_index
        reference[frame, joints, :] = (1.0 - alpha) * frozen_pose + alpha * target_pose[blend_index]


def _freeze_chain_segment_motion(
    reference: np.ndarray,
    source_motion: np.ndarray,
    start: int,
    end: int,
    anchor_joint: int,
    chain_segment: np.ndarray,
    anchor_frame: int,
) -> None:
    if end <= start or chain_segment.size == 0:
        return

    anchor_frame = int(np.clip(anchor_frame, 0, source_motion.shape[0] - 1))
    anchor_pose = source_motion[anchor_frame]
    chain_anchor = int(np.clip(anchor_joint, 0, reference.shape[1] - 1))

    frozen_offsets: list[np.ndarray] = []
    previous_joint = chain_anchor
    for joint in chain_segment:
        joint_index = int(joint)
        frozen_offsets.append(anchor_pose[joint_index, POS_CHANNELS] - anchor_pose[previous_joint, POS_CHANNELS])
        previous_joint = joint_index

    current_parent_positions = reference[start:end, chain_anchor, POS_CHANNELS].copy()
    for segment_index, joint in enumerate(chain_segment):
        joint_index = int(joint)
        current_positions = current_parent_positions + frozen_offsets[segment_index][None, :]
        reference[start:end, joint_index, POS_CHANNELS] = current_positions
        reference[start:end, joint_index, ROT_CHANNELS] = anchor_pose[joint_index, ROT_CHANNELS]
        reference[start:end, joint_index, VEL_CHANNELS] = 0.0
        if reference.shape[2] > CONTACT_CHANNEL:
            reference[start:end, joint_index, CONTACT_CHANNEL] = anchor_pose[joint_index, CONTACT_CHANNEL]
        current_parent_positions = current_positions


class MotionCorruptor:
    def __init__(
        self,
        curriculum_stage: int = 1,
        seed: int | None = None,
    ) -> None:
        self.curriculum_stage = int(curriculum_stage)
        self.rng = np.random.default_rng(seed)

    @property
    def preset(self) -> CorruptionPreset:
        return CURRICULUM_PRESETS.get(self.curriculum_stage, CURRICULUM_PRESETS[2])

    def set_stage(self, curriculum_stage: int) -> None:
        self.curriculum_stage = int(curriculum_stage)

    def corrupt(
        self,
        clean_motion: np.ndarray,
        length: int,
        kinematic_chains: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if clean_motion.ndim != 3:
            raise ValueError(f"Expected clean motion with shape [T, J, F], got {clean_motion.shape}")

        valid_length = min(int(length), clean_motion.shape[0])
        if valid_length <= 0:
            empty_reference = np.zeros_like(clean_motion, dtype=np.float32)
            empty_confidence = np.zeros((clean_motion.shape[0], clean_motion.shape[1], 1), dtype=np.float32)
            return empty_reference, empty_confidence, {"stage": self.preset.name, "applied": []}

        reference = clean_motion.astype(np.float32, copy=True)
        source_motion = reference.copy()
        parent_lookup = _build_parent_lookup(kinematic_chains, reference.shape[1])
        confidence = np.zeros((clean_motion.shape[0], clean_motion.shape[1], 1), dtype=np.float32)
        confidence[:valid_length, :, 0] = 1.0
        applied: list[str] = []
        preset = self.preset

        self._apply_temporally_contiguous_dropout(reference, source_motion, confidence, valid_length, preset, applied, parent_lookup)
        self._apply_per_frame_dropout(reference, source_motion, confidence, valid_length, preset, applied, parent_lookup)
        self._apply_gaussian_jitter(reference, confidence, valid_length, preset, applied)
        self._apply_low_frequency_drift(reference, confidence, valid_length, preset, applied)
        self._apply_short_flicker(reference, source_motion, confidence, valid_length, preset, applied, parent_lookup)
        self._apply_local_limb_corruption(reference, source_motion, confidence, valid_length, preset, applied, kinematic_chains, parent_lookup)
        self._apply_root_trajectory_noise(reference, confidence, valid_length, preset, applied)
        self._apply_terminal_joint_freeze(reference, source_motion, confidence, valid_length, preset, applied, kinematic_chains)

        confidence = np.clip(confidence, 0.0, 1.0)
        reference[valid_length:] = 0.0
        confidence[valid_length:] = 0.0
        metadata = {
            "stage": preset.name,
            "applied": applied,
            "valid_length": valid_length,
            "mean_confidence": float(confidence[:valid_length].mean()),
        }
        return reference, confidence, metadata

    def _apply_temporally_contiguous_dropout(
        self,
        reference: np.ndarray,
        source_motion: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
        parent_lookup: np.ndarray,
    ) -> None:
        n_joints = reference.shape[1]
        num_segments = _sample_int(self.rng, preset.joint_dropout_segments)
        for _ in range(num_segments):
            if valid_length < 2:
                break
            dropout_len = min(valid_length, _sample_int(self.rng, preset.joint_dropout_length))
            start = int(self.rng.integers(0, max(1, valid_length - dropout_len + 1)))
            joint_count = int(self.rng.integers(1, min(max(1, n_joints // 8), n_joints) + 1))
            joints = np.sort(self.rng.choice(np.arange(n_joints), size=joint_count, replace=False))
            _freeze_joint_window(reference, source_motion, start, start + dropout_len, joints, anchor_frame=start - 1, parent_lookup=parent_lookup)
            _apply_recovery_blend(reference, source_motion, start + dropout_len, joints)
            _reduce_confidence(confidence, start, start + dropout_len, joints, 0.0)
            applied.append("joint_dropout")

    def _apply_per_frame_dropout(
        self,
        reference: np.ndarray,
        source_motion: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
        parent_lookup: np.ndarray,
    ) -> None:
        drop_prob = _sample_uniform(self.rng, preset.per_frame_dropout_prob)
        if drop_prob <= 0.0:
            return
        drop_mask = self.rng.random((valid_length, reference.shape[1])) < drop_prob
        if not np.any(drop_mask):
            return
        confidence_slice = confidence[:valid_length, :, 0]
        for frame_idx in np.nonzero(np.any(drop_mask, axis=1))[0]:
            joints = np.nonzero(drop_mask[frame_idx])[0].astype(np.int64)
            _freeze_joint_window(reference, source_motion, frame_idx, frame_idx + 1, joints, anchor_frame=frame_idx - 1, parent_lookup=parent_lookup)
            confidence_slice[frame_idx, joints] = 0.0
        applied.append("frame_visibility_dropout")

    def _apply_gaussian_jitter(
        self,
        reference: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
    ) -> None:
        jitter_std = _sample_uniform(self.rng, preset.jitter_std)
        if jitter_std <= 0.0:
            return
        active_mask = self.rng.random((valid_length, reference.shape[1])) < 0.18
        if not np.any(active_mask):
            return
        confidence_value = _sample_uniform(self.rng, (0.70, 0.92))
        position_noise = self.rng.normal(0.0, jitter_std, size=(valid_length, reference.shape[1], 3)).astype(np.float32)
        velocity_noise = self.rng.normal(0.0, jitter_std * 0.35, size=(valid_length, reference.shape[1], 3)).astype(np.float32)
        active_mask_expanded = active_mask[..., None]
        reference[:valid_length, :, POS_CHANNELS] += position_noise * active_mask_expanded
        reference[:valid_length, :, VEL_CHANNELS] += velocity_noise * active_mask_expanded
        confidence_slice = confidence[:valid_length, :, 0]
        confidence_slice[active_mask] = np.minimum(confidence_slice[active_mask], confidence_value)
        applied.append("gaussian_jitter")

    def _apply_low_frequency_drift(
        self,
        reference: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
    ) -> None:
        drift_scale = _sample_uniform(self.rng, preset.drift_scale)
        if drift_scale <= 0.0 or valid_length < 3:
            return
        segment_length = max(3, int(round(valid_length * _sample_uniform(self.rng, (0.12, 0.35)))))
        segment_length = min(valid_length, segment_length)
        start = int(self.rng.integers(0, max(1, valid_length - segment_length + 1)))
        joint_count = int(self.rng.integers(1, min(max(1, reference.shape[1] // 10), reference.shape[1]) + 1))
        joints = np.sort(self.rng.choice(np.arange(reference.shape[1]), size=joint_count, replace=False))
        drift = _low_frequency_noise(self.rng, segment_length, dims=3, scale=drift_scale)
        drift_velocity = np.diff(np.concatenate([drift[:1], drift], axis=0), axis=0)
        reference[start:start + segment_length, joints, POS_CHANNELS] += drift[:, None, :]
        reference[start:start + segment_length, joints, VEL_CHANNELS] += drift_velocity[:, None, :]
        confidence_value = _sample_uniform(self.rng, (0.55, 0.85))
        _reduce_confidence(confidence, start, start + segment_length, joints, confidence_value)
        applied.append("low_frequency_drift")

    def _apply_short_flicker(
        self,
        reference: np.ndarray,
        source_motion: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
        parent_lookup: np.ndarray,
    ) -> None:
        num_segments = _sample_int(self.rng, preset.flicker_segments)
        for _ in range(num_segments):
            flicker_len = min(valid_length, _sample_int(self.rng, preset.flicker_length))
            start = int(self.rng.integers(0, max(1, valid_length - flicker_len + 1)))
            joint_count = int(self.rng.integers(1, min(max(1, reference.shape[1] // 10), reference.shape[1]) + 1))
            joints = np.sort(self.rng.choice(np.arange(reference.shape[1]), size=joint_count, replace=False))
            _freeze_joint_window(reference, source_motion, start, start + flicker_len, joints, anchor_frame=start - 1, parent_lookup=parent_lookup)
            _apply_recovery_blend(reference, source_motion, start + flicker_len, joints)
            _reduce_confidence(confidence, start, start + flicker_len, joints, 0.0)
            applied.append("short_flicker")

    def _apply_local_limb_corruption(
        self,
        reference: np.ndarray,
        source_motion: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
        kinematic_chains: Any,
        parent_lookup: np.ndarray,
    ) -> None:
        num_segments = _sample_int(self.rng, preset.limb_corruption_segments)
        for _ in range(num_segments):
            joints = _extract_chain_joints(kinematic_chains, self.rng, reference.shape[1])
            if joints.size == 0:
                continue
            segment_length = min(valid_length, max(3, int(round(valid_length * _sample_uniform(self.rng, (0.10, 0.35))))))
            start = int(self.rng.integers(0, max(1, valid_length - segment_length + 1)))
            corruption_mode = "dropout" if self.rng.random() < 0.5 else "jitter"
            if corruption_mode == "dropout":
                _freeze_joint_window(reference, source_motion, start, start + segment_length, joints, anchor_frame=start - 1, parent_lookup=parent_lookup)
                _apply_recovery_blend(reference, source_motion, start + segment_length, joints)
                _reduce_confidence(confidence, start, start + segment_length, joints, 0.0)
            else:
                jitter_std = _sample_uniform(self.rng, preset.jitter_std) * 0.6
                noise = self.rng.normal(0.0, jitter_std, size=(segment_length, joints.size, 3)).astype(np.float32)
                velocity_noise = np.diff(
                    np.concatenate([np.zeros_like(noise[:1]), noise], axis=0),
                    axis=0,
                )
                reference[start:start + segment_length, joints, POS_CHANNELS] += noise
                reference[start:start + segment_length, joints, VEL_CHANNELS] += velocity_noise
                confidence_value = _sample_uniform(self.rng, (0.50, 0.80))
                _reduce_confidence(confidence, start, start + segment_length, joints, confidence_value)
            applied.append("local_limb_corruption")

    def _apply_root_trajectory_noise(
        self,
        reference: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
    ) -> None:
        root_noise_std = _sample_uniform(self.rng, preset.root_noise_std)
        if root_noise_std <= 0.0:
            return
        drift = _low_frequency_noise(self.rng, valid_length, dims=3, scale=root_noise_std, num_knots=5)
        drift_velocity = np.diff(np.concatenate([drift[:1], drift], axis=0), axis=0)
        reference[:valid_length, 0, POS_CHANNELS] += drift
        reference[:valid_length, 0, VEL_CHANNELS] += drift_velocity
        confidence[:valid_length, 0, 0] = np.minimum(
            confidence[:valid_length, 0, 0],
            _sample_uniform(self.rng, (0.65, 0.90)),
        )
        applied.append("root_trajectory_noise")

    def _apply_terminal_joint_freeze(
        self,
        reference: np.ndarray,
        source_motion: np.ndarray,
        confidence: np.ndarray,
        valid_length: int,
        preset: CorruptionPreset,
        applied: list[str],
        kinematic_chains: Any,
    ) -> None:
        segments = _select_terminal_chain_segments(
            kinematic_chains,
            self.rng,
            reference.shape[1],
            preset.terminal_freeze_segments,
            preset.terminal_freeze_depth,
        )
        if not segments:
            return

        confidence_value = _sample_uniform(self.rng, (0.02, 0.1))
        for anchor_joint, chain_segment in segments:
            _freeze_chain_segment_motion(
                reference,
                source_motion,
                0,
                valid_length,
                anchor_joint,
                chain_segment,
                anchor_frame=0,
            )
            _reduce_confidence(confidence, 0, valid_length, chain_segment, confidence_value)
        applied.append("terminal_joint_freeze")
