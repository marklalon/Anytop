"""Shared motion geometry losses; diffusion weighting stays in GaussianDiffusion."""
import math
import numpy as np
import torch
import torch as th

from diffusion.nn import sum_flat
from diffusion.losses import geodesic_distance
from utils.rotation_conversions import rotation_6d_to_matrix_safe
from utils.device_transfer import host_to_device
from data_loaders.truebones.truebones_utils.canonical_features import (
    CANONICAL_DPOS_STD_KEY, CANONICAL_STD_KEY, DPOS_LOSS_WEIGHT_KEY,
    PARENT_INDEX_KEY, ROOT_STAT_SLICE,
    _length_scale_from_cond, get_canonical_global_stats, _view_stat_torch,
    _check_root_stats, canonical_to_physical_hml, root_relative_positions,
)

# Floor of the rest length that normalizes the bone losses, as a fraction
# of the skeleton's length scale L: bones shorter than this (finger tips, toe
# segments) are weighted as if they had this length, which bounds how much
# louder the shortest bones can be than an L-long one.
BONE_REST_FLOOR_FRAC = 0.06
# Lags of the limb correlation loss, as fractions of the window: zero lag reads
# whether limbs move together, the others which one leads.
LIMB_CORR_LAG_FRACS = (0.0, 0.125, 0.25)
# Mean squared limb-tip step, in L per frame, below which a limb counts as
# still: it sits between a breathing idle's tips and a walk's.
LIMB_CORR_ENERGY_FLOOR = 1e-4
# Fixed angular unit: no inverse target-energy weighting, so nearly static
# joints cannot amplify annotation noise. Chordal error is bounded on SO(3).
ROTATION_STEP_SCALE_RAD = 0.1

class MotionGeometryLosses:
    drot_joint_boost = 0.0

    @staticmethod
    def _physical_rotation_matrices(physical):
        """Local rotation matrices [bs, T, J, 3, 3] of a physical [bs, J, F, T] feature."""
        return rotation_6d_to_matrix_safe(physical.permute(0, 3, 1, 2)[..., 3:9].float())

    def geodesic_loss(self, a, b, spat_mask, lengths, n_joints, rotations=None):
        # assuming a.shape == b.shape == bs, J, Jdim, seqlen
        # assuming spat_mask.shape == bs, 1, 1, max_joints
        # rotations: optional (target, pred) _physical_rotation_matrices of a, b
        if rotations is None:
            rotations = (self._physical_rotation_matrices(a), self._physical_rotation_matrices(b))
        rots_target, rots_pred = rotations
        loss = geodesic_distance(rots_pred, rots_target).permute(0, 2, 3, 1)
        spat_masked_loss = (loss * spat_mask.float().transpose(1,3))
        loss = sum_flat(spat_masked_loss)  # gives \sigma_euclidean over unmasked elements
        non_zero_elements = (lengths * n_joints).float()
        loss_val = loss / non_zero_elements
        return loss_val

    def rotation_step_loss(self, pred_physical, target_physical, spat_mask, lengths, y,
                           rotations=None):
        """Target-relative SO(3) increments, one loss per sample.

        With no explicit rotations, reads physical local rotations. Training
        always supplies global matrices for the global-step objective.

        Delta[t] = R[t]^T R[t+1]. The squared Frobenius error / 2 is the
        squared chordal angular distance (approximately radians squared for
        small errors), divided by a fixed 0.1-radian unit squared. It avoids
        acos' singular gradient at an exact match and any division by a
        target joint's near-zero motion energy. This matches motion, rather
        than penalizing motion magnitude. The original pose losses retain
        responsibility for a time-constant orientation offset.

        Padding never contributes. A loop's last VALID frame steps to frame
        zero; an open window excludes that step. At high diffusion noise the
        x0 estimate can average several motions; diffusion applies its noise
        weighting outside this shared loss.
        """
        batch, joints, _, frames = pred_physical.shape
        if frames < 2:
            return pred_physical.new_zeros(batch)
        if rotations is None:
            rotations = (self._physical_rotation_matrices(target_physical),
                         self._physical_rotation_matrices(pred_physical))
        target_rot, pred_rot = rotations
        device = pred_rot.device
        lengths = th.as_tensor(lengths, device=device).long().reshape(batch, 1).clamp(0, frames)
        frame = th.arange(frames, device=device).view(1, frames)
        next_frame = th.where(frame + 1 < lengths, frame + 1, 0)
        gather_index = next_frame[:, :, None, None, None].expand(batch, frames, joints, 3, 3)

        def increments(rot):
            return rot.transpose(-1, -2) @ rot.gather(1, gather_index)

        target_step = increments(target_rot)
        error = (increments(pred_rot) - target_step).square().sum(dim=(-1, -2))
        error = error / (2.0 * ROTATION_STEP_SCALE_RAD ** 2)
        is_loop = self._coerce_bool_batch((y or {}).get('is_loop'), batch, device, default=False)
        step_valid = ((frame + 1 < lengths)
                      | (is_loop[:, None] & (frame == lengths - 1) & (lengths >= 2)))
        joint_valid = spat_mask.to(device=device, dtype=error.dtype).reshape(batch, joints)
        weight = step_valid[:, :, None].to(error.dtype) * joint_valid[:, None, :]
        if self.drot_joint_boost > 0.:
            # Detached GT RMS chordal speed, computed only on valid steps.
            # No inverse-energy weighting: static clips retain uniform weights.
            with th.no_grad():
                identity = th.eye(3, device=device, dtype=target_step.dtype)
                energy = (target_step.detach() - identity).square().sum(dim=(-1, -2)) / 2.0
                rms = ((energy * weight).sum(dim=1)
                       / weight.sum(dim=1).clamp_min(1.0)).sqrt()
                relative = rms / rms.amax(dim=1, keepdim=True).clamp_min(1e-8)
                joint_weight = 1.0 + self.drot_joint_boost * relative
            weight = weight * joint_weight[:, None, :]
        loss = (error * weight).sum(dim=(1, 2)) / weight.sum(dim=(1, 2)).clamp_min(1.0)
        return loss

    @staticmethod
    def _canonical_global_rotation_matrices(canonical, y):
        # Unstandardized canonical rotations are Q_j = G_j A_j, with a
        # time-constant rest frame A_j. Q[t]^T Q[t+1] = A^T Delta_G A.
        # Chordal error and activity are invariant under this conjugation,
        # so no FK or removal of A is needed. Never read standardized 6D
        # values directly as a rotation matrix.
        stats = get_canonical_global_stats(y)
        if stats is None:
            raise KeyError("canonical rotation decode requires canonical_feature_mean/std")
        mean, std = (th.as_tensor(stat, device=canonical.device, dtype=canonical.dtype)
                     for stat in stats)
        _check_root_stats(mean.shape[-1])
        _check_root_stats(std.shape[-1])
        # Slice before arithmetic: positions, velocity and root stats are unused.
        mean_rot = _view_stat_torch(mean, canonical.shape, canonical.shape[2])[:, :, 3:9]
        std_rot = _view_stat_torch(std, canonical.shape, canonical.shape[2])[:, :, 3:9]
        rot6d = canonical[:, :, 3:9] * std_rot + mean_rot
        return rotation_6d_to_matrix_safe(rot6d.permute(0, 3, 1, 2).float())

    def _per_sample_y_scalar(self, y, key, batch_size, device, default=None):
        """A ``[B]`` float tensor from ``y[key]`` (``default`` when absent)."""
        value = y.get(key) if isinstance(y, dict) else None
        if value is None:
            if default is None:
                raise KeyError(
                    f"y[{key!r}] is missing; the training loader collates it from cond.npy "
                    "(regenerate the cond if it predates it)."
                )
            return th.full((batch_size,), float(default), device=device, dtype=th.float32)
        value = value if torch.is_tensor(value) else th.as_tensor(np.asarray(value, dtype=np.float32))
        value = value.to(device=device, dtype=th.float32).reshape(-1)
        if value.numel() == 1 and batch_size != 1:
            value = value.expand(batch_size)
        return value

    def _position_step_stats(self, y, batch_size, device):
        """The per-sample canonical std ``[B, F]`` and position-step RMS ``[B]``
        the position-step losses measure in."""
        std = y.get(CANONICAL_STD_KEY)
        if std is None:
            raise KeyError(f"y[{CANONICAL_STD_KEY!r}] is missing; the position-step loss needs it.")
        std = std.to(device=device, dtype=th.float32)
        std = std.reshape(1, -1).expand(batch_size, -1) if std.dim() == 1 else std.reshape(batch_size, -1)
        dpos_std = self._per_sample_y_scalar(y, CANONICAL_DPOS_STD_KEY, batch_size, device)
        return std, dpos_std.clamp_min(1e-8)

    @staticmethod
    def _position_steps(pos):
        """Frame-to-frame steps along the last axis, the wrap step
        ``pos[0] - pos[-1]`` last."""
        inner = pos[..., 1:] - pos[..., :-1]
        wrap = pos[..., 0:1] - pos[..., -1:]
        return th.cat([inner, wrap], dim=-1)

    def _position_step_frame_weight(self, y, batch_size, n_frames, device, dtype):
        """``[B, T]`` step weights: every inner step, plus the wrap step on a loop."""
        is_loop = self._coerce_bool_batch(y.get('is_loop'), batch_size, device, default=False)
        frame_weight = th.ones(batch_size, n_frames, device=device, dtype=dtype)
        frame_weight[:, -1] = is_loop.to(dtype=dtype)
        return frame_weight

    def position_step_loss(self, target, model_output, spat_mask, n_joints, y):
        """Per-sample MSE of every joint's frame-to-frame position step against
        the target's, in units of the corpus position-step RMS. A loop sample
        also counts its wrap step ``pos[0] - pos[-1]``. The root's step is its
        world XZ step, so this is also the root trajectory's step and, on a
        loop, its closure. Scaled by the species' ``dpos_loss_weight``. ``[B]``."""
        batch_size, max_joints, _n_feats, n_frames = target.shape
        device = target.device
        if n_frames < 2:
            return target.new_zeros(batch_size)
        y = y if isinstance(y, dict) else {}
        std, dpos_std = self._position_step_stats(y, batch_size, device)
        # Joint 0 is standardized by the root's own stats (canonical_features, "Root").
        pos_std = th.cat([
            std[:, ROOT_STAT_SLICE].unsqueeze(1),
            std[:, 0:3].unsqueeze(1).expand(-1, max_joints - 1, -1),
        ], dim=1)                                                                   # [B, J, 3]
        scale = (pos_std / dpos_std.view(batch_size, 1, 1)).unsqueeze(-1)

        steps = self._position_steps
        err = ((steps(model_output[:, :, 0:3, :]) - steps(target[:, :, 0:3, :])) * scale) ** 2
        frame_weight = self._position_step_frame_weight(y, batch_size, n_frames, device, err.dtype)
        joint_weight = spat_mask.float().transpose(1, 3).reshape(batch_size, -1, 1, 1)
        weighted = err * joint_weight * frame_weight.view(batch_size, 1, 1, n_frames)
        n_joints_f = th.as_tensor(n_joints, device=device, dtype=err.dtype).reshape(-1)
        denom = (frame_weight.sum(dim=1) * n_joints_f * 3.0).clamp_min(1.0)
        per_sample = sum_flat(weighted) / denom
        weight = self._per_sample_y_scalar(y, DPOS_LOSS_WEIGHT_KEY, batch_size, device, default=1.0)
        return per_sample * weight

    def position_acceleration_loss(self, target, model_output, spat_mask, n_joints, y):
        """Target-relative position second differences, one value per sample.

        Penalize acceleration of the reconstruction ERROR, not acceleration
        magnitude. Use the same physical position / corpus step-RMS units and
        species weights as dpos_loss (no per-joint motion-energy division).
        Open clips need three valid frames; loops wrap at their valid length.
        Joint/time padding never contributes, including across the loop seam.
        """
        batch, joints, _, frames = target.shape
        if frames < 3:
            return model_output.float().sum((1, 2, 3)) * 0
        std, dpos_std = self._position_step_stats(y, batch, target.device)
        pos_std = th.cat([
            std[:, ROOT_STAT_SLICE].unsqueeze(1),
            std[:, 0:3].unsqueeze(1).expand(-1, joints - 1, -1),
        ], dim=1)
        scale = (pos_std / dpos_std[:, None, None]).unsqueeze(-1)
        residual = (model_output[:, :, :3].float() - target[:, :, :3].float()) * scale
        lengths = th.as_tensor(y.get('lengths', [frames] * batch), device=target.device).reshape(batch)
        time = th.arange(frames, device=target.device)[None].expand(batch, -1)
        loop = self._coerce_bool_batch(y.get('is_loop'), batch, target.device, default=False)
        valid = (time < lengths[:, None]) & (lengths[:, None] >= 3)
        valid = valid & (loop[:, None] | (time + 2 < lengths[:, None]))

        def shifted(offset):
            index = (time + offset) % lengths[:, None].clamp_min(1)
            return residual.gather(-1, index[:, None, None].expand(batch, joints, 3, frames))

        acceleration = shifted(2) - 2 * shifted(1) + residual
        joint_valid = spat_mask.bool().reshape(batch, joints, 1, 1)
        err = acceleration.masked_fill(~(joint_valid & valid[:, None, None]), 0).square()
        counts = th.as_tensor(n_joints, device=target.device).reshape(batch)
        denom = (valid.sum(-1) * counts * 3).clamp_min(1)
        weight = self._per_sample_y_scalar(y, DPOS_LOSS_WEIGHT_KEY, batch, target.device, default=1.0)
        return err.sum((1, 2, 3)) / denom * weight

    def root_position_loss(self, target, model_output, y):
        """Root XYZ position MSE in canonical standardized units, per sample.

        Uses the root's own axis statistics, just like its share of pose MSE,
        without dilution by the joint count or motion-energy species weights.
        """
        batch, _, _, frames = target.shape
        lengths = th.as_tensor(y.get('lengths', [frames] * batch), device=target.device).reshape(batch)
        valid = th.arange(frames, device=target.device)[None] < lengths[:, None]
        error = (model_output[:, 0, :3].float() - target[:, 0, :3].float()).square()
        return error.masked_fill(~valid[:, None], 0).sum((1, 2)) / (3 * valid.sum(1)).clamp_min(1)

    def root_position_step_loss(self, target, model_output, y, *, channels=(0, 1, 2)):
        """Root step MSE in corpus position-step RMS units, per sample.

        Shares wrap handling and species weights with position_step_loss.
        Both diffusion and AE supervise all three root position channels.
        """
        batch_size, _max_joints, _n_feats, n_frames = target.shape
        if n_frames < 2:
            return target.new_zeros(batch_size)
        y = y if isinstance(y, dict) else {}
        err = self._root_step_error(target, model_output, y, channels) ** 2
        frame_weight = self._position_step_frame_weight(y, batch_size, n_frames, target.device, err.dtype)
        return self._root_weighted_mean(err, frame_weight, y)

    def root_acceleration_loss(self, target, model_output, y, *, channels=(0, 1, 2)):
        """MSE of the root's selected-axis acceleration, the step of its step, in
        :meth:`root_position_step_loss`'s units and species weight. A loop
        counts every circular second difference, an open clip only those of
        its inner steps. The step loss weighs an in-stride sway component by
        the square of its frequency and this one by the fourth power, so it
        reaches the stride band of fast gaits that the step loss underweights.
        ``[B]``."""
        batch_size, _max_joints, _n_feats, n_frames = target.shape
        if n_frames < 3:
            return target.new_zeros(batch_size)
        y = y if isinstance(y, dict) else {}
        step_err = self._root_step_error(target, model_output, y, channels)
        # acc[t] = step[t + 1] - step[t]; the last two read the wrap step.
        err = (th.roll(step_err, shifts=-1, dims=-1) - step_err) ** 2
        is_loop = self._coerce_bool_batch(y.get('is_loop'), batch_size, target.device, default=False)
        frame_weight = th.ones(batch_size, n_frames, device=target.device, dtype=err.dtype)
        frame_weight[:, -2:] = is_loop.to(dtype=err.dtype).unsqueeze(1)
        return self._root_weighted_mean(err, frame_weight, y)

    def _root_step_error(self, target, model_output, y, channels):
        """The output's minus the target's selected root-axis steps
        (:meth:`_position_steps`), in units of the position-step RMS.
        ``[B, len(channels), T]``."""
        batch_size = target.shape[0]
        std, dpos_std = self._position_step_stats(y, batch_size, target.device)
        axes = list(channels)
        scale = (std[:, ROOT_STAT_SLICE][:, axes] / dpos_std.view(batch_size, 1)).unsqueeze(-1)
        steps = self._position_steps
        return (steps(model_output[:, 0, axes, :]) - steps(target[:, 0, axes, :])) * scale

    def _root_weighted_mean(self, err, frame_weight, y):
        """Per-sample mean over weighted frames and axes, with species dpos weights."""
        batch_size = err.shape[0]
        denom = (frame_weight.sum(dim=1) * float(err.shape[1])).clamp_min(1.0)
        per_sample = (err * frame_weight.unsqueeze(1)).sum(dim=(1, 2)) / denom
        weight = self._per_sample_y_scalar(y, DPOS_LOSS_WEIGHT_KEY, batch_size, err.device, default=1.0)
        return per_sample * weight

    def _masked_smooth_l1(self, x, mask, beta=0.1, weight=None):
        """Smooth-L1 (Huber) of ``x`` averaged over entries where ``mask`` > 0.

        ``beta`` is small so the relative bone errors (typically << 1) stay in
        the quadratic regime while a few gross outliers are linearized (robust).
        ``weight`` scales entries without entering the denominator, so a weight
        below one lowers the loss instead of renormalizing it away. ``mask``,
        ``weight`` and ``x`` broadcast to the same shape.
        """
        absx = x.abs()
        huber = th.where(absx < beta, 0.5 * x * x / beta, absx - 0.5 * beta)
        denom = mask.sum().clamp(min=1.0)
        if weight is not None:
            huber = huber * weight
        return (huber * mask).sum() / denom

    def bone_vector_consistency_loss(self, pred_physical, target_physical, spat_mask, y, rotations=None, sample_weight=None):
        """Target-relative, rest-length-normalized parent->child bone-vector loss.

        For every bone ``v_j = p_j - p_parent`` on the position channel it
        penalizes ``|v_pred - v_gt| / max(rest_j, BONE_REST_FLOOR_FRAC * L)``
        through a smooth-L1. The vector difference carries both the bone's
        length error and its direction error (about ``length * angle``), so a
        bone that keeps its length but points the wrong way is penalized too.

        Why normalized by rest length: ``l_simple`` weights every joint's
        absolute position uniformly in standardized space, so a fixed position
        error is a tiny fraction of a long proximal bone but several bone
        lengths on a short distal one -- distal bones are left nearly
        unconstrained in both length and direction. Dividing by the rest bone
        length gives them proportionally larger gradient, the signal
        ``l_simple`` structurally omits (and the reason an unnormalized bone
        L2 is redundant with it). The floor at ``BONE_REST_FLOOR_FRAC * L``
        bounds that amplification so the shortest bones cannot dominate.

        Why the target is GT (not rest): anchoring on the ground-truth per-frame
        bone vector preserves genuinely animated bone-length deformation.

        Optional ``sample_weight`` scales samples before the existing global reduction.

        ``rotations`` optionally passes the (target, pred)
        ``_physical_rotation_matrices`` already computed for another loss.

        Leaf virtual bones: a joint's rotation is only visible on the position
        channel through its children, so a leaf's rotation (a hand, a foot, a
        head, a single-bone wing -- end sites are dropped in preprocessing) has
        no bone above to carry it and is held only by its own rotation channel.
        Every non-root leaf therefore adds one virtual child bone of unit length
        in its local frame, entering the same average as a real bone. Its
        direction is unknown (the mesh it carries is not in the features), so it
        is isotropic: ``|R_pred - R_gt|_F / sqrt(3)``, the RMS tip error over all
        unit directions, which is the sphere average of a real bone's relative
        direction error under the same rotation.

        Returns a scalar loss.
        """
        return self._bone_consistency_loss(
            pred_physical, target_physical, spat_mask, y, rotations, sample_weight)

    def bone_length_loss(self, pred_physical, target_physical, spat_mask, y, sample_weight=None):
        """Smooth-L1 of signed GT-relative bone-length error / floored rest length.

        Match each frame's target length, preserving animated stretch. Direction
        and rotations (including virtual leaf bones) do not contribute.
        """
        return self._bone_consistency_loss(
            pred_physical, target_physical, spat_mask, y,
            sample_weight=sample_weight, length_only=True)

    def _bone_consistency_loss(self, pred_physical, target_physical, spat_mask, y,
                               rotations=None, sample_weight=None, length_only=False):
        bs, max_joints, _n_feats, n_frames = pred_physical.shape
        device = pred_physical.device
        parents_list = (y or {}).get('parents')
        if parents_list is None:
            raise ValueError(
                "bone_vector_consistency_loss requires y['parents'] (per-sample "
                "parent arrays from the collate); none were provided."
            )
        # Rest bone lengths MUST come from the PHYSICAL rest pose: y['rest_geom']
        # is divided by L, so it carries only relative lengths.
        rest_pos = (y or {}).get('rest_pos_ric_hml')
        if rest_pos is None:
            rest_physical = (y or {}).get('rest_pose_physical')
            if rest_physical is not None:
                rest_pos = rest_physical[..., 0:3]
        if rest_pos is None:
            raise ValueError(
                "bone_vector_consistency_loss requires physical rest positions "
                "(y['rest_pos_ric_hml'] or y['rest_pose_physical']) to normalize "
                "by rest bone length."
            )
        if not th.is_tensor(rest_pos):
            rest_pos = th.as_tensor(rest_pos, dtype=pred_physical.dtype)
        rest_pos = rest_pos.to(device=device, dtype=pred_physical.dtype)  # [bs, J, 3]

        # Padded [bs, max_joints] parent-index tensor + per-joint bone validity
        # (False for the root, which has no parent bone). The collate's
        # PARENT_INDEX_KEY already is that tensor (-1 at roots and padding) and
        # arrives with the batch; without it the per-sample arrays are padded on
        # the host and copied once, not once per sample (each a stream sync).
        parent_index = (y or {}).get(PARENT_INDEX_KEY)
        if parent_index is None or tuple(parent_index.shape) != (bs, max_joints):
            padded = np.full((bs, max_joints), -1, dtype=np.int64)
            for b, p in enumerate(parents_list):
                p = np.asarray(p).reshape(-1)[:max_joints]
                padded[b, :p.size] = p
            parent_index = host_to_device(padded, device)
        parent_index = parent_index.to(device=device, dtype=th.long)
        parents_idx = parent_index.clamp(min=0)
        parent_is_bone = parent_index >= 0

        # The all-RIC view: a bone off the root must not carry its world XZ.
        pos_pred = root_relative_positions(pred_physical[:, :, 0:3, :])    # [bs, J, 3, T]
        pos_tgt = root_relative_positions(target_physical[:, :, 0:3, :])

        gather_bt = parents_idx.view(bs, max_joints, 1, 1).expand(-1, -1, 3, n_frames)
        vec_pred = pos_pred - th.gather(pos_pred, 1, gather_bt)               # [bs, J, 3, T]
        vec_tgt = pos_tgt - th.gather(pos_tgt, 1, gather_bt)
        gather_r = parents_idx.view(bs, max_joints, 1).expand(-1, -1, 3)
        rest_len = (rest_pos - th.gather(rest_pos, 1, gather_r)).norm(dim=2)    # [bs, J]

        # Valid bones: real (unpadded) joint, has a parent, and non-degenerate
        # rest length (skip zero-length bones, e.g. zero-offset leaves).
        joint_valid = spat_mask.float().transpose(1, 3).reshape(bs, max_joints) > 0.5
        bone_valid = joint_valid & parent_is_bone & (rest_len > 1e-4)          # [bs, J]

        length_scale = th.as_tensor(
            _length_scale_from_cond(y, like=pred_physical),
            dtype=pred_physical.dtype, device=device,
        ).reshape(-1)
        length_scale = length_scale.expand(bs) if length_scale.numel() == 1 else length_scale
        rest_floor = (BONE_REST_FLOOR_FRAC * length_scale).view(bs, 1)
        inv_rest = (1.0 / th.maximum(rest_len, rest_floor)).unsqueeze(-1)       # [bs, J, 1]

        if length_only:
            rel = (vec_pred.norm(dim=2) - vec_tgt.norm(dim=2)) * inv_rest
            valid_bt = bone_valid.unsqueeze(-1).to(rel.dtype).expand(-1, -1, n_frames)
            return self._masked_smooth_l1(rel, valid_bt, weight=sample_weight)

        rel = (vec_pred - vec_tgt).norm(dim=2) * inv_rest                     # [bs, J, T]

        # Leaf virtual bones (see docstring): a real, non-root joint no real
        # joint names as its parent.
        child_count = th.zeros(bs, max_joints, dtype=pred_physical.dtype, device=device)
        child_count.scatter_add_(1, parents_idx, (joint_valid & parent_is_bone).to(pred_physical.dtype))
        leaf = joint_valid & parent_is_bone & (child_count == 0)                # [bs, J]
        if rotations is None:
            rotations = (
                self._physical_rotation_matrices(target_physical),
                self._physical_rotation_matrices(pred_physical),
            )
        rot_tgt, rot_pred = rotations                                         # [bs, T, J, 3, 3]
        leaf_rel = ((rot_pred - rot_tgt).flatten(-2).norm(dim=-1) / math.sqrt(3.0)).transpose(1, 2)  # [bs, J, T]

        rel = th.cat([rel, leaf_rel.to(rel.dtype)], dim=1)
        valid_bt = th.cat([bone_valid, leaf], dim=1).unsqueeze(-1).to(pred_physical.dtype).expand(-1, -1, n_frames)
        return self._masked_smooth_l1(rel, valid_bt, weight=sample_weight)

    def limb_correlation_loss(self, pred_physical, target_physical, y, lengths):
        """Inter-limb coordination: the prediction's lagged limb-tip correlations
        against the target's. ``[B]``.

        Each limb tip (``y['limb_tip_mask']``) contributes its frame-to-frame
        step relative to the root joint, in units of L. For every ordered pair
        of tips and every lag of ``LIMB_CORR_LAG_FRACS`` the normalized
        correlation ``<v_i(t), v_j(t + lag)> / (|v_i| |v_j|)`` is compared with
        the target's; positive lags over ordered pairs also cover the negative
        ones. It matches whatever structure the target has -- antiphase gait,
        a jump's in-phase push, a one-sided strike -- and adds no gait prior.

        * Shifting the whole clip in time keeps it, so absolute timing stays
          with ``l_simple``; scaling the amplitude keeps it, so the blurred
          high-noise x0 is not pulled away from the posterior mean.
        * Pairs are weighted by ``sqrt(E_i E_j)`` of the target's tip energies:
          a limb the target holds still has no meaningful correlation and
          drops out. The norms carry ``LIMB_CORR_ENERGY_FLOOR`` so a still
          predicted limb gives a bounded gradient.
        * Per sample the weights are normalized by their sum, floored at the
          pair count times ``LIMB_CORR_ENERGY_FLOOR``: a many-legged rig weighs
          what a quadruped does, and a near-still clip fades out.
        """
        batch_size, max_joints, _n_feats, n_frames = pred_physical.shape
        tips = (y or {}).get('limb_tip_mask')
        if tips is None or n_frames < 2:
            return pred_physical.new_zeros(batch_size)
        device, dtype = pred_physical.device, pred_physical.dtype
        tips = tips.to(device=device, dtype=dtype).reshape(batch_size, max_joints)
        length_scale = th.as_tensor(
            _length_scale_from_cond(y, like=pred_physical), dtype=dtype, device=device,
        ).reshape(-1).expand(batch_size)
        n_steps = n_frames - 1
        step_valid = (
            th.arange(n_steps, device=device).view(1, n_steps)
            < (th.as_tensor(lengths, device=device).reshape(-1, 1) - 1)
        ).to(dtype)                                                              # [B, T-1]
        step_count = step_valid.sum(dim=1).clamp_min(1.0)
        norm_floor = (step_count * LIMB_CORR_ENERGY_FLOOR).view(batch_size, 1)

        def tip_steps(physical):
            pos = root_relative_positions(physical[:, :, 0:3, :])
            pos = pos - pos[:, 0:1]
            steps = (pos[..., 1:] - pos[..., :-1]) / length_scale.view(batch_size, 1, 1, 1)
            return steps * step_valid.view(batch_size, 1, 1, n_steps) * tips.view(batch_size, max_joints, 1, 1)

        def correlations(steps):
            sq_norm = steps.pow(2).sum(dim=(2, 3))                               # [B, J]
            inv_norm = (sq_norm + norm_floor).rsqrt()
            out = []
            for frac in LIMB_CORR_LAG_FRACS:
                lag = int(round(frac * n_steps))
                if lag >= n_steps:
                    continue
                gram = th.einsum('bict,bjct->bij', steps[..., :n_steps - lag], steps[..., lag:])
                out.append(gram * inv_norm.unsqueeze(2) * inv_norm.unsqueeze(1))
            return th.stack(out, dim=1), sq_norm                                 # [B, lags, J, J]

        corr_pred, _ = correlations(tip_steps(pred_physical))
        corr_tgt, tgt_sq_norm = correlations(tip_steps(target_physical))
        energy = tgt_sq_norm / step_count.view(batch_size, 1)
        pair = tips.unsqueeze(2) * tips.unsqueeze(1) * (1.0 - th.eye(max_joints, device=device, dtype=dtype))
        weight = (energy.unsqueeze(2) * energy.unsqueeze(1)).clamp_min(0.0).sqrt() * pair
        err = (corr_pred - corr_tgt).pow(2).mean(dim=1)                          # [B, J, J]
        denom = th.maximum(
            weight.sum(dim=(1, 2)),
            pair.sum(dim=(1, 2)).clamp_min(1.0) * LIMB_CORR_ENERGY_FLOOR,
        )
        return (err * weight).sum(dim=(1, 2)) / denom

    def _coerce_bool_batch(self, value, batch_size, device, default=False):
        if value is None:
            return th.full((batch_size,), bool(default), device=device, dtype=th.bool)
        value = th.as_tensor(value, device=device, dtype=th.bool).reshape(-1)
        if value.numel() == 1 and batch_size != 1:
            value = value.expand(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                f"Boolean batch value has length {value.numel()} but expected {batch_size}."
            )
        return value

    def loop_wrap_loss(self, model_output, y, n_joints):
        """Rotation seam continuity of loop samples, on denormalized outputs.

        A loop window holds no closing key (dataset._drop_loop_closing_frame)
        and is resampled periodically, so its last frame is one ordinary step
        BEFORE frame 0 -- the seam is a frame step like any other, not a
        repeated pose. The wrap step's geodesic angle ``(R[-1], R[0])`` must
        match the mean of its neighbours ``(R[-2], R[-1])`` and ``(R[0], R[1])``.
        The position seam is position_step_loss's wrap step.
        """
        batch_size, max_joints, _n_feats, n_frames = model_output.shape
        device = model_output.device
        zero = model_output.new_zeros(())
        if n_frames < 2:
            return {'loop_wrap_loss': zero}
        active = self._coerce_bool_batch(y.get('is_loop'), batch_size, device, default=False)
        n_joints_long = th.as_tensor(n_joints, device=device, dtype=th.long).reshape(-1)

        valid_joints = n_joints_long.clamp(min=0, max=max_joints)
        active_valid = active & (valid_joints > 0)
        active_weight = active_valid.to(dtype=model_output.dtype)
        active_denom = active_weight.sum().clamp(min=1.0)

        joint_mask = th.arange(max_joints, device=device).view(1, max_joints) < valid_joints.view(batch_size, 1)
        joint_weight = joint_mask.to(dtype=model_output.dtype)

        # Stacked, not indexed with a host list (a blocking upload per step).
        # At n_frames == 2 both neighbours are the wrap step itself and the
        # residual is zero: two frames have no seam to judge.
        seam_rot6d = th.stack(
            (
                model_output[:, :, 3:9, -2],
                model_output[:, :, 3:9, -1],
                model_output[:, :, 3:9, 0],
                model_output[:, :, 3:9, 1],
            ),
            dim=1,
        )                                                                    # [bs, 4, J, 6]
        seam_rots = rotation_6d_to_matrix_safe(seam_rot6d)                   # [bs, 4, J, 3, 3]
        step_before = geodesic_distance(seam_rots[:, 0], seam_rots[:, 1]).squeeze(-1)   # [bs, J]
        step_wrap = geodesic_distance(seam_rots[:, 1], seam_rots[:, 2]).squeeze(-1)
        step_after = geodesic_distance(seam_rots[:, 2], seam_rots[:, 3]).squeeze(-1)
        rot_residual = (step_wrap - 0.5 * (step_before + step_after)).abs()
        rot_per_sample = (rot_residual * joint_weight).sum(dim=1) / joint_weight.sum(dim=1).clamp(min=1.0)
        rot_per_sample = th.where(active_valid, rot_per_sample, th.zeros_like(rot_per_sample))
        rot_loss = (rot_per_sample * active_weight).sum() / active_denom

        return {'loop_wrap_loss': rot_loss}


LOSS_WEIGHTS = {
    'dpos_loss': 'lambda_dpos',
    'acc_loss': 'lambda_acc',
    'drot_loss': 'lambda_drot',
    'root_pos_loss': 'lambda_root_pos',
    'root_dpos_loss': 'lambda_root_dpos',
    'root_acc_loss': 'lambda_root_acc',
    'geodesic_loss': 'lambda_geo',
    'bone_loss': 'lambda_bone',
    'limb_corr_loss': 'lambda_limb_corr',
    'loop_wrap_loss': 'lambda_loop_wrap',
}


class MotionReconstructionLosses(MotionGeometryLosses):
    """AE objective using the existing motion losses, with no noise weighting.

    This fixed-window baseline requires equal valid lengths. Bone/global and
    loop/active-sample reductions retain the existing diffusion semantics.
    """

    def __init__(self, drot_joint_boost=0.0, **weights):
        unknown = weights.keys() - set(LOSS_WEIGHTS.values())
        if unknown:
            raise ValueError(f'Unknown loss weights: {sorted(unknown)}')
        self.weights = {key: float(weights.get(argument, 0.0)) for key, argument in LOSS_WEIGHTS.items()}
        self.drot_joint_boost = float(drot_joint_boost)
        if any(not math.isfinite(v) or v < 0 for v in (*self.weights.values(), self.drot_joint_boost)):
            raise ValueError('Loss weights and drot_joint_boost must be finite and nonnegative')

    def __call__(self, prediction, target, y, step_loss_scale=1.0):
        """Scale only dpos/drot/acc contributions; returned raw terms stay comparable."""
        if not math.isfinite(step_loss_scale) or not 0 <= step_loss_scale <= 1:
            raise ValueError('step_loss_scale must be finite and in [0, 1]')
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return self._losses(prediction.float(), target.float(), y, step_loss_scale)

    def _losses(self, prediction, target, y, step_loss_scale):
        if prediction.shape != target.shape or target.shape[2] != 9:
            raise ValueError('Expected matching [B,J,9,T] motion tensors')
        batch, joints, _, frames = target.shape
        lengths, n_joints = y['lengths'], y['n_joints']
        if not torch.all(lengths == frames):
            raise ValueError('Shared AE losses currently require fixed windows without time padding')
        valid = torch.arange(joints, device=target.device)[None] < n_joints[:, None]
        spat_mask = valid[:, None, None, :]
        per_sample = ((prediction - target).square() * valid[:, :, None, None]).sum((1, 2, 3))
        terms = {'pose_mse': (per_sample / (n_joints * frames * 9)).mean()}
        weights = self.weights

        if weights['dpos_loss']:
            terms['dpos_loss'] = self.position_step_loss(target, prediction, spat_mask, n_joints, y).mean()
        if weights['acc_loss']:
            terms['acc_loss'] = self.position_acceleration_loss(target, prediction, spat_mask, n_joints, y).mean()
        if weights['root_dpos_loss']:
            terms['root_dpos_loss'] = self.root_position_step_loss(target, prediction, y, channels=(0, 1, 2)).mean()
        if weights['root_acc_loss']:
            terms['root_acc_loss'] = self.root_acceleration_loss(target, prediction, y, channels=(0, 1, 2)).mean()
        if weights['root_pos_loss']:
            terms['root_pos_loss'] = self.root_position_loss(target, prediction, y).mean()
        if any(weights[key] for key in ('drot_loss', 'geodesic_loss', 'bone_loss', 'limb_corr_loss', 'loop_wrap_loss')):
            zeros = prediction.new_zeros(batch, joints, 3, frames)
            physical_gt, global_gt = canonical_to_physical_hml(
                torch.cat((target, zeros), dim=2), y, return_canonical_rotations=True)
            physical_pred, global_pred = canonical_to_physical_hml(
                torch.cat((prediction, zeros), dim=2), y, return_canonical_rotations=True)
            local = None
            if weights['geodesic_loss'] or weights['bone_loss']:
                local = (self._physical_rotation_matrices(physical_gt),
                         self._physical_rotation_matrices(physical_pred))
            if weights['drot_loss']:
                terms['drot_loss'] = self.rotation_step_loss(
                    physical_pred, physical_gt, spat_mask, lengths, y,
                    rotations=(global_gt, global_pred)).mean()
            if weights['geodesic_loss']:
                terms['geodesic_loss'] = self.geodesic_loss(
                    physical_gt, physical_pred, spat_mask, lengths, n_joints, rotations=local).mean()
            if weights['bone_loss']:
                terms['bone_loss'] = self.bone_vector_consistency_loss(
                    physical_pred, physical_gt, spat_mask, y, rotations=local)
            if weights['limb_corr_loss']:
                if 'limb_tip_mask' not in y:
                    raise KeyError('limb_corr_loss requires limb_tip_mask')
                terms['limb_corr_loss'] = self.limb_correlation_loss(physical_pred, physical_gt, y, lengths).mean()
            if weights['loop_wrap_loss']:
                terms.update(self.loop_wrap_loss(physical_pred, y, n_joints))
        terms['loss'] = terms['pose_mse'] + sum(
            weights[key] * terms[key] * (step_loss_scale if key in ('dpos_loss', 'drot_loss', 'acc_loss') else 1.0)
            for key in weights if weights[key])
        return terms

