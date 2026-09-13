"""fk_direction_loss: global-rotation supervision through bone directions.

Synthetic checks use a small tree whose position channel is exactly FK(rotation),
so the loss must be zero on the ground truth, positive when an ANCESTOR's rotation
is perturbed, and untouched for a LEAF (a leaf points no bone). The masking rules
(GT-inconsistent, collapsed, padding) are exercised one by one. The real-data check
certifies the FK convention on a rotation-only rig (loss ~0) and a Biped rig
(only translation-keyed bones masked).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion.gaussian_diffusion import (  # noqa: E402
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)

def _axis_angle_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _random_rotations(rng, shape):
    """Uniform-ish random rotation matrices of shape (*shape, 3, 3)."""
    q = rng.normal(size=shape + (4,))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(shape + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _matrix_to_6d_columns(m):
    """The dataset's 6D convention: the first two COLUMNS of the matrix
    (``Quaternions.rotation_matrix(cont6d=True)``), which is what
    ``rotation_6d_to_matrix_safe`` reads back."""
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def _fk_positions(local_rot, parents, rest_pos):
    """Reference FK under the own-rotation encoding: G[j] = G[p] @ R[j],
    pos[j] = pos[p] + G[p] @ (rest[j] - rest[p]); root at its rest position."""
    T, J = local_rot.shape[:2]
    G = np.zeros_like(local_rot)
    pos = np.zeros((T, J, 3))
    for j in range(J):  # parents precede children in this fixture
        p = parents[j]
        if p < 0:
            G[:, j] = local_rot[:, j]
            pos[:, j] = rest_pos[j]
        else:
            G[:, j] = G[:, p] @ local_rot[:, j]
            pos[:, j] = pos[:, p] + np.einsum('tij,j->ti', G[:, p], rest_pos[j] - rest_pos[p])
    return pos, G


class _Fixture:
    """A 7-joint tree: root -> spine -> head; root -> hipL -> kneeL -> footL; root -> hipR."""

    parents = np.array([-1, 0, 1, 0, 3, 4, 0], dtype=np.int64)
    rest_pos = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 1.3, 0.0],
        [0.0, 1.6, 0.05],
        [0.1, 0.9, 0.0],
        [0.1, 0.5, 0.02],
        [0.1, 0.1, 0.0],
        [-0.1, 0.9, 0.0],
    ])

    def __init__(self, n_frames=4, seed=0):
        rng = np.random.default_rng(seed)
        self.n_frames = n_frames
        J = len(self.parents)
        self.local_rot = _random_rotations(rng, (n_frames, J))
        self.pos, self.global_rot = _fk_positions(self.local_rot, self.parents, self.rest_pos)

    def features(self, local_rot=None, pos=None):
        """[1, J, 12, T] physical feature tensor (pos | rot6d | vel=0)."""
        local_rot = self.local_rot if local_rot is None else local_rot
        pos = self.pos if pos is None else pos
        T, J = local_rot.shape[:2]
        f = np.zeros((T, J, 12), dtype=np.float32)
        f[:, :, 0:3] = pos
        f[:, :, 3:9] = _matrix_to_6d_columns(local_rot)
        return torch.from_numpy(f).permute(1, 2, 0).unsqueeze(0).contiguous()

    def y(self, batch_size=1, max_joints=None):
        J = len(self.parents)
        max_joints = J if max_joints is None else max_joints
        rest = torch.zeros(batch_size, max_joints, 3)
        rest[:, :J] = torch.from_numpy(self.rest_pos.astype(np.float32))
        return {
            'parents': [self.parents.copy() for _ in range(batch_size)],
            'rest_pos_ric_hml': rest,
        }

    def spat_mask(self, batch_size=1, max_joints=None):
        J = len(self.parents)
        max_joints = J if max_joints is None else max_joints
        m = torch.zeros(batch_size, 1, 1, max_joints)
        m[..., :J] = 1.0
        return m


def _make_diffusion(lambda_fk=1.0):
    return GaussianDiffusion(
        betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        lambda_fk=lambda_fk,
    )


def _pad_features(feat, max_joints):
    b, J, F, T = feat.shape
    out = torch.zeros(b, max_joints, F, T, dtype=feat.dtype)
    out[:, :J] = feat
    return out


class FkDirectionLossTest(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()
        self.diffusion = _make_diffusion()

    def _loss(self, pred, target, fx=None, batch_size=1, max_joints=None):
        fx = self.fx if fx is None else fx
        return self.diffusion.fk_direction_loss(
            pred, target, fx.spat_mask(batch_size, max_joints), fx.y(batch_size, max_joints)
        )

    def test_constructor_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            _make_diffusion(lambda_fk=-0.1)

    def test_ground_truth_is_self_consistent(self):
        target = self.fx.features()
        terms = self._loss(target.clone(), target)
        self.assertLess(float(terms['fk_loss']), 1e-9)
        # fp32 acos next to 1 is only good to ~sqrt(eps) radians.
        self.assertLess(float(terms['fk_angle_deg']), 0.05)
        self.assertLess(float(terms['fk_gt_masked_frac']), 1e-9)

    def test_chain_global_rotations_matches_reference_fk(self):
        # Reverse the joint order so every child precedes its parent, then check
        # the iterative chain product still lands on the reference FK.
        fx = self.fx
        J = len(fx.parents)
        perm = np.arange(J)[::-1]
        inv = np.empty(J, dtype=np.int64)
        inv[perm] = np.arange(J)
        parents_perm = np.array([inv[fx.parents[j]] if fx.parents[j] >= 0 else -1 for j in perm])
        local = torch.from_numpy(fx.local_rot[:, perm]).unsqueeze(0).float()      # [1, T, J, 3, 3]
        y = {'parents': [parents_perm], 'rest_pos_ric_hml': torch.zeros(1, J, 3)}
        parents_idx, is_bone, depth = self.diffusion._padded_parents(y, 1, J, local.device)
        self.assertEqual(depth, 3)
        G = self.diffusion._chain_global_rotations(local, parents_idx, is_bone, depth)[0]
        expected = torch.from_numpy(fx.global_rot[:, perm]).float()
        self.assertTrue(torch.allclose(G, expected, atol=1e-5))

    def test_leaf_rotation_does_not_change_loss(self):
        target = self.fx.features()
        perturbed = self.fx.local_rot.copy()
        for leaf in (2, 5, 6):
            perturbed[:, leaf] = perturbed[:, leaf] @ _axis_angle_matrix([0, 0, 1], math.radians(40))
        pred = self.fx.features(local_rot=perturbed)
        terms = self._loss(pred, target)
        self.assertLess(float(terms['fk_loss']), 1e-9)

    def test_ancestor_rotation_error_is_graded_by_its_global_consequence(self):
        target = self.fx.features()
        perturbed = self.fx.local_rot.copy()
        # 10 degrees at hipL: the thigh->knee and knee->foot directions both move.
        perturbed[:, 3] = perturbed[:, 3] @ _axis_angle_matrix([1, 0, 0], math.radians(10))
        pred = self.fx.features(local_rot=perturbed)
        terms = self._loss(pred, target)
        self.assertGreater(float(terms['fk_loss']), 0.0)
        # Only the two bones below hipL are affected: the mean over the 6 graded
        # bones is 2/6 of the per-bone angle (<= 10 degrees).
        self.assertGreater(float(terms['fk_angle_deg']), 0.5)
        self.assertLess(float(terms['fk_angle_deg']), 10.0 * 2 / 6 + 1e-3)
        # The loss is 2(1 - cos) averaged over graded bone-frames.
        expected_max = 2 * (1 - math.cos(math.radians(10))) * 2 / 6
        self.assertLessEqual(float(terms['fk_loss']), expected_max + 1e-6)

    def test_gradient_reaches_rotation_channels_only(self):
        target = self.fx.features()
        perturbed = self.fx.local_rot.copy()
        perturbed[:, 0] = perturbed[:, 0] @ _axis_angle_matrix([0, 1, 0], math.radians(5))
        pred_data = self.fx.features(local_rot=perturbed)
        pred = pred_data.clone().requires_grad_(True)
        terms = self._loss(pred, target)
        terms['fk_loss'].backward()
        grad = pred.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad[:, :, 3:9].abs().sum().item(), 0.0)
        self.assertEqual(grad[:, :, 0:3].abs().sum().item(), 0.0)
        self.assertEqual(grad[:, :, 9:12].abs().sum().item(), 0.0)
        # A leaf's rotation points no bone: it gets no gradient.
        for leaf in (2, 5, 6):
            self.assertEqual(grad[:, leaf, 3:9].abs().sum().item(), 0.0)

    def test_gt_inconsistent_bone_is_masked_not_fitted(self):
        # Displace the knee LATERALLY in GT (a Biped-style translation key): its
        # own bone and the foot below it drop out of the loss, and pred == target
        # still scores zero.
        pos = self.fx.pos.copy()
        thigh = pos[:, 4] - pos[:, 3]
        lateral = np.cross(thigh, [0.0, 0.0, 1.0])
        lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True)
        pos[:, 4] += 0.4 * np.linalg.norm(thigh, axis=-1, keepdims=True) * lateral  # ~22 degrees
        target = self.fx.features(pos=pos)
        terms = self._loss(target.clone(), target)
        self.assertLess(float(terms['fk_loss']), 1e-9)
        # 2 of the 6 graded bones (knee, foot) are gated out on every frame.
        self.assertAlmostEqual(float(terms['fk_gt_masked_frac']), 2 / 6, places=5)

    def test_pure_bone_stretch_is_not_masked_and_not_penalized(self):
        # Stretch the thigh along its own axis: direction is unchanged, so the
        # bone stays graded and a matching prediction costs nothing.
        pos = self.fx.pos.copy()
        thigh = pos[:, 4] - pos[:, 3]
        delta = 0.35 * thigh
        pos[:, 4] += delta
        pos[:, 5] += delta  # the foot rides along, so its own bone keeps its direction
        target = self.fx.features(pos=pos)
        terms = self._loss(target.clone(), target)
        self.assertLess(float(terms['fk_loss']), 1e-9)
        self.assertLess(float(terms['fk_gt_masked_frac']), 1e-9)

    def test_collapsed_gt_bone_is_masked_without_nan(self):
        pos = self.fx.pos.copy()
        pos[:, 6] = pos[:, 0]  # hipR sits on the root in the clip
        target = self.fx.features(pos=pos)
        terms = self._loss(target.clone(), target)
        self.assertTrue(torch.isfinite(terms['fk_loss']))
        self.assertLess(float(terms['fk_loss']), 1e-9)
        self.assertAlmostEqual(float(terms['fk_gt_masked_frac']), 1 / 6, places=5)

    def test_padding_joints_and_mixed_batch(self):
        max_joints = 11
        target = _pad_features(self.fx.features(), max_joints)
        # Garbage in the padding rows must not leak into the loss.
        pred = target.clone()
        pred[:, 7:, :, :] = 3.0
        batch = torch.cat([pred, pred], dim=0)
        tgt = torch.cat([target, target], dim=0)
        terms = self._loss(batch, tgt, batch_size=2, max_joints=max_joints)
        self.assertLess(float(terms['fk_loss']), 1e-9)
        self.assertLess(float(terms['fk_gt_masked_frac']), 1e-9)

    def test_training_losses_reports_the_term_when_enabled(self):
        # Wire-through: the term shows up in training_losses under its flag,
        # and is absent when the flag is 0.
        class _Identity(torch.nn.Module):
            def forward(self, x, t, **kw):
                return x

        n_joints, n_frames = len(self.fx.parents), self.fx.n_frames
        x_start = self.fx.features()
        y = self.fx.y()
        y.update({
            'lengths': torch.full((1,), n_frames, dtype=torch.int64),
            'n_joints': torch.full((1,), n_joints, dtype=torch.int64),
            'joints_padding_mask': torch.ones(1, 1, 1, n_joints + 1, n_joints + 1),
            'canonical_feature_mean': torch.zeros(12),
            'canonical_feature_std': torch.ones(12),
        })
        # Identity model returns x_t, so target/pred differ by the noise; we only
        # check the key is wired through.
        t = torch.zeros(1, dtype=torch.long)
        terms = _make_diffusion(lambda_fk=0.3).training_losses(_Identity(), x_start, t, model_kwargs={'y': y})
        for key in ('fk_loss', 'fk_angle_deg', 'fk_gt_masked_frac'):
            self.assertIn(key, terms)
            self.assertTrue(torch.isfinite(terms[key]).all())
        terms_off = _make_diffusion(lambda_fk=0.0).training_losses(_Identity(), x_start, t, model_kwargs={'y': y})
        self.assertNotIn('fk_loss', terms_off)


_COND = REPO_ROOT / 'dataset' / 'merged' / 'cond.npy'
_UB_CLIP = REPO_ROOT / 'dataset' / 'unitybundles' / 'processed' / 'motions' / 'KI_Human_Walk01Forwards.npy'
_TB_CLIP = REPO_ROOT / 'dataset' / 'truebones' / 'zoo' / 'truebones_processed' / 'motions' / 'Buffalo_WalkLoop.npy'


@unittest.skipUnless(_COND.is_file() and _UB_CLIP.is_file() and _TB_CLIP.is_file(),
                     'merged cond / sample clips not available')
class FkDirectionLossRealDataTest(unittest.TestCase):
    """The loss reads stored features under the exporter's FK convention:
    a rotation-only rig scores ~0 on its own GT, a Biped rig masks exactly its
    translation-keyed bones."""

    @classmethod
    def setUpClass(cls):
        cls.cond = np.load(_COND, allow_pickle=True).item()
        cls.diffusion = _make_diffusion()

    def _terms(self, species, clip):
        c = self.cond[species]
        parents = np.asarray(c['parents'], dtype=np.int64)
        J = len(parents)
        m = np.load(clip).astype(np.float32)[:, :J]
        feat = torch.from_numpy(m).permute(1, 2, 0).unsqueeze(0).contiguous()
        y = {
            'parents': [parents],
            'rest_pos_ric_hml': torch.from_numpy(np.asarray(c['rest_pos_ric_hml'], dtype=np.float32)).unsqueeze(0),
        }
        mask = torch.ones(1, 1, 1, J)
        return self.diffusion.fk_direction_loss(feat.clone(), feat, mask, y)

    def test_rotation_only_rig_ground_truth_scores_zero(self):
        terms = self._terms('unitybundles/KI_Human', _UB_CLIP)
        self.assertLess(float(terms['fk_loss']), 1e-6)
        self.assertLess(float(terms['fk_angle_deg']), 0.1)
        self.assertLess(float(terms['fk_gt_masked_frac']), 1e-6)

    def test_biped_rig_masks_only_its_translation_keyed_bones(self):
        terms = self._terms('truebones/zoo/Buffalo', _TB_CLIP)
        # Graded bones may disagree with FK by up to FK_GT_AGREEMENT_DEG, so the
        # ground truth carries a small floor, not an exact zero.
        self.assertLess(float(terms['fk_loss']), 2 * (1 - math.cos(math.radians(5.0))))
        self.assertLess(float(terms['fk_angle_deg']), 2.0)
        # Clavicles, Spine, the Thighs and a tongue helper: a handful of the
        # 42 bones, on some or all frames.
        self.assertGreater(float(terms['fk_gt_masked_frac']), 0.02)
        self.assertLess(float(terms['fk_gt_masked_frac']), 0.25)


if __name__ == '__main__':
    unittest.main()
