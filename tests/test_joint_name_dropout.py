"""Whole-joint name dropout: the model must be trained without the joint's name.

``--dropout_prob`` thins the 768-dim name vector elementwise, so every joint keeps
its identity and the model never learns to place a joint from rest_pose /
graph_dist / joints_relations alone. These cover the substitution that does hide
it, and the two ways it could silently do nothing: leaking the name back through
the species FiLM pathway, and writing content into padding slots.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from model.anytop import AnyTop  # noqa: E402


T5_DIM = 512


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


class _ProbeHead(torch.nn.Module):
    """Wraps a head to capture the exact tensor it was called with."""

    def __init__(self, real):
        super().__init__()
        self.real = real
        self.last_input = None

    def forward(self, inp):
        self.last_input = inp
        return self.real(inp)


def _make_model(joint_name_drop_prob=0.0, joint_name_drop_all_prob=0.0,
                species_joint_cond=False, max_joints=4):
    return AnyTop(
        max_joints=max_joints,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        species_joint_cond=species_joint_cond,
        joint_name_drop_prob=joint_name_drop_prob,
        joint_name_drop_all_prob=joint_name_drop_all_prob,
    )


def _joint_struct(batch=2, joints=4):
    """Stand-in structural descriptors; only their shape matters here."""
    return torch.randn(batch, joints, JOINT_STRUCT_DIM)


def _joint_cond_inputs(batch=2, joints=4, frames=3):
    """Shapes InputProcess is called with from AnyTop.forward."""
    return (
        torch.randn(batch, joints, 13, frames),
        torch.randn(1, batch, joints, 13),
        torch.randn(batch, joints, T5_DIM),
        torch.randn(batch, T5_DIM),
    )


def _padding_mask(n_joints, max_joints):
    """Same outer-product mask data_loaders.tensors.n_joints_to_mask builds."""
    n_joints = torch.as_tensor(n_joints, dtype=torch.int64)
    valid = torch.arange(max_joints + 1).expand(len(n_joints), max_joints + 1) < (
        n_joints.unsqueeze(1) + 1
    )
    mask = valid.unsqueeze(2).float() * valid.unsqueeze(1).float()
    return mask.unsqueeze(1).unsqueeze(1)


def _make_y(n_joints=(4, 4), max_joints=4, **extra):
    batch = len(n_joints)
    y = {
        'joints_padding_mask': _padding_mask(n_joints, max_joints),
        'rest_pose': torch.randn(batch, max_joints, 13, dtype=torch.float32),
        'n_joints': torch.tensor(n_joints, dtype=torch.int64),
        'joints_names_embs': torch.randn(batch, max_joints, T5_DIM, dtype=torch.float32),
        'joint_struct': _joint_struct(batch, max_joints),
        'lengths': torch.tensor([3] * batch, dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(13, dtype=torch.float32),
        'canonical_feature_std': torch.ones(13, dtype=torch.float32),
    }
    y.update(extra)
    return y


class JointNameDropoutTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(max_joints=4, feature_len=13, latent_dim=8, ff_size=32,
                       num_layers=1, num_heads=2, dropout=0.0, cross_limb=True)
        self.assertEqual(model.joint_name_drop_prob, 0.0)
        self.assertEqual(model.joint_name_drop_all_prob, 0.0)
        self.assertIsNone(model.input_process.unknown_joint_name)

    def test_invalid_probs_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(joint_name_drop_prob=1.5)
        with self.assertRaises(ValueError):
            _make_model(joint_name_drop_all_prob=-0.1)

    def test_state_dict_key_only_when_enabled(self):
        """An existing checkpoint must rebuild with no extra key: load_model is
        strict about missing keys outside the QK-norm allowlist, so creating the
        parameter unconditionally would refuse every pre-C1 checkpoint."""
        off = _make_model()
        on = _make_model(joint_name_drop_prob=0.15)
        self.assertNotIn('input_process.unknown_joint_name', off.state_dict())
        self.assertIn('input_process.unknown_joint_name', on.state_dict())

    def test_unknown_vector_is_not_zero_initialized(self):
        """Zero-init would place the substitute inside the padding signal and
        start the parameter with no gradient signal to leave it."""
        ip = _make_model(joint_name_drop_prob=0.15).input_process
        self.assertGreater(float(ip.unknown_joint_name.detach().norm()), 1.0)

    def test_eval_keeps_every_name(self):
        ip = _make_model(joint_name_drop_prob=1.0, joint_name_drop_all_prob=1.0).input_process
        ip.eval()
        _, _, joints, _ = _joint_cond_inputs()
        valid = torch.ones(joints.shape[:2], dtype=torch.bool)
        self.assertTrue(torch.equal(ip._drop_joint_names(joints, valid), joints))

    def test_drop_all_replaces_every_valid_name(self):
        ip = _make_model(joint_name_drop_all_prob=1.0).input_process
        ip.train()
        _, _, joints, _ = _joint_cond_inputs()
        valid = torch.ones(joints.shape[:2], dtype=torch.bool)
        dropped = ip._drop_joint_names(joints, valid)
        expected = ip.unknown_joint_name.expand_as(joints)
        self.assertTrue(torch.allclose(dropped, expected))

    def test_substitute_is_not_rescaled(self):
        """A token substitution, not a variance-preserving dropout: the dropped
        row must be the parameter itself, never parameter/(1-p)."""
        ip = _make_model(joint_name_drop_prob=1.0).input_process
        ip.train()
        _, _, joints, _ = _joint_cond_inputs()
        valid = torch.ones(joints.shape[:2], dtype=torch.bool)
        dropped = ip._drop_joint_names(joints, valid)
        self.assertTrue(torch.allclose(dropped[0, 0], ip.unknown_joint_name))

    def test_padding_rows_are_untouched(self):
        ip = _make_model(joint_name_drop_prob=1.0, joint_name_drop_all_prob=1.0).input_process
        ip.train()
        _, _, joints, _ = _joint_cond_inputs()
        valid = torch.tensor([[True, True, False, False],
                              [True, True, True, True]])
        dropped = ip._drop_joint_names(joints, valid)
        self.assertTrue(torch.equal(dropped[0, 2:], joints[0, 2:]))
        self.assertTrue(torch.allclose(dropped[0, :2], ip.unknown_joint_name.expand(2, T5_DIM)))

    def test_per_joint_rate_matches_prob(self):
        ip = _make_model(joint_name_drop_prob=0.5).input_process
        ip.train()
        joints = torch.randn(256, 8, T5_DIM)
        valid = torch.ones(joints.shape[:2], dtype=torch.bool)
        dropped = ip._drop_joint_names(joints, valid)
        rate = (dropped != joints).all(dim=-1).float().mean().item()
        self.assertGreater(rate, 0.4)
        self.assertLess(rate, 0.6)

    def test_film_sees_the_dropped_names(self):
        """The leak this ordering exists to prevent: if the species FiLM head read
        the pre-drop copy it would hand the model the very name just withheld,
        and the whole mechanism would be a no-op under --species_joint_cond."""
        model = _make_model(joint_name_drop_all_prob=1.0, species_joint_cond=True)
        ip = model.input_process
        probe = _ProbeHead(ip.species_film_j)
        ip.species_film_j = probe
        ip.train()
        x, rest_pose, joints, species = _joint_cond_inputs()
        with torch.no_grad():
            ip(x, rest_pose, joints, species,
               torch.ones(joints.shape[:2], dtype=torch.bool),
               _joint_struct(*joints.shape[:2]))
        joint_part = probe.last_input[..., :T5_DIM]
        self.assertFalse(torch.allclose(joint_part, joints))
        self.assertTrue(torch.allclose(joint_part, ip.unknown_joint_name.expand_as(joints)))

    def test_forward_derives_joint_valid_from_padding_mask(self):
        """End to end: AnyTop.forward must read per-joint validity off the
        padding mask's [1:, 1:] diagonal, so the padded joint of the second
        sample keeps its (zero) name row."""
        model = _make_model(joint_name_drop_all_prob=1.0, max_joints=4)
        model.train()
        captured = {}
        real = model.input_process.forward

        def spy(x, rest_pose, joints_embedded_names, species_emb=None, joint_valid=None,
                joint_struct=None):
            captured['joint_valid'] = joint_valid
            return real(x, rest_pose, joints_embedded_names, species_emb, joint_valid,
                        joint_struct)

        model.input_process.forward = spy
        model.seqTransDecoder = _CaptureDecoder()
        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        out = model(x, ts, y=_make_y(n_joints=(4, 2)))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(
            captured['joint_valid'],
            torch.tensor([[True, True, True, True], [True, True, False, False]]),
        ))


if __name__ == '__main__':
    unittest.main()
