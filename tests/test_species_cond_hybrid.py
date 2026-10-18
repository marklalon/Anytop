from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from model.anytop import AnyTop  # noqa: E402
from model.cfg_sampler import ClassifierFreeSpeciesModel  # noqa: E402


T5_DIM = 512


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


class _TimestepEchoDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs['tgt'] + kwargs['timesteps_embs'].sum()


class _ProbeHead(torch.nn.Module):
    """Wraps a head to capture the exact tensor it was called with."""

    def __init__(self, real):
        super().__init__()
        self.real = real
        self.last_input = None

    def forward(self, inp):
        self.last_input = inp
        return self.real(inp)


def _make_model(species_cfg_drop_prob=0.15, species_joint_cond=False):
    return AnyTop(
        max_joints=4,
        feature_len=12,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        species_cfg_drop_prob=species_cfg_drop_prob,
        species_joint_cond=species_joint_cond,
    )


def _joint_cond_inputs(batch=1, joints=4, frames=3):
    """Shapes InputProcess is called with from AnyTop.forward (anytop.py:796)."""
    return (
        torch.randn(batch, joints, 12, frames),
        torch.randn(1, batch, joints, 12),
        torch.randn(batch, joints, T5_DIM),
        torch.randn(batch, T5_DIM),
    )


def _joint_struct(batch=1, joints=4):
    """Stand-in structural descriptors; only their shape matters here."""
    return torch.randn(batch, joints, JOINT_STRUCT_DIM)


def _make_y(species_emb=None, **extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 12, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        # The output coordinate frame is an unconditional model input: every
        # forward reads it, so a hand-built y has to carry it.
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    if species_emb is not None:
        y['species_emb'] = species_emb
    y.update(extra)
    return y


class SpeciesHybridTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(max_joints=4, feature_len=12, latent_dim=8, ff_size=32,
                       num_layers=1, num_heads=2, dropout=0.0, cross_limb=True)
        self.assertFalse(hasattr(model, "species_film"))
        self.assertFalse(model.species_joint_cond)
        self.assertFalse(model.input_process.species_joint_cond)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(species_joint_cond=True, species_cfg_drop_prob=1.5)

    def test_cfg_drop_bypasses_entire_joint_modulation(self):
        model = _make_model(species_joint_cond=True).eval()
        ip = model.input_process
        torch.nn.init.normal_(ip.species_film_j[-1].weight, std=0.1)
        x, rest, joints, species = _joint_cond_inputs(batch=2)
        struct = _joint_struct(2)
        mixed = ip(x, rest, joints, species, joint_struct=struct,
                   species_active=torch.tensor([True, False]))
        ip.species_joint_cond = False
        plain = ip(x, rest, joints, joint_struct=struct)
        self.assertFalse(torch.allclose(mixed[:, 0], plain[:, 0]))
        self.assertTrue(torch.equal(mixed[:, 1], plain[:, 1]))

    def test_full_drop_in_training_matches_explicit_bypass(self):
        model = _make_model(species_joint_cond=True, species_cfg_drop_prob=1.0).train()
        torch.nn.init.normal_(model.input_process.species_film_j[-1].weight, std=0.1)
        model.seqTransDecoder = _CaptureDecoder()
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([1, 2])
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        dropped = model(x, t, y=y)
        explicit = model(x, t, y=dict(y, species_active=False))
        kept = model(x, t, y=dict(y, species_active=True))
        self.assertTrue(torch.equal(dropped, explicit))
        self.assertFalse(torch.allclose(dropped, kept))

    def test_eval_keeps_and_validates_species_mask(self):
        model = _make_model(species_cfg_drop_prob=1.0).eval()
        self.assertTrue(model._resolve_species_active(None, 2, 'cpu').all())
        with self.assertRaisesRegex(ValueError, 'batch dimension'):
            model._resolve_species_active([True, False, True], 2, 'cpu')

    def test_joint_cond_film_keeps_text_embedding_dim(self):
        plain = _make_model()
        fused = _make_model(species_joint_cond=True)
        self.assertEqual(plain.input_process.text_embedding.in_features, T5_DIM)
        self.assertEqual(fused.input_process.text_embedding.in_features, T5_DIM)
        self.assertIsNone(plain.input_process.species_film_j)
        # concat([joint_emb, species_emb]) in, (gamma_residual, beta) out.
        self.assertEqual(fused.input_process.species_film_j[0].in_features, 2 * T5_DIM)
        self.assertEqual(fused.input_process.species_film_j[0].out_features, 128)
        self.assertEqual(fused.input_process.species_film_j[-1].in_features, 128)
        self.assertEqual(fused.input_process.species_film_j[-1].out_features, 2 * T5_DIM)

    def test_joint_cond_film_starts_at_identity(self):
        model = _make_model(species_joint_cond=True)
        model.eval()
        ip = model.input_process
        x, rest_pose, joints, species = _joint_cond_inputs()
        joint_struct = _joint_struct(*joints.shape[:2])
        with torch.no_grad():
            fused = ip(x, rest_pose, joints, species, None, joint_struct)
            ip.species_joint_cond = False  # bypass the FiLM fusion entirely
            plain = ip(x, rest_pose, joints, None, None, joint_struct)
        self.assertTrue(torch.allclose(fused, plain))

    def test_joint_cond_film_differentiates_across_joints(self):
        """The whole point of the FiLM head: gamma must vary per joint. The additive
        form it replaced could not -- it added one per-species constant to every joint."""
        model = _make_model(species_joint_cond=True)
        head = model.input_process.species_film_j
        torch.nn.init.normal_(head[-1].weight, std=0.5)
        joints = torch.randn(1, 4, T5_DIM)
        species = torch.randn(1, T5_DIM).unsqueeze(1).expand(-1, 4, -1)
        with torch.no_grad():
            gamma_residual, _ = head(torch.cat([joints, species], dim=-1)).chunk(2, dim=-1)
        self.assertGreater(gamma_residual.std(dim=1).mean().item(), 1e-4)

    def test_joint_cond_film_reads_pre_dropout_joint_embeddings(self):
        """The FiLM head must read the *pre-dropout* joint embeddings. With
        dropout p=1 the dropout-ified copy is all zeros, so a head that
        accidentally consumed it would receive a zero joint part. The old
        per-joint spread assertion on the full output could not catch that
        (pose data alone spreads the joints), so probe the head's input
        directly instead."""
        model = _make_model(species_joint_cond=True)
        ip = model.input_process
        ip.joints_names_dropout.p = 1.0
        torch.nn.init.normal_(ip.species_film_j[-1].weight, std=0.5)
        probe = _ProbeHead(ip.species_film_j)
        ip.species_film_j = probe
        model.train()
        x, rest_pose, joints, species = _joint_cond_inputs()
        with torch.no_grad():
            ip(x, rest_pose, joints, species, None, _joint_struct(*joints.shape[:2]))
        # Head input: [B, J, 2*T5] = [joint_emb, species_emb]. The joint part
        # must be the un-dropped input, not the zeroed dropout copy.
        joint_part = probe.last_input[..., :T5_DIM]
        self.assertTrue(torch.allclose(joint_part, joints))

    def test_joint_cond_requires_species_emb(self):
        model = _make_model(species_joint_cond=True)
        model.eval()
        model.seqTransDecoder = _CaptureDecoder()
        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        with self.assertRaises(ValueError):
            model(x, ts, y=_make_y())  # no species_emb -> InputProcess raises

    def test_full_forward_joint_condition(self):
        model = _make_model(species_cfg_drop_prob=0.0,
                            species_joint_cond=True)
        model.eval()
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        out = model(x, ts, y=y)
        self.assertIsNotNone(capture.last_kwargs)
        self.assertEqual(out.shape, x.shape)

    def test_species_cfg_matches_explicit_film_branches(self):
        model = _make_model(species_joint_cond=True)
        model.eval()
        torch.nn.init.normal_(model.input_process.species_film_j[-1].weight, std=0.1)
        model.seqTransDecoder = _TimestepEchoDecoder()
        x = torch.randn(2, 4, 12, 3)
        t = torch.tensor([1, 2])
        y = _make_y(species_emb=torch.randn(2, T5_DIM))
        with torch.no_grad():
            cond = model(x, t, y=dict(y, species_active=torch.ones(2, dtype=torch.bool)))
            uncond = model(x, t, y=dict(y, species_active=torch.zeros(2, dtype=torch.bool)))
            guided = ClassifierFreeSpeciesModel(model, 2.0)(x, t, y=y)
        self.assertFalse(torch.allclose(cond, uncond))
        self.assertTrue(torch.allclose(guided, uncond + 2 * (cond - uncond)))
        self.assertNotIn('species_active', y)

    def test_species_cfg_requires_trained_joint_condition(self):
        from sample.conditioning import _wrap_species_cfg
        args = types.SimpleNamespace(species_cfg_scale=2.0, species_cfg_drop_prob=0.15)
        with self.assertRaises(SystemExit):
            _wrap_species_cfg(_make_model(), args)
        args.species_cfg_drop_prob = 0.0
        with self.assertRaises(SystemExit):
            _wrap_species_cfg(_make_model(species_joint_cond=True), args)
        args.species_cfg_drop_prob = 0.15
        self.assertIsInstance(
            _wrap_species_cfg(_make_model(species_joint_cond=True), args),
            ClassifierFreeSpeciesModel,
        )

    def test_disabled_species_cfg_preserves_forward_count(self):
        from sample.conditioning import _wrap_species_cfg
        from model.cfg_sampler import ClassifierFreeActionModel
        for scale in (0.0, 2.0, float('nan')):
            for action_cfg in (False, True):
                with self.subTest(scale=scale, action_cfg=action_cfg):
                    inner = Mock(return_value=torch.ones(2, 1, 1, 1))
                    model = ClassifierFreeActionModel(inner, 2.0) if action_cfg else inner
                    args = types.SimpleNamespace(species_active=False, species_cfg_scale=scale,
                                                 species_cfg_drop_prob=0.0)
                    wrapped = _wrap_species_cfg(model, args)
                    self.assertIs(wrapped, model)
                    y = {'species_active': torch.zeros(2, dtype=torch.bool)}
                    wrapped(torch.zeros(2, 1, 1, 1), torch.zeros(2), y=y)
                    self.assertEqual(inner.call_count, 2 if action_cfg else 1)


if __name__ == '__main__':
    unittest.main()
