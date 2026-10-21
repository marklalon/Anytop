from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import Mock
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.truebones.truebones_utils.dataset_tags import (  # noqa: E402
    SPECIES_VARIANTS,
    species_code_dim,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.truebones.truebones_utils.canonical_features import (  # noqa: E402
    JOINT_MODEL_CHANNELS,
    ROOT_MODEL_CHANNELS,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN  # noqa: E402
from model.anytop import AnyTop, InputProcess  # noqa: E402
from model.cfg_sampler import ClassifierFreeSpeciesModel  # noqa: E402


T5_DIM = 512
CODE_DIM = species_code_dim(T5_DIM)


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


def _make_model(species_cfg_drop_prob=0.15, species_cond=False, **extra):
    return AnyTop(
        max_joints=4,
        feature_len=12,
        latent_dim=8,
        ff_size=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=T5_DIM,
        species_cfg_drop_prob=species_cfg_drop_prob,
        species_cond=species_cond,
        **extra,
    )


def _make_y(species_code=None, **extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_pose': torch.randn(2, 4, 12, dtype=torch.float32),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'graph_dist': torch.zeros(2, 4, 4, dtype=torch.int64),
        'joints_relations': torch.zeros(2, 4, 4, dtype=torch.int64),
        'lengths': torch.tensor([3, 3], dtype=torch.int64),
        # The output coordinate frame is an unconditional model input: every
        # forward reads it, so a hand-built y has to carry it.
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    if species_code is not None:
        y['species_code'] = species_code
    y.update(extra)
    return y


def _open_species_heads(model, std=0.1):
    """Leave zero-init so the species path actually changes the output."""
    torch.nn.init.normal_(model.species_null_emb, std=std)
    torch.nn.init.normal_(model.seqTransDecoder.species_adaln[-1].weight, std=std)


class SpeciesConditionTest(unittest.TestCase):
    def test_disabled_by_default(self):
        model = AnyTop(max_joints=4, feature_len=12, latent_dim=8, ff_size=32,
                       num_layers=1, num_heads=2, dropout=0.0, cross_limb=True)
        self.assertFalse(model.species_cond)
        self.assertIsNone(model.species_projection)
        self.assertIsNone(model.species_null_emb)
        self.assertIsNone(model.seqTransDecoder.species_adaln)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(species_cond=True, species_cfg_drop_prob=1.5)

    def test_input_process_species_modulation_follows_species_cond(self):
        self.assertIsNone(_make_model(species_cond=False).input_process.species_joint_modulation)
        head = _make_model(species_cond=True).input_process.species_joint_modulation
        # Reads [species token || static joint embedding], emits (gamma, beta).
        self.assertEqual(head[0].in_features, 2 * 8)
        self.assertEqual(head[-1].out_features, 2 * 8)

    def test_projection_reads_body_plan_and_variant_axes(self):
        model = _make_model(species_cond=True)
        self.assertEqual(model.species_projection[0].in_features, T5_DIM + len(SPECIES_VARIANTS))

    def test_heads_start_as_identity(self):
        """Zero-init null and AdaLN output: a fresh model ignores the species."""
        model = _make_model(species_cond=True)
        head = model.seqTransDecoder.species_adaln
        self.assertEqual(torch.count_nonzero(head[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(head[-1].bias), 0)
        joint_head = model.input_process.species_joint_modulation
        self.assertEqual(torch.count_nonzero(joint_head[-1].weight), 0)
        self.assertEqual(torch.count_nonzero(joint_head[-1].bias), 0)
        self.assertEqual(torch.count_nonzero(model.species_null_emb), 0)

    def test_variant_axis_moves_the_species_token(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0).eval()
        code = torch.zeros(2, CODE_DIM)
        code[:, :T5_DIM] = torch.randn(T5_DIM)
        code[1, T5_DIM + SPECIES_VARIANTS.index('Fluttering')] = 1.0
        with torch.no_grad():
            token = model._build_species_token({'species_code': code}, 2, 'cpu', torch.float32)
        self.assertFalse(torch.allclose(token[0], token[1]))

    def test_inactive_rows_carry_the_null_token(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0).eval()
        _open_species_heads(model)
        y = {'species_code': torch.randn(2, CODE_DIM),
             'species_active': torch.tensor([True, False])}
        with torch.no_grad():
            token = model._build_species_token(y, 2, 'cpu', torch.float32)
        self.assertTrue(torch.equal(token[1], model.species_null_emb))
        self.assertFalse(torch.equal(token[0], model.species_null_emb))

    def test_species_token_drives_the_decoder_adaln(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0).eval()
        capture = _CaptureDecoder()
        model.seqTransDecoder = capture
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([1, 2])
        with torch.no_grad():
            model(x, t, y=_make_y(species_code=torch.randn(2, CODE_DIM)))
        cond = capture.last_kwargs['species_adaln_cond']
        self.assertEqual(tuple(cond.shape), (2, 8))

    def test_species_adaln_changes_the_full_decoder_output(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0).eval()
        _open_species_heads(model, std=0.5)
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([1, 2])
        code = torch.randn(2, CODE_DIM)
        with torch.no_grad():
            kept = model(x, t, y=_make_y(species_code=code))
            # Same additive token, AdaLN head switched off: the head must matter.
            head = model.seqTransDecoder.species_adaln
            model.seqTransDecoder.species_adaln = None
            no_head = model(x, t, y=_make_y(species_code=code))
            model.seqTransDecoder.species_adaln = head
        self.assertFalse(torch.allclose(kept, no_head))

    def test_full_drop_in_training_matches_explicit_null(self):
        model = _make_model(species_cond=True, species_cfg_drop_prob=1.0).train()
        _open_species_heads(model)
        model.seqTransDecoder = _TimestepEchoDecoder()
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([1, 2])
        y = _make_y(species_code=torch.randn(2, CODE_DIM))
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

    def test_species_cond_requires_species_code(self):
        model = _make_model(species_cond=True).eval()
        model.seqTransDecoder = _CaptureDecoder()
        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        ts = torch.tensor([1, 2], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'species_code'):
            model(x, ts, y=_make_y())

    def test_species_cfg_matches_explicit_branches(self):
        model = _make_model(species_cond=True).eval()
        _open_species_heads(model)
        x = torch.randn(2, 4, 12, 3)
        t = torch.tensor([98, 99])
        y = _make_y(species_code=torch.randn(2, CODE_DIM))
        with torch.no_grad():
            cond = model(x, t, y=dict(y, species_active=torch.ones(2, dtype=torch.bool)))
            uncond = model(x, t, y=dict(y, species_active=torch.zeros(2, dtype=torch.bool)))
            guided = ClassifierFreeSpeciesModel(model, 2.0, 100)(x, t, y=y)
        self.assertFalse(torch.allclose(cond, uncond))
        self.assertTrue(torch.allclose(guided, uncond + 2 * (cond - uncond), atol=1e-6))
        self.assertNotIn('species_active', y)

    def test_species_cfg_requires_trained_condition(self):
        from sample.conditioning import _wrap_species_cfg
        args = types.SimpleNamespace(species_cfg_scale=2.0, species_cfg_drop_prob=0.15,
                                     diffusion_steps=100)
        with self.assertRaises(SystemExit):
            _wrap_species_cfg(_make_model(), args)
        args.species_cfg_drop_prob = 0.0
        with self.assertRaises(SystemExit):
            _wrap_species_cfg(_make_model(species_cond=True), args)
        args.species_cfg_drop_prob = 0.15
        self.assertIsInstance(
            _wrap_species_cfg(_make_model(species_cond=True), args),
            ClassifierFreeSpeciesModel,
        )

    def test_disabled_species_cfg_preserves_forward_count(self):
        from sample.conditioning import _wrap_species_cfg
        from model.cfg_sampler import ClassifierFreeActionModel
        for scale in (0.0, 2.0, float('nan')):
            for action_cfg in (False, True):
                with self.subTest(scale=scale, action_cfg=action_cfg):
                    inner = Mock(return_value=torch.ones(2, 1, 1, 1))
                    model = ClassifierFreeActionModel(inner, 2.0, 100) if action_cfg else inner
                    args = types.SimpleNamespace(species_active=False, species_cfg_scale=scale,
                                                 species_cfg_drop_prob=0.0)
                    wrapped = _wrap_species_cfg(model, args)
                    self.assertIs(wrapped, model)
                    y = {'species_active': torch.zeros(2, dtype=torch.bool)}
                    wrapped(torch.zeros(2, 1, 1, 1), torch.full((2,), 99), y=y)
                    self.assertEqual(inner.call_count, 2 if action_cfg else 1)


class SpeciesJointModulationTest(unittest.TestCase):
    """InputProcess' per-joint (gamma, beta) driven by the species token."""

    LATENT = 8
    JOINTS = 4
    FRAMES = 3

    def _process(self):
        torch.manual_seed(0)
        return InputProcess(len(JOINT_MODEL_CHANNELS), len(ROOT_MODEL_CHANNELS), self.LATENT, T5_DIM,
                            dropout_prob=0.0, species_joint_modulation=True).eval()

    def _inputs(self, valid=None):
        torch.manual_seed(1)
        if valid is None:
            valid = torch.ones(2, self.JOINTS, dtype=torch.bool)
        return dict(
            x=torch.randn(2, self.JOINTS, FEATS_LEN, self.FRAMES),
            rest_pose=torch.randn(1, 2, self.JOINTS, FEATS_LEN),
            joints_embedded_names=torch.randn(2, self.JOINTS, T5_DIM) * valid.unsqueeze(-1),
            joint_valid=valid,
            joint_struct=torch.randn(2, self.JOINTS, JOINT_STRUCT_DIM) * valid.unsqueeze(-1),
            species_token=torch.randn(2, self.LATENT),
        )

    def _open(self, process, gamma, beta):
        """Random weights on the chosen halves of the output layer, zeros elsewhere."""
        out = process.species_joint_modulation[-1]
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
            if gamma:
                out.weight[:self.LATENT].normal_(std=0.5)
            if beta:
                out.weight[self.LATENT:].normal_(std=0.5)

    def _delta(self, process, **inputs):
        """Modulated minus unmodulated tokens, [frames + 1, B, J, d]. Both passes
        draw the same dropout masks, so in training mode they cancel too."""
        with torch.no_grad():
            torch.manual_seed(2)
            modulated = process(**inputs)
            head = process.species_joint_modulation
            out = head[-1]
            saved = out.weight.clone(), out.bias.clone()
            out.weight.zero_()
            out.bias.zero_()
            torch.manual_seed(2)
            plain = process(**inputs)
            out.weight.copy_(saved[0])
            out.bias.copy_(saved[1])
        return modulated - plain

    def test_zero_init_is_identity(self):
        process = self._process()
        self.assertEqual(float(self._delta(process, **self._inputs()).abs().max()), 0.0)

    def test_beta_shifts_every_frame_including_rest_equally(self):
        process = self._process()
        self._open(process, gamma=False, beta=True)
        delta = self._delta(process, **self._inputs())
        self.assertGreater(float(delta[0].abs().max()), 0.0)
        for frame in range(1, delta.shape[0]):
            torch.testing.assert_close(delta[frame], delta[0])

    def test_gamma_scales_motion_frames_only(self):
        process = self._process()
        self._open(process, gamma=True, beta=False)
        delta = self._delta(process, **self._inputs())
        self.assertEqual(float(delta[0].abs().max()), 0.0)
        self.assertGreater(float(delta[1:].abs().max()), 0.0)

    def test_modulation_is_joint_dependent(self):
        process = self._process()
        self._open(process, gamma=False, beta=True)
        delta = self._delta(process, **self._inputs())[0, 0]
        self.assertFalse(torch.allclose(delta[1], delta[2]))

    def test_padded_joints_are_untouched(self):
        process = self._process()
        self._open(process, gamma=True, beta=True)
        valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
        delta = self._delta(process, **self._inputs(valid))
        self.assertEqual(float(delta[:, 0, 2:].abs().max()), 0.0)
        self.assertEqual(float(delta[:, 1, 3:].abs().max()), 0.0)
        self.assertGreater(float(delta[:, 0, :2].abs().max()), 0.0)

    def test_modulation_ignores_the_elementwise_name_dropout(self):
        """The head reads the name before the train-only elementwise dropout."""
        torch.manual_seed(0)
        process = InputProcess(len(JOINT_MODEL_CHANNELS), len(ROOT_MODEL_CHANNELS), self.LATENT, T5_DIM,
                               dropout_prob=0.5, species_joint_modulation=True)
        self._open(process, gamma=False, beta=True)
        inputs = self._inputs()
        process.eval()
        eval_delta = self._delta(process, **inputs)
        process.train()
        # The rest frame's delta is beta alone (gamma is closed); it must not
        # depend on whether the elementwise dropout is active.
        train_delta = self._delta(process, **inputs)
        torch.testing.assert_close(train_delta[0], eval_delta[0])

    def test_whole_joint_name_drop_reaches_the_modulation(self):
        process = self._process()
        self._open(process, gamma=False, beta=True)
        inputs = self._inputs()
        kept = self._delta(process, **inputs)
        drop = torch.zeros(2, self.JOINTS, dtype=torch.bool)
        drop[0, 1] = True
        dropped = self._delta(process, joint_name_drop=drop, **inputs)
        self.assertFalse(torch.allclose(dropped[0, 0, 1], kept[0, 0, 1]))
        torch.testing.assert_close(dropped[0, 0, 2], kept[0, 0, 2])

    def test_species_token_must_match_the_module(self):
        process = self._process()
        inputs = self._inputs()
        inputs.pop('species_token')
        with self.assertRaises(ValueError):
            process(**inputs)
        plain = InputProcess(len(JOINT_MODEL_CHANNELS), len(ROOT_MODEL_CHANNELS), self.LATENT, T5_DIM,
                             dropout_prob=0.0).eval()
        with self.assertRaises(ValueError):
            plain(**self._inputs())

    def test_anytop_feeds_the_species_token_to_the_input(self):
        """Same species code, per-joint head open vs closed: the head must matter."""
        model = _make_model(species_cond=True, species_cfg_drop_prob=0.0).eval()
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([1, 2])
        y = _make_y(species_code=torch.randn(2, CODE_DIM))
        with torch.no_grad():
            closed = model(x, t, y=y)
            torch.nn.init.normal_(model.input_process.species_joint_modulation[-1].weight, std=0.5)
            opened = model(x, t, y=y)
        self.assertFalse(torch.allclose(opened, closed))


if __name__ == '__main__':
    unittest.main()
