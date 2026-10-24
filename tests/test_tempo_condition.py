"""The tempo condition: resample_speed_cond enters as each window frame's
physical time, added per frame (never through the timestep token), and is
CFG-droppable via tempo_cfg_drop_prob / tempo_active."""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.reference import REST_GEOM_DIM  # noqa: E402
from model.anytop import AnyTop, create_sin_embedding  # noqa: E402
from model.cfg_sampler import ClassifierFreeTempoModel  # noqa: E402

T5_DIM = 512
LATENT = 8


class _TimestepEchoDecoder(torch.nn.Module):
    def forward(self, **kwargs):
        return kwargs['tgt'] + kwargs['timesteps_embs'].sum()


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_model(tempo_cfg_drop_prob=0.1):
    model = AnyTop(
        max_joints=4, feature_len=12, latent_dim=LATENT, ff_size=32, num_layers=1,
        num_heads=2, dropout=0.0, cross_limb=True, t5_out_dim=T5_DIM,
        tempo_cfg_drop_prob=tempo_cfg_drop_prob,
    )
    # A trained projection: the zero-init last layer would hide every effect.
    torch.nn.init.normal_(model.tempo_projection[-1].weight, std=0.5)
    return model


def _make_y(frames=3, **extra):
    y = {
        'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
        'rest_geom': torch.linspace(-1.0, 1.0, 2 * 4 * REST_GEOM_DIM).view(2, 4, REST_GEOM_DIM),
        'n_joints': torch.tensor([4, 3], dtype=torch.int64),
        'joints_names_embs': torch.zeros(2, 4, T5_DIM, dtype=torch.float32),
        'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
        'lengths': torch.tensor([frames, frames], dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
        'canonical_feature_std': torch.ones(12, dtype=torch.float32),
    }
    y.update(extra)
    return y


def _decoder_input(model, y, frames=3, x=None):
    capture = _CaptureDecoder()
    model.seqTransDecoder = capture
    if x is None:
        x = torch.zeros(2, 4, 12, frames)
    with torch.no_grad():
        model(x, torch.tensor([1, 2]), y=y)
    return capture.last_kwargs


class TempoConditionTest(unittest.TestCase):
    def test_fresh_projection_is_a_no_op(self):
        model = AnyTop(max_joints=4, feature_len=12, latent_dim=LATENT, ff_size=32,
                       num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
                       t5_out_dim=T5_DIM).eval()
        emb = model._tempo_frame_embedding(
            _make_y(resample_speed_cond=torch.tensor([0.5, 2.0])), 2, 3, torch.device('cpu'))
        self.assertTrue(torch.equal(emb, torch.zeros(3, 2, LATENT)))

    def test_speed_never_reaches_the_timestep_token(self):
        model = _make_model().eval()
        slow = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([0.5, 0.5])))
        fast = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(slow['timesteps_embs'], fast['timesteps_embs']))
        self.assertFalse(torch.allclose(slow['tgt'], fast['tgt']))

    def test_frame_times_follow_the_loader_resample_step(self):
        model = _make_model().eval()
        frames, speed = 5, torch.tensor([2.0, 2.0])
        y = _make_y(frames, resample_speed_cond=speed, is_loop=torch.tensor([True, False]))
        emb = model._tempo_frame_embedding(y, 2, frames, torch.device('cpu'))
        # Loop: step L/T = 2; open: (L-1)/(T-1) = 9/4.
        steps = torch.tensor([2.0, 9.0 / 4.0])
        times = torch.arange(frames, dtype=torch.float32).view(1, -1, 1) * steps.view(-1, 1, 1)
        expected = model.tempo_projection(create_sin_embedding(times, LATENT)).transpose(0, 1)
        self.assertTrue(torch.allclose(emb, expected, atol=1e-6))

    def test_rest_pose_slot_carries_no_tempo(self):
        model = _make_model().eval()
        slow = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([0.5, 0.5])))
        fast = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(slow['tgt'][0], fast['tgt'][0]))

    def test_dropped_rows_ignore_the_speed(self):
        model = _make_model().eval()
        off = torch.zeros(2, dtype=torch.bool)
        slow = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([0.5, 0.5]), tempo_active=off))
        fast = _decoder_input(model, _make_y(resample_speed_cond=torch.tensor([2.0, 2.0]), tempo_active=off))
        self.assertTrue(torch.equal(slow['tgt'], fast['tgt']))

    def test_training_drop_and_eval_keep(self):
        model = _make_model(tempo_cfg_drop_prob=1.0)
        cpu = torch.device('cpu')
        model.train()
        self.assertFalse(bool(model._resolve_tempo_active(None, 4, cpu).any()))
        model.eval()
        self.assertTrue(bool(model._resolve_tempo_active(None, 4, cpu).all()))
        with self.assertRaises(ValueError):
            model._resolve_tempo_active(torch.ones(3, dtype=torch.bool), 2, cpu)

    def test_invalid_drop_prob_rejected(self):
        with self.assertRaises(ValueError):
            _make_model(tempo_cfg_drop_prob=1.5)

    def test_tempo_cfg_matches_explicit_branches(self):
        model = _make_model().eval()
        model.seqTransDecoder = _TimestepEchoDecoder()
        x, t = torch.randn(2, 4, 12, 3), torch.tensor([98, 99])
        y = _make_y(resample_speed_cond=torch.tensor([0.5, 2.0]))
        with torch.no_grad():
            cond = model(x, t, y=dict(y, tempo_active=torch.ones(2, dtype=torch.bool)))
            uncond = model(x, t, y=dict(y, tempo_active=torch.zeros(2, dtype=torch.bool)))
            guided = ClassifierFreeTempoModel(model, 0.5, 100)(x, t, y=y)
        self.assertFalse(torch.allclose(cond, uncond))
        self.assertTrue(torch.allclose(guided, uncond + 0.5 * (cond - uncond), atol=1e-6))
        self.assertNotIn('tempo_active', y)

    def test_wrap_requires_a_trained_drop(self):
        from sample.conditioning import _wrap_tempo_cfg
        model = _make_model()
        args = types.SimpleNamespace(tempo_cfg_scale=1.0, tempo_cfg_drop_prob=0.0, diffusion_steps=100)
        self.assertIs(_wrap_tempo_cfg(model, args), model)
        args.tempo_cfg_scale = 0.5
        with self.assertRaises(SystemExit):
            _wrap_tempo_cfg(model, args)
        args.tempo_cfg_drop_prob = 0.1
        self.assertIsInstance(_wrap_tempo_cfg(model, args), ClassifierFreeTempoModel)
        args.tempo_cfg_scale = float('nan')
        with self.assertRaises(SystemExit):
            _wrap_tempo_cfg(model, args)


if __name__ == '__main__':
    unittest.main()
