from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from data_loaders.tensors import truebones_batch_collate  # noqa: E402
from data_loaders.truebones.data.dataset import resample_motion_features  # noqa: E402
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.motion_transformer import circular_phase_embedding  # noqa: E402
from utils.model_util import create_gaussian_diffusion  # noqa: E402


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_batch_item(
    is_loop: bool,
    loop_full_cycle: bool,
    loop_phase_length: float | None = None,
    playspeed_cond: float = 1.0,
):
    n_frames = 5
    n_joints = 2
    n_feats = 13
    max_joints = 3
    motion = np.zeros((n_frames, n_joints, n_feats), dtype=np.float32)
    tpose = np.zeros((n_joints, n_feats), dtype=np.float32)
    offsets = np.zeros((n_joints, 3), dtype=np.float32)
    graph = np.zeros((n_joints, n_joints), dtype=np.int64)
    relations = np.zeros((n_joints, n_joints), dtype=np.int64)
    names = np.zeros((n_joints, 4), dtype=np.float32)
    if loop_phase_length is None:
        loop_phase_length = float(n_frames)
    metadata = {
        'action_group': 'locomotion',
        'action_label': 'run, gallops forward',
        'translation_root_index': 0,
        'is_loop': is_loop,
        'loop_full_cycle': loop_full_cycle,
        'loop_phase_length': loop_phase_length,
        'playspeed_cond': playspeed_cond,
    }
    extra_cond = {'joint_mask_candidate_roots': np.zeros((n_joints,), dtype=np.bool_)}
    return (
        motion,
        n_frames,
        [-1, 0],
        tpose,
        offsets,
        graph,
        relations,
        'Horse',
        names,
        0,
        max_joints,
        metadata,
        'Horse_walk.npy',
        extra_cond,
    )


class NativeLoopTests(unittest.TestCase):
    def _make_diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(
            betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_LARGE,
            loss_type=LossType.MSE,
            lambda_loop_wrap=1.0,
        )

    def test_periodic_resample_preserves_loop_endpoints(self):
        motion = np.zeros((4, 1, 1), dtype=np.float32)
        motion[:, 0, 0] = np.array([0.0, 1.0, 2.0, 0.0], dtype=np.float32)

        resampled = resample_motion_features(motion, 7)

        self.assertAlmostEqual(float(resampled[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(resampled[-1, 0, 0]), 0.0)

    def test_circular_phase_gives_loop_endpoints_same_phase(self):
        emb = circular_phase_embedding(
            length=6,
            dim=8,
            batch_size=1,
            device=torch.device('cpu'),
            dtype=torch.float32,
            lengths=torch.tensor([5]),
        )

        self.assertTrue(torch.allclose(emb[1, 0], emb[-1, 0], atol=1e-6))

    def test_circular_phase_can_repeat_multiple_cycles(self):
        atol = 3e-6
        emb = circular_phase_embedding(
            length=8,
            dim=8,
            batch_size=1,
            device=torch.device('cpu'),
            dtype=torch.float32,
            lengths=torch.tensor([4.0]),
        )

        self.assertTrue(torch.allclose(emb[1, 0], emb[4, 0], atol=atol))
        self.assertTrue(torch.allclose(emb[4, 0], emb[7, 0], atol=atol))
        self.assertTrue(torch.allclose(emb[2, 0], emb[5, 0], atol=atol))
        self.assertTrue(torch.allclose(emb[3, 0], emb[6, 0], atol=atol))

    def test_truebones_collate_forwards_loop_flags_as_bool_tensors(self):
        _, cond = truebones_batch_collate([
            _make_batch_item(True, True, loop_phase_length=3.0, playspeed_cond=0.5),
            _make_batch_item(False, False, loop_phase_length=5.0, playspeed_cond=2.0),
        ])

        self.assertEqual(cond['y']['is_loop'].dtype, torch.bool)
        self.assertEqual(cond['y']['loop_full_cycle'].dtype, torch.bool)
        self.assertEqual(cond['y']['is_loop'].tolist(), [True, False])
        self.assertEqual(cond['y']['loop_full_cycle'].tolist(), [True, False])
        self.assertTrue(torch.equal(cond['y']['loop_phase_lengths'], torch.tensor([3.0, 5.0], dtype=torch.float32)))
        self.assertTrue(torch.equal(cond['y']['playspeed_cond'], torch.tensor([0.5, 2.0], dtype=torch.float32)))

    def test_anytop_coerces_default_playspeed_to_one(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )

        value = model._coerce_playspeed_cond(None, batch_size=2, device=torch.device('cpu'), dtype=torch.float32)

        self.assertTrue(torch.equal(value, torch.ones(2, 1, dtype=torch.float32)))

    def test_velocity_consistency_scales_physical_velocity_by_playspeed(self):
        diffusion = self._make_diffusion()
        model_output = torch.zeros(1, 1, 13, 7, dtype=torch.float32)
        model_output[0, 0, 0, :] = torch.linspace(0.0, 3.0, steps=7)
        model_output[0, 0, 9, :] = 1.0
        spat_mask = torch.ones(1, 1, 1, 1, dtype=torch.float32)

        loss = diffusion.velocity_consistency_loss(
            model_output,
            spat_mask,
            n_joints=torch.tensor([1]),
            y={'playspeed_cond': torch.tensor([4.0 / 7.0], dtype=torch.float32)},
        )

        self.assertLess(float(loss.item()), 1e-6)

    def test_loop_wrap_loss_skips_non_loop_samples(self):
        diffusion = self._make_diffusion()
        model_output = torch.zeros(2, 3, 13, 6, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0
        model_output[1, :, 0:3, -2:] = 100.0

        y = {
            'is_loop': torch.tensor([True, False]),
            'loop_full_cycle': torch.tensor([True, True]),
            'translation_root_index': [0, 0],
        }
        terms = diffusion.loop_wrap_loss(
            model_output,
            y,
            n_joints=torch.tensor([3, 3]),
        )

        self.assertLess(float(terms['loop_wrap_loss'].item()), 1e-6)

    def test_loop_wrap_components_use_the_actual_seam(self):
        diffusion = self._make_diffusion()
        model_output = torch.zeros(1, 2, 13, 6, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0

        model_output[:, :, 0:3, 1:3] = 10.0
        model_output[:, :, 0:3, 3:5] = -10.0
        model_output[:, :, 9:12, 0] = 5.0
        model_output[:, :, 9:12, -1] = 0.0
        model_output[:, :, 12:13, 0:2] = 1.0
        model_output[:, :, 12:13, 4:6] = 0.0

        model_output[:, :, 3:9, 1] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        model_output[:, :, 3:9, 4] = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])

        y = {
            'is_loop': torch.tensor([True]),
            'loop_full_cycle': torch.tensor([True]),
            'translation_root_index': [0],
        }
        terms = diffusion.loop_wrap_loss(
            model_output,
            y,
            n_joints=torch.tensor([2]),
        )

        self.assertLess(float(terms['loop_wrap_pose'].item()), 1e-6)
        self.assertLess(float(terms['loop_wrap_rot'].item()), 1e-6)
        self.assertNotIn('loop_wrap_vel', terms)
        self.assertNotIn('loop_wrap_contact', terms)
        self.assertLess(float(terms['loop_wrap_terminal_vel'].item()), 1e-6)

    def test_loop_wrap_terminal_velocity_uses_physical_step_scale(self):
        diffusion = self._make_diffusion()
        model_output = torch.zeros(1, 1, 13, 7, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0
        model_output[0, 0, 0, -1] = -0.5
        model_output[0, 0, 9, -1] = 1.0
        y = {
            'is_loop': torch.tensor([True]),
            'loop_full_cycle': torch.tensor([True]),
            'translation_root_index': [0],
            'playspeed_cond': torch.tensor([4.0 / 7.0], dtype=torch.float32),
        }

        terms = diffusion.loop_wrap_loss(model_output, y, n_joints=torch.tensor([1]))

        self.assertLess(float(terms['loop_wrap_terminal_vel'].item()), 1e-6)

    def test_create_gaussian_diffusion_preserves_loop_args(self):
        class Args:
            noise_schedule = 'cosine'
            diffusion_steps = 10
            timestep_respacing = ''
            sigma_small = True
            lambda_geo = 0.0
            lambda_vel = 0.0
            lambda_loop_wrap = 0.75
            temporal_span_seam_loss_weight = 0.0
            temporal_span_seam_width = 2

        diffusion = create_gaussian_diffusion(Args())
        self.assertEqual(diffusion.lambda_loop_wrap, 0.75)

    def test_anytop_forwards_loop_phase_metadata(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
            loop_cond_prob=1.0,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder

        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        y = {
            'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            'rest_pose': torch.randn(2, 4, 13, dtype=torch.float32),
            'n_joints': torch.tensor([4, 3], dtype=torch.int64),
            'joints_names_embs': torch.zeros(2, 4, 512, dtype=torch.float32),
            # Unconditional model input -- every forward reads the frame.
            'canonical_feature_mean': torch.zeros(13, dtype=torch.float32),
            'canonical_feature_std': torch.ones(13, dtype=torch.float32),
            'is_loop': torch.tensor([True, False]),
            'lengths': torch.tensor([3, 3], dtype=torch.int64),
            'loop_phase_lengths': torch.tensor([2.0, 3.0], dtype=torch.float32),
        }

        model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs['loop_phase_mask'], y['is_loop']))
        self.assertTrue(torch.equal(capture_decoder.last_kwargs['lengths'], y['loop_phase_lengths']))

    def test_anytop_loop_phase_requires_full_cycle_when_available(self):
        model = AnyTop(
            max_joints=4,
            feature_len=13,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder

        x = torch.randn(2, 4, 13, 3, dtype=torch.float32)
        y = {
            'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            'rest_pose': torch.randn(2, 4, 13, dtype=torch.float32),
            'n_joints': torch.tensor([4, 3], dtype=torch.int64),
            'joints_names_embs': torch.zeros(2, 4, 512, dtype=torch.float32),
            # Unconditional model input -- every forward reads the frame.
            'canonical_feature_mean': torch.zeros(13, dtype=torch.float32),
            'canonical_feature_std': torch.ones(13, dtype=torch.float32),
            'is_loop': torch.tensor([True, True]),
            'loop_full_cycle': torch.tensor([True, False]),
            'lengths': torch.tensor([3, 3], dtype=torch.int64),
        }

        model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs['loop_phase_mask'], torch.tensor([True, False])))


    # ------------------------------------------------------------------
    # Generation-side loop_phase_length (the value training always supplied
    # and generation, until this was wired, never did).
    # ------------------------------------------------------------------

    def test_generation_phase_length_inverts_the_training_identity(self):
        """k = round(playspeed*T/L) and phase_len = (T-1)/k + 1 -- the exact
        inverse of what the loader computes from its tile count."""
        from sample.generate import _resolve_loop_phase_length

        entry = {'loop_period_by_action': {'walk': 25.0, 'run': 20.0}, 'loop_period_median': 25.0}
        # 60 output frames over a 25-frame walk cycle -> 2 cycles.
        phase_length, cycles = _resolve_loop_phase_length(entry, 60, 1.0, 'walk, strides forward')
        self.assertEqual(cycles, 2)
        self.assertAlmostEqual(phase_length, (60 - 1) / 2 + 1, places=6)
        # A 20-frame run cycle in the same window is 3, not 2: the period is an
        # action property, so the label -- not just the species -- selects it.
        phase_length, cycles = _resolve_loop_phase_length(entry, 60, 1.0, 'run, forward, fast')
        self.assertEqual(cycles, 3)
        self.assertAlmostEqual(phase_length, (60 - 1) / 3 + 1, places=6)
        # playspeed scales the source length, so it scales the cycle count too.
        _, cycles = _resolve_loop_phase_length(entry, 60, 2.0, 'walk')
        self.assertEqual(cycles, 5)

    def test_generation_phase_length_falls_back_to_one_cycle(self):
        """No baked period -> the pre-fix value, so the change can only help."""
        from sample.generate import _resolve_loop_phase_length

        phase_length, cycles = _resolve_loop_phase_length({}, 60, 1.0, 'walk')
        self.assertEqual(phase_length, 60.0)
        self.assertIsNone(cycles)
        # A label naming no core word falls through to the species median.
        entry = {'loop_period_by_action': {'walk': 25.0}, 'loop_period_median': 30.0}
        _, cycles = _resolve_loop_phase_length(entry, 60, 1.0, 'wobbles about')
        self.assertEqual(cycles, 2)

    def test_generation_phase_length_never_asks_for_a_fractional_cycle(self):
        """A window the model is told is closed must hold a whole number of
        cycles; a fractional request makes it warp phase to close the seam."""
        from sample.generate import _resolve_loop_phase_length

        for period in (13.0, 17.0, 23.0, 41.0, 57.0):
            entry = {'loop_period_by_action': {}, 'loop_period_median': period}
            phase_length, cycles = _resolve_loop_phase_length(entry, 60, 1.0, '')
            self.assertEqual(cycles, int(cycles))
            self.assertGreaterEqual(cycles, 1)
            expected = ((60 - 1) / cycles + 1) if cycles > 1 else 60.0
            self.assertAlmostEqual(phase_length, expected, places=6)

    def test_collate_turns_generation_metadata_into_loop_phase_lengths(self):
        """The metadata key only reaches the model if the collate emits the
        plural tensor name the forward reads."""
        items = [
            _make_batch_item(True, True, loop_phase_length=30.5),
            _make_batch_item(True, True, loop_phase_length=20.0),
        ]
        _, cond = truebones_batch_collate(items)
        self.assertIn('loop_phase_lengths', cond['y'])
        self.assertTrue(torch.equal(
            cond['y']['loop_phase_lengths'], torch.tensor([30.5, 20.0], dtype=torch.float32)
        ))



if __name__ == '__main__':
    unittest.main()
