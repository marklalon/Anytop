from __future__ import annotations

import math
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
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.motion_transformer import circular_phase_embedding  # noqa: E402
from utils.model_util import create_gaussian_diffusion, load_model  # noqa: E402


# circular_phase_embedding's window closure is exact only in real arithmetic:
# in fp32 the closing phase misses 2*pi*f by ~(2*pi*f)*eps, and with the top
# frequency at dim // 2 the residual is bounded by pi * dim * eps -- it grows
# with dim. A flat 1e-6 passes at (6, 8) (7.0e-07) but fails at the production
# shape (6.7e-05 at (61, 256)). 2x margin: measured worst case over length
# 6..481, dim 8..1024 is 0.70 of the bound.
def _closure_atol(dim: int) -> float:
    return 2.0 * math.pi * dim * torch.finfo(torch.float32).eps


# Toy shapes plus the production one (--num_frames 60 -> T = 61, latent_dim 256).
_PHASE_SHAPES = ((6, 8), (8, 8), (61, 256))


class _CaptureDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs['tgt']


def _make_batch_item(
    is_loop: bool,
    resample_speed_cond: float = 1.0,
):
    n_frames = 5
    n_joints = 2
    n_feats = FEATS_LEN
    max_joints = 3
    motion = np.zeros((n_frames, n_joints, n_feats), dtype=np.float32)
    tpose = np.zeros((n_joints, n_feats), dtype=np.float32)
    offsets = np.zeros((n_joints, 3), dtype=np.float32)
    graph = np.zeros((n_joints, n_joints), dtype=np.int64)
    relations = np.zeros((n_joints, n_joints), dtype=np.int64)
    names = np.zeros((n_joints, 4), dtype=np.float32)
    metadata = {
        'action_group': 'locomotion',
        'action_label': 'run, gallops forward',
        'translation_root_index': 0,
        'is_loop': is_loop,
        'resample_speed_cond': resample_speed_cond,
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
        for length, dim in _PHASE_SHAPES:
            with self.subTest(length=length, dim=dim):
                emb = circular_phase_embedding(
                    length=length,
                    dim=dim,
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                )

                self.assertEqual(tuple(emb.shape), (length, dim))
                # Slot 0 is the T-pose token; motion frames 1..length-1 close on
                # themselves.
                self.assertTrue(torch.equal(emb[0], torch.zeros(dim)))
                atol = _closure_atol(dim)
                self.assertTrue(
                    torch.allclose(emb[1], emb[-1], atol=atol),
                    f'closure residual {(emb[1] - emb[-1]).abs().max().item():.3e} '
                    f'exceeds atol={atol:.3e} at (length={length}, dim={dim})',
                )

    def test_circular_phase_wraps_exactly_once_per_window(self):
        """The period is the window, never a cycle count: no interior frame
        repeats the first frame's phase, so the embedding cannot tell the
        model how many gait cycles the window holds."""
        for length, dim in _PHASE_SHAPES:
            with self.subTest(length=length, dim=dim):
                emb = circular_phase_embedding(
                    length=length,
                    dim=dim,
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                )

                first = emb[1]
                for frame in range(2, length - 1):
                    self.assertFalse(
                        torch.allclose(emb[frame], first, atol=1e-3), frame
                    )
                self.assertTrue(
                    torch.allclose(emb[length - 1], first, atol=_closure_atol(dim))
                )

    def test_truebones_collate_forwards_loop_flags_as_bool_tensors(self):
        _, cond = truebones_batch_collate([
            _make_batch_item(True, resample_speed_cond=0.5),
            _make_batch_item(False, resample_speed_cond=2.0),
        ])

        self.assertEqual(cond['y']['is_loop'].dtype, torch.bool)
        self.assertEqual(cond['y']['is_loop'].tolist(), [True, False])
        self.assertNotIn('loop_full_cycle', cond['y'])
        self.assertNotIn('loop_phase_lengths', cond['y'])
        self.assertTrue(torch.equal(cond['y']['resample_speed_cond'], torch.tensor([0.5, 2.0], dtype=torch.float32)))

    def test_anytop_coerces_default_resample_speed_to_one(self):
        model = AnyTop(
            max_joints=4,
            feature_len=12,
            latent_dim=8,
            ff_size=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            cross_limb=True,
        )

        value = model._coerce_resample_speed_cond(None, batch_size=2, device=torch.device('cpu'), dtype=torch.float32)

        self.assertTrue(torch.equal(value, torch.ones(2, 1, dtype=torch.float32)))

    def test_load_model_remaps_legacy_playspeed_projection_keys(self):
        # Checkpoints written before the rename carry ``playspeed_projection.*``;
        # they must load into ``resample_speed_projection`` unchanged.
        kwargs = dict(
            max_joints=4, feature_len=12, latent_dim=8, ff_size=32,
            num_layers=1, num_heads=2, dropout=0.0, cross_limb=True,
        )
        source = AnyTop(**kwargs)
        legacy_state = {
            (key.replace('resample_speed_projection.', 'playspeed_projection.', 1)
             if key.startswith('resample_speed_projection.') else key): value
            for key, value in source.state_dict().items()
        }
        self.assertTrue(any(key.startswith('playspeed_projection.') for key in legacy_state))

        restored = AnyTop(**kwargs)
        load_model(restored, legacy_state)

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(restored.state_dict()[key], value), key)

    def test_velocity_consistency_compares_root_relative_xz_at_physical_step_scale(self):
        # Joint 0 is the translation root, joint 1 a child. The root sways in
        # world X (vel ch9) while its RIC X stays structurally zero; the child's
        # RIC X path advances by its WORLD velocity minus the root's sway, and
        # every Y path by its own velocity (get_rifke leaves Y alone). At
        # resample_speed 4/7 over 7 frames the physical step scale is 0.5.
        diffusion = self._make_diffusion()
        n_frames = 7
        step_scale = 0.5
        ramp = torch.arange(n_frames, dtype=torch.float32)
        model_output = torch.zeros(1, 2, 12, n_frames, dtype=torch.float32)
        model_output[0, 0, 9, :] = 0.25
        model_output[0, 0, 10, :] = 0.5
        model_output[0, 0, 1, :] = ramp * 0.5 * step_scale
        model_output[0, 1, 9, :] = 1.0
        model_output[0, 1, 0, :] = ramp * (1.0 - 0.25) * step_scale
        model_output[0, 1, 10, :] = 0.5
        model_output[0, 1, 1, :] = ramp * 0.5 * step_scale
        spat_mask = torch.ones(1, 1, 1, 2, dtype=torch.float32)
        n_joints = torch.tensor([2])
        y = {
            'resample_speed_cond': torch.tensor([4.0 / 7.0], dtype=torch.float32),
            'translation_root_index': [0],
        }

        loss = diffusion.velocity_consistency_loss(model_output, spat_mask, n_joints, y=y)
        self.assertLess(float(loss.item()), 1e-6)

        # Negative controls, so the pass above cannot come from a mask: the
        # same tensor read at resample_speed 1 ...
        loss_unit_step = diffusion.velocity_consistency_loss(
            model_output, spat_mask, n_joints, y={'translation_root_index': [0]},
        )
        self.assertGreater(float(loss_unit_step.item()), 1e-3)
        # ... or with the child's RIC X following its world velocity as if the
        # root did not move.
        world_frame = model_output.clone()
        world_frame[0, 1, 0, :] = ramp * 1.0 * step_scale
        loss_world = diffusion.velocity_consistency_loss(world_frame, spat_mask, n_joints, y=y)
        self.assertGreater(float(loss_world.item()), 1e-3)

    def test_velocity_consistency_is_zero_on_a_stored_clip_layout(self):
        # Ground-truth tensors satisfy ric[t+1]-ric[t] == vel[t]-vel_root[t]
        # (XZ) exactly, for every joint including the root. Build one the way
        # preprocessing does -- world positions, then get_rifke and world deltas
        # -- with the root as joint 1 to cover a non-zero translation_root_index.
        diffusion = self._make_diffusion()
        rng = np.random.default_rng(0)
        n_frames, n_joints, root = 9, 3, 1
        world = rng.normal(size=(n_frames, n_joints, 3)).astype(np.float32)
        ric = world.copy()
        ric[..., 0] -= world[:, root:root + 1, 0]
        ric[..., 2] -= world[:, root:root + 1, 2]
        vel = np.zeros_like(world)
        vel[:-1] = world[1:] - world[:-1]
        feats = np.zeros((n_frames, n_joints, 12), dtype=np.float32)
        feats[..., 0:3] = ric
        feats[..., 9:12] = vel
        model_output = torch.as_tensor(feats).permute(1, 2, 0).unsqueeze(0)
        spat_mask = torch.ones(1, 1, 1, n_joints, dtype=torch.float32)

        loss = diffusion.velocity_consistency_loss(
            model_output, spat_mask, torch.tensor([n_joints]), y={'translation_root_index': [root]},
        )
        self.assertLess(float(loss.item()), 1e-10)

        wrong_root = diffusion.velocity_consistency_loss(
            model_output, spat_mask, torch.tensor([n_joints]), y={'translation_root_index': [0]},
        )
        self.assertGreater(float(wrong_root.item()), 1e-3)

    def test_loop_wrap_loss_skips_non_loop_samples(self):
        diffusion = self._make_diffusion()
        model_output = torch.zeros(2, 3, 12, 6, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0
        model_output[1, :, 0:3, -2:] = 100.0

        y = {
            'is_loop': torch.tensor([True, False]),
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
        model_output = torch.zeros(1, 2, 12, 6, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0

        model_output[:, :, 0:3, 1:3] = 10.0
        model_output[:, :, 0:3, 3:5] = -10.0
        model_output[:, :, 9:12, 0] = 5.0
        model_output[:, :, 9:12, -1] = 0.0

        model_output[:, :, 3:9, 1] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
        model_output[:, :, 3:9, 4] = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])

        y = {
            'is_loop': torch.tensor([True]),
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
        # The root's RIC X/Z are structurally zero and masked out of the
        # terminal term, so the seam has to be checked on a child joint (X)
        # and on the root's height (Y): pos[0] - pos[-1] == vel[-1] * 0.5 at
        # resample_speed 4/7 over 7 frames.
        diffusion = self._make_diffusion()
        model_output = torch.zeros(1, 2, 12, 7, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0
        model_output[0, 1, 0, -1] = -0.5
        model_output[0, 1, 9, -1] = 1.0
        model_output[0, 0, 1, -1] = -0.25
        model_output[0, 0, 10, -1] = 0.5
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
            'resample_speed_cond': torch.tensor([4.0 / 7.0], dtype=torch.float32),
        }

        terms = diffusion.loop_wrap_loss(model_output, y, n_joints=torch.tensor([2]))
        self.assertLess(float(terms['loop_wrap_terminal_vel'].item()), 1e-6)

        # Same tensor at resample_speed 1 leaves the seam open.
        y_unit = dict(y)
        y_unit.pop('resample_speed_cond')
        terms_unit = diffusion.loop_wrap_loss(model_output, y_unit, n_joints=torch.tensor([2]))
        self.assertGreater(float(terms_unit['loop_wrap_terminal_vel'].item()), 1e-3)

    def _closure_output(self, root_vel_x, *, n_joints=2, root_index=0):
        """[1, n_joints, 12, T] physical output whose root X velocity rows are ``root_vel_x``."""
        n_frames = len(root_vel_x)
        model_output = torch.zeros(1, n_joints, 12, n_frames, dtype=torch.float32)
        model_output[:, :, 3, :] = 1.0
        model_output[:, :, 7, :] = 1.0
        model_output[0, root_index, 9, :] = torch.tensor(root_vel_x, dtype=torch.float32)
        return model_output

    def test_loop_root_xz_closure_is_zero_when_all_rows_sum_to_zero(self):
        diffusion = self._make_diffusion()
        # Four visible steps of +1 closed by a terminal wrap row of -4: the
        # identity every stored loop tensor satisfies.
        model_output = self._closure_output([1.0, 1.0, 1.0, 1.0, -4.0])
        # A DC bias on a NON-root joint and on the root's Y velocity must not
        # count: only the root's XZ path is integrated by the exporter.
        model_output[0, 1, 9:12, :] = 0.3
        model_output[0, 0, 10, :] = 0.7
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
        }

        terms = diffusion.loop_root_xz_closure_loss(model_output, y, n_joints=torch.tensor([2]))

        self.assertLess(float(terms['loop_root_xz_closure'].item()), 1e-6)
        self.assertLess(float(terms['loop_root_xz_drift'].item()), 1e-6)

    def test_loop_root_xz_closure_measures_the_per_cycle_dc_bias(self):
        diffusion = self._make_diffusion()
        # 0.01 per frame over 6 rows: the seam pop is 0.06, the loss its square.
        model_output = self._closure_output([0.01] * 6)
        model_output[0, 0, 11, :] = -0.01  # Z bias of the same size
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
        }

        terms = diffusion.loop_root_xz_closure_loss(model_output, y, n_joints=torch.tensor([2]))

        expected_drift = float(np.hypot(0.06, 0.06))
        self.assertAlmostEqual(float(terms['loop_root_xz_drift'].item()), expected_drift, places=6)
        self.assertAlmostEqual(float(terms['loop_root_xz_closure'].item()), expected_drift ** 2, places=6)

    def test_loop_root_xz_closure_skips_non_loop_and_averages_over_active(self):
        diffusion = self._make_diffusion()
        closed = self._closure_output([1.0, 1.0, -2.0])
        drifting = self._closure_output([1.0, 1.0, 1.0])
        one_shot = self._closure_output([5.0, 5.0, 5.0])
        model_output = torch.cat([closed, drifting, one_shot], dim=0)
        y = {
            'is_loop': torch.tensor([True, True, False]),
            'translation_root_index': [0, 0, 0],
        }

        terms = diffusion.loop_root_xz_closure_loss(model_output, y, n_joints=torch.tensor([2, 2, 2]))

        # (0 + 3^2) / 2 active samples; the one-shot's 15 never enters.
        self.assertAlmostEqual(float(terms['loop_root_xz_closure'].item()), 4.5, places=6)
        self.assertAlmostEqual(float(terms['loop_root_xz_drift'].item()), 1.5, places=6)

    def test_loop_root_xz_closure_uses_physical_step_scale(self):
        diffusion = self._make_diffusion()
        model_output = self._closure_output([1.0] * 7)
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
            # 7 output frames drawn from 4 source frames: step_scale = 3 / 6.
            'resample_speed_cond': torch.tensor([4.0 / 7.0], dtype=torch.float32),
        }

        terms = diffusion.loop_root_xz_closure_loss(model_output, y, n_joints=torch.tensor([2]))

        self.assertAlmostEqual(float(terms['loop_root_xz_drift'].item()), 7.0 * 0.5, places=6)

    def test_loop_root_xz_closure_follows_translation_root_index_and_masks_invalid(self):
        diffusion = self._make_diffusion()
        # Joint 1 is the translation root here; joint 0 carries a bias that
        # must be ignored.
        model_output = self._closure_output([1.0] * 4, root_index=1)
        model_output[0, 0, 9, :] = 9.0
        y_valid = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [1],
        }
        y_invalid = dict(y_valid, translation_root_index=[5])

        valid = diffusion.loop_root_xz_closure_loss(model_output, y_valid, n_joints=torch.tensor([2]))
        invalid = diffusion.loop_root_xz_closure_loss(model_output, y_invalid, n_joints=torch.tensor([2]))

        self.assertAlmostEqual(float(valid['loop_root_xz_drift'].item()), 4.0, places=6)
        self.assertEqual(float(invalid['loop_root_xz_closure'].item()), 0.0)

    def test_training_losses_adds_root_closure_only_when_weighted_but_always_logs_drift(self):
        class _DriftingModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                out = torch.zeros_like(x)
                out[:, :, 3, :] = 1.0
                out[:, :, 7, :] = 1.0
                out[:, 0, 9, :] = 0.1  # root X velocity: constant drift
                return out

        batch_size, n_joints, n_feats, n_frames = 1, 2, FEATS_LEN, 5
        model_kwargs = {
            'y': {
                'lengths': torch.full((batch_size,), n_frames, dtype=torch.int64),
                'n_joints': torch.full((batch_size,), n_joints, dtype=torch.int64),
                'joints_padding_mask': torch.ones(batch_size, 1, 1, n_joints + 1, n_joints + 1),
                'rest_pos_ric_hml': torch.zeros(n_joints, 3),
                'canonical_feature_mean': torch.zeros(n_feats),
                'canonical_feature_std': torch.ones(n_feats),
                'is_loop': torch.tensor([True]),
                'translation_root_index': [0],
            }
        }
        x_start = torch.zeros(batch_size, n_joints, n_feats, n_frames)
        x_start[:, :, 3, :] = 1.0
        x_start[:, :, 7, :] = 1.0
        t = torch.tensor([0], dtype=torch.int64)

        def _terms(**weights):
            diffusion = GaussianDiffusion(
                betas=np.array([0.001, 0.002, 0.003], dtype=np.float64),
                model_mean_type=ModelMeanType.START_X,
                model_var_type=ModelVarType.FIXED_LARGE,
                loss_type=LossType.MSE,
                **weights,
            )
            return diffusion.training_losses(_DriftingModel(), x_start, t, model_kwargs=model_kwargs)

        expected_drift = 0.1 * n_frames
        baseline = _terms(lambda_loop_wrap=0.04)
        self.assertAlmostEqual(float(baseline['loop_root_xz_drift'].item()), expected_drift, places=5)
        unweighted_total = float(baseline['loss'].sum().item())

        weighted = _terms(lambda_loop_wrap=0.04, lambda_loop_root_closure=2.0)
        self.assertAlmostEqual(
            float(weighted['loss'].sum().item()) - unweighted_total,
            2.0 * expected_drift ** 2,
            places=5,
        )

        closure_only = _terms(lambda_loop_root_closure=1.0)
        self.assertNotIn('loop_wrap_loss', closure_only)
        self.assertAlmostEqual(float(closure_only['loop_root_xz_drift'].item()), expected_drift, places=5)

        off = _terms()
        self.assertNotIn('loop_root_xz_drift', off)

    def test_create_gaussian_diffusion_preserves_loop_args(self):
        class Args:
            noise_schedule = 'cosine'
            diffusion_steps = 10
            timestep_respacing = ''
            sigma_small = True
            lambda_geo = 0.0
            lambda_vel = 0.0
            lambda_loop_wrap = 0.75
            lambda_loop_root_closure = 1.5
            temporal_span_seam_loss_weight = 0.0
            temporal_span_seam_width = 2

        diffusion = create_gaussian_diffusion(Args())
        self.assertEqual(diffusion.lambda_loop_wrap, 0.75)
        self.assertEqual(diffusion.lambda_loop_root_closure, 1.5)

    def test_anytop_forwards_is_loop_as_the_only_loop_condition(self):
        model = AnyTop(
            max_joints=4,
            feature_len=12,
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

        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        y = {
            'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            'rest_pose': torch.randn(2, 4, 12, dtype=torch.float32),
            'n_joints': torch.tensor([4, 3], dtype=torch.int64),
            'joints_names_embs': torch.zeros(2, 4, 512, dtype=torch.float32),
            'joint_struct': torch.zeros(2, 4, JOINT_STRUCT_DIM, dtype=torch.float32),
            # Unconditional model input -- every forward reads the frame.
            'canonical_feature_mean': torch.zeros(12, dtype=torch.float32),
            'canonical_feature_std': torch.ones(12, dtype=torch.float32),
            'is_loop': torch.tensor([True, False]),
            'lengths': torch.tensor([3, 3], dtype=torch.int64),
        }

        model(x, torch.tensor([1, 2], dtype=torch.int64), y=y)

        self.assertIsNotNone(capture_decoder.last_kwargs)
        self.assertTrue(torch.equal(capture_decoder.last_kwargs['loop_phase_mask'], y['is_loop']))
        self.assertNotIn('lengths', capture_decoder.last_kwargs)

    def test_decoder_loop_tables_are_per_sample_and_period_free(self):
        """Loop samples get the circular table, one wrap per window; non-loop
        samples get a zero phase and the absolute table. No per-sample period
        exists any more, so the picture is the same for every batch."""
        from model.motion_transformer import (
            GraphMotionDecoder,
            GraphMotionDecoderLayer,
            _sin_time_embedding,
        )

        D, H, J, T = 8, 2, 3, 6
        layer = GraphMotionDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        dec = GraphMotionDecoder(layer, num_layers=1, cross_limb=True, cross_limb_latents=2, cross_limb_dim=8)
        seen = {}

        def stub(output, *a, **kw):
            seen['phase'] = kw['loop_phase_embedding']
            seen['cross_limb'] = kw['cross_limb_time_embedding']
            return output

        dec.layers[0].forward = stub
        y = {
            'graph_dist': torch.zeros(2, J, J, dtype=torch.int64),
            'joints_relations': torch.zeros(2, J, J, dtype=torch.int64),
        }
        dec.forward(
            tgt=torch.zeros(T, 2, J, D),
            timesteps_embs=torch.zeros(2, D),
            memory=None,
            y=y,
            loop_phase_mask=torch.tensor([True, False]),
        )

        cpu = torch.device('cpu')
        circular = circular_phase_embedding(T, D, cpu, torch.float32)
        self.assertEqual(tuple(seen['phase'].shape), (T, 2, D))
        self.assertTrue(torch.equal(seen['phase'][:, 0], circular))
        self.assertTrue(torch.equal(seen['phase'][:, 1], torch.zeros(T, D)))
        cl_dim = dec.cross_limb_blocks[0].latent_dim
        self.assertEqual(tuple(seen['cross_limb'].shape), (T, 2, cl_dim))
        self.assertTrue(torch.equal(
            seen['cross_limb'][:, 0], circular_phase_embedding(T, cl_dim, cpu, torch.float32)
        ))
        self.assertTrue(torch.equal(
            seen['cross_limb'][:, 1], _sin_time_embedding(T, cl_dim, cpu, torch.float32)
        ))


if __name__ == '__main__':
    unittest.main()
