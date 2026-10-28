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
from data_loaders.reference import REST_GEOM_DIM  # noqa: E402
from data_loaders.truebones.data.dataset import resample_motion_features  # noqa: E402
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_DIM,
)
from data_loaders.truebones.truebones_utils.param_utils import FEATS_LEN  # noqa: E402
from data_loaders.truebones.truebones_utils.canonical_features import CANONICAL_STAT_LEN, ROOT_STAT_SLICE  # noqa: E402
from model.anytop import AnyTop  # noqa: E402
from model.motion_transformer import circular_phase_embedding  # noqa: E402
from utils.model_util import create_gaussian_diffusion  # noqa: E402


# circular_phase_embedding's window closure is exact only in real arithmetic:
# in fp32 a phase near 2*pi*f misses by ~(2*pi*f)*eps, and with the top
# frequency at dim // 2 the residual is bounded by pi * dim * eps -- it grows
# with dim. A flat 1e-6 passes at (6, 8) but fails at the production shape
# (61, 256), so tolerances scale with dim.
def _closure_atol(dim: int) -> float:
    return 2.0 * math.pi * dim * torch.finfo(torch.float32).eps


def _phase_step(emb: torch.Tensor, a: int, b: int) -> float:
    return float(torch.linalg.norm(emb[a].double() - emb[b].double()))


# Toy shapes plus the production one (--num_frames 60 -> T = 61, latent_dim 256).
_PHASE_SHAPES = ((6, 8), (8, 8), (60, 256))


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
    tpose = np.zeros((n_joints, REST_GEOM_DIM - 1), dtype=np.float32)
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
    extra_cond = {'joint_mask_limb_roots': np.zeros((n_joints,), dtype=np.bool_)}
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

    def test_circular_phase_wrap_is_one_ordinary_step(self):
        """A loop window holds no closing key: its last frame is one step
        BEFORE frame 0. So the phase of the last frame differs from the first
        by exactly one frame step, the same as any two neighbours -- not the
        zero step a period of motion_frames - 1 gave, which taught the model
        to copy frame 0 into the last frame."""
        for length, dim in _PHASE_SHAPES:
            with self.subTest(length=length, dim=dim):
                emb = circular_phase_embedding(
                    length=length,
                    dim=dim,
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                )

                self.assertEqual(tuple(emb.shape), (length, dim))
                ordinary = _phase_step(emb, 1, 0)
                self.assertGreater(ordinary, 1e-2)
                atol = _closure_atol(dim)
                for a, b in ((length - 1, length - 2), (0, length - 1)):
                    self.assertAlmostEqual(_phase_step(emb, a, b), ordinary, delta=atol)
                # The frame after the last is frame 0: phase 0 is cos = 1, sin = 0.
                half = dim // 2
                expected_first = torch.cat([torch.ones(half), torch.zeros(dim - half)])
                self.assertTrue(torch.equal(emb[0], expected_first))

    def test_circular_phase_wraps_exactly_once_per_window(self):
        """The period is the window, never a cycle count: no frame after the
        first repeats its phase, so the embedding cannot tell the model how
        many gait cycles the window holds."""
        for length, dim in _PHASE_SHAPES:
            with self.subTest(length=length, dim=dim):
                emb = circular_phase_embedding(
                    length=length,
                    dim=dim,
                    device=torch.device('cpu'),
                    dtype=torch.float32,
                )

                first = emb[0]
                for frame in range(1, length):
                    self.assertFalse(
                        torch.allclose(emb[frame], first, atol=1e-3), frame
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

        self.assertTrue(torch.equal(value, torch.ones(2, dtype=torch.float32)))

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

    @staticmethod
    def _rotating_cycle(n_frames, *, n_joints=2, stall_seam=False):
        """[1, n_joints, 12, T] physical output: every joint turns about Z at a
        constant rate round one cycle per window, positions ride a circle, and
        the velocity rows are the steps to the next frame of the cycle (the
        terminal row the wrap delta), at step scale 1. ``stall_seam`` copies
        frame 0 into the last frame -- the old closing-key convention."""
        angles = 2.0 * math.pi * torch.arange(n_frames, dtype=torch.float32) / n_frames
        if stall_seam:
            angles[-1] = angles[0]
        model_output = torch.zeros(1, n_joints, 12, n_frames, dtype=torch.float32)
        # 6D = first two matrix rows of Rz(angle): (cos, -sin, 0), (sin, cos, 0).
        model_output[0, :, 3, :] = torch.cos(angles)
        model_output[0, :, 4, :] = -torch.sin(angles)
        model_output[0, :, 6, :] = torch.sin(angles)
        model_output[0, :, 7, :] = torch.cos(angles)
        model_output[0, 1:, 0, :] = torch.cos(angles)
        model_output[0, :, 1, :] = torch.sin(angles)
        positions = model_output[0, :, 0:3, :]
        model_output[0, :, 9:12, :] = torch.roll(positions, -1, dims=-1) - positions
        return model_output

    def test_loop_wrap_components_use_the_actual_seam(self):
        diffusion = self._make_diffusion()
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
        }
        terms = diffusion.loop_wrap_loss(
            self._rotating_cycle(8), y, n_joints=torch.tensor([2]),
        )

        # The last frame is one ordinary step before frame 0; nothing pulls
        # it onto frame 0.
        self.assertNotIn('loop_wrap_pose', terms)
        self.assertNotIn('loop_wrap_vel', terms)
        self.assertNotIn('loop_wrap_contact', terms)
        self.assertNotIn('loop_wrap_terminal_vel', terms)
        self.assertLess(float(terms['loop_wrap_rot'].item()), 1e-5)

    def test_loop_wrap_rotation_penalizes_a_stalled_seam(self):
        # The last frame copying frame 0 is what the old period and pose/rot
        # terms produced. Its wrap step is zero while its neighbours are two
        # steps (frame 6 -> the copied angle 0) and one step (frame 0 -> 1):
        # 1.5 steps short of their mean.
        diffusion = self._make_diffusion()
        y = {
            'is_loop': torch.tensor([True]),
            'translation_root_index': [0],
        }
        terms = diffusion.loop_wrap_loss(
            self._rotating_cycle(8, stall_seam=True), y, n_joints=torch.tensor([2]),
        )
        step = 2.0 * math.pi / 8.0
        self.assertAlmostEqual(float(terms['loop_wrap_rot'].item()), 1.5 * step, places=4)

    def test_position_step_loss_counts_the_wrap_step_only_on_loops(self):
        # Moving the last frame by one unit changes two steps: the last inner
        # step and the wrap step pos[0] - pos[-1]. A loop sample pays for both,
        # a one-shot only for the inner one.
        diffusion = self._make_diffusion()
        n_frames, n_joints = 6, 2
        target = torch.zeros(2, n_joints, 12, n_frames, dtype=torch.float32)
        target[:, :, 0, :] = torch.arange(n_frames, dtype=torch.float32)
        prediction = target.clone()
        prediction[:, :, 0, -1] += 1.0
        y = {
            'is_loop': torch.tensor([True, False]),
            'canonical_feature_std': torch.ones(CANONICAL_STAT_LEN, dtype=torch.float32),
            'canonical_dpos_std': torch.ones(2, dtype=torch.float32),
        }
        spat_mask = torch.ones(2, 1, 1, n_joints, dtype=torch.float32)
        per_sample = diffusion.position_step_loss(
            target, prediction, spat_mask, torch.tensor([n_joints, n_joints]), y,
        )
        self.assertAlmostEqual(float(per_sample[0]), 2.0 * n_joints / (n_frames * n_joints * 3), places=6)
        self.assertAlmostEqual(float(per_sample[1]), 1.0 * n_joints / ((n_frames - 1) * n_joints * 3), places=6)

        # The species weight scales the term; a matching prediction costs nothing.
        y['dpos_loss_weight'] = torch.tensor([0.5, 0.5])
        weighted = diffusion.position_step_loss(
            target, prediction, spat_mask, torch.tensor([n_joints, n_joints]), y,
        )
        self.assertTrue(torch.allclose(weighted, 0.5 * per_sample))
        self.assertEqual(float(diffusion.position_step_loss(
            target, target, spat_mask, torch.tensor([n_joints, n_joints]), y,
        ).abs().max()), 0.0)

    def test_position_step_loss_reads_the_root_trajectory_in_its_own_units(self):
        # The root's canonical position is standardized by the root's own std,
        # so the same canonical step error weighs root_std / pos_std more there.
        diffusion = self._make_diffusion()
        n_frames, n_joints = 5, 2
        target = torch.zeros(1, n_joints, 12, n_frames, dtype=torch.float32)
        std = torch.ones(CANONICAL_STAT_LEN, dtype=torch.float32)
        std[ROOT_STAT_SLICE] = 3.0
        y = {
            'is_loop': torch.tensor([True]),
            'canonical_feature_std': std,
            'canonical_dpos_std': torch.ones(1, dtype=torch.float32),
        }
        spat_mask = torch.ones(1, 1, 1, n_joints, dtype=torch.float32)
        losses = []
        for joint in (0, 1):
            prediction = target.clone()
            prediction[:, joint, 0, -1] += 1.0
            losses.append(float(diffusion.position_step_loss(
                target, prediction, spat_mask, torch.tensor([n_joints]), y,
            )[0]))
        self.assertAlmostEqual(losses[0], 9.0 * losses[1], places=5)

    def test_root_position_step_loss_is_the_root_xz_share_of_the_position_step_loss(self):
        # Only the root's world X/Z steps count, in the root's own units, with
        # the wrap step on loops; the same error is the all-joint mean's
        # 2 / (3 * n_joints) share of it.
        diffusion = self._make_diffusion()
        n_frames, n_joints = 6, 3
        target = torch.zeros(2, n_joints, 12, n_frames, dtype=torch.float32)
        std = torch.ones(CANONICAL_STAT_LEN, dtype=torch.float32)
        std[ROOT_STAT_SLICE] = 3.0
        y = {
            'is_loop': torch.tensor([True, False]),
            'canonical_feature_std': std,
            'canonical_dpos_std': torch.ones(2, dtype=torch.float32),
        }
        spat_mask = torch.ones(2, 1, 1, n_joints, dtype=torch.float32)

        prediction = target.clone()
        prediction[:, 0, 0, -1] += 1.0
        root = diffusion.root_position_step_loss(target, prediction, y)
        self.assertAlmostEqual(float(root[0]), 9.0 * 2.0 / (n_frames * 2), places=5)
        self.assertAlmostEqual(float(root[1]), 9.0 * 1.0 / ((n_frames - 1) * 2), places=5)
        dpos = diffusion.position_step_loss(
            target, prediction, spat_mask, torch.tensor([n_joints, n_joints]), y,
        )
        self.assertTrue(torch.allclose(dpos, root * 2.0 / (3.0 * n_joints)))

        # The root's height and every other joint are not its business.
        for joint, channel in ((0, 1), (1, 0), (2, 2)):
            prediction = target.clone()
            prediction[:, joint, channel, -1] += 1.0
            self.assertEqual(float(diffusion.root_position_step_loss(target, prediction, y).abs().max()), 0.0)

        # The species weight scales it like the position-step loss.
        prediction = target.clone()
        prediction[:, 0, 2, -1] += 1.0
        unweighted = diffusion.root_position_step_loss(target, prediction, y)
        y['dpos_loss_weight'] = torch.tensor([0.5, 0.5])
        self.assertTrue(torch.allclose(diffusion.root_position_step_loss(target, prediction, y), 0.5 * unweighted))

    def test_root_acceleration_loss_counts_second_differences_of_the_root_xz(self):
        # A one-frame bump on the root's X gives the scaled steps (+3, -3) and
        # the accelerations (3, -6, 3); a loop averages over every frame, an
        # open clip over its inner second differences only.
        diffusion = self._make_diffusion()
        n_frames, n_joints = 6, 3
        target = torch.zeros(2, n_joints, 12, n_frames, dtype=torch.float32)
        std = torch.ones(CANONICAL_STAT_LEN, dtype=torch.float32)
        std[ROOT_STAT_SLICE] = 3.0
        y = {
            'is_loop': torch.tensor([True, False]),
            'canonical_feature_std': std,
            'canonical_dpos_std': torch.ones(2, dtype=torch.float32),
        }
        prediction = target.clone()
        prediction[:, 0, 0, 2] += 1.0
        acc = diffusion.root_acceleration_loss(target, prediction, y)
        self.assertAlmostEqual(float(acc[0]), 54.0 / (n_frames * 2), places=5)
        self.assertAlmostEqual(float(acc[1]), 54.0 / ((n_frames - 2) * 2), places=5)

        # A constant speed error has no acceleration on an open clip.
        prediction = target.clone()
        prediction[:, 0, 2, :] += torch.arange(n_frames, dtype=torch.float32)
        self.assertEqual(float(diffusion.root_acceleration_loss(target, prediction, y)[1]), 0.0)

        # The root's height and every other joint are not its business.
        for joint, channel in ((0, 1), (1, 0), (2, 2)):
            prediction = target.clone()
            prediction[:, joint, channel, 2] += 1.0
            self.assertEqual(float(diffusion.root_acceleration_loss(target, prediction, y).abs().max()), 0.0)

        prediction = target.clone()
        prediction[:, 0, 2, 2] += 1.0
        unweighted = diffusion.root_acceleration_loss(target, prediction, y)
        y['dpos_loss_weight'] = torch.tensor([0.5, 0.5])
        self.assertTrue(torch.allclose(diffusion.root_acceleration_loss(target, prediction, y), 0.5 * unweighted))

    def test_create_gaussian_diffusion_preserves_loop_args(self):
        class Args:
            noise_schedule = 'cosine'
            diffusion_steps = 10
            timestep_respacing = ''
            sigma_small = True
            lambda_geo = 0.0
            lambda_loop_wrap = 0.75
            lambda_root_dpos = 0.17
            lambda_root_acc = 0.23
            temporal_span_seam_loss_weight = 0.0
            temporal_span_seam_width = 2

        diffusion = create_gaussian_diffusion(Args())
        self.assertEqual(diffusion.lambda_loop_wrap, 0.75)
        self.assertEqual(diffusion.lambda_root_dpos, 0.17)
        self.assertEqual(diffusion.lambda_root_acc, 0.23)

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
        )
        capture_decoder = _CaptureDecoder()
        model.seqTransDecoder = capture_decoder

        x = torch.randn(2, 4, 12, 3, dtype=torch.float32)
        y = {
            'joints_padding_mask': torch.ones(2, 1, 1, 5, 5, dtype=torch.float32),
            'rest_geom': torch.randn(2, 4, REST_GEOM_DIM, dtype=torch.float32),
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
        """Loop samples get the circular phase, one wrap per window; non-loop
        samples get a zero phase. No per-sample period exists any more, so the
        picture is the same for every batch."""
        from model.motion_transformer import (
            GraphMotionDecoder,
            GraphMotionDecoderLayer,
        )

        D, H, J, T = 8, 2, 3, 6
        layer = GraphMotionDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        dec = GraphMotionDecoder(layer, num_layers=1, cross_limb=True, cross_limb_latents=2, cross_limb_dim=8)
        seen = {}

        def stub(output, *a, **kw):
            seen['phase'] = kw['loop_phase_embedding']
            seen['kwargs'] = kw
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
        # The cross-limb blocks get no time table of their own any more.
        self.assertNotIn('cross_limb_time_embedding', seen['kwargs'])


if __name__ == '__main__':
    unittest.main()
