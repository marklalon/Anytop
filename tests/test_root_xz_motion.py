"""Root XZ retention: the strip gate, the provenance flag, and tiling.

See docs/root_xz_motion_refactor.md. Only clips that genuinely travel
(``xz_extent > ROOT_XZ_STRIP_THRESHOLD``) lose their root XZ; everything below
that keeps it verbatim. ``root_xz_stripped`` marks the ones this pipeline
zeroed, so the model can discount a trajectory it knows is fabricated.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_lib.Animation import Animation
from motion_lib.Quaternions import Quaternions

from data_loaders.truebones.truebones_utils.motion_process import (
    ROOT_XZ_STRIP_THRESHOLD,
)
from data_loaders.truebones.truebones_utils.features import (
    extract_motion_features_from_aligned_anims,
)
from data_loaders.truebones.data.dataset import _tile_loop_motion


def _straight_line_anim(n_frames: int, path_xz: np.ndarray) -> Animation:
    """A two-joint skeleton whose root walks ``path_xz`` while the child bobs.

    The child's own motion is what makes the clip's first and last POSE close
    up, which is exactly the condition the deleted ambiguous-band gate keyed on.
    """
    parents = np.array([-1, 0], dtype=np.int64)
    offsets = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rotations = Quaternions(
        np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, len(parents), 1))
    )
    positions = np.zeros((n_frames, len(parents), 3), dtype=np.float64)
    positions[:, 0, 0] = path_xz[:, 0]
    positions[:, 0, 2] = path_xz[:, 1]
    positions[:, 1] = offsets[1]
    return Animation(rotations, positions, Quaternions.id(len(parents)), offsets, parents)


def _extract(anim):
    features, _max_joints, _m, _e, is_loop, stripped = extract_motion_features_from_aligned_anims(
        anim,
        anim,
        foot_contact_vel_thresh=0.01,
        object_type='TestSkeleton',
        max_joints=8,
        foot_indices=[1],
        orientation_quat=Quaternions.id(1).qs[0],
        translation_root_index=0,
    )
    return features, is_loop, stripped


def _closed_excursion(n_frames: int, amplitude: float) -> np.ndarray:
    """Go out and come back: net displacement 0, extent ``amplitude``."""
    t = np.linspace(0.0, 2.0 * np.pi, num=n_frames, endpoint=True)
    return np.stack([amplitude * np.sin(t), np.zeros_like(t)], axis=-1)


def _travelling_path(n_frames: int, distance: float = 4.0) -> np.ndarray:
    return np.stack(
        [np.linspace(0.0, distance, num=n_frames), np.zeros(n_frames)], axis=-1
    )


# ── C1: the ambiguous band is no longer stripped ───────────────────────────

@pytest.mark.parametrize('amplitude', [0.1, 0.3, 0.59])
def test_ambiguous_band_keeps_its_root_xz(amplitude):
    """A closed excursion inside [0.08, 0.6] used to be zeroed wholesale.

    These are strikes, dodges and idle sways -- the root drift IS the action.
    """
    assert amplitude < ROOT_XZ_STRIP_THRESHOLD
    features, _, stripped = _extract(_straight_line_anim(40, _closed_excursion(40, amplitude)))

    assert stripped is False
    assert np.abs(features[:, 0, [9, 11]]).max() > 1e-4
    # And it integrates back to the excursion it came from.
    travelled = np.abs(np.cumsum(features[:-1, 0, 9])).max()
    assert travelled == pytest.approx(amplitude, rel=0.05)


def test_true_locomotion_is_still_stripped():
    """Gate A survives: a clip that actually travels loses its root XZ."""
    features, _, stripped = _extract(_straight_line_anim(40, _travelling_path(40)))

    assert stripped is True
    assert np.abs(features[:, 0, [9, 11]]).max() == 0.0


def test_root_ric_xz_is_structurally_zero():
    """The exporter's unconditional RIC cleanup rests on this identity."""
    features, _, _ = _extract(_straight_line_anim(30, _closed_excursion(30, 0.4)))
    np.testing.assert_array_equal(features[:, 0, [0, 2]], 0.0)


# ── C2: the flag is PROVENANCE, not content ────────────────────────────────

def test_flag_is_false_for_a_natively_in_place_clip():
    """A clip an artist authored in place is honest data: its root really does
    not travel, so there is nothing to warn the model about. Only a zero this
    code wrote is a fabrication, and only that gets flagged."""
    n_frames = 30
    features, _, stripped = _extract(
        _straight_line_anim(n_frames, np.zeros((n_frames, 2)))
    )

    assert stripped is False
    assert np.abs(features[:, 0, [9, 11]]).max() == pytest.approx(0.0, abs=1e-6)


def test_stripped_flag_marks_only_the_fabricated_zeros():
    """Both clips end up with a zero root trajectory; only one is a lie."""
    n_frames = 30
    _honest, _, honest_stripped = _extract(
        _straight_line_anim(n_frames, np.zeros((n_frames, 2)))
    )
    _fabricated, _, fabricated_stripped = _extract(
        _straight_line_anim(n_frames, _travelling_path(n_frames))
    )

    assert honest_stripped is False
    assert fabricated_stripped is True


def test_strip_writes_exact_zeros():
    """What lets the validator check the flag without an epsilon."""
    features, _, stripped = _extract(_straight_line_anim(40, _travelling_path(40)))
    assert stripped is True
    # Every frame, terminal row included.
    assert (features[:, 0, [9, 11]] == 0.0).all()


# ── §1.4: a travelling loop still tiles without a seam ─────────────────────

def test_travelling_loop_tiles_without_a_jump():
    """Position is the cumulative sum of the velocity rows, and the terminal row
    holds the cycle's wrap delta (its last step), so joining two copies is just
    concatenating velocities -- the seam step must match the stride.
    """
    n_frames = 24
    step = 0.05
    motion = np.zeros((n_frames, 2, 13), dtype=np.float32)
    motion[:, 1, 9] = step           # a constant stride in +X, terminal row included

    tiled = _tile_loop_motion(motion, 2)
    assert tiled.shape[0] == 2 * n_frames

    path = np.cumsum(tiled[:-1, 1, 9])
    steps = np.diff(path)
    np.testing.assert_allclose(steps, step, atol=1e-6)
    assert path[-1] == pytest.approx(step * (2 * n_frames - 1), rel=1e-5)


# ── C2 model side: the flag reaches the timestep token ─────────────────────

def _tiny_model():
    from model.anytop import AnyTop
    return AnyTop(
        max_joints=4,
        feature_len=13,
        latent_dim=8,
        ff_size=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        cross_limb=True,
        t5_out_dim=512,
    )


def test_root_xz_strip_projection_is_unconditional():
    """No flag gates it, so the parameter shape is a constant of the model."""
    import torch

    model = _tiny_model()
    assert model.root_xz_strip_projection is not None
    token = model.root_xz_strip_projection(torch.zeros(2, 1, dtype=torch.float32))
    assert tuple(token.shape) == (2, 8)


def test_root_xz_strip_projection_separates_the_two_classes():
    import torch

    model = _tiny_model()
    model.eval()
    torch.nn.init.normal_(model.root_xz_strip_projection[-1].weight, std=0.5)
    torch.nn.init.normal_(model.root_xz_strip_projection[-1].bias, std=0.5)

    flags = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    token = model.root_xz_strip_projection(flags)
    assert not torch.allclose(token[0], token[1])


def test_root_xz_stripped_coercion_accepts_bools_and_broadcasts():
    import torch

    model = _tiny_model()
    coerced = model._coerce_loop_condition(
        torch.tensor([True, False, True]), 3, torch.device('cpu'), torch.float32,
        field_name='root_xz_stripped',
    )
    assert tuple(coerced.shape) == (3, 1)
    np.testing.assert_array_equal(coerced.numpy().reshape(-1), [1.0, 0.0, 1.0])

    broadcast = model._coerce_loop_condition(
        torch.tensor([True]), 4, torch.device('cpu'), torch.float32,
        field_name='root_xz_stripped',
    )
    assert tuple(broadcast.shape) == (4, 1)

    # A missing flag is the "not stripped" default, never a crash.
    default = model._coerce_loop_condition(
        None, 2, torch.device('cpu'), torch.float32, field_name='root_xz_stripped',
    )
    np.testing.assert_array_equal(default.numpy().reshape(-1), [0.0, 0.0])


def test_root_xz_stripped_coercion_names_itself_in_the_error():
    import torch

    model = _tiny_model()
    with pytest.raises(ValueError, match='root_xz_stripped'):
        model._coerce_loop_condition(
            torch.tensor([True, False]), 3, torch.device('cpu'), torch.float32,
            field_name='root_xz_stripped',
        )


def test_collate_carries_root_xz_stripped_into_y():
    """The loader writes the flag into motion_metadata; the collate must
    surface it in ``y`` the way it does is_loop, or the model reads None."""
    import torch
    from data_loaders.tensors import truebones_collate

    def _item(flag):
        return {
            'inp': torch.zeros(3, 13, 5),
            'rest_pose': torch.zeros(3, 13),
            'n_joints': 3,
            'lengths': 5,
            'parents': np.array([-1, 0, 1]),
            'graph_dist': torch.zeros(4, 4),
            'joints_relations': torch.zeros(4, 4),
            'object_type': 'TestSkeleton',
            'joints_names_embs': torch.zeros(4, 512),
            'joints_padding_mask': torch.ones(1, 1, 4, 4),
            'root_xz_stripped': flag,
            'is_loop': False,
        }

    _motion, cond = truebones_collate([_item(True), _item(False)])
    assert 'root_xz_stripped' in cond['y']
    np.testing.assert_array_equal(
        cond['y']['root_xz_stripped'].numpy(), np.array([True, False])
    )


# ── inference: no switch, and the honest value is what gets asked for ──────

def test_generation_never_asks_for_a_stripped_root():
    """There is no --in_place: how far the root should travel is already implied
    by the action label and --loop, and a second knob could only be combined
    into a request no training sample ever looked like."""
    import argparse
    import inspect

    from sample import generate as generate_module
    from utils import parser_util

    assert 'in_place' not in inspect.signature(generate_module.create_condition).parameters

    parser = argparse.ArgumentParser()
    parser_util.add_sampling_options(parser)
    parser_util.add_generate_options(parser)
    options = {opt for action in parser._actions for opt in action.option_strings}
    assert '--in_place' not in options

    source = inspect.getsource(generate_module.create_condition)
    assert "'root_xz_stripped': False," in source


# ── validator: a flagged clip must really carry no root XZ ─────────────────

def _run_flag_validator(motion, metadata, root_index, capsys):
    from utils.validate_anytop_dataset import _validate_root_xz_stripped_flag

    _validate_root_xz_stripped_flag(motion, 'Clip_Test.npy', metadata, root_index)
    return capsys.readouterr().out


def test_validator_accepts_a_correctly_stripped_clip(capsys):
    motion = np.zeros((10, 3, 13), dtype=np.float32)
    assert _run_flag_validator(motion, {'root_xz_stripped': True}, 1, capsys) == ''


def test_validator_flags_a_stripped_clip_that_still_moves(capsys):
    motion = np.zeros((10, 3, 13), dtype=np.float32)
    motion[:, 1, [9, 11]] = 0.02
    out = _run_flag_validator(motion, {'root_xz_stripped': True}, 1, capsys)
    assert 'root_xz_stripped=True' in out


def test_validator_says_nothing_about_an_unstripped_clip(capsys):
    """An artist-authored in-place clip sits at zero and is NOT stripped; that
    is legitimate, so the one-directional check must stay quiet."""
    still = np.zeros((10, 3, 13), dtype=np.float32)
    assert _run_flag_validator(still, {'root_xz_stripped': False}, 1, capsys) == ''

    moving = np.zeros((10, 3, 13), dtype=np.float32)
    moving[:, 1, [9, 11]] = 0.02
    assert _run_flag_validator(moving, {'root_xz_stripped': False}, 1, capsys) == ''
    assert _run_flag_validator(moving, {}, 1, capsys) == ''


def test_validator_reports_a_root_index_the_features_disagree_with(capsys):
    """A clip's features can be built around a different joint than the
    per-species canonical root in metadata; say so instead of blaming the
    flag."""
    motion = np.zeros((10, 3, 13), dtype=np.float32)
    motion[:, 1, 0] = 0.5           # joint 1's RIC XZ is NOT structurally zero
    motion[:, 1, [9, 11]] = 0.02
    out = _run_flag_validator(motion, {'root_xz_stripped': True}, 1, capsys)
    assert 'not the joint the features were built around' in out
