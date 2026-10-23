"""The rest-geometry channel, y['rest_geom']: rest_geom_features plus a present flag.

The loader and generation must hand the model the same rows, the rows must
follow the augmented skeleton, the training-only drop must hide a joint's
geometry together with its present flag, and no other input may carry the rest
geometry past that drop.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data_loaders.truebones.data.dataset as dataset_module  # noqa: E402
from data_loaders.reference import REST_GEOM_DIM, rest_geom_features  # noqa: E402
from data_loaders.tensors import truebones_batch_collate  # noqa: E402
from data_loaders.truebones.truebones_utils.leaf_drop import drop_joints_from_cond  # noqa: E402
from sample.conditioning import create_condition  # noqa: E402
from tests.test_bone_length_aug import make_dataset  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_struct_features import (  # noqa: E402
    JOINT_STRUCT_FEATURE_NAMES,
)
from tests.test_joint_name_dropout import (  # noqa: E402
    _joint_cond_inputs, _joint_struct, _make_model,
)


def _plain_dataset():
    dataset, data = make_dataset()
    dataset.opt.bone_length_aug_prob = 0
    return dataset, data


def test_loader_and_generation_build_identical_rest_geom():
    dataset, data = _plain_dataset()
    cond = dataset.cond_dict['test']
    random.seed(0)
    _, loaded = truebones_batch_collate([dataset._prepare_sample('test', data)])
    _, generated = create_condition(['test'], {'test': cond}, 30, dataset.opt.max_joints, 12)
    assert torch.equal(loaded['y']['rest_geom'], generated['y']['rest_geom'])
    rest_geom = generated['y']['rest_geom'][0]
    assert rest_geom.shape == (dataset.opt.max_joints, REST_GEOM_DIM)
    np.testing.assert_array_equal(rest_geom[:, :-1].numpy(), rest_geom_features(cond))
    assert rest_geom[:, -1].tolist() == [1.0] * len(cond['parents'])


def test_padding_rows_are_zero_and_absent():
    dataset, data = _plain_dataset()
    n_joints = len(dataset.cond_dict['test']['parents'])
    dataset.opt.max_joints = n_joints + 3
    _, cond = truebones_batch_collate([dataset._prepare_sample('test', data)])
    rest_geom = cond['y']['rest_geom'][0]
    assert torch.equal(rest_geom[n_joints:], torch.zeros(3, REST_GEOM_DIM))
    assert bool((rest_geom[:n_joints, -1] == 1).all())


def test_leaf_drop_rebuilds_rest_geom(monkeypatch):
    dataset, data = _plain_dataset()
    dataset.opt.leaf_drop_prob = 1.0
    monkeypatch.setattr(dataset_module, 'sample_leaf_drop', lambda *a: [3])
    random.seed(0)
    sample = dataset._prepare_sample('test', data)
    _, kept_cond = drop_joints_from_cond(dataset.cond_dict['test'], [3])
    np.testing.assert_array_equal(sample[3], rest_geom_features(kept_cond))


def test_rest_geom_drop_is_training_only():
    model = _make_model()
    model.rest_geom_drop_prob = model.rest_geom_joint_drop_prob = 1.0
    model.eval()
    assert model._sample_rest_geom_drop(2, 4, torch.device('cpu')) is None
    model.train()
    assert bool(model._sample_rest_geom_drop(2, 4, torch.device('cpu')).all())
    model.rest_geom_drop_prob = model.rest_geom_joint_drop_prob = 0.0
    assert model._sample_rest_geom_drop(2, 4, torch.device('cpu')) is None


def test_rest_geom_drop_rates():
    model = _make_model().train()
    model.rest_geom_drop_prob, model.rest_geom_joint_drop_prob = 0.0, 0.5
    torch.manual_seed(0)
    assert abs(float(model._sample_rest_geom_drop(256, 8, torch.device('cpu')).float().mean()) - 0.5) < 0.05
    model.rest_geom_drop_prob, model.rest_geom_joint_drop_prob = 0.5, 0.0
    drop = model._sample_rest_geom_drop(256, 8, torch.device('cpu'))
    # Whole-sample drop: every joint of a row agrees.
    assert torch.equal(drop, drop[:, :1].expand_as(drop))


def test_dropped_joints_read_as_absent():
    model = _make_model()
    ip = model.input_process.eval()
    x, rest_geom, names = _joint_cond_inputs()
    struct = _joint_struct()
    drop = torch.tensor([[False, True, False, True], [True, False, False, False]])
    hidden = rest_geom.masked_fill(drop[:, :, None], 0.0)
    assert bool((hidden[drop][:, -1] == 0).all())
    with torch.no_grad():
        dropped = ip(x, rest_geom, names, None, struct, rest_geom_drop=drop)
        reference = ip(x, hidden, names, None, struct)
    assert torch.equal(dropped, reference)


def test_joint_struct_geometry_columns_never_reach_the_denoiser():
    """Only the topology columns and the side sign of joint_struct are read, so
    the rest geometry has one path in, the one rest_geom_drop hides."""
    ip = _make_model().input_process.eval()
    x, rest_geom, names = _joint_cond_inputs()
    struct = _joint_struct()
    moved = struct.clone()
    for name in ('height_n', 'fore_aft_n', 'attach_h_n'):
        moved[..., JOINT_STRUCT_FEATURE_NAMES.index(name)] += 0.37
    side = JOINT_STRUCT_FEATURE_NAMES.index('lateral_signed')
    moved[..., side] *= 3.0  # same sign, other magnitude
    with torch.no_grad():
        torch.testing.assert_close(ip(x, rest_geom, names, None, struct),
                                   ip(x, rest_geom, names, None, moved), rtol=0, atol=0)


def test_fully_dropped_rest_geom_hides_the_rest():
    ip = _make_model().input_process.eval()
    x, rest_geom, names = _joint_cond_inputs()
    struct = _joint_struct()
    drop = torch.ones(rest_geom.shape[:2], dtype=torch.bool)
    with torch.no_grad():
        a = ip(x, rest_geom, names, None, struct, rest_geom_drop=drop)
        b = ip(x, torch.randn_like(rest_geom), names, None, struct, rest_geom_drop=drop)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_every_frame_token_reads_the_rest_geometry():
    """No leading rest frame: the output keeps the input's frame count and
    every frame changes with the rest geometry."""
    ip = _make_model().input_process.eval()
    x, rest_geom, names = _joint_cond_inputs()
    struct = _joint_struct()
    with torch.no_grad():
        a = ip(x, rest_geom, names, None, struct)
        b = ip(x, rest_geom + 1.0, names, None, struct)
    assert a.shape[0] == x.shape[-1]
    assert bool(((a - b).abs().flatten(1).amax(1) > 0).all())
