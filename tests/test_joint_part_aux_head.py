"""The auxiliary part / contact head: a training target, never a condition.

Pins what the rest of the pipeline relies on -- the head adds nothing to a
model built without it, ``return_aux=False`` returns exactly the plain x0, the
loss sees only the joints it is meant to (a blanked name for the part, a
labelled joint for the contact), and the loader's targets are always present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_loaders.tensors import _joint_part_targets  # noqa: E402
from data_loaders.truebones.truebones_utils.joint_parts import (  # noqa: E402
    HELPER_PART_ID,
    JOINT_CONTACT_KEY,
    JOINT_PARTS,
    JOINT_PARTS_KEY,
    PART_IDS,
)
from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM  # noqa: E402
from diffusion import gaussian_diffusion as gd  # noqa: E402
from model.anytop import AnyTop  # noqa: E402

T5_DIM = 16


def _model(part_head=True, joint_name_drop_prob=0.0, num_layers=2, part_head_layer=0):
    torch.manual_seed(0)
    return AnyTop(
        max_joints=4, feature_len=12, latent_dim=8, ff_size=16, num_layers=num_layers,
        num_heads=2, dropout=0.0, t5_out_dim=T5_DIM, part_head=part_head,
        part_head_layer=part_head_layer, joint_name_drop_prob=joint_name_drop_prob,
    )


def _y(batch=2, joints=4, frames=3):
    valid = torch.ones(batch, joints + 1)
    mask = (valid.unsqueeze(2) * valid.unsqueeze(1)).unsqueeze(1).unsqueeze(1)
    return {
        'joints_padding_mask': mask,
        'rest_pose': torch.randn(batch, joints, 12),
        'n_joints': torch.full((batch,), joints, dtype=torch.int64),
        'joints_names_embs': torch.randn(batch, joints, T5_DIM),
        'joint_struct': torch.randn(batch, joints, JOINT_STRUCT_DIM),
        'lengths': torch.full((batch,), frames, dtype=torch.int64),
        'canonical_feature_mean': torch.zeros(12),
        'canonical_feature_std': torch.ones(12),
        'graph_dist': torch.zeros(batch, joints, joints, dtype=torch.long),
        'joints_relations': torch.zeros(batch, joints, joints, dtype=torch.long),
    }


def test_a_model_without_the_head_has_none_of_its_parameters():
    plain = _model(part_head=False)
    assert not any(key.startswith('joint_part_head') for key in plain.state_dict())
    with pytest.raises(ValueError, match='part_head'):
        plain(torch.randn(2, 4, 12, 3), torch.tensor([1, 2]), y=_y(), return_aux=True)


def test_the_head_does_not_touch_the_other_parameters():
    plain, headed = _model(part_head=False), _model(part_head=True)
    for key, value in plain.state_dict().items():
        assert torch.equal(value, headed.state_dict()[key]), key


def test_return_aux_keeps_x0_bit_identical_and_shapes_the_logits():
    model = _model().eval()
    x, t, y = torch.randn(2, 4, 12, 3), torch.tensor([1, 2]), _y()
    with torch.no_grad():
        plain = model(x, t, y=y)
        x0, aux = model(x, t, y=y, return_aux=True)
    assert torch.equal(plain, x0)
    assert aux['part_logits'].shape == (2, 4, len(JOINT_PARTS))
    assert aux['contact_logit'].shape == (2, 4)
    assert not aux['joint_name_drop'].any(), 'eval keeps every name'


def test_the_loss_reads_the_mask_the_forward_blanked_names_with():
    model = _model(joint_name_drop_prob=1.0).train()
    _, aux = model(torch.randn(2, 4, 12, 3), torch.tensor([1, 2]), y=_y(), return_aux=True)
    assert aux['joint_name_drop'].all()


def test_the_tap_layer_is_validated():
    with pytest.raises(ValueError, match='part_head_layer'):
        _model(num_layers=2, part_head_layer=3)
    assert _model(num_layers=1).part_head_layer == 1
    assert _model(num_layers=4).part_head_layer == 2


def test_collate_targets_leave_out_helpers_and_padding():
    part_ids = np.array([PART_IDS['trunk'], HELPER_PART_ID, PART_IDS['foot']], dtype=np.int16)
    contact = np.array([False, False, True])
    item = _joint_part_targets({JOINT_PARTS_KEY: part_ids, JOINT_CONTACT_KEY: contact}, 5, 3)
    assert item['joint_part_target'].tolist() == [PART_IDS['trunk'], -1, PART_IDS['foot'], -1, -1]
    assert item['joint_contact_target'].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert item['joint_contact_valid'].tolist() == [True, False, True, False, False]
    empty = _joint_part_targets({JOINT_PARTS_KEY: None}, 5, 3)
    assert empty['joint_part_target'].tolist() == [-1] * 5
    assert not empty['joint_contact_valid'].any()


def _diffusion(weights=None):
    return gd.GaussianDiffusion(
        betas=np.linspace(1e-4, 0.02, 10), model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL, loss_type=gd.LossType.MSE,
        lambda_part=1.0, lambda_contact=1.0, part_class_weights=weights,
    )


def test_part_loss_covers_blanked_labelled_joints_only():
    logits = torch.zeros(2, 3, len(JOINT_PARTS))
    logits[0, 0, PART_IDS['trunk']] = 10.0       # right, blanked
    logits[0, 1, PART_IDS['trunk']] = 10.0       # wrong, but its name is visible
    aux = {
        'part_logits': logits,
        'contact_logit': torch.zeros(2, 3),
        'joint_name_drop': torch.tensor([[True, False, True], [False, False, False]]),
    }
    y = {
        'joint_part_target': torch.tensor([[PART_IDS['trunk'], PART_IDS['leg'], -1], [0, 0, 0]]),
        'joint_contact_target': torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        'joint_contact_valid': torch.tensor([[True, True, False], [False, False, False]]),
    }
    terms = _diffusion().joint_part_losses(aux, y)
    # Sample 0: only joint 0 counts (joint 1 shows its name, joint 2 is unlabelled).
    assert float(terms['part_loss'][0]) < 1e-3
    assert float(terms['part_loss'][1]) == 0.0, 'nothing blanked, nothing learned'
    # Contacts: joints 0 and 1 of sample 0, logits at 0 -> log 2 each.
    assert float(terms['contact_loss'][0]) == pytest.approx(np.log(2.0), rel=1e-5)
    assert float(terms['contact_loss'][1]) == 0.0


def test_class_weights_reweigh_the_part_loss():
    logits = torch.zeros(1, 2, len(JOINT_PARTS))
    logits[0, :, PART_IDS['trunk']] = 10.0       # right on joint 0, wrong on joint 1
    aux = {
        'part_logits': logits,
        'contact_logit': torch.zeros(1, 2),
        'joint_name_drop': torch.tensor([[True, True]]),
    }
    y = {
        'joint_part_target': torch.tensor([[PART_IDS['trunk'], PART_IDS['soft']]]),
        'joint_contact_target': torch.zeros(1, 2),
        'joint_contact_valid': torch.zeros(1, 2, dtype=torch.bool),
    }
    weights = [1.0] * len(JOINT_PARTS)
    weights[PART_IDS['soft']] = 3.0
    plain = float(_diffusion().joint_part_losses(aux, y)['part_loss'][0])
    weighted = float(_diffusion(weights).joint_part_losses(aux, y)['part_loss'][0])
    # (0 + ce) / 2 unweighted against (0 + 3 ce) / 4: the rare class's miss counts more.
    assert weighted == pytest.approx(1.5 * plain, rel=1e-3)
