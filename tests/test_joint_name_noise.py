"""Training-time name noise in the names' principal subspace (model/joint_name_noise.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.anytop import AnyTop  # noqa: E402
from model.joint_name_noise import JointNameNoise, add_joint_name_noise, joint_name_noise_basis  # noqa: E402

DIM = 16


def _cond_dict(seed=0):
    rng = np.random.default_rng(seed)
    # Names on a 6-dim subspace, a few duplicated and a few blank.
    mixing = rng.standard_normal((6, DIM))
    names = rng.standard_normal((40, 6)) @ mixing
    names[5] = names[3]
    names[10] = 0
    return {'a': {'joints_names_embs': names[:25]}, 'b': {'joints_names_embs': names[25:]}}


def test_basis_is_orthonormal_sorted_and_deterministic():
    basis, std = joint_name_noise_basis(_cond_dict(), 4)
    assert basis.shape == (4, DIM) and std.shape == (4,)
    torch.testing.assert_close(basis @ basis.T, torch.eye(4), atol=1e-5, rtol=0)
    assert bool((std[:-1] >= std[1:]).all())
    again, again_std = joint_name_noise_basis(_cond_dict(), 4)
    assert torch.equal(basis, again) and torch.equal(std, again_std)


def test_basis_needs_more_names_than_its_rank():
    with pytest.raises(ValueError):
        joint_name_noise_basis(_cond_dict(), 64)


def test_noise_stays_in_the_subspace_and_skips_blank_rows():
    basis, std = joint_name_noise_basis(_cond_dict(), 4)
    names = torch.randn(3, 5, DIM)
    names[0, 2] = 0
    names[2] = 0
    noisy = add_joint_name_noise(names, 0.5, basis, std)
    assert torch.equal(noisy[0, 2], torch.zeros(DIM))
    assert torch.equal(noisy[2], torch.zeros(5, DIM))
    delta = noisy - names
    outside = delta - (delta @ basis.T) @ basis
    assert float(outside.abs().max()) < 1e-5
    assert float(delta[1].abs().max()) > 0


def test_noise_is_training_only():
    module = JointNameNoise(0.2)
    module.set_basis(*joint_name_noise_basis(_cond_dict(), 4))
    names = torch.randn(2, 3, DIM)
    assert torch.equal(module.eval()(names), names)
    assert not torch.equal(module.train()(names), names)


def test_training_without_a_basis_is_an_error():
    module = JointNameNoise(0.2).train()
    with pytest.raises(ValueError):
        module(torch.randn(2, 3, DIM))
    names = torch.randn(2, 3, DIM)
    assert torch.equal(module.eval()(names), names)
    assert torch.equal(JointNameNoise(0.0).train()(names), names)


def test_model_attaches_one_basis_to_both_name_paths_without_new_state():
    kwargs = dict(max_joints=4, feature_len=12, latent_dim=16, ff_size=32, num_layers=1,
                  num_heads=2, t5_out_dim=DIM, ref_encoder=True, ref_width=8)
    plain = AnyTop(**kwargs)
    model = AnyTop(joint_name_noise_std=0.1, **kwargs)
    basis, std = joint_name_noise_basis(_cond_dict(), 4)
    model.set_joint_name_noise_basis(basis, std)
    assert torch.equal(model.input_process.joint_name_noise.basis, basis)
    assert torch.equal(model.motion_encoder.joint_name_noise.std, std)
    assert model.motion_encoder.joint_name_noise.noise_std == 0.1
    assert model.state_dict().keys() == plain.state_dict().keys()
