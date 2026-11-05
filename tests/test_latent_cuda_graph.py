"""CUDA graph buckets accept collator metadata and refresh tensor inputs."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion.latent_diffusion import create_latent_diffusion
from train.latent_cuda_graph import EagerCudaGraphSteps


def test_numpy_condition_signature_and_capture_copy():
    original = {'parents': [np.array([-1, 0, 1], dtype=np.int64)],
                'nested': (np.arange(6).reshape(2, 3)[:, ::2],)}
    cloned = EagerCudaGraphSteps._clone_tree(original)
    signature = EagerCudaGraphSteps._signature(original)
    assert {signature: 'cached'}[EagerCudaGraphSteps._signature(cloned)] == 'cached'
    original['parents'][0][1] = -1
    assert cloned['parents'][0].tolist() == [-1, 0, 1]
    assert EagerCudaGraphSteps._signature(original) != signature
    for changed in (np.array([-1, 0, 1], dtype=np.int32),
                    np.array([[-1, 0, 1]], dtype=np.int64)):
        assert EagerCudaGraphSteps._signature(changed) != EagerCudaGraphSteps._signature(cloned['parents'][0])


class ConditionedLinear(torch.nn.Linear):
    def forward(self, z, timesteps, y):
        return super().forward(z) + y['resample_speed_cond']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires CUDA')
def test_graph_replay_with_numpy_parents_matches_eager_updates():
    device = torch.device('cuda')
    model = ConditionedLinear(3, 3).to(device)
    eager = copy.deepcopy(model)
    diffusion = create_latent_diffusion(steps=10, loss_weights=torch.tensor([[.5, 1., 1.5], [1.5, 1., .5]], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    steps = EagerCudaGraphSteps(model, diffusion, optimizer, None, device,
                               torch.amp.GradScaler('cuda', enabled=False))
    for index in range(12):
        target = torch.randn(2, 60, 2, 3, device=device)
        noise = torch.randn_like(target)
        t = torch.randint(10, (2,), device=device)
        y = dict(lengths=torch.full((2,), 60, device=device),
                 parents=[np.array([-1, index], dtype=np.int64)],
                 motion_name=[f'clip-{index}'], action_label=[f'label-{index}'],
                 unused_tensor=torch.empty(index + 1, device=device),
                 resample_speed_cond=torch.full((2, 1, 1, 1), float(index), device=device))
        eager_optimizer.zero_grad(set_to_none=True)
        terms = diffusion.training_losses(eager, target, t, {'y': y}, noise)
        expected = terms['loss'].mean()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        eager_optimizer.step()
        actual = steps.step(target, t, noise, y)
        steps.optimizer_step()
        torch.testing.assert_close(actual, expected)
        for key, value in terms.items():
            torch.testing.assert_close(steps.last_metrics[key], value.detach().mean())
        for captured, reference in zip(model.parameters(), eager.parameters()):
            torch.testing.assert_close(captured, reference)
    assert len(steps.graphs) == 1
    assert set(next(iter(steps.graphs.values()))[4]) == {'lengths', 'resample_speed_cond'}


def test_host_conditions_are_dynamic_tensors_and_metadata_is_excluded():
    steps = EagerCudaGraphSteps(None, None, None, None, torch.device('cpu'), None)
    y = dict(resample_speed_cond=np.array([1., 2.]), is_loop=[True, False],
             action_word_ids=np.array([[1, 2], [2, 3]]),
             parents=[np.array([-1, 0])], motion_name=['first', 'second'])
    first = steps._conditions(y)
    assert set(first) == {'resample_speed_cond', 'is_loop', 'action_word_ids'}
    assert all(torch.is_tensor(value) for value in first.values())
    y['resample_speed_cond'] = np.array([3., 4.])
    second = steps._conditions(y)
    assert steps._signature(first) == steps._signature(second)
    static = steps._clone_tree(first)
    steps._copy_tree(static, second)
    torch.testing.assert_close(static['resample_speed_cond'], second['resample_speed_cond'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires CUDA')
def test_cache_limit_falls_back_to_eager_and_cached_graph_can_resume():
    device = torch.device('cuda')
    model = ConditionedLinear(3, 3).to(device)
    eager = copy.deepcopy(model)
    diffusion = create_latent_diffusion(steps=10, loss_weights=torch.tensor([[.5, 1., 1.5], [1.5, 1., .5]], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eager_optimizer = torch.optim.AdamW(eager.parameters(), lr=1e-3)
    steps = EagerCudaGraphSteps(model, diffusion, optimizer, None, device,
                               torch.amp.GradScaler('cuda', enabled=False), max_graphs=2)
    for batch in (1, 2, 3, 4, 1, 2):
        target = torch.randn(batch, 60, 2, 3, device=device)
        noise = torch.randn_like(target)
        t = torch.randint(10, (batch,), device=device)
        y = dict(lengths=torch.full((batch,), 60, device=device), resample_speed_cond=1.)
        eager_optimizer.zero_grad(set_to_none=True)
        expected = diffusion.training_losses(eager, target, t, {'y': y}, noise)['loss'].mean()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        eager_optimizer.step()
        if batch == 3:
            with pytest.warns(RuntimeWarning, match='additional input signatures run eagerly'):
                actual = steps.step(target, t, noise, y)
        else:
            actual = steps.step(target, t, noise, y)
        steps.optimizer_step()
        torch.testing.assert_close(actual, expected)
        for captured, reference in zip(model.parameters(), eager.parameters()):
            torch.testing.assert_close(captured, reference)
        assert len(steps.graphs) <= 2
    assert len(steps.graphs) == 2


@pytest.mark.parametrize('field', ['ref_active', 'joint_mask'])
def test_filter_does_not_hide_unsupported_conditions(field):
    steps = EagerCudaGraphSteps(None, None, None, None, torch.device('cpu'), None)
    with pytest.raises(ValueError, match='unsupported'):
        steps.step(torch.zeros(1, 60, 2, 3), None, None, {field: torch.ones(1)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph requires CUDA')
def test_real_denoiser_buckets_and_memory_stay_bounded():
    from tests.action_label_test_utils import make_test_bundle, action_cond_fields
    from data_loaders.reference import REST_GEOM_DIM
    from data_loaders.truebones.truebones_utils.joint_struct_features import JOINT_STRUCT_DIM
    from data_loaders.truebones.truebones_utils.dataset_tags import species_code_dim
    from model.latent_motion_denoiser import LatentMotionDenoiser

    device = torch.device('cuda')
    model = LatentMotionDenoiser(
        t5_out_dim=768, slots=2, latent_channels=3, width=144, layers=1,
        heads=4, ff_size=32, skeleton_layers=1, dropout=0,
        joint_name_drop_prob=0, rest_geom_joint_drop_prob=0,
        action_label_cfg_drop_prob=0, action_conditioning=make_test_bundle()).to(device).train()
    eager = copy.deepcopy(model)
    diffusion = create_latent_diffusion(steps=10, loss_weights=torch.tensor([[.5, 1., 1.5], [1.5, 1., .5]], device=device))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=1e-3)
    steps = EagerCudaGraphSteps(model, diffusion, optimizer, None, device,
                               torch.amp.GradScaler('cuda', enabled=False))
    memory = []
    for index in range(20):
        joints = 4 if index % 2 else 6
        y = dict(
            lengths=torch.full((2,), 60, device=device),
            n_joints=torch.tensor([joints - 1, joints], device=device),
            joints_names_embs=torch.randn(2, joints, 768, device=device),
            rest_geom=torch.randn(2, joints, REST_GEOM_DIM, device=device),
            joint_struct=torch.randn(2, joints, JOINT_STRUCT_DIM, device=device),
            graph_dist=torch.zeros(2, joints, joints, dtype=torch.long, device=device),
            joints_relations=torch.zeros(2, joints, joints, dtype=torch.long, device=device),
            species_code=torch.randn(2, species_code_dim(768), device=device),
            is_loop=torch.tensor([bool(index % 3), True], device=device),
            loop_tile_count=torch.tensor([index % 3 + 1, 2], device=device),
            resample_speed_cond=torch.tensor([1. + index / 20, 1.5], device=device),
            temporal_rope_off=torch.tensor([bool(index % 2), False], device=device),
            parents=[np.array([-1, index, index + 1])],
            motion_name=[f'clip-{index}', f'other-{index}'],
            ae_only_tensor=torch.empty(index + 1, device=device))
        labels = ['walk', 'shake'] if index % 3 else ['shake', 'walk']
        y.update({key: value.to(device) if torch.is_tensor(value) else value
                  for key, value in action_cond_fields(labels, ['locomotion', 'stationary']).items()})
        y['action_label_active'] = torch.tensor([bool(index % 2), True], device=device)
        target = torch.randn(2, 60, 2, 3, device=device)
        noise = torch.randn_like(target)
        t = torch.randint(10, (2,), device=device)
        eager_optimizer.zero_grad(set_to_none=True)
        expected = diffusion.training_losses(eager, target, t, {'y': y}, noise)['loss'].mean()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(eager.parameters(), 1.0)
        eager_optimizer.step()
        actual = steps.step(target, t, noise, y)
        steps.optimizer_step()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        torch.cuda.synchronize()
        if index >= 6:  # Both buckets and eager allocator have warmed up.
            memory.append((torch.cuda.memory_allocated(), torch.cuda.memory_reserved()))
        assert len(steps.graphs) <= 2
    assert len(steps.graphs) == 2
    for column in zip(*memory):
        assert max(column) - min(column) < 2 * 1024**2
    for captured, reference in zip(model.parameters(), eager.parameters()):
        torch.testing.assert_close(captured, reference, rtol=1e-3, atol=1e-5)
