"""Mixed graph replays preserve objectives, gradients and optimizer progress."""
import copy

import pytest
import torch

from model.motion_detail_refiner import MotionDetailRefiner, RefinerConfig
from sample.detail_refinement import tree_to
from tests.test_motion_detail_refiner import fixture
from train.detail_cuda_graph import CudaGraphDetailTrainer
from train.detail_runtime import BatchedDetailRuntime
from train.detail_gradients import DetailGradientMonitor
from train.train_motion_detail import (group_loss, matched_identity_loss,
                                       prepare_groups, training_context)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('identity_weight,max_groups', [(1., 256), (0., 1)])
def test_mixed_cuda_graph_gradients_updates_and_recapture(identity_weight, max_groups):
    groups = []
    for i, (loop, frames) in enumerate([(True, 37), (False, 23), (False, 1)]):
        config, base, y = fixture(loop, frames)
        if i == 1:
            y['parents'] = [[-1, 3, 0, 2]]
        g = tree_to({'y': y, 'gt': base.repeat(i % 2 + 1, 1, 1, 1),
                     'train_base': base.repeat(3, 1, 1, 1), 'validation_base': base}, 'cuda')
        g['y'] = training_context(g['y'], 1)
        groups.append(g)
    scales = prepare_groups(groups, config, batched=True)
    actual = MotionDetailRefiner(config).cuda()
    expected = copy.deepcopy(actual)
    optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-4)
    weights = dict(keep=1., geo=1., reg=.01, stat=1., identity=identity_weight)
    trainer = CudaGraphDetailTrainer(actual, weights, max_groups)
    mixer = DetailGradientMonitor(expected)
    runtime = BatchedDetailRuntime(expected)
    # Replay earlier groups after different topologies/lengths/batch sizes have
    # reused the activation pool; recreate graphs as on checkpoint resume.
    for step, i in enumerate([0, 1, 2, 0, 2, 1, 0, 1]):
        g, scale, seed = groups[i], scales[i], 913 + step
        if step == 4:
            model_state = copy.deepcopy(actual.state_dict())
            optimizer_state = copy.deepcopy(optimizer.state_dict())
            actual = MotionDetailRefiner(config).cuda()
            actual.load_state_dict(model_state)
            optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-4)
            optimizer.load_state_dict(optimizer_state)
            trainer = CudaGraphDetailTrainer(actual, weights, max_groups)
        before = [p.detach().clone() for p in actual.parameters()]
        metrics = trainer.backward(g, scale, seed)
        for p, previous in zip(actual.parameters(), before):
            torch.testing.assert_close(p, previous, rtol=0, atol=0)
        losses, _, _ = group_loss(expected, g, 'train_base', seed, scale, runtime)
        losses['identity'] = (matched_identity_loss(expected, g, seed, scale, runtime)[0]
                              if identity_weight else losses['stat'].new_zeros(()))
        loss = sum(weights[k] * losses[k] for k in trainer.keys)
        diagnostics = mixer.backward(sum(weights[k] * losses[k] for k in trainer.keys if k != 'identity'),
            weights['identity'] * losses['identity'] if identity_weight else None)
        torch.testing.assert_close(metrics, torch.stack([loss.detach()] +
            [losses[k].detach() for k in trainer.keys] +
            [diagnostics[k] for k in trainer.gradient_keys]), atol=2e-6, rtol=2e-5)
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=1e-3)
        for net, opt in [(actual, optimizer), (expected, reference_optimizer)]:
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        for a, b in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    assert len(trainer.graphs) == min(max_groups, 3)
