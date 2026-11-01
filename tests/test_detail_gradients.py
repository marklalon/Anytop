"""Task diagnostics must preserve the weighted objective gradient exactly."""
import pytest
import torch
import copy

from train.detail_gradients import DetailGradientMonitor


@pytest.mark.parametrize('a,b', [([1., 0.], [-1., 1.]), ([1., 2.], [2., 1.]),
                                ([1., 0.], [-3., 0.]), ([0., 0.], [1., 2.]),
                                ([1., 2.], [0., 0.])])
def test_weighted_sum_and_diagnostics_include_conflicts_and_zero_gradients(a, b):
    net = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(0.)) for _ in a])
    mixer = DetailGradientMonitor(net)
    ga, gb = torch.tensor(a), torch.tensor(b)
    metrics = mixer.backward(sum(p * g for p, g in zip(net, ga)),
                             sum(p * g for p, g in zip(net, gb)))
    merged = torch.stack([p.grad for p in net])
    assert torch.isfinite(merged).all()
    torch.testing.assert_close(metrics['grad_merged_main_dot'], merged @ ga)
    torch.testing.assert_close(metrics['grad_merged_identity_dot'], merged @ gb)
    torch.testing.assert_close(merged, ga + gb, rtol=0, atol=0)
    # A fresh backward must not accumulate prior gradients or scratch state.
    mixer.backward(sum(p * g for p, g in zip(net, ga)))
    torch.testing.assert_close(torch.stack([p.grad for p in net]), ga)


def test_sum_mode_preserves_conflicting_weighted_gradients():
    net = torch.nn.Linear(2, 1, bias=False)
    mixer = DetailGradientMonitor(net)
    metrics = mixer.backward(net.weight.sum(), -2 * net.weight.sum())
    torch.testing.assert_close(net.weight.grad, -torch.ones_like(net.weight))
    assert metrics['grad_conflict'] == 1


def test_monitor_matches_one_backward_of_the_original_real_objective():
    from tests.test_motion_detail_refiner import fixture
    from model.motion_detail_refiner import MotionDetailRefiner
    from train.train_motion_detail import prepare_groups, group_loss, matched_identity_loss
    from train.detail_runtime import BatchedDetailRuntime
    config, base, y = fixture(True, 30)
    generated = base.repeat(2, 1, 1, 1)
    generated[:, 1, 0] += .1
    generated[:, 1, 3] += .02
    group = {'y': y, 'gt': base, 'train_base': generated, 'validation_base': base}
    scale = prepare_groups([group], config, batched=True)[0]
    actual = MotionDetailRefiner(config)
    expected = copy.deepcopy(actual)
    def objectives(net):
        runtime = BatchedDetailRuntime(net)
        losses = group_loss(net, group, 'train_base', 99, scale, runtime)[0]
        main = losses['stat'] + losses['keep'] + losses['geo'] + .01 * losses['reg']
        identity = matched_identity_loss(net, group, 99, scale, runtime)[0]
        return main, identity
    main, identity = objectives(expected)
    (main + identity).backward()
    main, identity = objectives(actual)
    DetailGradientMonitor(actual).backward(main, identity)
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-4)
