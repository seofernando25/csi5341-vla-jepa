import pytest
import torch

from evaluation.gradient_alignment import alignment, query_residual


def test_gradient_alignment_handles_conflict_and_unused_parameters():
    result = alignment([torch.tensor([3., 4.]), None],
                       [torch.tensor([-6., -8.]), torch.tensor([2.])])
    assert result['action_gradient_l2'] == pytest.approx(5.)
    assert result['weighted_world_gradient_l2'] == pytest.approx(104 ** .5)
    assert result['cosine'] == pytest.approx(-50 / (5 * 104 ** .5))
    assert result['action_parameters_with_grad'] == 1
    assert result['world_parameters_with_grad'] == 2
    assert result['finite']
    assert alignment([torch.zeros(2)], [torch.zeros(2)])['cosine'] is None


def test_query_probe_preserves_other_tokens_and_connects_only_matching_rows():
    ids = torch.tensor([[1, 8, 9, 8]])
    query_ids = torch.tensor([8, 9, 10])
    hidden = torch.randn(1, 4, 2)
    delta = torch.zeros(3, 2, requires_grad=True)
    result = query_residual(ids, query_ids, delta, hidden)
    assert torch.equal(result, hidden)
    result.sum().backward()
    assert torch.equal(delta.grad, torch.tensor([[2., 2.], [1., 1.], [0., 0.]]))
    with torch.no_grad():
        delta.fill_(1.)
    changed = query_residual(ids, query_ids, delta, hidden)
    assert torch.equal(changed[:, :1], hidden[:, :1])
    assert torch.equal(changed[:, 1:], hidden[:, 1:] + 1.)
