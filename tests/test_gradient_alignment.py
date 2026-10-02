import pytest
import torch

from evaluation.gradient_alignment import alignment


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
