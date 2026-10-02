import pytest
import torch
from torch import nn

from evaluation.models import cast_inference_policy


def fixture_policy():
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(2))
    model.rotary_emb = nn.Module()
    freq = torch.tensor([0.001388192, 0.9237417], dtype=torch.float32)
    model.rotary_emb.register_buffer('inv_freq', freq.clone(), persistent=False)
    model.rotary_emb.register_buffer('original_inv_freq', freq.clone(), persistent=False)
    return model, freq


def test_native_frequency_values_survive_parameter_cast():
    model, freq = fixture_policy()
    preserved = cast_inference_policy(model, 'cpu', 'native_rope')
    assert len(preserved) == 2
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.rotary_emb.inv_freq, freq)
    assert model.rotary_emb.inv_freq.dtype == torch.float32
    assert 'rotary_emb.inv_freq' not in model.state_dict()
    assert not model.weight.requires_grad


def test_legacy_and_already_rounded_buffers_are_explicit():
    model, freq = fixture_policy()
    assert cast_inference_policy(model, 'cpu') == {}
    assert model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert not torch.equal(model.rotary_emb.inv_freq.float(), freq)
    with pytest.raises(ValueError, match='already rounded'):
        cast_inference_policy(model, 'cpu', 'native_rope')
