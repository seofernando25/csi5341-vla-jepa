import torch
from safetensors.torch import load_file, save_file

from evaluation.query_adapter import QueryTokenResidual


def test_query_rows_are_only_trainable_additions_and_survive_round_trip(tmp_path):
    adapter = QueryTokenResidual([8, 9], 3)
    ids = torch.tensor([[1, 8, 9, 8]])
    hidden = torch.randn(1, 4, 3, dtype=torch.bfloat16)
    assert torch.equal(adapter(ids, hidden), hidden)
    adapter(ids, hidden).sum().backward()
    assert torch.equal(adapter.delta.grad, torch.tensor([[2.] * 3, [1.] * 3]))
    assert set(adapter.state_dict()) == {'delta'}
    with torch.no_grad():
        adapter.delta.copy_(torch.tensor([[.125] * 3, [-.25] * 3]))
    path = tmp_path / 'query.safetensors'
    save_file(adapter.state_dict(), path)
    restored = QueryTokenResidual([8, 9], 3)
    restored.load_state_dict(load_file(path), strict=True)
    assert torch.equal(restored.delta, adapter.delta)
    assert torch.equal(restored(ids, hidden), adapter(ids, hidden))
    assert torch.equal(restored(ids, hidden)[:, :1], hidden[:, :1])
