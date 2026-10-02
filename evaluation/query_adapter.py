"""A zero-initialized residual for explicitly selected input-query tokens."""

from __future__ import annotations

import torch
from torch import nn


class QueryTokenResidual(nn.Module):
    def __init__(self, token_ids: list[int], hidden_size: int):
        super().__init__()
        if not token_ids or len(set(token_ids)) != len(token_ids) or hidden_size <= 0:
            raise ValueError('Query IDs must be unique and hidden width positive')
        self.register_buffer('token_ids', torch.tensor(token_ids, dtype=torch.long), persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(token_ids), hidden_size, dtype=torch.float32))

    def forward(self, ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape != (*ids.shape, self.delta.shape[1]):
            raise ValueError('Input-query IDs and embedding shape differ')
        matches = ids[..., None] == self.token_ids
        index = matches.to(torch.int64).argmax(-1)
        residual = torch.nn.functional.embedding(index, self.delta).to(hidden.dtype)
        return hidden + residual * matches.any(-1)[..., None]
