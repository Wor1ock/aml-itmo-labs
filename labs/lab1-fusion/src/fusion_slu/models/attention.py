from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastDropout(nn.Module):
    """Dropout with a uniform-mask implementation.

    ``nn.Dropout`` calls ``aten::bernoulli_``, whose CPU random generator is
    roughly twice as slow as ``torch.rand`` on Windows. The mask below has the
    same Bernoulli(keep) distribution and the same inverted scaling, so it is a
    drop-in replacement that noticeably shortens an epoch on CPU-only machines.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = torch.rand_like(x) < keep
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, hidden: int, dropout: float, max_len: int) -> None:
        super().__init__()
        if hidden % 2 != 0:
            raise ValueError("hidden size must be even for sinusoidal positional encoding")
        self.dropout = FastDropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            -torch.arange(0, hidden, 2, dtype=torch.float32) * math.log(10000.0) / hidden
        )
        encoding = torch.zeros(1, max_len, hidden)
        encoding[0, :, 0::2] = torch.sin(position * div_term)
        encoding[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds positional encoding capacity")
        return cast(torch.Tensor, self.dropout(x + self.pe[:, : x.size(1)].to(dtype=x.dtype)))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with an optional key-padding mask.

    Args:
        query: [batch, query_len, hidden]
        key: [batch, key_len, hidden]
        value: [batch, key_len, hidden]
        valid_mask: bool [batch, key_len], True for real tokens.
    """

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.attention_dropout = FastDropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_len = query.shape[0], query.shape[1]
        q = self.query(query).view(batch, query_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / self.scale
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = self.attention_dropout(attention) @ v
        context = context.transpose(1, 2).reshape(batch, query_len, self.heads * self.head_dim)
        return self.output(context), attention.mean(dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm self- or cross-attention block."""

    def __init__(self, hidden: int, heads: int, feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(hidden, heads, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, feed_forward),
            nn.GELU(),
            nn.Linear(feed_forward, hidden),
        )
        self.dropout = FastDropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_value: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = x if key_value is None else key_value
        attended, attention = self.attention(x, source, source, valid_mask)
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x, attention
