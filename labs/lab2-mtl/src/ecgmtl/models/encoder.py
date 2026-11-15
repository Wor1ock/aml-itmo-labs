"""Shared encoder: a small 1D CNN over 12 leads with attention pooling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import ModelConfig


class ConvBlock(nn.Module):
    """Two convolutions, batch norm, GELU and max-pooling in time."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Learned weighted average over time."""

    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, D)
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)  # (B, T)
        return torch.einsum("bt,btd->bd", weights, x)


class SharedEncoder(nn.Module):
    """Maps (B, leads, T) to a single common representation h of shape (B, hidden)."""

    def __init__(self, cfg: ModelConfig, n_leads: int, signal_len: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = n_leads
        for out_channels, kernel in zip(cfg.channels, cfg.kernels, strict=True):
            blocks.append(ConvBlock(in_channels, out_channels, kernel))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool: nn.Module = (
            AttentionPool(cfg.channels[-1]) if cfg.pool == "attention" else nn.Identity()
        )
        self.proj = nn.Sequential(
            nn.Linear(cfg.channels[-1], cfg.hidden),
            nn.LayerNorm(cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.hidden = cfg.hidden
        self.out_time = signal_len // (2 ** len(cfg.channels))

    def forward(self, signal: Tensor) -> Tensor:
        x = self.blocks(signal)  # (B, C, T')
        x = x.transpose(1, 2)  # (B, T', C)
        if isinstance(self.pool, nn.Identity):
            x = x.mean(dim=1)  # (B, C)
        else:
            x = self.pool(x)  # (B, C)
        return self.proj(x)  # (B, hidden)
