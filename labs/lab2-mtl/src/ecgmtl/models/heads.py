"""Task-specific heads: one small MLP per task on top of the shared representation."""

from __future__ import annotations

from torch import Tensor, nn


class TaskHead(nn.Module):
    """Two-layer MLP with its own capacity, independent of the other heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)  # (B, out_dim)


def make_heads(hidden: int, n_diag: int, n_rhythm: int, cfg_hidden: int, dropout: float) -> nn.ModuleDict:
    """diag: multi-label logits, rhythm: single-label logits, age: one scalar."""
    return nn.ModuleDict(
        {
            "diag": TaskHead(hidden, n_diag, cfg_hidden, dropout),
            "rhythm": TaskHead(hidden, n_rhythm, cfg_hidden, dropout),
            "age": TaskHead(hidden, 1, cfg_hidden // 2, dropout),
        }
    )
