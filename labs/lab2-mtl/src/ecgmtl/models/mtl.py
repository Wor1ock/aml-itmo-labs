"""Multi-task model: one shared encoder, one head per task."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import DataConfig, ModelConfig
from .encoder import SharedEncoder
from .heads import make_heads


class MtlEcg(nn.Module):
    """Shared representation ``h`` followed by independent task heads."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        n_diag: int,
        n_rhythm: int,
        tasks: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(model_cfg, data_cfg.n_leads, data_cfg.signal_len)
        self.heads = make_heads(
            model_cfg.hidden, n_diag, n_rhythm, model_cfg.head_hidden, model_cfg.dropout
        )
        self.tasks = tasks
        self.disabled = [name for name in self.heads if name not in tasks]

    def forward(self, signal: Tensor) -> dict[str, Tensor]:
        h = self.encoder(signal)  # (B, hidden)
        return {name: self.heads[name](h) for name in self.tasks}

    def shared_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def head_parameters(self, task: str) -> list[nn.Parameter]:
        return list(self.heads[task].parameters())

    def head_parameters_all(self) -> list[nn.Parameter]:
        return [p for name in self.tasks for p in self.head_parameters(name)]

    def count_parameters(self) -> dict[str, int]:
        shared = sum(p.numel() for p in self.shared_parameters())
        per_task = {
            name: sum(p.numel() for p in self.head_parameters(name)) for name in self.tasks
        }
        return {"shared": shared, **per_task, "total": sum(per_task.values()) + shared}


def build_model(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    n_diag: int,
    n_rhythm: int,
    tasks: tuple[str, ...],
) -> MtlEcg:
    return MtlEcg(model_cfg, data_cfg, n_diag, n_rhythm, tasks)
