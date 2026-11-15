"""Uncertainty weighting (Kendall et al., 2018).

``L = sum_i L_i / (2 * sigma_i^2) + log sigma_i`` with learnable ``log sigma_i^2``.
The regression term is scaled down automatically because its raw magnitude differs from a
cross-entropy by orders of magnitude.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .masking import TaskLoss


class UncertaintyWeighting(nn.Module):
    """Learns one log-variance per task; parameters are trained with the model."""

    def __init__(self, tasks: tuple[str, ...]) -> None:
        super().__init__()
        self.tasks = tasks
        self.log_var = nn.Parameter(torch.zeros(len(tasks)))

    def forward(self, losses: dict[str, TaskLoss]) -> Tensor:
        total: Tensor | None = None
        for i, name in enumerate(self.tasks):
            task_loss = losses.get(name)
            if task_loss is None or task_loss.n_valid == 0:
                continue
            s = self.log_var[i]
            term = task_loss.loss * torch.exp(-s) / 2.0 + s / 2.0
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError("no task has valid labels in this batch")
        return total

    def weights(self) -> dict[str, float]:
        """Effective 1 / (2 sigma^2) per task, for logging."""
        with torch.no_grad():
            factors = torch.exp(-self.log_var) / 2.0
        return {name: float(factors[i]) for i, name in enumerate(self.tasks)}
