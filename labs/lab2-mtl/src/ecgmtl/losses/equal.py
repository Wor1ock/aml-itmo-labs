"""Equal and manually weighted loss aggregation: ``L = sum_i w_i * L_i``."""

from __future__ import annotations

from torch import Tensor

from .masking import TaskLoss


def weighted_sum(losses: dict[str, TaskLoss], weights: dict[str, float]) -> Tensor:
    """Sum of weighted task losses; tasks without valid labels in the batch are skipped."""
    total: Tensor | None = None
    for name, task_loss in losses.items():
        if task_loss.n_valid == 0:
            continue
        term = weights.get(name, 1.0) * task_loss.loss
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("no task has valid labels in this batch")
    return total
