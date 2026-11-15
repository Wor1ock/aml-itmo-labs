"""Loss aggregation strategies and masking."""

from .masking import TaskLoss, compute_task_losses

__all__ = ["TaskLoss", "compute_task_losses"]
