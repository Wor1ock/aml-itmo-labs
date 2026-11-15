"""Per-task losses with masking of missing labels.

A record without a rhythm statement or without a plausible age must not push its head:
``n_valid`` counts the rows that actually carry the label, and a task with no valid row
in the batch is dropped from the total loss instead of contributing a fake zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import TASKS

DIAG, RHYTHM, AGE = 0, 1, 2


@dataclass(frozen=True)
class TaskLoss:
    loss: Tensor
    n_valid: int

    def detached(self) -> dict[str, float | int]:
        return {"loss": float(self.loss.detach()), "n_valid": self.n_valid}


def _masked_mean(values: Tensor, mask: Tensor) -> TaskLoss:
    """Average per-sample values over rows where the label exists."""
    n_valid = int(mask.sum().item())
    if n_valid == 0:
        return TaskLoss(loss=torch.zeros((), device=values.device), n_valid=0)
    return TaskLoss(loss=values[mask].mean(), n_valid=n_valid)


def diag_loss(logits: Tensor, targets: Tensor, mask: Tensor, pos_weight: Tensor | None) -> TaskLoss:
    # logits: (B, 5), targets: (B, 5), mask: (B,)
    raw = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    return _masked_mean(raw.mean(dim=1), mask)


def rhythm_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> TaskLoss:
    # logits: (B, K), targets: (B,) long
    raw = F.cross_entropy(logits, targets, reduction="none")
    return _masked_mean(raw, mask)


def age_loss(pred: Tensor, targets: Tensor, mask: Tensor) -> TaskLoss:
    # pred: (B, 1) -> (B,), targets: (B,)
    raw = F.smooth_l1_loss(pred.squeeze(-1), targets, reduction="none")
    return _masked_mean(raw, mask)


def compute_task_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    tasks: tuple[str, ...],
    diag_pos_weight: Tensor | None = None,
) -> dict[str, TaskLoss]:
    """Loss for every active task; tasks absent from ``outputs`` are skipped."""
    mask = batch["mask"]
    losses: dict[str, TaskLoss] = {}
    if "diag" in tasks:
        losses["diag"] = diag_loss(outputs["diag"], batch["diag"], mask[:, DIAG], diag_pos_weight)
    if "rhythm" in tasks:
        losses["rhythm"] = rhythm_loss(outputs["rhythm"], batch["rhythm"], mask[:, RHYTHM])
    if "age" in tasks:
        losses["age"] = age_loss(outputs["age"], batch["age"], mask[:, AGE])
    return losses


def active_tasks(losses: dict[str, TaskLoss]) -> tuple[str, ...]:
    """Task order that is actually present in this batch."""
    return tuple(name for name in TASKS if name in losses and losses[name].n_valid > 0)
