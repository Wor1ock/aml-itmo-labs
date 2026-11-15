"""Loss aggregation, masking and weighting."""

from __future__ import annotations

import torch

from ecgmtl.losses.equal import weighted_sum
from ecgmtl.losses.masking import TaskLoss, age_loss, compute_task_losses, diag_loss, rhythm_loss
from ecgmtl.losses.uncertainty import UncertaintyWeighting


def _batch(b: int = 4) -> dict[str, torch.Tensor]:
    return {
        "mask": torch.ones(b, 3, dtype=torch.bool),
        "diag": torch.tensor([[1.0, 0, 0, 0, 0]] * b),
        "rhythm": torch.zeros(b, dtype=torch.long),
        "age": torch.full((b,), 50.0),
    }


def test_equal_weights_sum() -> None:
    losses = {"diag": TaskLoss(torch.tensor(2.0), 4), "rhythm": TaskLoss(torch.tensor(3.0), 4)}
    assert float(weighted_sum(losses, {"diag": 1.0, "rhythm": 2.0})) == 8.0


def test_masked_task_is_skipped() -> None:
    losses = {"diag": TaskLoss(torch.tensor(2.0), 4), "age": TaskLoss(torch.zeros(()), 0)}
    assert float(weighted_sum(losses, {"diag": 1.0, "age": 1.0})) == 2.0


def test_all_tasks_masked_raises() -> None:
    losses = {"age": TaskLoss(torch.zeros(()), 0)}
    try:
        weighted_sum(losses, {"age": 1.0})
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_masked_rows_do_not_change_loss() -> None:
    logits = torch.zeros(4, 5)
    target = torch.zeros(4, 5)
    mask = torch.tensor([True, True, False, False])
    masked = diag_loss(logits, target, mask, None)
    only_valid = diag_loss(logits[:2], target[:2], torch.ones(2, dtype=torch.bool), None)
    assert masked.n_valid == 2
    assert torch.isclose(masked.loss, only_valid.loss)


def test_rhythm_and_age_losses_report_valid_counts() -> None:
    logits = torch.zeros(4, 3)
    mask = torch.tensor([True, False, True, False])
    assert rhythm_loss(logits, torch.zeros(4, dtype=torch.long), mask).n_valid == 2
    pred = torch.zeros(4, 1)
    assert age_loss(pred, torch.zeros(4), mask).n_valid == 2


def test_compute_task_losses_respects_task_list() -> None:
    outputs = {"diag": torch.zeros(4, 5), "rhythm": torch.zeros(4, 3), "age": torch.zeros(4, 1)}
    losses = compute_task_losses(outputs, _batch(), ("diag", "age"))
    assert set(losses) == {"diag", "age"}


def test_uncertainty_weighting_is_finite_and_learns() -> None:
    module = UncertaintyWeighting(("diag", "rhythm", "age"))
    losses = {
        "diag": TaskLoss(torch.tensor(0.7), 4),
        "rhythm": TaskLoss(torch.tensor(1.9), 4),
        "age": TaskLoss(torch.tensor(12.0), 4),
    }
    total = module(losses)
    assert torch.isfinite(total)
    total.backward()
    assert module.log_var.grad is not None
    assert float(module.log_var.grad.abs().sum()) > 0
    weights = module.weights()
    assert set(weights) == {"diag", "rhythm", "age"}


def test_uncertainty_skips_missing_task() -> None:
    module = UncertaintyWeighting(("diag", "age"))
    losses = {"diag": TaskLoss(torch.tensor(0.5), 4), "age": TaskLoss(torch.zeros(()), 0)}
    assert torch.isfinite(module(losses))
