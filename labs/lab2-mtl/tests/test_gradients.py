"""Gradient flow, conflict computation and PCGrad projection."""

from __future__ import annotations

import pytest
import torch

from ecgmtl.losses.masking import compute_task_losses
from ecgmtl.losses.pcgrad import (
    conflict_matrix,
    cosine_similarity,
    pcgrad_backward,
    project_conflicts,
)
from ecgmtl.models.mtl import build_model


@pytest.fixture
def model_and_batch(tiny_config):
    torch.manual_seed(0)
    model = build_model(tiny_config.model, tiny_config.data, 5, 3, ("diag", "rhythm", "age"))
    batch = {
        "signal": torch.randn(4, 12, 1000),
        "diag": torch.tensor([[1.0, 0, 0, 0, 0]] * 2 + [[0, 1, 0, 0, 0]] * 2),
        "rhythm": torch.tensor([0, 1, 2, 0]),
        "age": torch.tensor([30.0, 40.0, 50.0, 60.0]),
        "mask": torch.ones(4, 3, dtype=torch.bool),
    }
    return model, batch


def test_gradients_reach_shared_encoder(model_and_batch) -> None:
    model, batch = model_and_batch
    outputs = model(batch["signal"])
    losses = compute_task_losses(outputs, batch, ("diag", "rhythm", "age"))
    total = sum(task.loss for task in losses.values() if task.n_valid > 0)
    total.backward()
    shared = model.shared_parameters()
    assert all(p.grad is not None for p in shared)
    assert sum(float(p.grad.abs().sum()) for p in shared) > 0


def test_pcgrad_projects_opposite_gradients() -> None:
    grads = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([-1.0, 0.0])}
    total, count = project_conflicts(grads)
    assert count == 2
    assert torch.allclose(total, torch.zeros(2)), total


def test_pcgrad_keeps_agreeing_gradients() -> None:
    grads = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([2.0, 0.0])}
    total, count = project_conflicts(grads)
    assert count == 0
    assert torch.allclose(total, torch.tensor([3.0, 0.0]))


def test_cosine_and_conflict_matrix() -> None:
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0
    assert cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    matrix = conflict_matrix(
        {"diag": torch.tensor([1.0, 0.0]), "age": torch.tensor([-1.0, 0.0])}
    )
    assert set(matrix) == {"diag__age"}
    assert matrix["diag__age"] == pytest.approx(-1.0)


def test_pcgrad_backward_fills_head_and_shared_grads(model_and_batch) -> None:
    model, batch = model_and_batch
    outputs = model(batch["signal"])
    losses = compute_task_losses(outputs, batch, ("diag", "rhythm", "age"))
    active = {name: task.loss for name, task in losses.items() if task.n_valid > 0}
    n_projected, total = pcgrad_backward(active, model.shared_parameters(), model.head_parameters_all())
    assert n_projected >= 0
    assert torch.isfinite(total)
    assert all(p.grad is not None for p in model.shared_parameters())
    for name in model.tasks:
        assert all(p.grad is not None for p in model.head_parameters(name))


def test_masked_task_gets_no_gradient(model_and_batch) -> None:
    model, batch = model_and_batch
    batch["mask"][:, 1] = False  # no rhythm labels in this batch
    outputs = model(batch["signal"])
    losses = compute_task_losses(outputs, batch, ("diag", "rhythm", "age"))
    active = {name: task.loss for name, task in losses.items() if task.n_valid > 0}
    assert "rhythm" not in active
    pcgrad_backward(active, model.shared_parameters(), model.head_parameters_all())
    assert all(p.grad is None for p in model.head_parameters("rhythm"))
