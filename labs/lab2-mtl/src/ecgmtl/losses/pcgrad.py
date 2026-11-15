"""PCGrad (Yu et al., 2020) — projection of conflicting task gradients.

Implemented by hand because the lab asks to explain the formula, not to call a library:
for every task gradient ``g_i`` and every other task gradient ``g_j`` with a negative dot
product, replace ``g_i`` with ``g_i - (g_i . g_j / ||g_j||^2) * g_j``. Only gradients of
the shared encoder are projected; head gradients never conflict by construction.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def flatten(tensors: Sequence[Tensor | None], params: Sequence[nn.Parameter]) -> Tensor:
    """Concatenate gradients (``None`` becomes zeros) into one vector."""
    parts = [
        t.reshape(-1) if t is not None else torch.zeros(p.numel(), device=p.device)
        for t, p in zip(tensors, params, strict=True)
    ]
    return torch.cat(parts) if parts else torch.zeros(0)


def unflatten(vector: Tensor, params: Sequence[nn.Parameter]) -> list[Tensor]:
    """Split a flat gradient vector back into per-parameter tensors."""
    grads = []
    offset = 0
    for param in params:
        size = param.numel()
        grads.append(vector[offset : offset + size].view_as(param))
        offset += size
    return grads


def task_gradients(
    losses: dict[str, Tensor],
    params: Sequence[nn.Parameter],
    retain_graph: bool = True,
) -> tuple[dict[str, Tensor], list[nn.Parameter]]:
    """Flat gradient of every task with respect to the shared parameters."""
    leaves = [p for p in params if p.requires_grad]
    grads: dict[str, Tensor] = {}
    for name, loss in losses.items():
        raw = torch.autograd.grad(loss, leaves, retain_graph=retain_graph, allow_unused=True)
        grads[name] = flatten(raw, leaves)
    return grads, leaves


def pcgrad_backward(
    losses: dict[str, Tensor],
    shared: Sequence[nn.Parameter],
    heads: Sequence[nn.Parameter],
    generator: torch.Generator | None = None,
) -> tuple[int, Tensor]:
    """Backward pass with PCGrad on shared parameters and plain sum on the heads.

    Returns the number of applied projections and the sum of task losses (for logging).
    """
    leaves = list(shared) + list(heads)
    raw = {
        name: torch.autograd.grad(
            loss, leaves, retain_graph=True, allow_unused=True, create_graph=False
        )
        for name, loss in losses.items()
    }
    n_shared = len(shared)
    shared_grads = {name: flatten(grads[:n_shared], shared) for name, grads in raw.items()}
    flat, n_projected = project_conflicts(shared_grads, generator)
    assign_gradients(shared, flat)
    for grads in raw.values():
        for grad, param in zip(grads[n_shared:], heads, strict=True):
            if grad is None:
                continue
            param.grad = grad.clone() if param.grad is None else param.grad + grad
    return n_projected, torch.stack(list(losses.values())).sum()


def project_conflicts(grads: dict[str, Tensor], generator: torch.Generator | None = None) -> tuple[Tensor, int]:
    """Return the summed shared gradient after projection and the number of projections."""
    names = list(grads)
    projected = {name: grads[name].clone() for name in names}
    n_projected = 0
    for i, name_i in enumerate(names):
        order = [n for n in names if n != name_i]
        if generator is not None:
            order = [order[j] for j in torch.randperm(len(order), generator=generator).tolist()]
        for name_j in order:
            g_i, g_j = projected[name_i], grads[name_j]
            dot = torch.dot(g_i, g_j)
            if dot < 0:
                projected[name_i] = g_i - (dot / (g_j.norm() ** 2 + 1e-12)) * g_j
                n_projected += 1
    return torch.stack([projected[name] for name in names]).sum(dim=0), n_projected


def assign_gradients(params: Sequence[nn.Parameter], flat: Tensor) -> None:
    for param, grad in zip(params, unflatten(flat, params), strict=True):
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad += grad


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def conflict_matrix(grads: dict[str, Tensor]) -> dict[str, float]:
    """Pairwise cosine similarity between task gradients on the shared encoder."""
    names = list(grads)
    out: dict[str, float] = {}
    for i, name_i in enumerate(names):
        for name_j in names[i + 1 :]:
            out[f"{name_i}__{name_j}"] = cosine_similarity(grads[name_i], grads[name_j])
    return out
