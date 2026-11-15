"""Head shapes and parameter accounting."""

from __future__ import annotations

import torch

from ecgmtl.data.labels import SUPERCLASSES
from ecgmtl.models.mtl import build_model


def test_head_shapes(tiny_config) -> None:
    model = build_model(tiny_config.model, tiny_config.data, 5, 3, ("diag", "rhythm", "age"))
    signal = torch.randn(2, 12, 1000)
    out = model(signal)
    assert out["diag"].shape == (2, len(SUPERCLASSES))
    assert out["rhythm"].shape == (2, 3)
    assert out["age"].shape == (2, 1)


def test_single_task_model_has_one_head(tiny_config) -> None:
    model = build_model(tiny_config.model, tiny_config.data, 5, 3, ("diag",))
    out = model(torch.randn(1, 12, 1000))
    assert set(out) == {"diag"}
    params = model.count_parameters()
    assert params["total"] == params["shared"] + params["diag"]


def test_shared_part_is_shared(tiny_config) -> None:
    model = build_model(tiny_config.model, tiny_config.data, 5, 3, ("diag", "rhythm", "age"))
    heads = {name: set(map(id, model.head_parameters(name))) for name in model.tasks}
    shared = set(map(id, model.shared_parameters()))
    for task, params in heads.items():
        assert not (params & shared), f"{task} head shares encoder parameters"
    assert heads["diag"].isdisjoint(heads["rhythm"])
    assert heads["rhythm"].isdisjoint(heads["age"])
