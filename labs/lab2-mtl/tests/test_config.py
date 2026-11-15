"""Config loading: defaults, CLI overrides, unknown keys, shipped files."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecgmtl.config import ConfigError, load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_fill_missing_sections(tmp_path) -> None:
    cfg = load_config(_write(tmp_path, "train:\n  epochs: 2\n"))
    assert cfg.train.epochs == 2
    assert cfg.loss.strategy == "equal"
    assert cfg.data.hz == 100
    assert len(cfg.split.test_folds) == len(cfg.split.val_folds) == 5


def test_cli_overrides_win(tmp_path) -> None:
    cfg = load_config(
        _write(tmp_path, "train:\n  epochs: 2\n"),
        {"train.epochs": "4", "loss.strategy": "pcgrad"},
    )
    assert cfg.train.epochs == 4
    assert cfg.loss.strategy == "pcgrad"


def test_unknown_key_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "train:\n  epoch: 2\n"))


def test_bad_strategy_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "loss:\n  strategy: magic\n"))


def test_overlapping_split_is_rejected(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, "split:\n  test_folds: [10, 9]\n  val_folds: [9, 8]\n"))


def test_shipped_configs_parse() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("default.yaml", "smoke.yaml"):
        path = root / "configs" / name
        if path.exists():
            cfg = load_config(path)
            assert cfg.train.epochs >= 1
            assert set(cfg.loss.weights) >= set(cfg.loss.tasks)


def test_smoke_config_is_small() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "smoke.yaml"
    if not path.exists():
        pytest.skip("smoke config is not part of the wheel")
    cfg = load_config(path)
    assert cfg.train.epochs <= 2
    assert 0 < cfg.train.limit_train <= 1024
    assert cfg.experiments.folds == 1
