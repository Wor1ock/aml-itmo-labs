from __future__ import annotations

from pathlib import Path

import pytest

from fusion_slu.config import ConfigError, load_config, parse_condition


def test_smoke_config_extends_default() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.data.cache_tag == "smoke"
    assert config.training.epochs == 2
    assert config.model.hid_dim == 64
    assert config.report.output.name == "smoke.html"


def test_cli_override() -> None:
    config = load_config("configs/smoke.yaml", ["training.epochs=1", "model.modality=audio"])
    assert config.training.epochs == 1
    assert config.model.modality == "audio"


def test_unknown_key_fails() -> None:
    with pytest.raises(ConfigError):
        load_config("configs/smoke.yaml", ["training.unknown_key=1"])


def test_condition_parser() -> None:
    assert parse_condition("clean").kind == "clean"
    assert parse_condition("text_drop_25").value == pytest.approx(0.25)
    assert parse_condition("audio_noise_5db").value == pytest.approx(5.0)
    with pytest.raises(ConfigError):
        parse_condition("nonsense")


def test_paths_are_absolute() -> None:
    config = load_config("configs/default.yaml")
    assert config.data.root.is_absolute()
    assert config.report.output.is_absolute()
    assert isinstance(config.results_dir, Path)
